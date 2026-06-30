# -*- coding: utf-8 -*-
import os
import sys
import time
import csv
import argparse
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

# ==========================================
#  ( HPO )
# ==========================================
def forceDeterministicSystem(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

forceDeterministicSystem(42)

# ==========================================
#  eval_utils
# ==========================================
currentDir = os.path.dirname(os.path.abspath(__file__))
evalDir = os.path.join(currentDir, 'eval')
if evalDir not in sys.path:
    sys.path.append(evalDir)

import eval_utils 

from data.dataset_decoupled_v10 import ProbForceDataset  
from models.decoupled_predictor_v2 import DecoupledForcePredictor  
from models.asgd_force_solver_v2 import BatchedASGDForceSolver as ASGDForceSolver
from losses.doubled_loss import IsomorphicRoleMetricLoss
from utils import seed_everything, load_config, save_config, EarlyStopping

# ==========================================
# 
# ==========================================
class UnifiedRankingOrbitalField(nn.Module):
    def __init__(self, margin_rank=0.05, sigma_ratio=1.0, neg_sample_multiplier=2.0, tau_sharp=0.05, alpha_deg=0.5, deg_reg_weight=0.1, eps=1e-8):
        super().__init__()
        self.marginRank = margin_rank
        self.sigmaRatio = sigma_ratio
        self.negSampleMultiplier = neg_sample_multiplier
        self.tauSharp = tau_sharp
        self.alphaDeg = alpha_deg
        self.degRegWeight = deg_reg_weight
        self.eps = eps
        self.deltaArBarrier = 1e-2
        self.hubBoostGain = 2.0
        self.tauSmoothAr = 2.0

    def forward(self, pos, edge_index, w_edge=None, l_edge=None, q_node=None,
                lambda_kl=1.0, lambda_lnp=0.0, lambda_ar=0.0, lambda_eu=0.0,
                lambda_scale_lock=0.3, lambda_reg=0.01):
        
        device = pos.device
        N = pos.shape[0]
        row, col = edge_index
        numEdges = row.shape[0]

        if N <= 1 or numEdges == 0:
            return torch.tensor(0.0, device=device), {
                "kl_loss": 0.0, "lnp_loss": 0.0, "ar_loss": 0.0,
                "eu_loss": 0.0, "scale_lock_loss": 0.0, "reg_loss": 0.0
            }

        posCentered = pos - pos.mean(dim=0, keepdim=True)
        distSq = ((posCentered.unsqueeze(0) - posCentered.unsqueeze(1)) ** 2).sum(dim=-1)
        distMatrix = torch.sqrt(torch.clamp(distSq, min=self.eps))
        eyeMat = torch.eye(N, dtype=torch.bool, device=device)

        adj = torch.zeros((N, N), device=device)
        adj[row, col] = 1.0
        adj[col, row] = 1.0
        adj.fill_diagonal_(0.0)
        
        deg = adj.sum(dim=-1, keepdim=True)       
        degFlat = deg.squeeze(-1)                
        meanDeg = deg.mean() + self.eps

        neighborMask = adj.bool()
        nonNeighborMask = (~neighborMask) & (~eyeMat)

        metrics = {}
        totalLoss = torch.tensor(0.0, device=device)

        edgeLens = distMatrix[row, col]
        meanEdgeLen = edgeLens.mean() + self.eps

        # 1. 
        lossScaleLock = (meanEdgeLen - 1.0) ** 2
        if lambda_scale_lock > 0:
            totalLoss = totalLoss + lambda_scale_lock * lossScaleLock
        metrics["scale_lock_loss"] = lossScaleLock.item()

        # 2. KL 
        if lambda_kl > 0:
            alpha = deg / meanDeg  
            P_uniform = neighborMask.float() / (neighborMask.sum(dim=-1, keepdim=True) + self.eps)
            bandwidthSqNorm = (1.0 * self.sigmaRatio) ** 2
            qUnnorm = 1.0 / (1.0 + distSq / bandwidthSqNorm)
            qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
            qCond = qUnnorm / (qUnnorm.sum(dim=-1, keepdim=True) + self.eps)
            klMatrix = P_uniform * torch.log((P_uniform + 1e-12) / (qCond + self.eps))
            lossKL = torch.mean(klMatrix.sum(dim=-1, keepdim=True) * alpha)
            totalLoss = totalLoss + lambda_kl * lossKL
            metrics["kl_loss"] = lossKL.item()
        else:
            metrics["kl_loss"] = 0.0
        
        # 3.  (LNP - STE)
        if lambda_lnp > 0:
            distSorted, _ = torch.sort(distMatrix, dim=1)
            tau = distSorted[torch.arange(N, device=device), degFlat.long().clamp(max=N-1)]
            tauInvExpanded = (1.0 / (tau + self.eps)).unsqueeze(1)
            distMatrixNorm = distMatrix * tauInvExpanded
            
            kMax = int(torch.clamp(degFlat.max() * self.negSampleMultiplier, min=5, max=N-1).item())
            negDist = distMatrix.clone().masked_fill_(~nonNeighborMask, 1e9)
            topkNonVals, _ = torch.topk(negDist, k=kMax, dim=1, largest=False)
            topkNonValsNorm = topkNonVals * tauInvExpanded
            
            targetKPerRow = (degFlat * self.negSampleMultiplier).long().clamp(min=1, max=kMax)
            colIndices = torch.arange(kMax, device=device).unsqueeze(0)
            dynamicDegMask = colIndices < targetKPerRow.unsqueeze(1)
            
            violation = self.marginRank - (topkNonValsNorm.unsqueeze(2) - distMatrixNorm.unsqueeze(1))
            
            nonMarginMask = (topkNonValsNorm < 1.0 + self.marginRank) & dynamicDegMask
            neighborEdgeMask = neighborMask & (distMatrixNorm > 1.0 - self.marginRank)
            pairMask = nonMarginMask.unsqueeze(2) & neighborEdgeMask.unsqueeze(1)
            
            nonMarginSoft = torch.sigmoid((1.0 + self.marginRank - topkNonValsNorm) / self.tauSharp)
            neighborEdgeSoft = torch.sigmoid((distMatrixNorm - (1.0 - self.marginRank)) / self.tauSharp)
            pairMaskSoft = nonMarginSoft.unsqueeze(2) * neighborEdgeSoft.unsqueeze(1) * dynamicDegMask.unsqueeze(2) * neighborMask.unsqueeze(1)
            
            pairMaskSTE = pairMask.float() + (pairMaskSoft - pairMaskSoft.detach())
            smoothViolation = F.softplus(violation, beta=1.0 / self.tauSharp)
            lossPerNode = (pairMaskSTE * smoothViolation).sum(dim=(1, 2)) / (degFlat + self.eps)
            
            hubThreshold = degFlat.mean() + degFlat.std()
            isHubMask = degFlat > hubThreshold
            if not isHubMask.any():
                isHubMask = (degFlat == degFlat.max())
            isNormalMask = ~isHubMask
            
            hasCandidatePairs = degFlat > 0
            maskNormalValid = isNormalMask & hasCandidatePairs
            maskHubValid = isHubMask & hasCandidatePairs
            
            lossNormal = lossPerNode[maskNormalValid].mean() if maskNormalValid.sum() > 0 else torch.tensor(0.0, device=device)
            lossHub = lossPerNode[maskHubValid].mean() if maskHubValid.sum() > 0 else torch.tensor(0.0, device=device)
            
            gammaG = max(0.55, min(0.95, 1.0 - (isHubMask.sum().item() / N)))
            lossLNP = (gammaG * lossHub + (1.0 - gammaG) * lossNormal)
            totalLoss = totalLoss + lambda_lnp * lossLNP
            metrics["lnp_loss"] = lossLNP.item()
        else:
            metrics["lnp_loss"] = 0.0

        # 4.  (AR)
#        if lambda_ar > 0:
#            tri = neighborMask.unsqueeze(2) * neighborMask.unsqueeze(1) * (~eyeMat.unsqueeze(0)).float()
#            diffNpd = posCentered.unsqueeze(0) - posCentered.unsqueeze(1)
#            unit = diffNpd / (distMatrix.unsqueeze(-1) + self.eps)
#            cos = torch.einsum("aid,ajd->aij", unit, unit)
#            violationAr = F.relu(cos - (1.0 - 0.05))
#            perNodeAr = (tri * violationAr).sum(dim=(1, 2)) / (degFlat * (degFlat - 1.0)).clamp(min=1.0)
#            wAr = torch.log(degFlat + 1.0)
#            wAr = wAr / (wAr.mean() + self.eps)
#            lossAr = (perNodeAr * wAr).mean()
#            totalLoss = totalLoss + lambda_ar * lossAr
#            metrics["ar_loss"] = lossAr.item()
#        else:
#            metrics["ar_loss"] = 0.
        # 4. AR+LAR: sin(theta/2)-space degree-weighted (gamma=2.0, no norm)
        if lambda_ar > 0:
            validMask = (degFlat >= 2)
            numValid = validMask.sum()
            if numValid > 0:
                edgeMask = neighborMask
                posAr = posCentered
                diffAr = posCentered.unsqueeze(0) - posCentered.unsqueeze(1)
                distAr = distMatrix
                safeDiff = torch.where(edgeMask.unsqueeze(-1), diffAr, torch.tensor([1.0, 0.0], device=device, dtype=posAr.dtype)).clone()
                safeDist = torch.where(edgeMask, distAr, torch.ones_like(distAr)).clone().clamp(min=self.eps)
                unit = safeDiff / safeDist.unsqueeze(-1)
                alpha = torch.atan2(unit[..., 1], unit[..., 0])
                alphaMasked = alpha + (~edgeMask).float() * 100.0
                alphaSorted, _ = torch.sort(alphaMasked, dim=1)
                thetaCons = alphaSorted[:, 1:] - alphaSorted[:, :-1]
                lastIdx = (degFlat.clamp(min=1) - 1).long()
                thetaWrap = alphaSorted[:, 0] - alphaSorted[torch.arange(N, device=device), lastIdx] + 2.0 * math.pi
                thetas = torch.zeros_like(alphaSorted)
                for i in range(N):
                    di = int(degFlat[i].clamp(min=1).item())
                    if di >= 2: thetas[i, :di] = torch.cat([thetaCons[i, :di-1], thetaWrap[i].unsqueeze(0)])
                colIdx = torch.arange(N, device=device).unsqueeze(0)
                validAngleMask = (colIdx < degFlat.clamp(min=1).unsqueeze(1)).float()
                phi = 2.0 * math.pi / degFlat.clamp(min=2.0)
                targetHalfSin = torch.sin(phi / 2.0)
                actualHalfSin = torch.sin(thetas / 2.0)
                error = targetHalfSin.unsqueeze(1) - actualHalfSin
                rawPairLoss = (error * error * validAngleMask).sum(dim=1)
                GAMMA_DEG = 2.0  # linear degree weight (best validated)
                degClamp = degFlat.clamp(min=1.0)
                weight = degClamp ** (GAMMA_DEG - 1.0)
                weight = weight * validMask.float()
                lossAr = (rawPairLoss * weight).sum() / (weight.sum() + self.eps)
                totalLoss = totalLoss + lambda_ar * lossAr
                metrics["ar_loss"] = lossAr.item()
            else: metrics["ar_loss"] = 0.0
        else: metrics["ar_loss"] = 0.0


        # 5.  (EU)
        if lambda_eu > 0:
            lossEu = (((edgeLens - meanEdgeLen) / (meanEdgeLen + self.eps)) ** 2).mean()
            totalLoss = totalLoss + lambda_eu * lossEu
            metrics["eu_loss"] = lossEu.item()
        else:
            metrics["eu_loss"] = 0.0

        # 6.  (Reg)
        lossRegTotal = torch.tensor(0.0, device=device)
        degReg = (((degFlat / (degFlat.mean() + self.eps)) - 1.0) ** 2).mean()
        lossRegTotal = lossRegTotal + self.degRegWeight * degReg
        if lambda_reg > 0 and (w_edge is not None or l_edge is not None or q_node is not None):
            regFeatures = 0.0
            if w_edge is not None: regFeatures += ((w_edge - 1.0) ** 2).mean()
            if l_edge is not None: regFeatures += ((l_edge - 1.0) ** 2).mean()
            if q_node is not None: regFeatures += ((q_node - 1.0) ** 2).mean()
            lossRegTotal = lossRegTotal + lambda_reg * regFeatures
        totalLoss = totalLoss + lossRegTotal
        metrics["reg_loss"] = lossRegTotal.item()

        return totalLoss, metrics

# ==========================================
# 
# ==========================================
def getCurriculumWeights(currentEpoch, cfg):
    useCurriculum = cfg['train'].get('use_curriculum', True)
    targetLnp = cfg['loss'].get('lambda_lnp', 2.0)
    targetAr = cfg['loss'].get('lambda_ar', 0.5)
    targetEu = cfg['loss'].get('lambda_eu', 0.1)

    if not useCurriculum:
        return targetLnp, targetAr, targetEu
        
    warmupStart = cfg['train'].get('warmup_epochs', 20)
    rampSteps = cfg['train'].get('ramp_epochs', 80)

    if currentEpoch < warmupStart:
        return 0.0, 0.0, 0.0
    
    rampEnd = warmupStart + rampSteps
    if currentEpoch >= rampEnd:
        return targetLnp, targetAr, targetEu
        
    progress = (currentEpoch - warmupStart) / float(rampSteps)
    scaleFactor = 0.5 * (1.0 - math.cos(math.pi * progress))
    
    return targetLnp * scaleFactor, targetAr * scaleFactor, targetEu * scaleFactor

# ==========================================
#  Epoch  ()
# ==========================================
def train_one_epoch(model, solver, layout_criterion, latent_criterion, optimizer, loader, device, cfg, currentEpoch):
    model.train()
    totalLossVal = 0
    logMeters = {
        'loss_total': 0, 'role_metric_loss': 0, 'kl_loss': 0,
        'lnp_loss': 0, 'ar_loss': 0, 'eu_loss': 0, 'scale_lock_loss': 0, 'reg_loss': 0
    }
    
    # 
    lambdaLnp, lambdaAr, lambdaEu = getCurriculumWeights(currentEpoch, cfg)
    lambdaKl = cfg['loss'].get('lambda_kl', 1.0)
    lambdaScaleLock = cfg['loss'].get('lambda_scale_lock', 0.3)
    lambdaReg = cfg['loss'].get('lambda_reg', 0.01)
    wLatent = cfg['loss'].get('lambda_latent', 1.0)

    solver.num_steps = cfg['solver'].get('equilibrium_iters', 500)
    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    
    count = 0
    for batchData in pbar:
        optimizer.zero_grad()
        batchData = batchData.to(device)
        
        # 1. 
        _, _, Z_role_batch, q_node_batch, w_edge_batch, l_edge_batch = model(
            batchData.x, batchData.edge_index, batchData.edge_attr
        )
        
        # 2. 
        pos_final_batch = solver(
            batchData.pos_init, w_edge_batch, l_edge_batch, q_node_batch, batchData.edge_index, batchData.batch  
        )

        graphs = batchData.to_data_list()
        batchCombinedLoss = 0
        batchMeters = {k: 0 for k in logMeters.keys() if k != 'loss_total'}
        hasNanInBatch = False

        nodeOffset = 0
        edgeOffset = 0

        # 3. 
        for g in graphs:
            N = g.num_nodes
            E = g.num_edges
            
            Z_role_i = Z_role_batch[nodeOffset : nodeOffset + N]
            pos_final_i = pos_final_batch[nodeOffset : nodeOffset + N]
            w_edge_i = w_edge_batch[edgeOffset : edgeOffset + E]
            l_edge_i = l_edge_batch[edgeOffset : edgeOffset + E]
            q_node_i = q_node_batch[nodeOffset : nodeOffset + N]
            
            nodeOffset += N
            edgeOffset += E
            
            # 
            roleMetricLoss = latent_criterion(Z_role_i, g.x)
            
            # 
            layoutLoss, lossDict = layout_criterion(
                pos=pos_final_i, edge_index=g.edge_index, w_edge=w_edge_i, l_edge=l_edge_i, q_node=q_node_i,
                lambda_kl=lambdaKl, lambda_lnp=lambdaLnp, lambda_ar=lambdaAr, lambda_eu=lambdaEu,
                lambda_scale_lock=lambdaScaleLock, lambda_reg=lambdaReg
            )
            
            graphTotalLoss = (layoutLoss + wLatent * roleMetricLoss) / len(graphs)
            if torch.isnan(graphTotalLoss) or torch.isinf(graphTotalLoss):
                hasNanInBatch = True
                break
                
            batchCombinedLoss = batchCombinedLoss + graphTotalLoss
            
            batchMeters['kl_loss'] += lossDict.get('kl_loss', 0)
            batchMeters['lnp_loss'] += lossDict.get('lnp_loss', 0)
            batchMeters['ar_loss'] += lossDict.get('ar_loss', 0)
            batchMeters['eu_loss'] += lossDict.get('eu_loss', 0)
            batchMeters['scale_lock_loss'] += lossDict.get('scale_lock_loss', 0)
            batchMeters['reg_loss'] += lossDict.get('reg_loss', 0)
            batchMeters['role_metric_loss'] += roleMetricLoss.item()

        if hasNanInBatch:
            optimizer.zero_grad()
            continue
            
        # 4.  ()
        if isinstance(batchCombinedLoss, torch.Tensor):
            batchCombinedLoss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('clip_grad', 10.0))
        optimizer.step()
        
        totalLossVal += batchCombinedLoss.item()
        for k in logMeters.keys():
            if k == 'loss_total': logMeters[k] += batchCombinedLoss.item()
            else: logMeters[k] += batchMeters[k] / len(graphs)
        
        count += 1
        if count % 5 == 0:
            pbar.set_postfix({'KL': f"{batchMeters['kl_loss']/len(graphs):.4f}", 
                              'LNP': f"{batchMeters['lnp_loss']/len(graphs):.4f}"})

    return totalLossVal / max(1, count), {k: v / max(1, count) for k, v in logMeters.items()}

# ==========================================
#  ()
# ==========================================
def evaluate(model, solver, layout_criterion, latent_criterion, loader, device, cfg, calc_metrics=False):
    model.eval()
    totalLossVal = 0
    logMeters = {
        'loss_total': 0, 'role_metric_loss': 0, 'kl_loss': 0,
        'lnp_loss': 0, 'ar_loss': 0, 'eu_loss': 0, 'scale_lock_loss': 0, 'reg_loss': 0,
        'np_metric': 0.0, 'ar_metric': 0.0, 'eu_metric': 0.0,
        'lnp_metric': 0.0, 'lar_metric': 0.0, 'leu_metric': 0.0
    }
    
    # 
    lambdaKl = cfg['loss'].get('lambda_kl', 1.0)
    lambdaLnp = cfg['loss'].get('lambda_lnp', 2.0)
    lambdaAr = cfg['loss'].get('lambda_ar', 0.5)
    lambdaEu = cfg['loss'].get('lambda_eu', 0.1)
    lambdaScaleLock = cfg['loss'].get('lambda_scale_lock', 0.3)
    lambdaReg = cfg['loss'].get('lambda_reg', 0.01)
    wLatent = cfg['loss'].get('lambda_latent', 1.0)
    
    totalGraphs = 0
    with torch.no_grad():
        for batchData in loader:
            batchData = batchData.to(device)
            _, _, Z_role_batch, q_node_batch, w_edge_batch, l_edge_batch = model(
                batchData.x, batchData.edge_index, batchData.edge_attr
            )
            pos_final_batch = solver(
                batchData.pos_init, w_edge_batch, l_edge_batch, q_node_batch, batchData.edge_index, batchData.batch
            )
            
            graphs = batchData.to_data_list()
            totalGraphs += len(graphs)
            
            nodeOffset = 0
            edgeOffset = 0
            
            for g in graphs:
                N = g.num_nodes
                E = g.num_edges
                Z_role_i = Z_role_batch[nodeOffset : nodeOffset + N]
                pos_final_i = pos_final_batch[nodeOffset : nodeOffset + N]
                w_edge_i = w_edge_batch[edgeOffset : edgeOffset + E]
                l_edge_i = l_edge_batch[edgeOffset : edgeOffset + E]
                q_node_i = q_node_batch[nodeOffset : nodeOffset + N]
                
                nodeOffset += N
                edgeOffset += E
                
                roleMetricLoss = latent_criterion(Z_role_i, g.x)
                layoutLoss, lossDict = layout_criterion(
                    pos=pos_final_i, edge_index=g.edge_index, w_edge=w_edge_i, l_edge=l_edge_i, q_node=q_node_i,
                    lambda_kl=lambdaKl, lambda_lnp=lambdaLnp, lambda_ar=lambdaAr, lambda_eu=lambdaEu,
                    lambda_scale_lock=lambdaScaleLock, lambda_reg=lambdaReg
                )

                if calc_metrics:
                    G_nx = to_networkx(g, to_undirected=True)
                    pos_dict = {idx: pos_final_i[idx].cpu().numpy() for idx in range(N)}
                    try:
                        np_v = eval_utils.calculate_neighborhood_preservation(G_nx, pos_dict)
                        ar_v = eval_utils.calculate_angular_resolution(G_nx, pos_dict)
                        eu_v = eval_utils.calculate_edge_length_distribution(G_nx, pos_dict)
                        lnp_v = eval_utils.calculate_LNP(G_nx, pos_dict)
                        lar_v = eval_utils.calculate_LAR(G_nx, pos_dict)
                        leu_v = eval_utils.calculate_LEU(G_nx, pos_dict)
                    except Exception:
                        np_v, ar_v, eu_v, lnp_v, lar_v, leu_v = 0.0, 0.0, 999.0, 0.0, 0.0, 999.0
                        
                    logMeters['np_metric'] += (np_v if np_v is not None else 0.0)
                    logMeters['ar_metric'] += (ar_v if ar_v is not None else 0.0)
                    logMeters['eu_metric'] += (eu_v if eu_v is not None else 999.0)
                    logMeters['lnp_metric'] += (lnp_v if lnp_v is not None else 0.0)
                    logMeters['lar_metric'] += (lar_v if lar_v is not None else 0.0)
                    logMeters['leu_metric'] += (leu_v if leu_v is not None else 999.0)
                
                graphLoss = layoutLoss + wLatent * roleMetricLoss
                totalLossVal += graphLoss.item()
                
                logMeters['kl_loss'] += lossDict.get('kl_loss', 0)
                logMeters['lnp_loss'] += lossDict.get('lnp_loss', 0)
                logMeters['ar_loss'] += lossDict.get('ar_loss', 0)
                logMeters['eu_loss'] += lossDict.get('eu_loss', 0)
                logMeters['scale_lock_loss'] += lossDict.get('scale_lock_loss', 0)
                logMeters['reg_loss'] += lossDict.get('reg_loss', 0)
                logMeters['role_metric_loss'] += roleMetricLoss.item()

    avgLoss = totalLossVal / totalGraphs
    avgMeters = {k: v / totalGraphs for k, v in logMeters.items() if k != 'loss_total'}
    avgMeters['loss_total'] = avgLoss
    return avgLoss, avgMeters

# ==========================================
#  HPO  ( Optuna )
# ==========================================
def run_training_pipeline(cfg, trial=None, force_restart=False):
    seed_everything(cfg['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    saveDir = os.path.join(cfg['log']['save_dir'], cfg['log']['exp_name'])
    if not os.path.exists(saveDir): 
        os.makedirs(saveDir)
    save_config(cfg, os.path.join(saveDir, 'config.yaml'))
    
    logFilePath = os.path.join(saveDir, 'loss_log.csv')
    if force_restart and os.path.exists(logFilePath):
        os.remove(logFilePath)
        
    print(f"[INFO] Device: {device} | DeepCS Formal HPO Setup v9")
    
    # 
    dataset = ProbForceDataset(root=cfg['data']['root'])
    indices = list(range(len(dataset)))
    
    try:
        labels = [data.y.item() for data in dataset]
        trainIdx, valIdx = train_test_split(
            indices, train_size=cfg['data']['split_ratio'], stratify=labels, random_state=cfg['train']['seed']
        )
    except (AttributeError, ValueError):
        trainLen = int(len(dataset) * cfg['data']['split_ratio'])
        valLen = len(dataset) - trainLen
        generator = torch.Generator().manual_seed(cfg['train']['seed'])
        train_set, val_set = torch.utils.data.random_split(dataset, [trainLen, valLen], generator=generator)
        trainIdx, valIdx = train_set.indices, val_set.indices

    trainSet = Subset(dataset, trainIdx)
    valSet = Subset(dataset, valIdx)
    
    trainLoader = DataLoader(trainSet, batch_size=cfg['data']['batch_size'], shuffle=True)
    valLoader = DataLoader(valSet, batch_size=cfg['data']['batch_size'], shuffle=False)
    
    # 
    sampleData = dataset[0]
    nodeInDim = sampleData.x.shape[1]
    edgeInDim = sampleData.edge_attr.shape[1]
    
    model = DecoupledForcePredictor(
        node_in_dim=nodeInDim, 
        edge_in_dim=edgeInDim, 
        hidden_dim=cfg['model']['hidden_channels'],
        latent_dim=cfg['model'].get('latent_dim', 128),
        num_layers=cfg['model'].get('num_layers', 3),
        heads=cfg['model']['heads'], 
        dropout=cfg['model']['dropout']
    ).to(device)
    
    solver = ASGDForceSolver(
        num_steps=cfg['solver'].get('equilibrium_iters', 300),
        initial_lr=cfg['solver'].get('initial_lr', 0.5),
        max_step_size=cfg['solver'].get('max_step_size', 2.0),
        min_step_size=cfg['solver'].get('min_step_size', 0.01),
        max_disp=cfg['solver'].get('max_disp', 1.0),
    ).to(device)
    
    layoutCriterion = UnifiedRankingOrbitalField(
        margin_rank=cfg['loss'].get('margin_rank', 0.05),
        sigma_ratio=cfg['loss'].get('sigma_ratio', 1.0),
        neg_sample_multiplier=cfg['loss'].get('neg_sample_multiplier', 2.0),
        tau_sharp=cfg['loss'].get('tau_sharp', 0.05),
        deg_reg_weight=cfg['loss'].get('deg_reg_weight', 0.1)
    ).to(device)
    
    latentCriterion = IsomorphicRoleMetricLoss(
        tau_f=cfg['loss'].get('tau_f', 0.1),
        tau_rw=cfg['loss'].get('tau_rw', 0.05),
        tau_z=cfg['loss'].get('tau_z', 1.0)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=float(cfg['train']['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    bestModelPath = os.path.join(saveDir, 'best_model.pth')
    lastCkptPath = os.path.join(saveDir, 'last.last')
    
    earlyStopping = EarlyStopping(
        patience=cfg['train'].get('patience', 15), 
        delta=cfg['train'].get('min_delta', 0.0001), 
        path=bestModelPath, 
        verbose=False
    )
    
    useCurriculum = cfg['train'].get('use_curriculum', True)
    fullLoadEpoch = (cfg['train'].get('warmup_epochs', 20) + cfg['train'].get('ramp_epochs', 80)) if useCurriculum else 0
    
    startEpoch = 0
    if os.path.exists(lastCkptPath) and not force_restart:
        checkpoint = torch.load(lastCkptPath, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startEpoch = checkpoint['epoch'] + 1

    bestValLoss = float('inf')
    bestRealMetrics = None  

    for epoch in range(startEpoch, cfg['train']['epochs']):
        startTime = time.time()
        
        try:
            trainLoss, trainMeters = train_one_epoch(
                model, solver, layoutCriterion, latentCriterion, optimizer, trainLoader, device, cfg, epoch
            )
        except ZeroDivisionError:
            # 
            print(f"\n[FATAL] Physics Collapse at Epoch {epoch+1}!")
            return {
                'np_metric': 0.0, 'ar_metric': 0.0, 'eu_metric': 999.0,
                'lnp_metric': 0.0, 'lar_metric': 0.0, 'leu_metric': 999.0,
                'hpo_fitness': -1998.0  # 
            }
        
        valLoss, valMeters = evaluate(
            model, solver, layoutCriterion, latentCriterion, valLoader, device, cfg, calc_metrics=False
        )
        
        if valLoss < bestValLoss and epoch >= fullLoadEpoch:
            bestValLoss = valLoss
            _, detailedValMeters = evaluate(
                model, solver, layoutCriterion, latentCriterion, valLoader, device, cfg, calc_metrics=True
            )
            bestRealMetrics = detailedValMeters
        
        elapsed = time.time() - startTime
        currentLr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR: {currentLr:.6f} | Trn Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f} | Time: {elapsed:.1f}s")
        
        #  CSV 
        logData = {'epoch': epoch + 1, 'train_total_loss': trainLoss, 'val_total_loss': valLoss, 'time_s': elapsed}
        for k, v in trainMeters.items(): logData[f'train_{k}'] = v
        for k, v in valMeters.items(): logData[f'val_{k}'] = v
        writeHeader = not os.path.exists(logFilePath)
        with open(logFilePath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=logData.keys())
            if writeHeader: writer.writeheader()
            writer.writerow(logData)
        
        if epoch >= fullLoadEpoch:
            scheduler.step(valLoss)
            earlyStopping(valLoss, model)
            if earlyStopping.early_stop:
                break
                
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()
        }, lastCkptPath)

    if bestRealMetrics is None:
        _, bestRealMetrics = evaluate(model, solver, layoutCriterion, latentCriterion, valLoader, device, cfg, calc_metrics=True)
        
    # =========================================================
    #  (Standardized Evaluation)
    # NP/LNP/AR/LAR EU/LEU 0
    # =========================================================
    fitnessScore = (
        (bestRealMetrics['np_metric'] + bestRealMetrics['lnp_metric']) +
        (bestRealMetrics['ar_metric'] + bestRealMetrics['lar_metric']) -
        (bestRealMetrics['eu_metric'] + bestRealMetrics['leu_metric'])
    )
    bestRealMetrics['hpo_fitness'] = fitnessScore
    
    print(f"[HPO OBJECTIVE SETTLED] Fitness Score: {fitnessScore:.4f} | (NP+LNP)={(bestRealMetrics['np_metric'] + bestRealMetrics['lnp_metric']):.4f} | (AR+LAR)={(bestRealMetrics['ar_metric'] + bestRealMetrics['lar_metric']):.4f} | (EU+LEU)={(bestRealMetrics['eu_metric'] + bestRealMetrics['leu_metric']):.4f}")
        
    return bestRealMetrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/prob_force_decoupled_v18_batched_kl_rangeplus.yaml')
    parser.add_argument('--force_restart', action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.config): 
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    cfg = load_config(args.config)
    run_training_pipeline(cfg, trial=None, force_restart=args.force_restart)