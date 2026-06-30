# -*- coding: utf-8 -*-
import copy
import warnings
import numpy as np
import networkx as nx
from typing import Dict, Any
import os
import sys
import math
import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import gd_models.dnn2.SOTA_layouts as SOTA
    import gd_models.dnn2.graph_preprocessing as gp
    print("[Info] DNN2 modules imported successfully.")
except ImportError as e:
    import traceback
    print(f"[Debug] DNN2 Import Failed. Reason: {e}")
    print("[Debug] Traceback details:")
    traceback.print_exc()
    SOTA = None
    gp = None


# --- Add this near your top imports in layout.py ---
try:
    import torch
    import torch_geometric as pyg
    from torch_geometric.data import Data, Batch
    import json
    
    # [FIX]: A native replacement for the broken 'attrdict' library
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    # Imports specific to the Core-GD / neuraldrawer codebase
    from gd_models.core_gd.train import get_model
    from gd_models.core_gd.preprocessing import preprocess_dataset
    # Fallback import if coarsening is needed
    try:
        from gd_models.core_gd.train import create_coarsened_dataset, go_to_coarser_graph
    except ImportError as e:
        import traceback
        print(f"[Warning] Coarsening import failed: {e}")
        create_coarsened_dataset = None
except ImportError as e:
    import traceback
    print(f"[Debug] Core-GD Import Failed. Reason: {e}")
    traceback.print_exc()
    torch = None

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ======================================================================
# [修改 2] 引入全新的解耦模型架构，移除旧版的 GraphPredictor 和 APSP
# ======================================================================
try:
    import torch
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import from_networkx
    from torch_scatter import scatter_add  # 用于特征提取

    # 精准导入根目录下的 utils 和 models
    from utils import compute_topological_status, seed_everything, load_config
    from models.decoupled_predictor_v2 import DecoupledForcePredictor, AblationForcePredictor
    from models.asgd_force_solver_v2 import BatchedASGDForceSolver as ASGDForceSolver

except ImportError as e:
    print(f"[Warning] Deep Learning dependencies not fully loaded: {e}")

# Core Fix 1: Globally mask sklearn's annoying FutureWarning
warnings.filterwarnings("ignore")

# Core Fix 2: Patch the old fa2 library for NetworkX 3.0+ compatibility
if not hasattr(nx, "to_scipy_sparse_matrix"):
    nx.to_scipy_sparse_matrix = nx.to_scipy_sparse_array
    
def extractGlobalHierarchy(graphObj, numNodes):
  prDict = nx.pagerank(graphObj, alpha=0.85)
  prTensor = torch.tensor([prDict[i] for i in range(numNodes)], dtype=torch.float32).view(numNodes, 1)
  prNorm = prTensor / (prTensor.max() + 1e-8)
  coreDict = nx.core_number(graphObj)
  coreTensor = torch.tensor([coreDict[i] for i in range(numNodes)], dtype=torch.float32).view(numNodes, 1)
  coreNorm = coreTensor / (coreTensor.max() + 1e-8)
  return torch.cat([prNorm, coreNorm], dim=1)
def extractStructuralFingerprints(graphObj, numNodes, pygEdgeIndex):
  rowIdx, colIdx = pygEdgeIndex[0], pygEdgeIndex[1]
  degNp = np.array([graphObj.degree(i) for i in range(numNodes)], dtype=float)
  degTensor = torch.from_numpy(degNp).float()
  
  feat1 = torch.log(degTensor + 1.0)
  arithMass = scatter_add(degTensor[colIdx], rowIdx, dim=0, dim_size=numNodes)
  feat2 = torch.log(arithMass + 1.0)
  
  invDeg = 1.0 / (degTensor + 1e-6)
  sumInvDeg = scatter_add(invDeg[colIdx], rowIdx, dim=0, dim_size=numNodes)
  harmonicMass = degTensor / (sumInvDeg + 1e-6)
  feat3 = torch.log(harmonicMass + 1.0)
  
  fingerprintMatrix = torch.stack([feat1, feat2, feat3], dim=1)
  return fingerprintMatrix

def computeRwpe(pygEdgeIndex, numNodes, walkLength=4):
  adjMatrix = torch.zeros((numNodes, numNodes), dtype=torch.float32)
  adjMatrix[pygEdgeIndex[0], pygEdgeIndex[1]] = 1.0
  degVector = adjMatrix.sum(dim=-1, keepdim=True)
  degVector[degVector == 0] = 1.0
  probMatrix = adjMatrix / degVector
        
  peList = []
  currProbMatrix = probMatrix.clone()
  for _ in range(walkLength):
    peList.append(torch.diag(currProbMatrix))
    currProbMatrix = torch.matmul(currProbMatrix, probMatrix)
  return torch.stack(peList, dim=1)
class AestheticMetricLoss(nn.Module):
    """
    [DeepCS V9.1] Ultimate Master Aesthetic Metric Loss
    Fully Differentiable Stress-Normalized Space. 
    Zero-gradient for global scaling guarantees NO EXPLOSIONS.
    """
    def __init__(self, 
                 w_stress=1.0, w_vr=0.1, w_anr=0.1, w_np=0.1,
                 vr_margin_ratio=1.0, 
                 anr_sensitivity=2.0,
                 np_margin=0.2):
        super(AestheticMetricLoss, self).__init__()
        self.weights = {'stress': w_stress, 'vr': w_vr, 'anr': w_anr, 'np': w_np}
        self.vr_margin_ratio = vr_margin_ratio
        self.anr_s = anr_sensitivity
        self.np_margin = np_margin

    def forward(self, final_pos, d_sp, weights, mask, adj=None, batch_data=None):
        B, N, _ = final_pos.shape
        device = final_pos.device
        
        mask_f = mask.float()
        mask_mat = mask_f.unsqueeze(2) * mask_f.unsqueeze(1)
        eye = torch.eye(N, device=device).bool().unsqueeze(0)
        valid_pair_mask = mask_mat * (~eye)
        
        dist_curr = torch.cdist(final_pos, final_pos, p=2).clamp(min=1e-8)
        
        total_loss = torch.tensor(0.0, device=device)
        loss_aux = torch.tensor(0.0, device=device)

        # ==========================================
        # [THE ANCHOR]: Global Scale Normalization Factor (s)
        # ==========================================
        num = (weights * d_sp * dist_curr * mask_mat).sum(dim=(1, 2), keepdim=True)
        den = (weights * (dist_curr**2) * mask_mat).sum(dim=(1, 2), keepdim=True)
        s_scale = (num / (den + 1e-12)).detach()
        norm_dist = dist_curr * s_scale

        # ==========================================
        # 1. Scale-Invariant Stress Loss
        # ==========================================
        if self.weights['stress'] > 0:
            stress_err = weights * (s_scale * dist_curr - d_sp)**2 * mask_mat
            actual_n = mask_f.sum(dim=-1).clamp(min=1.0).unsqueeze(-1) # [B, 1]
            loss_stress = (stress_err.sum(dim=(1, 2)) / 2.0) / actual_n.view(B)
            loss_stress = loss_stress.mean()
            total_loss += self.weights['stress'] * loss_stress
        else:
            loss_stress = torch.tensor(0.0, device=device)

        # ==========================================
        # 2. VR Loss (Evaluated in Normalized Space)
        # ==========================================
        loss_vr = torch.tensor(0.0, device=device)
        if self.weights['vr'] > 0 and adj is not None:
            non_adj_mask = valid_pair_mask.bool() & (~(adj > 0.5))
            d_max_theo = d_sp.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            num_nodes = mask_f.sum(dim=1, keepdim=True).unsqueeze(2)
            target_vr = self.vr_margin_ratio * d_max_theo / torch.sqrt(num_nodes.clamp(min=1.0))
            
            masked_dist = norm_dist.masked_fill(~non_adj_mask, float('inf'))
            flat_dist = masked_dist.view(B, -1)
            
            for b in range(B):
                valid_dist = flat_dist[b][flat_dist[b] < float('inf')]
                if len(valid_dist) > 0:
                    temperature = 0.05 * d_max_theo[b,0,0] 
                    weights_vr = F.softmax(-valid_dist / temperature, dim=0)
                    soft_min_dist = (valid_dist * weights_vr).sum()
                    loss_vr += F.relu(target_vr[b,0,0] - soft_min_dist) ** 2
                    
            loss_vr = loss_vr / B
            total_loss += self.weights['vr'] * loss_vr

        # ==========================================
        # 3. ANR Loss (Evaluated in Normalized Space)
        # ==========================================
        loss_anr = torch.tensor(0.0, device=device)
        if self.weights['anr'] > 0 and adj is not None:
            adj_mask = (adj > 0.5)
            degrees = adj_mask.sum(dim=-1).float().clamp(min=1.0)
            ideal_angle = (2 * math.pi / degrees).unsqueeze(2).unsqueeze(3)
            
            target_chord = torch.sqrt(2.0 - 2.0 * torch.cos(ideal_angle) + 1e-8)
            actual_chord = norm_dist.unsqueeze(1) 
            
            mask_vw = adj_mask.unsqueeze(3) & adj_mask.unsqueeze(2)
            eye_vw = torch.eye(N, device=device).bool().unsqueeze(0).unsqueeze(0)
            mask_vw = mask_vw & (~eye_vw)
            
            violation = F.relu(target_chord - actual_chord)
            loss_anr = (violation ** 2 * mask_vw).sum() / mask_vw.sum().clamp(min=1.0)
            
            total_loss += self.weights['anr'] * loss_anr

        # ==========================================
        # 4. Hub-Centric NP Loss
        # ==========================================
        loss_np = torch.tensor(0.0, device=device)
        if self.weights['np'] > 0 and adj is not None:
            adj_true = (adj > 0.5) & valid_pair_mask.bool()
            adj_false = (d_sp > 2.5) & valid_pair_mask.bool()
            
            dist_true_masked = norm_dist.masked_fill(~adj_true, -1e9)
            max_true_dist = dist_true_masked.max(dim=-1, keepdim=True)[0]
            max_true_dist = max_true_dist.masked_fill(max_true_dist == -1e9, 0.0)
            
            dist_false_masked = norm_dist.masked_fill(~adj_false, 1e9)
            k_false = min(5, max(1, N // 4))
            bottomk_false_dist, _ = torch.topk(dist_false_masked, k_false, dim=-1, largest=False)
            
            base_violation = F.relu(max_true_dist.detach() + self.np_margin - bottomk_false_dist)
            
            degrees = adj_true.sum(dim=-1).float() 
            hub_weights = F.relu(degrees - 1.0)
            hub_weights = torch.sqrt(hub_weights + 1e-8).unsqueeze(-1) 
            
            weighted_violation = base_violation * hub_weights
            valid_rows = (adj_true.sum(dim=-1) > 0).unsqueeze(-1)
            weight_sum = (hub_weights * valid_rows).sum().clamp(min=1.0)
            loss_np = (weighted_violation * valid_rows).sum() / weight_sum
            
            total_loss += self.weights['np'] * loss_np

        aux_dict = {
            'vr': loss_vr if self.weights.get('vr', 0) > 0 else 0.0,
            'anr': loss_anr if self.weights.get('anr', 0) > 0 else 0.0,
            'np': loss_np if self.weights.get('np', 0) > 0 else 0.0
        }
        return total_loss, loss_stress, aux_dict


class SandboxOptimizer(nn.Module):
    def __init__(self, init_pos):
        super(SandboxOptimizer, self).__init__()
        self.pos = nn.Parameter(init_pos.clone())

    def forward(self):
        return self.pos

class LayoutGenerator:

    _dnn2_model = None
    _coregd_model = None
    _coregd_config = None
    _coregd_device = 'cpu'
    
    deepcs_components = None
    
    def __init__(self, json_data: Dict[str, Any]):
        self.original_data = copy.deepcopy(json_data)
        self.graph_id = self.original_data.get("graph_id", "unknown_graph")
        self.G = self._parse_to_nx(self.original_data)
        
    def _parse_to_nx(self, data: Dict[str, Any]) -> nx.Graph:
        G = nx.Graph()
        for node in data.get("nodes", []):
            G.add_node(str(node["id"]))
        for edge in data.get("edges", []):
            u, v = str(edge["source"]), str(edge["target"])
            weight = float(edge.get("weight", 1.0))
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, weight=weight)
        return G

    def _export_json(self, pos: Dict[str, np.ndarray], algo_name: str) -> Dict[str, Any]:
        export_data = copy.deepcopy(self.original_data)
        export_data["algorithm"] = algo_name
        
        pos_serializable = {}
        for k, v in pos.items():
            pos_serializable[k] = {"x": float(v[0]), "y": float(v[1])}
        
        for node in export_data.get("nodes", []):
            node_id = str(node["id"])
            if node_id in pos_serializable:
                node["x"] = pos_serializable[node_id]["x"]
                node["y"] = pos_serializable[node_id]["y"]
                
        return export_data

    @classmethod
    def load_deepcs_model(cls, cfg_path: str, ckpt_path: str, model_name: str = "DeepCS"):
        """
        Loads the Decoupled DeepCS model and Batched ASGD solver into memory.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading {model_name} to {device}...")
        
        if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing config or ckpt files for {model_name}.")
            
        cfg = load_config(cfg_path)
        seed_everything(cfg['train']['seed'])
        
        # 1. ????? Decoupled Predictor
        abl_cfg = cfg.get('ablation', None)
        if abl_cfg is not None:
            model = AblationForcePredictor(
                node_in_dim=cfg['model'].get('node_in_dim', 11),
                edge_in_dim=cfg['model'].get('edge_in_dim', 1),
                hidden_dim=cfg['model']['hidden_channels'],
                latent_dim=cfg['model'].get('latent_dim', 128),
                num_layers=cfg['model'].get('num_layers', 3),
                heads=cfg['model']['heads'],
                dropout=cfg['model'].get('dropout', 0.0),
                spatial_type=abl_cfg.get('spatial_type', 'GATv2'),
                use_role_stream=abl_cfg.get('use_role_stream', True),
                isolate_gradient=abl_cfg.get('isolate_gradient', True),
                capacity_match=abl_cfg.get('capacity_match', False)
            ).to(device)
        else:
            model = DecoupledForcePredictor(
                node_in_dim=cfg['model'].get('node_in_dim', 11),
                edge_in_dim=cfg['model'].get('edge_in_dim', 1),
                hidden_dim=cfg['model']['hidden_channels'],
                latent_dim=cfg['model'].get('latent_dim', 128),
                num_layers=cfg['model'].get('num_layers', 3),
                heads=cfg['model']['heads'],
                dropout=cfg['model'].get('dropout', 0.0)
            ).to(device)
            
        # 2. ????? ASGD ?????????
        solver = ASGDForceSolver(
            num_steps=cfg['solver'].get('equilibrium_iters', 500),
            initial_lr=cfg['solver'].get('initial_lr', 0.5),
            max_step_size=cfg['solver'].get('max_step_size', 2.0),
            min_step_size=cfg['solver'].get('min_step_size', 0.01),
            max_disp=cfg['solver'].get('max_disp', 2.0)
        ).to(device)

        # 3. ???????
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model.eval()
        
        # ?????? apsp ????
        cls.deepcs_components = {
            'model_name': model_name,
            'model': model,
            'solver': solver,
            'cfg': cfg,
            'device': device
        }
        print(f"[INFO] Successfully loaded {model_name}.")

    def generate_deepcs(self):
        """
        Generates layout using the loaded Decoupled DeepCS model.
        """
        if self.deepcs_components is None:
            raise RuntimeError("DeepCS model is not loaded. Call load_deepcs_model() first.")
            
        components = self.deepcs_components
        model = components['model']
        solver = components['solver']
        device = components['device']
        model_name = components['model_name']

        # 1. 准备图结构与映射
        G = self.G.copy()
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
        nodes_list = list(G.nodes())
        node_mapping = {n: i for i, n in enumerate(nodes_list)}
        reverse_mapping = {i: n for i, n in enumerate(nodes_list)}
        G_int = nx.relabel_nodes(G, node_mapping)
        N = G_int.number_of_nodes()
        
        pyg_edge_index = from_networkx(G_int).edge_index

        # 2. 动态特征提取 (严格对齐训练时的 11 维特征)
        # 2.1 提取 Laplacian 初始坐标
        L_norm = nx.normalized_laplacian_matrix(G_int).toarray().astype(np.float32)
        try:
            _, eig_vecs = np.linalg.eigh(L_norm)
            pos_init_base = eig_vecs[:, 1:3] if N > 2 else np.random.randn(N, 2)
            pos_init = pos_init_base + np.random.normal(0, 0.01, size=pos_init_base.shape)
        except Exception:
            pos_init = np.random.randn(N, 2) * 0.1
        pos_init_tensor = torch.from_numpy(pos_init).float()

        # 2.2 提取拓扑指纹
        fingerprintsData = extractStructuralFingerprints(G_int, N, pyg_edge_index)
        
        # 2.3 提取相对度数
        degNpData = np.array([G_int.degree(i) for i in range(N)], dtype=float)
        relDegTensor = torch.from_numpy(degNpData / float(N)).float().view(N, 1)
        
        # 2.4 提取全局层次 (PageRank & Core)
        globalHierData = extractGlobalHierarchy(G_int, N)
        
        # 2.5 提取随机游走位置编码 (RWPE)
        rwpeData = computeRwpe(pyg_edge_index, N, walkLength=4)
        
        # 2.6 提取拓扑状态 (Status)
        statusData = compute_topological_status(pyg_edge_index, N)
        if not isinstance(statusData, torch.Tensor):
            statusData = torch.tensor(statusData, dtype=torch.float32)
        statusData = statusData.view(N, -1)
        
        # 特征拼接 (总维度 = 3 + 1 + 2 + 4 + 1 = 11)
        x = torch.cat([fingerprintsData, relDegTensor, globalHierData, rwpeData, statusData], dim=1)
        
        numEdges = pyg_edge_index.shape[1]
        edgeAttrMatrix = torch.ones((numEdges, 1), dtype=torch.float32)

        # 构建 PyG Data
        data = Data(x=x, edge_index=pyg_edge_index, edge_attr=edgeAttrMatrix, pos_init=pos_init_tensor, num_nodes=N).to(device)

        # 3. 模型推理 (禁止梯度计算)
        with torch.no_grad():
            batch_data = Batch.from_data_list([data])
            
            # GNN 前向传播，输出物理参数
            out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
            q_nodes, w_edges, l_edges = out[3], out[4], out[5]
            
            # 物理求解器隐式求解坐标
            pos_final = solver(batch_data.pos_init, w_edges, l_edges, q_nodes, batch_data.edge_index, batch_data.batch)
            pos_deepcs = pos_final.cpu().numpy()

        # 4. 坐标归一化与字典格式化
        center_of_mass = pos_deepcs.mean(axis=0)
        pos_deepcs = pos_deepcs - center_of_mass
        
        layout_result = {
            "algorithm": model_name,
            "nodes": [],
            "edges": []  
        }
        
        for idx in range(N):
            original_node_id = reverse_mapping[idx]
            layout_result["nodes"].append({
                "id": str(original_node_id),
                "x": float(pos_deepcs[idx][0]),
                "y": float(pos_deepcs[idx][1])
            })
            
        for u, v in self.G.edges():
            layout_result["edges"].append({
                "source": str(u),
                "target": str(v),
                "weight": float(self.G[u][v].get("weight", 1.0))
            })
            
        return layout_result

    @classmethod
    def load_dnn2_model(cls, model_path: str, train_infos_path: str, n_max: int = -1):
        """
        Statically load the DNN2 model. Should be called once before the batch loop.
        """
        if cls._dnn2_model is None:
            if SOTA is None:
                raise ImportError("SOTA module not found. Please ensure the model codebase is in the PYTHONPATH.")
            cls._dnn2_model = SOTA.DNN2(model_path, train_infos_path, n_max)

    def generate_dnn2(self) -> Dict[str, Any]:
        """
        Algorithm: (DNN)2 Deep Learning Layout
        """
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "(DNN)2")

        if self._dnn2_model is None:
            print("[Warning] DNN2 model not loaded. Call load_dnn2_model() before execution. Falling back to FR.")
            fallback = self.generate_fr()
            fallback["algorithm"] = "(DNN)2 (Fallback)"
            return fallback

        import gd_models.dnn2.graph_preprocessing as gp
        
        # 拦截器 1：强制阻断推理阶段的节点随机置换，确保坐标输出顺序与输入一致
        original_swap_matrix = gp.swapMatrix
        original_swap_vector = gp.swapVector
        
        # 拦截器 2：动态尺度对齐 (Dynamic Scale Alignment)
        original_random = nx.random_layout
        original_kk = nx.kamada_kawai_layout

        try:
            gp.swapMatrix = lambda M, real_n=None, permutation=None: (M, np.ones_like(M), np.arange(M.shape[0]))
            gp.swapVector = lambda V, real_n=None, permutation=None: (V, np.ones_like(V), np.arange(V.shape[0]))
            
            # 【终极修复 2：DNN2 动态缩放保护】
            # 小图保持 1000 倍对齐训练域；大图根据节点数 N 平滑衰减倍率，防止数值溢出致盲网络
            n = self.G.number_of_nodes()
            scale_factor = 1000.0 if n <= 50 else 1000.0 * (50.0 / n)
            # 保证一个最小下限，避免极端大图完全失去尺度
            scale_factor = max(scale_factor, 100.0) 

            nx.random_layout = lambda G: {k: v * scale_factor for k, v in original_random(G).items()}
            nx.kamada_kawai_layout = lambda G: {k: v * scale_factor for k, v in original_kk(G).items()}
            # 1. 构建节点映射
            nodes = list(self.G.nodes())
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            idx_to_node = {i: node for i, node in enumerate(nodes)}
            n = len(nodes)
            
            # 2. 构建邻接矩阵
            am_matrix = nx.to_numpy_array(self.G, nodelist=nodes)

            # 3. 转换图对象
            mask = np.ones((n, 1))
            if hasattr(gp, 'AM2tlp'):
                target_g, _ = gp.AM2tlp(am_matrix, mask)
            else:
                target_g = nx.relabel_nodes(self.G, node_to_idx)

            # 4. 执行模型推理
            predicted_pos = self._dnn2_model.layout(target_g)

            # 5. 坐标映射回传
            final_pos = {}
            if isinstance(predicted_pos, dict):
                for int_idx, coords in predicted_pos.items():
                    idx = int_idx.id if hasattr(int_idx, 'id') else int(int_idx)
                    if idx in idx_to_node:
                        final_pos[idx_to_node[idx]] = np.array([float(coords[0]), float(coords[1])])
            else:
                for i in range(n):
                    final_pos[idx_to_node[i]] = np.array([float(predicted_pos[i][0]), float(predicted_pos[i][1])])

            return self._export_json(final_pos, "(DNN)2")

        except Exception as e:
            print(f"[Error] DNN2 inference failed: {e}. Falling back to FR.")
            fallback = self.generate_fr()
            fallback["algorithm"] = "(DNN)2 (Fallback)"
            return fallback
            
        finally:
            # 还原底层函数，防止污染后续处理流水线
            gp.swapMatrix = original_swap_matrix
            gp.swapVector = original_swap_vector
            nx.random_layout = original_random
            nx.kamada_kawai_layout = original_kk

    def generate_fr(self, iterations: int = 50) -> Dict[str, Any]:
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "Fruchterman-Reingold")
        pos = nx.spring_layout(self.G, iterations=iterations, seed=42, weight='weight')
        return self._export_json(pos, "Fruchterman-Reingold")

    def generate_kk(self) -> Dict[str, Any]:
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "Kamada-Kawai")
        pos = {}
        for component in nx.connected_components(self.G):
            sub_G = self.G.subgraph(component)
            if sub_G.number_of_nodes() == 1:
                pos[list(sub_G.nodes())[0]] = np.array([0.0, 0.0])
            else:
                sub_pos = nx.kamada_kawai_layout(sub_G, weight='weight')
                pos.update(sub_pos)
        return self._export_json(pos, "Kamada-Kawai")

    def generate_spectral(self) -> Dict[str, Any]:
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "Spectral")
        pos = nx.spectral_layout(self.G, weight='weight')
        return self._export_json(pos, "Spectral")

    def generate_mds(self) -> Dict[str, Any]:
        try:
            from sklearn.manifold import MDS
        except ImportError:
            return self.generate_kk()

        nodes = list(self.G.nodes())
        n = len(nodes)
        if n < 3:
            return self.generate_kk()

        path_lengths = dict(nx.shortest_path_length(self.G))
        dist_matrix = np.zeros((n, n))
        
        max_dist = 0.0
        for p_dict in path_lengths.values():
            if p_dict:
                max_dist = max(max_dist, max(p_dict.values()))
                
        disconnected_penalty = max_dist * 2.0 if max_dist > 0 else 10.0

        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = path_lengths.get(nodes[i], {}).get(nodes[j], disconnected_penalty)

        mds_solver = MDS(n_components=2, dissimilarity="precomputed", random_state=42, max_iter=300, normalized_stress='auto')
        coordinates = mds_solver.fit_transform(dist_matrix)
        
        pos = {nodes[i]: coordinates[i] for i in range(n)}
        return self._export_json(pos, "MDS (Scikit-Learn)")

    def generate_fa2(self, iterations: int = 100) -> Dict[str, Any]:
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "ForceAtlas2")
        try:
            from fa2 import ForceAtlas2
            fa2 = ForceAtlas2(
                edgeWeightInfluence=1.0, jitterTolerance=1.0, 
                barnesHutOptimize=True, barnesHutTheta=1.2, multiThreaded=False, 
                scalingRatio=2.0, strongGravityMode=False, gravity=1.0, verbose=False
            )
            positions = fa2.forceatlas2_networkx_layout(self.G, pos=None, iterations=iterations)
            return self._export_json(positions, "ForceAtlas2")
        except ImportError:
            fallback_json = self.generate_fr(iterations=iterations)
            fallback_json["algorithm"] = "ForceAtlas2 (Fallback to FR)"
            return fallback_json

    def generate_neato(self) -> Dict[str, Any]:
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "Neato")
        try:
            pos = nx.nx_agraph.graphviz_layout(self.G, prog="neato")
            pos_array = np.array(list(pos.values()))
            if len(pos_array) > 0:
                min_val, max_val = pos_array.min(axis=0), pos_array.max(axis=0)
                range_val = np.maximum(max_val - min_val, 1e-9)
                pos = {k: (np.array(v) - min_val) / range_val for k, v in pos.items()}
            return self._export_json(pos, "Neato")
        except (ImportError, ModuleNotFoundError):
            fallback_json = self.generate_kk()
            fallback_json["algorithm"] = "Neato (Fallback to KK)"
            return fallback_json

    # ==========================================
    # Algorithm 7: D3.js Force-Directed (Cross-stack Node.js call)
    # ==========================================
    def generate_d3(self) -> Dict[str, Any]:
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "D3-Force")
            
        import subprocess
        import json
        
        # 1. Prepare minimal data for JS
        input_data = {
            "nodes": [{"id": str(n)} for n in self.G.nodes()],
            "edges": [{"source": str(u), "target": str(v)} for u, v in self.G.edges()]
        }
        
        try:
            # 2. Start Node.js subprocess
            process = subprocess.Popen(
                ['node', 'd3_solver.js'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send data and wait for result
            stdout, stderr = process.communicate(input=json.dumps(input_data))
            
            if process.returncode != 0:
                print(f"[Warning] D3 engine execution failed: {stderr.strip()}. Falling back to FR.")
                fallback = self.generate_fr()
                fallback["algorithm"] = "D3-Force (Fallback to FR)"
                return fallback
                
            # 3. Parse coordinates calculated by D3
            d3_result = json.loads(stdout)
            pos = {}
            for node in d3_result.get('nodes', []):
                node_id = str(node['id'])
                pos[node_id] = np.array([float(node['x']), float(node['y'])])
                
            return self._export_json(pos, "D3-Force")
            
        except FileNotFoundError:
            print("[Warning] 'node' command not found in system, ensure Node.js is installed. Falling back to FR.")
            fallback = self.generate_fr()
            fallback["algorithm"] = "D3-Force (Fallback to FR)"
            return fallback
        except Exception as e:
            print(f"[Warning] D3 call exception: {e}. Falling back to FR.")
            fallback = self.generate_fr()
            fallback["algorithm"] = "D3-Force (Fallback to FR)"
            return fallback
    # =========================================================
    # Aesthetic Baseline Methods (Integration of Sandbox)
    # =========================================================
    def _run_aesthetic_opt(self, config: dict, algo_name: str, iterations: int = 800) -> Dict[str, Any]:
    
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, algo_name)
            
        pos = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 逐个处理连通分量 (与 generate_kk 的逻辑对齐)
        for component in nx.connected_components(self.G):
            sub_G = self.G.subgraph(component)
            N = sub_G.number_of_nodes()
            
            if N == 1:
                pos[list(sub_G.nodes())[0]] = np.array([0.0, 0.0])
                continue
                
            nodes = list(sub_G.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            
            # 动态构建 PyTorch Tensors
            adj = torch.zeros(1, N, N, device=device)
            d_sp = torch.zeros(1, N, N, device=device)
            
            # 使用 networkx 计算最短路径
            paths = dict(nx.shortest_path_length(sub_G))
            for u in nodes:
                for v in nodes:
                    i, j = node_to_idx[u], node_to_idx[v]
                    if v in paths[u]:
                        d = paths[u][v]
                        d_sp[0, i, j] = d
                        if d == 1:
                            adj[0, i, j] = 1.0
                            
            eye = torch.eye(N, device=device).bool()
            # 加上 1e-5 防止除零
            weights = 1.0 / (d_sp ** 2 + 1e-5)
            weights[0, eye] = 0.0 
            mask = torch.ones(1, N, device=device).bool()
            
            # 初始化坐标 (加入随机种子保证可重复性)
            torch.manual_seed(42)
            init_pos = torch.randn(1, N, 2, device=device) * 2.0
            
            model = SandboxOptimizer(init_pos).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.1)
            criterion = AestheticMetricLoss(**config).to(device)
            
            # 迭代优化
            for _ in range(iterations):
                optimizer.zero_grad()
                p = model()
                total_loss, _, _ = criterion(p, d_sp, weights, mask, adj)
                total_loss.backward()
                optimizer.step()
                
            # 提取坐标并居中
            final_pos = model().detach().cpu().numpy()[0]
            final_pos = final_pos - final_pos.mean(axis=0)
            
            # 映射回原始节点ID
            for i, n in enumerate(nodes):
                pos[n] = final_pos[i]
            
            try:
                del adj, d_sp, weights, mask, eye
                del init_pos, model, optimizer, criterion
                del p, total_loss  # 彻底摧毁计算图引用
            except NameError:
                pass
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return self._export_json(pos, algo_name)

    # ---------------- 8 个 Baseline 对外接口 ----------------
    def generate_aes_base(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 0.0, 'w_anr': 0.0, 'w_np': 0.0}
        return self._run_aesthetic_opt(cfg, "Aes_Base_PureStress")
        
    def generate_aes_exp1(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 1.0, 'w_anr': 0.0, 'w_np': 0.0}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_1_VR")
        
    def generate_aes_exp2(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 0.0, 'w_anr': 1.0, 'w_np': 0.0}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_2_ANR")
        
    def generate_aes_exp3(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 0.0, 'w_anr': 0.0, 'w_np': 0.3}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_3_NP")
        
    def generate_aes_exp4(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 1.0, 'w_anr': 1.0, 'w_np': 0.0}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_4_VR_ANR")
        
    def generate_aes_exp5(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 1.0, 'w_anr': 0.0, 'w_np': 0.3}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_5_VR_NP")
        
    def generate_aes_exp6(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 0.0, 'w_anr': 1.0, 'w_np': 1.0}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_6_ANR_NP")
        
    def generate_aes_exp7(self) -> Dict[str, Any]:
        cfg = {'w_stress': 1.0, 'w_vr': 1.0, 'w_anr': 1.0, 'w_np': 0.3}
        return self._run_aesthetic_opt(cfg, "Aes_Exp_7_ALL")
    
        
    @classmethod
    def load_coregd_model(cls, model_path: str, config_path: str):
        """
        Statically load the Core-GD PyTorch model. 
        """
        if cls._coregd_model is None:
            if torch is None:
                raise ImportError("PyTorch or PyG ecosystem is not installed.")
                
            cls._coregd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                cls._coregd_config = AttrDict(config_dict)
                
            cls._coregd_config.use_cupy = False
            model = get_model(cls._coregd_config)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(cls._coregd_device)))
            model.eval()
            model = model.to(cls._coregd_device)
            
            cls._coregd_model = model

    def generate_coregd(self) -> Dict[str, Any]:
        """
        Algorithm: Core-GD (NeuralDrawer) Deep Learning Layout
        """
        if self.G.number_of_nodes() == 0:
            return self._export_json({}, "Core-GD")

        if self._coregd_model is None:
            fallback = self.generate_fr()
            fallback["algorithm"] = "Core-GD (Fallback)"
            return fallback

        try:
            # 1. 构建带有完整初始特征的 PyG Data
            nodes = list(self.G.nodes())
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            idx_to_node = {i: node for i, node in enumerate(nodes)}
            n_nodes = len(nodes)
            
            edge_index_src = []
            edge_index_dst = []
            for u, v in self.G.edges():
                edge_index_src.extend([node_to_idx[u], node_to_idx[v]])
                edge_index_dst.extend([node_to_idx[v], node_to_idx[u]])
                
            if len(edge_index_src) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            
            # 【核心修复】：必须初始化节点特征 x (提供随机的二维初始坐标)，否则模型必崩
            x = torch.rand((n_nodes, 2), dtype=torch.float)
            
            pyg_data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
            pyg_data.index = 0  # 绑定索引，防越界
            
            dataset_nd = [pyg_data]
            
            coarsened = None
            coarsening_matrices = None
            
            if self._coregd_config.get('coarsen', False):
                if create_coarsened_dataset is None:
                    raise RuntimeError("Missing create_coarsened_dataset function for Core-GD")
                # 开启粗化后，模型会先在骨架上画，再上采样填充细节，彻底解决毛线团问题
                dataset_nd, coarsened, coarsening_matrices = create_coarsened_dataset(self._coregd_config, dataset_nd)
            else:
                dataset_nd = preprocess_dataset(dataset_nd, self._coregd_config)
                
            processed_data = dataset_nd[0].to(self._coregd_device)
            batch = Batch.from_data_list([processed_data])
            
            # 3. 闭环执行多尺度前向传播 (Forward Pass)
            with torch.no_grad():
                layer_num = max(int(self._coregd_config.get('iter_mean', 10)), 1)
                
                # 初始层推演
                pred, states = self._coregd_model(batch, layer_num, return_layers=True)
                
                # U-Net 架构的多尺度上采样推演 (如果开启了粗化)
                if self._coregd_config.get('coarsen', False) and coarsening_matrices is not None:
                    matrices_for_graph = coarsening_matrices[0]
                    noise_val = self._coregd_config.get('coarsen_noise', 0.01)
                    for i in range(1, len(matrices_for_graph) + 1):
                        batch = go_to_coarser_graph(
                            batch, states[-1], self._coregd_device, 
                            True, coarsened, coarsening_matrices, noise=noise_val
                        )
                        pred, states = self._coregd_model(batch, layer_num, encode=False, return_layers=True)
            
            # 4. 解析输出坐标并转码导出
            if isinstance(pred, tuple) or isinstance(pred, list):
                coords = pred[0] 
            else:
                coords = pred
                
            coords = coords.cpu().numpy()
            
            final_pos = {}
            for i in range(n_nodes):
                final_pos[idx_to_node[i]] = np.array([float(coords[i][0]), float(coords[i][1])])
                
            return self._export_json(final_pos, "Core-GD")

        except Exception as e:
            # 暴露完整的追踪栈以防万一
            import traceback
            traceback.print_exc()
            print(f"[Error] Core-GD inference failed: {e}. Falling back to FR.")
            fallback = self.generate_fr()
            fallback["algorithm"] = "Core-GD (Fallback)"
            return fallback