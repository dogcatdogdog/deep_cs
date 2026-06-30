import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProbabilisticLayoutLoss(nn.Module):
    """
    [DeepCS V10.0] The Probabilistic Topological Supervisor.
    Optimizes the ASGD Solver output without using any rigid distance constraints.
    Updated for Pure Metric Decoding: ALL direct L2 parameter regularizations are REMOVED.
    """
    def __init__(self, lambda_anchor=1.0, eps=1e-8):
        super(ProbabilisticLayoutLoss, self).__init__()
        # 锚点惩罚系数，防止整个系统在二维空间发生无限平移
        self.lambda_anchor = lambda_anchor
        # 数值稳定常数
        self.eps = eps

    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z=None, edge_index=None):
        N = pos.shape[0]
        device = pos.device

        # ==========================================
        # [核心修复 1]: 几何学强制归中，代替梯度惩罚
        # ==========================================
        # 直接把坐标系拉回原点，不产生任何回传梯度
        pos = pos - pos.mean(dim=0, keepdim=True)

        # ==========================================
        # Module 3: The Topological Compass (KL Divergence)
        # ==========================================
        # 1. Calculate pairwise squared distances
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 2]
        dist_sq = (diff ** 2).sum(dim=-1)           # [N, N]

        # 2. Student t-distribution (1 degree of freedom)
        q_unnorm = 1.0 / (1.0 + dist_sq)

        # 3. Clear diagonal
        eye = torch.eye(N, dtype=torch.bool, device=device)
        q_unnorm = q_unnorm.masked_fill(eye, 0.0)

        # 4. Global normalization
        q_sum = q_unnorm.sum().clamp(min=self.eps)
        Q_pred = q_unnorm / q_sum

        # 5. KL Divergence calculation
        valid_mask = P_target > 0
        P_valid = P_target[valid_mask]
        Q_valid = Q_pred[valid_mask].clamp(min=self.eps)

        loss_kl = torch.sum(P_valid * torch.log(P_valid / Q_valid))

        # ==========================================
        # Final Loss Fusion
        # ==========================================
        # [核心修复 2]: 总损失彻底剔除 loss_anchor
        total_loss = loss_kl

        loss_dict = {
            'kl_loss': loss_kl.item(),
            'reg_loss': 0.0,  
            'anchor_loss': 0.0  # 因为在计算图中被移除了，所以这里合法输出 0
        }
        
        return total_loss, loss_dict

class UmapLayoutLoss(nn.Module):

    def __init__(self, numNegSamples=5, gamma=1.0, eps=1e-8):
        super(UmapLayoutLoss, self).__init__()
        self.numNegSamples = numNegSamples
        self.gamma = gamma
        self.eps = eps
        
        # 物理碰撞体积参数 (a, b)
        # 通过非线性拟合获得，对应 UMAP 中 min_dist=0.1, spread=1.0 的拓扑流形
        # b < 1 是在微观尺度切断无限引力的数学核心
        self.aParam = 1
        self.bParam = 1

    def forward(self, pos, edgeWeight, w_edge, l_edge, q_node, Z=None, edgeIndex=None):
        if edgeIndex is None or edgeWeight is None:
            raise ValueError("edgeIndex and edgeWeight are required for UmapLayoutLoss.")
        
        N = pos.shape[0]
        device = pos.device

        # ==========================================
        # 1. 几何学强制归中 (平移不变性)
        # ==========================================
        pos = pos - pos.mean(dim=0, keepdim=True)

        row, col = edgeIndex
        numEdges = row.size(0)

        # ==========================================
        # 2. Module A: Attractive Force (正样本引力)
        # ==========================================
        posSrc = pos[row]
        posDst = pos[col]

        # 计算距离平方
        # [严谨数学防御]：由于后续有 bParam < 1 的分数幂运算，
        # 如果 distSqPos 绝对为 0，反向传播时导数会趋向无穷大 (NaN)。必须提前注入 eps。
        distSqPos = ((posSrc - posDst) ** 2).sum(dim=-1) + self.eps
        
        # 带有物理体积感知的高原概率核 (Plateau Kernel)
        qPos = 1.0 / (1.0 + self.aParam * (distSqPos ** self.bParam))

        lossPos = - torch.sum(edgeWeight * torch.log(qPos.clamp(min=self.eps)))

        # ==========================================
        # 3. Module B: Repulsive Force (负采样斥力)
        # ==========================================
        negIdx = torch.randint(0, N, (numEdges, self.numNegSamples), device=device)

        posSrcNeg = posSrc.unsqueeze(1)
        posDstNeg = pos[negIdx]

        distSqNeg = ((posSrcNeg - posDstNeg) ** 2).sum(dim=-1) + self.eps

        # 斥力场同样遵循该度量空间
        qNeg = 1.0 / (1.0 + self.aParam * (distSqNeg ** self.bParam))

        lossNegSum = - torch.sum(torch.log((1.0 - qNeg).clamp(min=self.eps)))
        lossNeg = (lossNegSum / self.numNegSamples) * self.gamma

        # ==========================================
        # 4. Final Loss Fusion
        # ==========================================
        totalLoss = lossPos + lossNeg

        lossDict = {
            'total_loss': totalLoss.item(),
            'pos_loss': lossPos.item(),
            'neg_loss': lossNeg.item(),
            'anchor_loss': 0.0 
        }
        
        return totalLoss, lossDict

        
        return totalLoss, lossDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticOrbitalLayoutLoss(nn.Module):

    def __init__(self, similarity_margin=0.7, sharpening_pow=3.0, gap_ratio=0.3, eps=1e-8):
        super(ProbabilisticOrbitalLayoutLoss, self).__init__()
        self.similarityMargin = similarity_margin
        self.sharpeningPow = sharpening_pow
        self.gapRatio = gap_ratio
        self.eps = eps

    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index=None, 
                lambda_cohesion=0.0, lambda_separation=0.0):
        N = pos.shape[0]
        device = pos.device

        # ==========================================
        # Module 1 & 2: 物理场生成 (视觉概率分布 Q)
        # ==========================================
        pos = pos - pos.mean(dim=0, keepdim=True)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        distSq = (diff ** 2).sum(dim=-1)
        
        qUnnorm = 1.0 / (1.0 + distSq)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        qUnnorm = qUnnorm.masked_fill(eye, 0.0)
        
        qSum = qUnnorm.sum().clamp(min=self.eps)
        Q_pred = qUnnorm / qSum

        # ==========================================
        # Module 3: 宏观 KL 损失
        # ==========================================
        validMask = P_target > 0
        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(P_target[validMask] / Q_pred[validMask].clamp(min=self.eps)))

        lossCohesion = torch.tensor(0.0, device=device)
        lossSeparation = torch.tensor(0.0, device=device)

        if Z_role is not None and (lambda_cohesion > 0 or lambda_separation > 0):
            zDetach = Z_role.detach()

            # ------------------------------------------
            # 步骤 A: 提取高对比度微观相似度 (S_role)
            # ------------------------------------------
            zNorm = torch.nn.functional.normalize(zDetach, p=2, dim=-1)
            cosSim = torch.matmul(zNorm, zNorm.transpose(0, 1))
            
            # ReLU 截断 + 归一化映射
            sRole = torch.relu(cosSim - self.similarityMargin) / (1.0 - self.similarityMargin + self.eps)
            sRole = torch.pow(sRole, self.sharpeningPow)

            # ------------------------------------------
            # 步骤 B: 轨道内聚 (Cohesion)
            # ------------------------------------------
            if lambda_cohesion > 0:
                Q_orbit = torch.matmul(torch.matmul(sRole.transpose(0, 1), Q_pred), sRole)
                Q_orbit = Q_orbit.masked_fill(eye, 0.0)
                Q_orbit = Q_orbit / Q_orbit.sum().clamp(min=self.eps)

                lossCohesion = torch.sum((Q_pred.clamp(min=self.eps) * torch.log(Q_pred.clamp(min=self.eps) / Q_orbit.clamp(min=self.eps)))[validMask])

            # ------------------------------------------
            # [核心优化] 步骤 C: 自适应统计轨道排斥 (Separation)
            # ------------------------------------------
            if lambda_separation > 0:
                adj = validMask.float()
                
                # 1. 计算局部邻域的平均概率密度 [N, 1]
                # 这反映了节点 i 周围“平均每个邻居分到了多少概率”
                neighborhoodQSum = (Q_pred * adj).sum(dim=-1, keepdim=True)
                deg = adj.sum(dim=-1, keepdim=True)
                avgNeighborhoodQ = neighborhoodQSum / (deg + self.eps)
                
                # 2. 生成基于比例的自适应 Margin [N, 1, 1]
                # 无论全图 Q 多么稀释，Margin 始终保持为邻域平均密度的固定比例
                adaptiveMargin = (self.gapRatio * avgNeighborhoodQ).unsqueeze(2)

                maskIjk = adj.unsqueeze(2) * adj.unsqueeze(1)
                deltaQ = torch.abs(Q_pred.unsqueeze(2) - Q_pred.unsqueeze(1))
                
                # 阶级鸿沟：0 代表兄弟，1 代表异类
                deltaZ = (1.0 - sRole).unsqueeze(0)
                
                # 违反约束的惩罚
                marginViolation = torch.relu(adaptiveMargin - deltaQ)
                # 使用 N 平方作为分母平滑梯度
                lossSeparation = torch.sum(maskIjk * deltaZ * marginViolation) / (N ** 2 + self.eps)

        totalLoss = lossGlobalKl + lambda_cohesion * lossCohesion + lambda_separation * lossSeparation

        lossDict = {
            'kl_loss': lossGlobalKl.item(),
            'cohesion_loss': lossCohesion.item(),
            'separation_loss': lossSeparation.item(),
        }
        return totalLoss, lossDict
        
        


class LocalTriadicOrbitalLoss(nn.Module):

    
    def __init__(self, sigmaRatio=0.3, contrastiveMargin=3.0, klMargin=1.5, bandWidth=0.8, beta=10.0, eps=1e-8):
        super(LocalTriadicOrbitalLoss, self).__init__()
        self.sigmaRatio = sigmaRatio
        self.contrastiveMargin = contrastiveMargin # 异类排斥的绝对裕度
        self.klMargin = klMargin                   # KL 散度停止拉扯的物理距离
        self.bandWidth = bandWidth                 # 同类节点允许错峰的软带宽
        self.beta = beta                           # 软化门控的陡峭系数
        self.eps = eps

    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
                lambda_cohesion=0.0, lambda_separation=0.0):
        numNodes = pos.shape[0]
        computeDevice = pos.device

        # ==========================================
        # 1. 宏观 KL 散度 (方案 2: Slack Margin)
        # ==========================================
        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
        distSqReal = (posDiff ** 2).sum(dim=-1)
        distMat = torch.sqrt(distSqReal + self.eps)
        
        # 方案 2 核心：物理距离进入 klMargin 后，有效距离视为 0，吸引梯度消失
        slackDist = F.relu(distMat - self.klMargin)
        qUnnorm = 1.0 / (1.0 + slackDist**2)
        
        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
        qSum = qUnnorm.sum().clamp(min=self.eps)
        qPred = qUnnorm / qSum

        validMask = P_target > 0
        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(P_target[validMask] / qPred[validMask].clamp(min=self.eps)))

        lossCohesion = torch.tensor(0.0, device=computeDevice)
        lossSeparation = torch.tensor(0.0, device=computeDevice)

        if Z_role is not None and (lambda_cohesion > 0 or lambda_separation > 0):
            # ==========================================
            # 2. RBF 相似度计算
            # ==========================================
            zDetach = Z_role.detach()
            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
            
            meanDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
            sigma = meanDist * self.sigmaRatio
            sRole = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))

            # ==========================================
            # 3. 三元组逻辑与 Hub 掩码
            # ==========================================
            adjMat = validMask.float()
            deg = adjMat.sum(dim=-1) 
            
            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()
            
            validTriplets = maskTriadic.sum().clamp(min=1.0)
            sRoleExp = sRole.unsqueeze(0).expand(numNodes, numNodes, numNodes)

            distKI = distMat.unsqueeze(2)
            distKJ = distMat.unsqueeze(1)
            deltaD = torch.abs(distKI - distKJ)

            # ==========================================
            # 4. 物理约束优化
            # ==========================================
            
            # (1) 软带宽凝聚 (Soft Bandwidth Cohesion)
            # 使用 Sigmoid 门控替代 ReLU，实现带宽边缘的平滑过渡
            # 只有当 deltaD 接近或超过 bandWidth 时，权重才迅速向 1 靠近
            if lambda_cohesion > 0:
                softGate = torch.sigmoid(self.beta * (deltaD - self.bandWidth))
                errCoh = torch.sum(maskTriadic * sRoleExp * softGate * (deltaD ** 2))
                lossCohesion = errCoh / validTriplets

            # (2) 对比分离 (Contrastive Separation)
            # 保持异类节点在物理空间中的最小间距要求
            if lambda_separation > 0:
                sepViolation = F.relu(self.contrastiveMargin - deltaD)
                errSep = torch.sum(maskTriadic * (1.0 - sRoleExp) * (sepViolation ** 2))
                lossSeparation = errSep / validTriplets

        totalLoss = lossGlobalKl + lambda_cohesion * lossCohesion + lambda_separation * lossSeparation

        return totalLoss, {
            'kl_loss': lossGlobalKl.item(),
            'cohesion_loss': lossCohesion.item(),
            'separation_loss': lossSeparation.item(),
        }
        
#class LatentBPROrbitalLoss(nn.Module):
#    def __init__(self, sigmaRatio=0.3, eps=1e-8):
#        super(LatentBPROrbitalLoss, self).__init__()
#        self.sigmaRatio = sigmaRatio
#        self.eps = eps
#
#    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
#                lambda_cohesion=0.0, lambda_separation=0.0):
#        numNodes = pos.shape[0]
#        computeDevice = pos.device
#        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
#
#        # ==========================================
#        # 1. 特征相似度场 (Z_role) 与 潜空间距离矩阵
#        # ==========================================
#        zDetach = Z_role.detach() if Z_role is not None else None
#        if zDetach is not None:
#            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
#            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
#            sigma = meanZDist * self.sigmaRatio
#            S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
#        else:
#            zDistSq = torch.zeros((numNodes, numNodes), device=computeDevice)
#            S_IJ = torch.ones((numNodes, numNodes), device=computeDevice)
#
#        # ==========================================
#        # 2. 纯粹拓扑 KL 散度 (宏观引力场)
#        # ==========================================
#        validMask = P_target > 0
#
#        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
#        distSqReal = (posDiff ** 2).sum(dim=-1)
#        
#        qUnnorm = 1.0 / (1.0 + distSqReal)
#        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
#        qSum = qUnnorm.sum().clamp(min=self.eps)
#        qPred = qUnnorm / qSum
#
#        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(
#            P_target[validMask] / qPred[validMask].clamp(min=self.eps)))
#
#        # ==========================================
#        # 3. 对数相对轨道物理约束 (Scale-Free Orbital Field)
#        # ==========================================
#        lossCohesion = torch.tensor(0.0, device=computeDevice)
#        lossSeparation = torch.tensor(0.0, device=computeDevice)
#
#        if zDetach is not None and (lambda_cohesion > 0 or lambda_separation > 0):
#            adjMat = validMask.float()
#            deg = adjMat.sum(dim=-1) 
#            
#            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
#            # Hub 定义：度数必须严格大于它的两个叶子邻居
#            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
#                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
#            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()
#            
#            distMat = torch.sqrt(distSqReal + self.eps)
#            logDistMat = torch.log(distMat)
#            
#            logDistKI = logDistMat.unsqueeze(2)
#            logDistKJ = logDistMat.unsqueeze(1)
#            deltaLogD = logDistKI - logDistKJ
#            
#            S_IJ_exp = S_IJ.unsqueeze(0).expand(numNodes, numNodes, numNodes)
#
#            hubDegrees = deg.clamp(min=1.0)
#            validHubCount = (maskTriadic.sum(dim=(1, 2)) > 0).float().sum().clamp(min=1.0)
#
#
#            if lambda_cohesion > 0:
#                # 强制同类半径相等。至于半径到底多大，交给 KL 引力和库仑斥力去博弈
#                errCohVar = maskTriadic * S_IJ_exp * (deltaLogD ** 2)
#                perHubCohVar = errCohVar.sum(dim=(1, 2)) / hubDegrees
#                
#                # [删除] 移除了强压迫性的 errCohPress，消除实心饼现象
#                
#                lossCohesion = perHubCohVar.sum() / validHubCount
#
#            # ---------------------------------------------------------
#            # (2) 分离项 (Separation)：基于 BPR 软排序与拓扑质量罗盘
#            # ---------------------------------------------------------
#            if lambda_separation > 0:
#                # [终极修正]：基于局部二跳拓扑质量的绝对排序罗盘
#                # mass = 自身度数 + 所有邻居度数之和 (Local Eigenvector Centrality)
#                mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
#                
#                massI = mass.view(1, -1, 1)
#                massJ = mass.view(1, 1, -1)
#                
#                # 罗盘判定：拓扑负载 (mass) 小的节点 I 必须在物理轨道内圈
#                dirMask = (massI < massJ).float()
#                
#                # BPR 排序推力 (Softplus 无阈值单向剥离)
#                violationIInner = F.softplus(logDistKI - logDistKJ)
#                violationJInner = F.softplus(logDistKJ - logDistKI)
#                
#                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
#                
#                # 仅在异类间 (1.0 - S_IJ_exp) 施加排斥力场
#                errSep = maskTriadic * (1.0 - S_IJ_exp) * directedViolation
#                
#                perHubSep = errSep.sum(dim=(1, 2)) / hubDegrees
#                lossSeparation = perHubSep.sum() / validHubCount
#
#        totalLoss = lossGlobalKl + lambda_cohesion * lossCohesion + lambda_separation * lossSeparation
#
#        return totalLoss, {
#            'kl_loss': lossGlobalKl.item(),
#            'cohesion_loss': lossCohesion.item(),
#            'separation_loss': lossSeparation.item(),
#        }

#class LatentBPROrbitalLoss(nn.Module):
#    def __init__(self, sigmaRatio=0.3, eps=1e-8):
#        super(LatentBPROrbitalLoss, self).__init__()
#        self.sigmaRatio = sigmaRatio
#        self.eps = eps
#
#    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
#                lambda_cohesion=0.0, lambda_separation=0.0):
#        numNodes = pos.shape[0]
#        computeDevice = pos.device
#        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
#
#        # ==========================================
#        # 1. 特征相似度场 (Z_role) 与 潜空间距离矩阵
#        # ==========================================
#        zDetach = Z_role.detach() if Z_role is not None else None
#        if zDetach is not None:
#            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
#            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
#            sigma = meanZDist * self.sigmaRatio
#            S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
#        else:
#            zDistSq = torch.zeros((numNodes, numNodes), device=computeDevice)
#            S_IJ = torch.ones((numNodes, numNodes), device=computeDevice)
#
#        # ==========================================
#        # 2. 纯粹拓扑 KL 散度 (宏观引力场)
#        # ==========================================
#        validMask = P_target > 0
#
#        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
#        distSqReal = (posDiff ** 2).sum(dim=-1)
#        
#        qUnnorm = 1.0 / (1.0 + distSqReal)
#        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
#        qSum = qUnnorm.sum().clamp(min=self.eps)
#        qPred = qUnnorm / qSum
#
#        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(
#            P_target[validMask] / qPred[validMask].clamp(min=self.eps)))
#
#        # ==========================================
#        # 3. 对数相对轨道物理约束 (Scale-Free Orbital Field)
#        # ==========================================
#        lossCohesion = torch.tensor(0.0, device=computeDevice)
#        lossSeparation = torch.tensor(0.0, device=computeDevice)
#
#        if zDetach is not None and (lambda_cohesion > 0 or lambda_separation > 0):
#            adjMat = validMask.float()
#            deg = adjMat.sum(dim=-1) 
#            
#            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
#            # Hub 定义：度数必须严格大于它的两个叶子邻居
#            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
#                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
#            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()
#            
#            distMat = torch.sqrt(distSqReal + self.eps)
#            logDistMat = torch.log(distMat)
#            
#            logDistKI = logDistMat.unsqueeze(2)
#            logDistKJ = logDistMat.unsqueeze(1)
#            deltaLogD = logDistKI - logDistKJ
#            
#            S_IJ_exp = S_IJ.unsqueeze(0).expand(numNodes, numNodes, numNodes)
#
#            hubDegrees = deg.clamp(min=1.0)
#            validHubCount = (maskTriadic.sum(dim=(1, 2)) > 0).float().sum().clamp(min=1.0)
#
#            # ---------------------------------------------------------
#            # (1) 凝聚项：方差对齐 + 1D 拓扑体积核盾 (打破径向单向塌缩)
#            # ---------------------------------------------------------
#            if lambda_cohesion > 0:
#                # 提取 Hub K 周围，与节点 I 同属一类的邻居数量 J (局部流形密度)
#                # shape: (K, I)
#                peerCount = (maskTriadic * S_IJ_exp).sum(dim=2)
#                
#                # Part A: 相对柔性牵引 (消除全局度数稀释，实施局部归一化)
#                # errCohVar shape: (K, I, J)
#                errCohVar = maskTriadic * S_IJ_exp * (deltaLogD ** 2)
#                
#                # [核心修改] 
#                # 1. 先对 J 求和，计算每个节点 I 偏离群体的方差总和
#                nodeCohVar = errCohVar.sum(dim=2) 
#                # 2. 局部归一化：除以该群体真实的兄弟数量，保证拉回梯度始终是饱满的 O(1)
#                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
#                # 3. 最后对 I 求和，汇总到 Hub K 层面
#                perHubCohVar = nodeCohVarNorm.sum(dim=1)
#                
#                # Part B: 绝对拓扑地板 (自适应 1D 几何膨胀)
#                maskHubToI = (maskTriadic.sum(dim=2) > 0).float()
#                
#                # 推演 1D 圆环最小安全对数半径: R = N / 2π => ln(R) = ln(N) - ln(2π)
#                mathLog2Pi = math.log(2 * math.pi)
#                shieldThreshold = torch.clamp(torch.log(peerCount.clamp(min=1.0)) - mathLog2Pi, min=0.0)
#                
#                # 只有当叶子被压到安全阈值以内时，才触发向外推力
#                errCoreShield = maskHubToI * F.relu(shieldThreshold - logDistMat) ** 2
#                
#                # 单求和项，绝不除以 hubDegrees，维持 O(1) 绝对刚性以对抗 KL 引力
#                perHubCoreShield = errCoreShield.sum(dim=1)
#                
#                lossCohesion = (perHubCohVar.sum() + perHubCoreShield.sum()) / validHubCount
#
##            # ---------------------------------------------------------
##            # (2) 分离项 (Separation)：基于 BPR 软排序与拓扑质量罗盘
##            # ---------------------------------------------------------
##            if lambda_separation > 0:
##                # mass = 自身度数 + 所有邻居度数之和 (Local Eigenvector Centrality)
##                mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
##                
##                massI = mass.view(1, -1, 1)
##                massJ = mass.view(1, 1, -1)
##                
##                # 罗盘判定：拓扑负载 (mass) 小的节点 I 必须在物理轨道内圈
##                dirMask = (massI < massJ).float()
##                
##                # BPR 排序推力 (Softplus 保证动量守恒，内圈受压由上方的核盾负责托底)
##                violationIInner = F.softplus(logDistKI - logDistKJ)
##                violationJInner = F.softplus(logDistKJ - logDistKI)
##                
##                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
##                
##                # 仅在异类间 (1.0 - S_IJ_exp) 施加排斥力场
##                errSep = maskTriadic * (1.0 - S_IJ_exp) * directedViolation
##                
##                perHubSep = errSep.sum(dim=(1, 2)) / hubDegrees
##                lossSeparation = perHubSep.sum() / validHubCount
#            # ---------------------------------------------------------
#            # (2) 分离项 (Separation)：结构对齐的拓扑力场 (V32)
#            # ---------------------------------------------------------
#            if lambda_separation > 0:
#                mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
#                massI = mass.view(1, -1, 1)
#                massJ = mass.view(1, 1, -1)
#                
#                # 罗盘判定
#                dirMask = (massI < massJ).float()
#                
#                # [修正 1]：使用线性质量差系数，提供可控的动力增量
#                # 质量差越大，推力越强，但不再使用平方项，防止 Epoch 0 爆炸
#                massDiff = (massJ - massI).clamp(min=1.0)
#                
#                # BPR 排序违规项
#                violation = F.softplus(logDistKI - logDistKJ) * massDiff
#                
#                # 提取异类有效配对掩码
#                heteroMask = maskTriadic * (1.0 - S_IJ_exp) * dirMask
#                
#                # [核心修正 2]：物理累加（移除 nodeSepNorm）
#                # 子 Hub 必须感受到所有邻居推力的总和，才能对抗同样是总和的 KL 引力
#                perHubSepSum = (heteroMask * violation).sum(dim=(1, 2))
#                
#                # [核心修正 3]：Hub 级量纲对齐
#                # 仅除以 hubDegrees，使 Hub 内部的平均排斥强度维持在 O(1)
#                perHubSep = perHubSepSum / hubDegrees
#                
#                # 全局 O(1) 约束：确保初始 SEP 与 KL (1.8) 量级接近，防止炸图 (KL > 2)
#                lossSeparation = perHubSep.sum() / validHubCount
#             
#
#        totalLoss = lossGlobalKl + lambda_cohesion * lossCohesion + lambda_separation * lossSeparation
#
#        return totalLoss, {
#            'kl_loss': lossGlobalKl.item(),
#            'cohesion_loss': lossCohesion.item(),
#            'separation_loss': lossSeparation.item(),
#        }


#            # ---------------------------------------------------------
#            # (2) 分离项 (Separation)：结构对齐的拓扑力场 (V32)
#            # ---------------------------------------------------------
#            if lambda_separation > 0:
#                mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
#                massI = mass.view(1, -1, 1)
#                massJ = mass.view(1, 1, -1)
#                
#                # 罗盘判定
#                dirMask = (massI < massJ).float()
#                
#                # [修正 1]：使用线性质量差系数，提供可控的动力增量
#                # 质量差越大，推力越强，但不再使用平方项，防止 Epoch 0 爆炸
#                massDiff = (massJ - massI).clamp(min=1.0)
#                
#                # BPR 排序违规项
#                violation = F.softplus(logDistKI - logDistKJ) * massDiff
#                
#                # 提取异类有效配对掩码
#                heteroMask = maskTriadic * (1.0 - S_IJ_exp) * dirMask
#                
#                # [核心修正 2]：物理累加（移除 nodeSepNorm）
#                # 子 Hub 必须感受到所有邻居推力的总和，才能对抗同样是总和的 KL 引力
#                perHubSepSum = (heteroMask * violation).sum(dim=(1, 2))
#                
#                # [核心修正 3]：Hub 级量纲对齐
#                # 仅除以 hubDegrees，使 Hub 内部的平均排斥强度维持在 O(1)
#                perHubSep = perHubSepSum / hubDegrees
#                
#                # 全局 O(1) 约束：确保初始 SEP 与 KL (1.8) 量级接近，防止炸图 (KL > 2)
#                lossSeparation = perHubSep.sum() / validHubCount

#class LatentBPROrbitalLoss(nn.Module):
#    # 新增 tauBase 参数，默认 0.5，控制对数空间下的基础容忍度
#    def __init__(self, sigmaRatio=0.3, tauBase=0.5, eps=1e-8):
#        super(LatentBPROrbitalLoss, self).__init__()
#        self.sigmaRatio = sigmaRatio
#        self.tauBase = tauBase
#        self.eps = eps
#
#    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
#                lambda_cohesion=0.0, lambda_separation=0.0):
#        numNodes = pos.shape[0]
#        computeDevice = pos.device
#        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
#
#        # ==========================================
#        # 1. 特征相似度场 (Z_role) 与 潜空间距离矩阵
#        # ==========================================
#        zDetach = Z_role.detach() if Z_role is not None else None
#        if zDetach is not None:
#            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
#            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
#            sigma = meanZDist * self.sigmaRatio
#            S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
#        else:
#            zDistSq = torch.zeros((numNodes, numNodes), device=computeDevice)
#            S_IJ = torch.ones((numNodes, numNodes), device=computeDevice)
#
#        # ==========================================
#        # 2. 纯粹拓扑 KL 散度 (宏观引力场)
#        # ==========================================
#        validMask = P_target > 0
#
#        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
#        distSqReal = (posDiff ** 2).sum(dim=-1)
#        
#        qUnnorm = 1.0 / (1.0 + distSqReal)
#        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
#        qSum = qUnnorm.sum().clamp(min=self.eps)
#        qPred = qUnnorm / qSum
#
#        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(
#            P_target[validMask] / qPred[validMask].clamp(min=self.eps)))
#
#        # ==========================================
#        # 3. 对数相对轨道物理约束 (Scale-Free Orbital Field)
#        # ==========================================
#        lossCohesion = torch.tensor(0.0, device=computeDevice)
#        lossSeparation = torch.tensor(0.0, device=computeDevice)
#
#        if zDetach is not None and (lambda_cohesion > 0 or lambda_separation > 0):
#            adjMat = validMask.float()
#            deg = adjMat.sum(dim=-1) 
#            
#            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
#            # Hub 定义：度数必须严格大于它的两个叶子邻居
#            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
#                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
#            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()
#            
#            distMat = torch.sqrt(distSqReal + self.eps)
#            logDistMat = torch.log(distMat)
#            
#            logDistKI = logDistMat.unsqueeze(2)
#            logDistKJ = logDistMat.unsqueeze(1)
#            deltaLogD = logDistKI - logDistKJ
#            
#            S_IJ_exp = S_IJ.unsqueeze(0).expand(numNodes, numNodes, numNodes)
#
#            hubDegrees = deg.clamp(min=1.0)
#            validHubCount = (maskTriadic.sum(dim=(1, 2)) > 0).float().sum().clamp(min=1.0)
#
#            # ---------------------------------------------------------
#            # (1) 凝聚项：2D 弹性方差对齐 + 2D 拓扑体积核盾
#            # ---------------------------------------------------------
#            if lambda_cohesion > 0:
#                # 提取 Hub K 周围，与节点 I 同属一类的邻居数量 J (局部流形密度)
#                peerCount = (maskTriadic * S_IJ_exp).sum(dim=2)
#                
#                # Part A: 相对柔性牵引 (引入 tauDynamic 实现 2D 轨道厚度容忍)
#                # tauDynamic 随拥挤度增加而收紧对数宽容度
#                tauDynamic = self.tauBase / peerCount.clamp(min=1.0).unsqueeze(2)
#                violationCoh = F.relu(torch.abs(deltaLogD) - tauDynamic)
#                errCohVar = maskTriadic * S_IJ_exp * (violationCoh ** 2)
#                
#                # 局部归一化拉回梯度
#                nodeCohVar = errCohVar.sum(dim=2) 
#                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
#                perHubCohVar = nodeCohVarNorm.sum(dim=1)
#                
#                # Part B: 绝对拓扑地板 (升级为 2D 面积计算: R = sqrt(N / pi))
#                maskHubToI = (maskTriadic.sum(dim=2) > 0).float()
#                mathLogPi = math.log(math.pi)
#                # 指数项转化为对数乘法: 0.5 * log(N) - 0.5 * log(pi)
#                shieldThreshold = torch.clamp(0.5 * torch.log(peerCount.clamp(min=1.0)) - 0.5 * mathLogPi, min=0.0)
#                
#                errCoreShield = maskHubToI * F.relu(shieldThreshold - logDistMat) ** 2
#                perHubCoreShield = errCoreShield.sum(dim=1)
#                
#                lossCohesion = (perHubCohVar.sum() + perHubCoreShield.sum()) / validHubCount
#
#            # ---------------------------------------------------------
#            # (2) 分离项：基于 Cauchy 分布的长程排斥力与拓扑罗盘
#            # ---------------------------------------------------------
#            if lambda_separation > 0:
#                # mass = 自身度数 + 所有邻居度数之和
#                mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
#                massI = mass.view(1, -1, 1)
#                massJ = mass.view(1, 1, -1)
#                
#                # 罗盘判定：拓扑负载 (mass) 小的节点 I 必须在物理轨道内圈
#                dirMask = (massI < massJ).float()
#                
#                # BPR 长程推力：利用 t-分布 (Cauchy) 的重尾特性替代 Softplus
#                # 提供 O(1/d^2) 级别的长程斥力以对抗 KL，劈开清晰的轨道间隙
#                mathPi = math.pi
#                violationIInner = -torch.log(0.5 + (1.0 / mathPi) * torch.atan(-deltaLogD) + self.eps)
#                violationJInner = -torch.log(0.5 + (1.0 / mathPi) * torch.atan(deltaLogD) + self.eps)
#                
#                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
#                
#                # 仅在异类间施加排斥力场
#                errSep = maskTriadic * (1.0 - S_IJ_exp) * directedViolation
#                
#                perHubSep = errSep.sum(dim=(1, 2)) / hubDegrees
#                lossSeparation = perHubSep.sum() / validHubCount
#              
#        totalLoss = lossGlobalKl + lambda_cohesion * lossCohesion + lambda_separation * lossSeparation
#
#        return totalLoss, {
#            'kl_loss': lossGlobalKl.item(),
#            'cohesion_loss': lossCohesion.item(),
#            'separation_loss': lossSeparation.item(),
#        }
#class LatentBPROrbitalLoss(nn.Module):
## 424
#    def __init__(self, sigmaRatio=0.3, tauBase=0.34, eps=1e-8):
#        super(LatentBPROrbitalLoss, self).__init__()
#        self.sigmaRatio = sigmaRatio
#        self.tauBase = tauBase
#        self.eps = eps
#
#    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
#                lambda_cohesion=0.0, lambda_separation=0.0):
#        numNodes = pos.shape[0]
#        computeDevice = pos.device
#        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
#
#        # ==========================================
#        # 1. 特征相似度场 (Z_role) 与 潜空间距离矩阵
#        # ==========================================
#        zDetach = Z_role.detach() if Z_role is not None else None
#        if zDetach is not None:
#            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
#            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
#            sigma = meanZDist * self.sigmaRatio
#            S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
#        else:
#            zDistSq = torch.zeros((numNodes, numNodes), device=computeDevice)
#            S_IJ = torch.ones((numNodes, numNodes), device=computeDevice)
#
#        # ==========================================
#        # 2. 纯粹拓扑 KL 散度 (宏观引力场)
#        # ==========================================
#        validMask = P_target > 0
#
#        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
#        distSqReal = (posDiff ** 2).sum(dim=-1)
#        
#        qUnnorm = 1.0 / (1.0 + distSqReal)
#        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
#        qSum = qUnnorm.sum().clamp(min=self.eps)
#        qPred = qUnnorm / qSum
#
#        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(
#            P_target[validMask] / qPred[validMask].clamp(min=self.eps)))
#
#        # ==========================================
#        # 3. 对数相对轨道物理约束 (Scale-Free Orbital Field)
#        # ==========================================
#        lossCohesion = torch.tensor(0.0, device=computeDevice)
#        lossSeparation = torch.tensor(0.0, device=computeDevice)
#
#        if zDetach is not None and (lambda_cohesion > 0 or lambda_separation > 0):
#            adjMat = validMask.float()
#            deg = adjMat.sum(dim=-1) 
#            
#            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
#            # Hub 定义：度数必须严格大于它的两个叶子邻居
#            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
#                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
#            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()
#            
#            distMat = torch.sqrt(distSqReal + self.eps)
#            logDistMat = torch.log(distMat)
#            
#            logDistKI = logDistMat.unsqueeze(2)
#            logDistKJ = logDistMat.unsqueeze(1)
#            deltaLogD = logDistKI - logDistKJ
#            
#            S_IJ_exp = S_IJ.unsqueeze(0).expand(numNodes, numNodes, numNodes)
#
#            hubDegrees = deg.clamp(min=1.0)
#            
#            # ==========================================
#            # 引入拓扑引力质量场 (Topological Mass Field)
#            # ==========================================
#            isValidHub = (maskTriadic.sum(dim=(1, 2)) > 0).float()
#            validHubCount = isValidHub.sum().clamp(min=1.0)
#            
#            # mass = 自身度数 + 所有邻居度数之和 (Local Eigenvector Centrality)
#            mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
#            
#            # 宏观广播权重计算与归一化
#            hubWeights = mass * isValidHub
#            totalWeight = hubWeights.sum().clamp(min=self.eps)
#            normHubWeights = hubWeights * (validHubCount / totalWeight)
#
##            # ---------------------------------------------------------
##            # (1) 凝聚项：2D 弹性方差对齐 + 2D 拓扑体积核盾
##            # ---------------------------------------------------------
##            if lambda_cohesion > 0:
##                # Part A: 相对柔性牵引 (2D 轨道厚度容忍)
##                peerCount = (maskTriadic * S_IJ_exp).sum(dim=2)
##                tauDynamic = self.tauBase / peerCount.clamp(min=1.0).unsqueeze(2)
##                violationCoh = F.relu(torch.abs(deltaLogD) - tauDynamic)
##                errCohVar = maskTriadic * S_IJ_exp * (violationCoh ** 2)
##                
##                # 局部微观方差汇总并完成 O(1) 归一化
##                nodeCohVar = errCohVar.sum(dim=2) 
##                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
##                perHubCohVarO1 = nodeCohVarNorm.sum(dim=1) / hubDegrees
##                
##                # Part B: 绝对拓扑地板 (2D 面积级安全半径)
##                maskHubToI = (maskTriadic.sum(dim=2) > 0).float()
##                mathLogPi = math.log(math.pi)
##                shieldThreshold = torch.clamp(0.5 * torch.log(peerCount.clamp(min=1.0)) - 0.5 * mathLogPi, min=0.0)
##                
##                errCoreShield = maskHubToI * F.relu(shieldThreshold - logDistMat) ** 2
##                perHubCoreShieldO1 = errCoreShield.sum(dim=1) / hubDegrees
##                
##                # 施加宏观拓扑质量广播权重
##                weightedHubCoh = (perHubCohVarO1 + perHubCoreShieldO1) * normHubWeights
##                lossCohesion = weightedHubCoh.sum() / validHubCount
#            # ---------------------------------------------------------
#            # (1) 凝聚项：2D 弹性局部宽容对齐 + 2D 热力学语义简并压力场
#            # ---------------------------------------------------------
#            if lambda_cohesion > 0:
#                # Part A: 相对柔性牵引 (2D 轨道厚度容忍，回归两两对比)
#                peerCount = (maskTriadic * S_IJ_exp).sum(dim=2)
#                
#                tauDynamic = self.tauBase
#                violationCoh = F.relu(torch.abs(deltaLogD) - tauDynamic)
#                errCohVar = maskTriadic * S_IJ_exp * (violationCoh ** 2)
#                
#                # 局部微观方差汇总并完成 O(1) 归一化
#                nodeCohVar = errCohVar.sum(dim=2) 
#                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
#                perHubCohVarO1 = nodeCohVarNorm.sum(dim=1) / hubDegrees
#                
#                # Part B: 热力学语义简并压力场 (Semantic Degeneracy Pressure)
#                # 彻底废除 ReLU 硬截断，采用连续的 Lennard-Jones 级排斥势能
#                maskHubToI = (maskTriadic.sum(dim=2) > 0).float()
#                
#                mathLogPi = math.log(math.pi)
#                # 动态临界对数半径 (复用 peerCount 作为语义有效 2D 面积)
#                uMin = 0.5 * torch.log(peerCount.clamp(min=1.0)) - 0.5 * mathLogPi
#                
#                # 连续指数势垒 (kappa = 4.0 决定力场刚度)
#                kappa = 4.0 
#                errDegeneracyPressure = maskHubToI * torch.exp(kappa * (uMin - logDistMat))
#                
#                perHubCoreShieldO1 = errDegeneracyPressure.sum(dim=1) / hubDegrees
#                
#                # 施加宏观拓扑质量广播权重
#                weightedHubCoh = (perHubCohVarO1 + perHubCoreShieldO1) * normHubWeights
#                lossCohesion = weightedHubCoh.sum() / validHubCount
#
##            # ---------------------------------------------------------
##            # (2) 分离项：长程排斥力 (Cauchy t-distribution) 与拓扑罗盘
##            # ---------------------------------------------------------
##            if lambda_separation > 0:
##                massI = mass.view(1, -1, 1)
##                massJ = mass.view(1, 1, -1)
##                
##                dirMask = (massI < massJ).float()
##                
##                # BPR 长程推力
##                mathPi = math.pi
##                violationIInner = -torch.log(0.5 + (1.0 / mathPi) * torch.atan(-deltaLogD) + self.eps)
##                violationJInner = -torch.log(0.5 + (1.0 / mathPi) * torch.atan(deltaLogD) + self.eps)
##                
##                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
##                errSep = maskTriadic * (1.0 - S_IJ_exp) * directedViolation
##                
##                # 微观排斥力 O(1) 归一化
##                perHubSepO1 = errSep.sum(dim=(1, 2)) / hubDegrees
##                
##                # 施加宏观拓扑质量广播权重
##                weightedHubSep = perHubSepO1 * normHubWeights
##                lossSeparation = weightedHubSep.sum() / validHubCount
#            # ---------------------------------------------------------
#            # (2) 分离项：Log-Lorentzian 拓扑排斥力 (严格对齐 KL 引力)
#            # ---------------------------------------------------------
#            if lambda_separation > 0:
#                massI = mass.view(1, -1, 1)
#                massJ = mass.view(1, 1, -1)
#                
#                dirMask = (massI < massJ).float()
#                
#                # 1. 提取轨道排序的绝对违规量 (引入 0.1 的物理 margin 确保轨道间隙)
#                margin = 0.1
#                # 如果 I 应该在内圈，但 d_I 超过了 d_J - margin，产生 >0 的违规量
#                violationIInner = F.relu(deltaLogD + margin)
#                # 如果 J 应该在内圈，但 d_J 超过了 d_I - margin，产生 >0 的违规量
#                violationJInner = F.relu(-deltaLogD + margin)
#                
#                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
#                
#                # 2. Log-Lorentzian 排斥势能 (重尾特性)
#                # 导数 f'(v) = 2v / (1+v^2)，最大推力严格锁定为 1.0，彻底消除梯度爆炸
#                errSep = maskTriadic * (1.0 - S_IJ_exp) * torch.log(1.0 + directedViolation ** 2)
#                
#                # 3. 微观排斥力 O(1) 归一化
#                perHubSepO1 = errSep.sum(dim=(1, 2)) / hubDegrees
#                
#                # 4. 施加宏观拓扑质量广播权重
#                weightedHubSep = perHubSepO1 * normHubWeights
#                lossSeparation = weightedHubSep.sum() / validHubCount
#        totalLoss = lossGlobalKl + lambda_cohesion * lossCohesion + lambda_separation * lossSeparation
#
#        return totalLoss, {
#            'kl_loss': lossGlobalKl.item(),
#            'cohesion_loss': lossCohesion.item(),
#            'separation_loss': lossSeparation.item(),
#        }



class LatentBPROrbitalLoss(nn.Module):
    # 426
    def __init__(self, sigmaRatio=0.3, tauBase=0.0, eps=1e-8):
        super(LatentBPROrbitalLoss, self).__init__()
        self.sigmaRatio = sigmaRatio
        self.tauBase = tauBase
        self.eps = eps

    # [核心修改 1]: 增加 lambda_kl=1.0 参数
    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
                lambda_cohesion=0.0, lambda_separation=0.0, lambda_kl=1.0):
        numNodes = pos.shape[0]
        computeDevice = pos.device
        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)

        # ==========================================
        # 1. 特征相似度场 (Z_role) 与 潜空间距离矩阵
        # ==========================================
        zDetach = Z_role.detach() if Z_role is not None else None
        if zDetach is not None:
            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
            sigma = meanZDist * self.sigmaRatio
            S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
        else:
            zDistSq = torch.zeros((numNodes, numNodes), device=computeDevice)
            S_IJ = torch.ones((numNodes, numNodes), device=computeDevice)

        # ==========================================
        # 2. 纯粹拓扑 KL 散度 (宏观引力场)
        # ==========================================
        validMask = P_target > 0

        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
        distSqReal = (posDiff ** 2).sum(dim=-1)
        
        qUnnorm = 1.0 / (1.0 + distSqReal)
        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
        qSum = qUnnorm.sum().clamp(min=self.eps)
        qPred = qUnnorm / qSum

        lossGlobalKl = torch.sum(P_target[validMask] * torch.log(
            P_target[validMask] / qPred[validMask].clamp(min=self.eps)))

        # ==========================================
        # 3. 对数相对轨道物理约束 (Scale-Free Orbital Field)
        # ==========================================
        lossCohesion = torch.tensor(0.0, device=computeDevice)
        lossSeparation = torch.tensor(0.0, device=computeDevice)

        if zDetach is not None and (lambda_cohesion > 0 or lambda_separation > 0):
            adjMat = validMask.float()
            deg = adjMat.sum(dim=-1) 
            
            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
            # Hub 定义：度数必须严格大于它的两个叶子邻居
            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()
            
            distMat = torch.sqrt(distSqReal + self.eps)
            logDistMat = torch.log(distMat)
            
            logDistKI = logDistMat.unsqueeze(2)
            logDistKJ = logDistMat.unsqueeze(1)
            deltaLogD = logDistKI - logDistKJ
            
            S_IJ_exp = S_IJ.unsqueeze(0).expand(numNodes, numNodes, numNodes)

            hubDegrees = deg.clamp(min=1.0)
            
            # ==========================================
            # 引入拓扑引力质量场 (Topological Mass Field)
            # ==========================================
            isValidHub = (maskTriadic.sum(dim=(1, 2)) > 0).float()
            validHubCount = isValidHub.sum().clamp(min=1.0)
            
            # mass = 自身度数 + 所有邻居度数之和 (Local Eigenvector Centrality)
            mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
            
            # 宏观广播权重计算与归一化
            hubWeights = mass * isValidHub
            totalWeight = hubWeights.sum().clamp(min=self.eps)
            normHubWeights = hubWeights * (validHubCount / totalWeight)

            # ---------------------------------------------------------
            # (1) 凝聚项：2D 弹性局部宽容对齐 + 2D 热力学语义简并压力场
            # ---------------------------------------------------------
            if lambda_cohesion > 0:
                # Part A: 相对柔性牵引 (2D 轨道厚度容忍，回归两两对比)
                peerCount = (maskTriadic * S_IJ_exp).sum(dim=2)
                
                tauDynamic = self.tauBase
                violationCoh = F.relu(torch.abs(deltaLogD) - tauDynamic)
                errCohVar = maskTriadic * S_IJ_exp * (violationCoh ** 2)
                
                # 局部微观方差汇总并完成 O(1) 归一化
                nodeCohVar = errCohVar.sum(dim=2) 
                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
                perHubCohVarO1 = nodeCohVarNorm.sum(dim=1) / hubDegrees
                
                # Part B: 热力学语义简并压力场 (Semantic Degeneracy Pressure)
                maskHubToI = (maskTriadic.sum(dim=2) > 0).float()
                
                mathLogPi = math.log(math.pi)
                uMin = 0.5 * torch.log(peerCount.clamp(min=1.0)) - 0.5 * mathLogPi
                
                kappa = 4.0 
                errDegeneracyPressure = maskHubToI * torch.exp(kappa * (uMin - logDistMat))
                
                perHubCoreShieldO1 = errDegeneracyPressure.sum(dim=1) / hubDegrees
                
                # 施加宏观拓扑质量广播权重
                weightedHubCoh = (perHubCohVarO1 + perHubCoreShieldO1) * normHubWeights
                lossCohesion = weightedHubCoh.sum() / validHubCount

            # ---------------------------------------------------------
            # (2) 分离项：Log-Lorentzian 拓扑排斥力 (严格对齐 KL 引力)
            # ---------------------------------------------------------
            if lambda_separation > 0:
                massI = mass.view(1, -1, 1)
                massJ = mass.view(1, 1, -1)
                
                dirMask = (massI < massJ).float()
                
                # 1. 提取轨道排序的绝对违规量 (引入 0.1 的物理 margin 确保轨道间隙)
                margin = 0.1
                violationIInner = F.relu(deltaLogD + margin)
                violationJInner = F.relu(-deltaLogD + margin)
                
                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
                
                # 2. Log-Lorentzian 排斥势能 
                errSep = maskTriadic * (1.0 - S_IJ_exp) * torch.log(1.0 + directedViolation ** 2)
                
                # 3. 微观排斥力 O(1) 归一化
                perHubSepO1 = errSep.sum(dim=(1, 2)) / hubDegrees
                
                # 4. 施加宏观拓扑质量广播权重
                weightedHubSep = perHubSepO1 * normHubWeights
                lossSeparation = weightedHubSep.sum() / validHubCount

        # [核心修改 2]: 让 lambda_kl 直接乘以带有梯度的张量
        totalLoss = (lambda_kl * lossGlobalKl) + (lambda_cohesion * lossCohesion) + (lambda_separation * lossSeparation)

        return totalLoss, {
            'kl_loss': lossGlobalKl.item(),
            'cohesion_loss': lossCohesion.item(),
            'separation_loss': lossSeparation.item(),
        }        


#class LocalRoleAwareKLLoss(nn.Module):
#    """
#    Unified Local KL Loss (coh + sep merged)
#
#    Core idea:
#    - Global KL: controls overall structure
#    - Local KL: refines neighborhood distribution around hubs
#
#    Key properties:
#    - Does NOT break global KL radial ordering
#    - Encourages same-role nodes to lie on similar radius (softly)
#    - Produces layered but thick rings
#    """
#
#    def __init__(
#        self,
#        sigmaRatio=0.3,     # role similarity bandwidth
#        tau_geom=0.5,       # geometry temperature (radius softness)
#        tau_sem=0.5,        # semantic temperature (role grouping sharpness)
#        eps=1e-8
#    ):
#        super().__init__()
#        self.sigmaRatio = sigmaRatio
#        self.tau_geom = tau_geom
#        self.tau_sem = tau_sem
#        self.eps = eps
#
#    def forward(
#        self,
#        pos,
#        P_target,
#        w_edge,
#        l_edge,
#        q_node,
#        Z_role,
#        edge_index,
#        lambda_local=1.0,
#        lambda_kl=1.0
#    ):
#        device = pos.device
#        N = pos.shape[0]
#
#        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
#
#        # ==========================================
#        # 1. Global KL (unchanged)
#        # ==========================================
#        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
#        distSq = (posDiff ** 2).sum(dim=-1)
#
#        qUnnorm = 1.0 / (1.0 + distSq)
#        qUnnorm = qUnnorm.masked_fill(eye_mask, 0.0)
#        qPred = qUnnorm / qUnnorm.sum().clamp(min=self.eps)
#
#        validMask = P_target > 0
#        lossGlobalKL = torch.sum(
#            P_target[validMask] *
#            torch.log(P_target[validMask] / qPred[validMask].clamp(min=self.eps))
#        )
#
#        # ==========================================
#        # 2. Role similarity S_ij (peer-based)
#        # ==========================================
#        z = Z_role.detach()
#
#        zDiff = z.unsqueeze(0) - z.unsqueeze(1)
#        zDistSq = (zDiff ** 2).sum(dim=-1)
#
#        meanDist = torch.sqrt(zDistSq + self.eps)[~eye_mask].mean().clamp(min=self.eps)
#        sigma = meanDist * self.sigmaRatio
#
#        S_ij = torch.exp(-zDistSq / (sigma ** 2 + self.eps))
#
#        # ==========================================
#        # 3. Graph structure
#        # ==========================================
#        adj = (P_target > 0).float()
#        deg = adj.sum(dim=-1)
#
#        # hub mask（与你原逻辑一致）
#        hubMask = (deg.unsqueeze(1) > deg.unsqueeze(0)).float()
#
#        # only keep valid hubs
#        isHub = (hubMask.sum(dim=1) > 0)
#
#        # mass weighting
#        mass = deg + torch.matmul(adj, deg.unsqueeze(1)).squeeze(1)
#        mass = mass * isHub.float()
#
#        if mass.sum() < self.eps:
#            lossLocal = torch.tensor(0.0, device=device)
#        else:
#            hubWeights = mass / mass.sum().clamp(min=self.eps)
#
#            # ==========================================
#            # 4. Local KL
#            # ==========================================
#            dist = torch.sqrt(distSq + self.eps)
#            logDist = torch.log(dist + self.eps)
#
#            lossLocal = 0.0
#            validHubCount = 0
#
#            for k in range(N):
#                if not isHub[k]:
#                    continue
#
#                neighbors = (adj[k] > 0).nonzero(as_tuple=False).squeeze(-1)
#
#                if neighbors.numel() < 2:
#                    continue
#
#                validHubCount += 1
#
#                # ------------------------------
#                # P(i|k): role-induced distribution
#                # ------------------------------
#                S_sub = S_ij[neighbors][:, neighbors]  # [M, M]
#                a_i = S_sub.sum(dim=1)  # [M]
#
#                p = F.softmax(a_i / self.tau_sem, dim=0)
#
#                # ------------------------------
#                # Q(i|k): geometry-induced distribution
#                # ------------------------------
#                log_r = logDist[neighbors, k]  # [M]
#                q = F.softmax(-log_r / self.tau_geom, dim=0)
#
#                # ------------------------------
#                # KL divergence
#                # ------------------------------
#                kl = torch.sum(p * (torch.log(p + self.eps) - torch.log(q + self.eps)))
#
#                lossLocal += hubWeights[k] * kl
#
#            if validHubCount > 0:
#                lossLocal = lossLocal
#            else:
#                lossLocal = torch.tensor(0.0, device=device)
#
#        # ==========================================
#        # 5. Total loss
#        # ==========================================
#        totalLoss = lambda_kl * lossGlobalKL + lambda_local * lossLocal
#
#        return totalLoss, {
#            "kl_global": lossGlobalKL.item(),
#            "kl_local": lossLocal.item()
#        }

class LocalRoleAwareKLLoss(nn.Module):
    def __init__(self, sigmaRatio=0.3, tau_geom=0.5, tau_sem=0.5, eps=1e-8):
        super().__init__()
        self.sigmaRatio = sigmaRatio
        self.tau_geom = tau_geom
        self.tau_sem = tau_sem
        self.eps = eps

    def forward(self, pos, P_target, Z_role,edge_index,w_edge,l_edge,q_node,lambda_kl=1.0, lambda_local=1.0):
        device = pos.device
        N = pos.shape[0]
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)

        # 1. Global KL (逻辑不变)
        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
        distSq = (posDiff ** 2).sum(dim=-1)
        qUnnorm = 1.0 / (1.0 + distSq)
        qUnnorm = qUnnorm.masked_fill(eye_mask, 0.0)
        qPred = qUnnorm / qUnnorm.sum().clamp(min=self.eps)

        validMask = P_target > 0
        lossGlobalKL = torch.sum(
            P_target[validMask] *
            torch.log((P_target[validMask] + self.eps) / (qPred[validMask] + self.eps))
        )

        # 2. Role Similarity & Density (关键重构)
        z = Z_role.detach()
        zDistSq = torch.cdist(z, z, p=2)**2
        meanDist = torch.sqrt(zDistSq + self.eps)[~eye_mask].mean().clamp(min=self.eps)
        sigma = meanDist * self.sigmaRatio
        S_ij = torch.exp(-zDistSq / (sigma ** 2 + self.eps))

        # 3. Hub Weighting
        adj = (P_target > 0).float()
        deg = adj.sum(dim=-1)
        isHub = (torch.matmul(adj, deg.unsqueeze(1)).squeeze(1) > 0) # 简化的 hub 判断
        mass = (deg + torch.matmul(adj, deg.unsqueeze(1)).squeeze(1)) * isHub.float()
        
        if mass.sum() < self.eps:
            return lambda_kl * lossGlobalKL, {"kl_global": lossGlobalKL.item(), "kl_local": 0.0}
        hubWeights = mass / mass.sum().clamp(min=self.eps)

        # 4. Local KL (密度校正版)
        dist = torch.sqrt(distSq + self.eps)
        lossLocal = 0.0
        
        for k in range(N):
            if not isHub[k]: continue
            
            neighbors = adj[k].nonzero(as_tuple=False).squeeze(-1)
            if neighbors.numel() < 2: continue

            # A. 提取邻域信息
            # n_P: 全局拓扑先验 [M]
            n_P = P_target[k, neighbors]
            # n_S_sub: 邻域内的相似度矩阵 [M, M]
            n_S_sub = S_ij[neighbors][:, neighbors]
            # rho: 邻域角色密度 (即凝聚力得分 a_i) [M]
            rho = n_S_sub.sum(dim=1).clamp(min=self.eps)
            
            # B. 密度校正的目标分布 P(i|k)
            # 公式: p \propto (P_target / rho) * exp(rho / tau_sem)
            # 注意: 使用 log-sum-exp 技巧提高数值稳定性
            log_p_unnorm = torch.log(n_P + self.eps) - torch.log(rho) + (rho / self.tau_sem)
            p_target_local = F.softmax(log_p_unnorm, dim=0)

            # C. 几何预测分布 Q(i|k)
            n_log_r = torch.log(dist[neighbors, k] + self.eps)
            q_pred_local = F.softmax(-n_log_r / self.tau_geom, dim=0)

            # D. 计算 KL
            kl = torch.sum(p_target_local * (
                torch.log(p_target_local + self.eps) - torch.log(q_pred_local + self.eps)
            ))
            lossLocal += hubWeights[k] * kl

        totalLoss = lambda_kl * lossGlobalKL + lambda_local * lossLocal
        return totalLoss, {"kl_global": lossGlobalKL.item(), "kl_local": lossLocal.item()}

import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentBPOrbitalRefineLoss(nn.Module):
    """
    KL-safe refinement loss

    Key idea:
    - Global KL: defines ALL ordering (sacred)
    - Cohesion: reduces within-role variance (no ordering impact)
    - Separation: only fixes KL violations (no independent ordering)
    """

    def __init__(self, sigmaRatio=0.3, eps=1e-8):
        super().__init__()
        self.sigmaRatio = sigmaRatio
        self.eps = eps

    def forward(
        self,
        pos,
        P_target,
        w_edge,
        l_edge,
        q_node,
        Z_role,
        edge_index,
        lambda_kl=1.0,
        lambda_coh=0.1,
        lambda_sep=0.1
    ):
        device = pos.device
        N = pos.shape[0]

        eye = torch.eye(N, dtype=torch.bool, device=device)

        # =========================
        # 1. Global KL (unchanged)
        # =========================
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist2 = (diff ** 2).sum(dim=-1)

        q = 1.0 / (1.0 + dist2)
        q = q.masked_fill(eye, 0.0)
        q = q / q.sum().clamp(min=self.eps)

        valid = P_target > 0
        loss_global = torch.sum(
            P_target[valid] *
            torch.log(P_target[valid] / q[valid].clamp(min=self.eps))
        )

        # =========================
        # 2. Role similarity
        # =========================
        z = Z_role.detach()
        zdist = ((z.unsqueeze(0) - z.unsqueeze(1)) ** 2).sum(-1)

        mean = torch.sqrt(zdist + self.eps)[~eye].mean().clamp(min=self.eps)
        sigma = mean * self.sigmaRatio

        S = torch.exp(-zdist / (sigma ** 2 + self.eps))

        # =========================
        # 3. Graph structure
        # =========================
        adj = (P_target > 0).float()
        deg = adj.sum(-1).clamp(min=1.0)

        hub = (deg.unsqueeze(1) > deg.unsqueeze(0)).float()
        is_hub = (hub.sum(dim=1) > 0)

        mass = (deg + torch.matmul(adj, deg.unsqueeze(1)).squeeze(1)) * is_hub.float()
        if mass.sum() < self.eps:
            return loss_global, {"kl": loss_global.item(), "coh": 0.0, "sep": 0.0}

        w_hub = mass / mass.sum().clamp(min=self.eps)

        # =========================
        # 4. geometry
        # =========================
        dist = torch.sqrt(dist2 + self.eps)
        log_r = torch.log(dist + self.eps)

        loss_coh = 0.0
        loss_sep = 0.0

        for k in range(N):
            if not is_hub[k]:
                continue

            nbr = (adj[k] > 0).nonzero(as_tuple=False).squeeze(-1)
            if nbr.numel() < 2:
                continue

            log_rk = log_r[nbr, k]

            # =========================
            # COH: variance only
            # =========================
            mu = log_rk.mean()
            coh = S[nbr][:, nbr] * (log_rk - mu).unsqueeze(1) ** 2
            loss_coh += w_hub[k] * coh.mean()

            # =========================
            # SEP: KL residual only
            # =========================
            Q_local = q[nbr][:, nbr]
            P_local = P_target[nbr][:, nbr]

            residual = F.relu(P_local - Q_local)

            sep = (1.0 - S[nbr][:, nbr]) * residual * (
                (log_rk.unsqueeze(0) - log_rk.unsqueeze(1)) ** 2
            )

            loss_sep += w_hub[k] * sep.mean()

        # =========================
        # total
        # =========================
        total = (
            lambda_kl * loss_global +
            lambda_coh * loss_coh +
            lambda_sep * loss_sep
        )

        return total, {
            "kl_loss": loss_global.item(),
            "cohesion_loss": loss_coh.item(),
            "separation_loss": loss_sep.item()
        }
        
class MetricModulatedKLLoss(nn.Module):

    def __init__(self, sigmaRatio=0.3, alpha=0.5, eps=1e-8):
        super().__init__()
        self.sigmaRatio = sigmaRatio
        self.alpha = alpha
        self.eps = eps

    def forward(self, pos, P_target, Z_role):

        device = pos.device
        N = pos.shape[0]
        eye = torch.eye(N, device=device, dtype=torch.bool)

        # =========================================================
        # 1. Global KL (unchanged structure)
        # =========================================================
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist2 = (diff ** 2).sum(dim=-1)

        q_unnorm = 1.0 / (1.0 + dist2)
        q_unnorm = q_unnorm.masked_fill(eye, 0.0)
        q = q_unnorm / q_unnorm.sum().clamp(min=self.eps)

        valid = P_target > 0
        loss_kl = torch.sum(
            P_target[valid] * torch.log(P_target[valid] / q[valid].clamp(min=self.eps))
        )

        # =========================================================
        # 2. Role similarity field (only defines geometry modulation)
        # =========================================================
        z = Z_role.detach()

        z_diff = z.unsqueeze(0) - z.unsqueeze(1)
        z_dist2 = (z_diff ** 2).sum(dim=-1)

        mean_dist = torch.sqrt(z_dist2 + self.eps)[~eye].mean().clamp(min=self.eps)
        sigma = mean_dist * self.sigmaRatio

        S = torch.exp(-z_dist2 / (sigma ** 2 + self.eps))

        # =========================================================
        # 3. Metric tensor field (关键核心)
        # =========================================================
        # direction structure: outer product in latent role space
        z_norm = z / (z.norm(dim=-1, keepdim=True) + self.eps)

        dir_tensor = torch.einsum('id,jd->ijd', z_norm, z_norm)

        # metric field: identity + low-rank role modulation
        M = S.unsqueeze(-1) * dir_tensor   # [N,N,D,D]

        I = torch.eye(pos.shape[-1], device=device)
        G = I + self.alpha * M.mean(dim=0)  # global approx metric

        # =========================================================
        # 4. Metric-modulated distance (NO extra loss)
        # =========================================================
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # [N,N,2]

        # x^T G x
        gx = torch.einsum('ijd,dd->ijd', diff, G)
        dist2_mod = (gx * diff).sum(dim=-1)

        # =========================================================
        # 5. KL under modified geometry
        # =========================================================
        q_mod = 1.0 / (1.0 + dist2_mod)
        q_mod = q_mod.masked_fill(eye, 0.0)
        q_mod = q_mod / q_mod.sum().clamp(min=self.eps)

        loss = torch.sum(
            P_target[valid] * torch.log(P_target[valid] / q_mod[valid].clamp(min=self.eps))
        )

        return loss, {
            "kl": loss_kl.item(),
            "kl_mod": loss.item()
        }
class LocalRoleMetricLoss(nn.Module):

    def __init__(self, max_hops=2, tau_mass=0.5):
        super(LocalRoleMetricLoss, self).__init__()
        # 局部地盘的界定范围。2 跳正好覆盖“叶子-Hub-叶子”的局部星系结构。
        self.max_hops = max_hops
        # 结构指纹相似度的温度系数
        self.tau_mass = tau_mass

    def forward(self, Z_role, edge_index):
        device = Z_role.device
        N = Z_role.shape[0]

        # =======================================================
        # 1. 构建绝对物理法则的 Target 矩阵 (无梯度)
        # =======================================================
        with torch.no_grad():
            adj = torch.zeros((N, N), device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            
            # --- A. 空间地盘约束 (S_target) ---
            # 严格限制在 max_hops 以内。跨星系的节点，S_target 绝对为 0。
            S_target = torch.eye(N, device=device)
            A_k = adj.clone()
            decay = 1.0
            for k in range(1, self.max_hops + 1):
                new_reached = (A_k > 0).float()
                S_target = torch.max(S_target, new_reached * decay)
                A_k = torch.matmul(A_k, adj)
                decay *= 0.5 

            # --- B. 纯结构身份约束 (C_target) ---
            # 提取算术-调和结构指纹，精确区分 Hub、分支与纯叶子
            deg = adj.sum(dim=-1)
            f1 = torch.log(deg + 1.0)
            
            arith_mass = torch.matmul(adj, deg)
            f2 = torch.log(arith_mass + 1.0)
            
            inv_deg = 1.0 / (deg + 1e-6)
            sum_inv_deg = torch.matmul(adj, inv_deg)
            harmonic_mass = deg / (sum_inv_deg + 1e-6)
            f3 = torch.log(harmonic_mass + 1.0)
            
            struct_feat = torch.stack([f1, f2, f3], dim=-1)
            feat_dist = torch.cdist(struct_feat, struct_feat, p=2)
            
            # 结构越相似，C_target 越接近 1；结构不同，迅速衰减为 0
            C_target = torch.exp(-feat_dist / self.tau_mass)
            
            # --- C. 终极局部轨道法则 ---
            # 只有在同一个地盘 (S>0) 且 扮演相同结构角色 (C>0) 时，才要求相似。
            # 任何一个条件不满足（比如跨 Hub，或者同 Hub 但一个是中心一个是叶子），
            # target_sim 都会被强制清零。
            target_sim = S_target * C_target

        # =======================================================
        # 2. 预测相似度与暴力 MSE 回归
        # =======================================================
        # 将 Z_role 投影到单位超球面上，计算两两余弦相似度
        Z_norm = F.normalize(Z_role, p=2, dim=-1)
        pred_sim = torch.matmul(Z_norm, Z_norm.T)
        
        # [核心杀招]：用 MSE 替换交叉熵
        # 当 target_sim = 0 时，如果 GIN 试图把它们输出成相似的向量 (pred_sim > 0)，
        # MSE 会产生极其巨大且确定方向的惩罚梯度，暴力撕裂特征坍缩。
        loss_metric = F.mse_loss(pred_sim, target_sim)

        return loss_metric
        
#class LocalRoleInfoNCELoss(nn.Module):
#
#    def __init__(self, max_hops=2, tau_mass=0.5, tau_nce=0.1):
#        super(LocalRoleInfoNCELoss, self).__init__()
#        self.max_hops = max_hops      # 局部地盘范围
#        self.tau_mass = tau_mass      # 物理结构指纹的敏感度温度
#        self.tau_nce = tau_nce        # InfoNCE 的对比温度 (控制排斥力的严苛程度)
#
#    def forward(self, Z_role, edge_index):
#        device = Z_role.device
#        N = Z_role.shape[0]
#
#        # =======================================================
#        # 1. 构建连续型软对比标签 (Soft Contrastive Targets)
#        # =======================================================
#        with torch.no_grad():
#            adj = torch.zeros((N, N), device=device)
#            adj[edge_index[0], edge_index[1]] = 1.0
#            
#            # --- A. 提取地盘 (S_target) ---
#            S_target = torch.eye(N, device=device)
#            A_k = adj.clone()
#            decay = 1.0
#            for k in range(1, self.max_hops + 1):
#                new_reached = (A_k > 0).float()
#                S_target = torch.max(S_target, new_reached * decay)
#                A_k = torch.matmul(A_k, adj)
#                decay *= 0.5 
#
#            # --- B. 提取指纹 (C_target) ---
#            deg = adj.sum(dim=-1)
#            f1 = torch.log(deg + 1.0)
#            
#            arith_mass = torch.matmul(adj, deg)
#            f2 = torch.log(arith_mass + 1.0)
#            
#            inv_deg = 1.0 / (deg + 1e-6)
#            sum_inv_deg = torch.matmul(adj, inv_deg)
#            harmonic_mass = deg / (sum_inv_deg + 1e-6)
#            f3 = torch.log(harmonic_mass + 1.0)
#            
#            struct_feat = torch.stack([f1, f2, f3], dim=-1)
#
#            # =========================================================
#            # [核心修复 1]：裁判视角的图内动态归一化
#            # 保证 feat_dist 的距离永远在合理范围内，使 tau_mass=0.5 重新生效
#            # =========================================================
#            f_min = struct_feat.min(dim=0, keepdim=True)[0]
#            f_max = struct_feat.max(dim=0, keepdim=True)[0]
#            struct_feat_norm = (struct_feat - f_min) / (f_max - f_min + 1e-8)
#            
#            # 使用归一化后的特征计算距离
#            feat_dist = torch.cdist(struct_feat_norm, struct_feat_norm, p=2)
#            C_target = torch.exp(-feat_dist / self.tau_mass)
#            
#            # --- C. 融合目标与【智能自环掩码】 ---
#            target_sim = S_target * C_target
#            
#            diag_mask = torch.eye(N, dtype=torch.bool, device=device)
#            
#            # [核心修复]：计算每个人除了自己之外，还有多少相似的兄弟
#            off_diag_sim = target_sim.clone()
#            off_diag_sim.masked_fill_(diag_mask, 0.0)
#            has_brothers = (off_diag_sim.sum(dim=-1) > 0.1) # 判断是否有实质性的兄弟
#            
#            # 只有当你有兄弟时，才把你的对角线（自环）清零，逼迫你去和兄弟对齐
#            # 如果你是孤独的 Hub (has_brothers=False)，保留你的自环，让你成为自己的正样本！
#            mask_to_apply = diag_mask & has_brothers.unsqueeze(-1)
#            target_sim.masked_fill_(mask_to_apply, 0.0)
#            
#            target_norm = target_sim / torch.clamp(target_sim.sum(dim=-1, keepdim=True), min=1e-8)
#
#        # =======================================================
#        # 2. 潜空间自适应温度对比
#        # =======================================================
#        Z_norm = F.normalize(Z_role, p=2, dim=-1)
#        logits = torch.matmul(Z_norm, Z_norm.T) / self.tau_nce
#        
#        # 同理，在 logit 预测端，只对有兄弟的节点屏蔽自环
#        logits.masked_fill_(mask_to_apply, -1e9)
#        
#        log_probs = F.log_softmax(logits, dim=-1)
#        
#        loss_nce = -torch.sum(target_norm * log_probs, dim=-1).mean()
#
#        return loss_nce
class LocalRoleInfoNCELoss(nn.Module):
    def __init__(self, max_hops=4, struct_tol=0.05, tau_nce=0.1):
        super(LocalRoleInfoNCELoss, self).__init__()
        self.max_hops = max_hops      
        self.struct_tol = struct_tol  # 严苛的硬阈值，拒绝和稀泥
        self.tau_nce = tau_nce

    def forward(self, Z_role, edge_index):
        device = Z_role.device
        N = Z_role.shape[0]

        with torch.no_grad():
            adj = torch.zeros((N, N), device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            
            S_target = torch.eye(N, device=device)
            A_k = adj.clone()
            decay = 1.0
            for k in range(1, self.max_hops + 1):
                new_reached = (A_k > 0).float()
                S_target = torch.max(S_target, new_reached * decay)
                A_k = torch.matmul(A_k, adj)
                decay *= 0.5 

            deg = adj.sum(dim=-1)
            f1 = torch.log(deg + 1.0)
            arith_mass = torch.matmul(adj, deg)
            f2 = torch.log(arith_mass + 1.0)
            inv_deg = 1.0 / (deg + 1e-6)
            sum_inv_deg = torch.matmul(adj, inv_deg)
            harmonic_mass = deg / (sum_inv_deg + 1e-6)
            f3 = torch.log(harmonic_mass + 1.0)
            
            struct_feat = torch.stack([f1, f2, f3], dim=-1)

            # [核心重构]：全局统一归一化，保留 Hub A 与 Hub B 在尺度上的绝对差异
            f_min = struct_feat.min(dim=0, keepdim=True)[0]
            f_max = struct_feat.max(dim=0, keepdim=True)[0]
            struct_feat_norm = (struct_feat - f_min) / (f_max - f_min + 1e-8)
            
            feat_dist = torch.cdist(struct_feat_norm, struct_feat_norm, p=2)
            
            # 使用布尔断言：只有结构指纹极其相似的，才是正样本
            C_mask = (feat_dist < self.struct_tol).float()
            S_mask = (S_target > 0).float()
            
            target_sim = C_mask * S_mask
            
            diag_mask = torch.eye(N, dtype=torch.bool, device=device)
            off_diag_sim = target_sim.clone()
            off_diag_sim.masked_fill_(diag_mask, 0.0)
            
            has_brothers = (off_diag_sim.sum(dim=-1) > 0.5)
            mask_to_apply = diag_mask & has_brothers.unsqueeze(-1)
            target_sim.masked_fill_(mask_to_apply, 0.0)
            
            target_norm = target_sim / torch.clamp(target_sim.sum(dim=-1, keepdim=True), min=1e-8)
            target_entropy = -torch.sum(target_norm * torch.log(torch.clamp(target_norm, min=1e-8)), dim=-1)

        Z_norm = F.normalize(Z_role, p=2, dim=-1)
        logits = torch.matmul(Z_norm, Z_norm.T) / self.tau_nce
        
        logits.masked_fill_(mask_to_apply, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        
        ce_loss = -torch.sum(target_norm * log_probs, dim=-1)
        true_nce_loss = (ce_loss - target_entropy).mean()

        return F.relu(true_nce_loss)
        
        
class InvariantTopologicalMetricLoss(nn.Module):

    def __init__(self, tau_f=0.1, tau_rw=0.05, tau_z=1.0):
        super(InvariantTopologicalMetricLoss, self).__init__()
        self.tau_f = tau_f
        self.tau_rw = tau_rw
        self.tau_z = tau_z

    def forward(self, Z_role, x_raw):
        device = Z_role.device
        N = Z_role.shape[0]

        with torch.no_grad():
            # 1. 宏观体量核 (f1, f2, f3)
            f_feat = x_raw[:, 0:3] 
            f_min = f_feat.min(dim=0, keepdim=True)[0]
            f_max = f_feat.max(dim=0, keepdim=True)[0]
            f_norm = (f_feat - f_min) / (f_max - f_min + 1e-8)
            f_dist_sq = torch.cdist(f_norm, f_norm, p=2) ** 2
            sim_f = torch.exp(-f_dist_sq / self.tau_f)
            
            # 2. 微观连接核 (RWPE) -> 极其敏锐地切分三角形与桥梁
            rw_feat = x_raw[:, 3:7]
            rw_dist_sq = torch.cdist(rw_feat, rw_feat, p=2) ** 2
            sim_rw = torch.exp(-rw_dist_sq / self.tau_rw)
            
            # 3. 乘法流形融合 (严格逻辑与)
            target_sim = sim_f * sim_rw
            
            diag_mask = torch.eye(N, dtype=torch.bool, device=device)
            target_sim.masked_fill_(diag_mask, 0.0)
            
            has_brothers = (target_sim.sum(dim=-1) > 0.01)
            mask_to_apply = diag_mask | (~has_brothers).unsqueeze(-1)
            target_sim.masked_fill_(mask_to_apply, 0.0)
            
            target_norm = target_sim / torch.clamp(target_sim.sum(dim=-1, keepdim=True), min=1e-8)
            target_entropy = -torch.sum(target_norm * torch.log(torch.clamp(target_norm, min=1e-8)), dim=-1)

        # 4. 驱动 64 维 GINE 潜变量 Z_role
        z_dist_sq = torch.cdist(Z_role, Z_role, p=2) ** 2
        logits = -z_dist_sq / self.tau_z
        
        logits.masked_fill_(mask_to_apply, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        
        ce_loss = -torch.sum(target_norm * log_probs, dim=-1)
        true_metric_loss = (ce_loss - target_entropy).mean()

        return F.relu(true_metric_loss)
class IsomorphicRoleMetricLoss(nn.Module):
    """
    [DeepCS V10.x] Structural Equivalence Distance Matching Loss.
    Replaces the old InvariantTopologicalMetricLoss.
    Directly aligns the L2 relative distance of the 64D Z_role manifold
    with the L2 relative distance of the pure structural feature space (Fingerprints + RWPE).
    """
    def __init__(self, tau_f=0.1, tau_rw=0.05, tau_z=1.0):
        super(IsomorphicRoleMetricLoss, self).__init__()
        # 保留原有参数签名以兼容外部调用，在此新逻辑中作为保留变量
        self.tau_f = tau_f
        self.tau_rw = tau_rw
        self.tau_z = tau_z

    def forward(self, Z_role, x_raw):
        # ==========================================================
        # 1. 结构真值提取 (Ground Truth Extraction)
        # ==========================================================
        with torch.no_grad():
            # [修正切片] 严格提取真正决定结构同构性的特征分量
            # x_raw 构成: fingerprints(0:3), rel_deg(3:4), global_hier(4:6), rwpe(6:10)
            fingerprints = x_raw[:, 0:3]  # [N, 3] 宏观度数质量指纹
            rwpe = x_raw[:, 6:10]         # [N, 4] 4步随机游走概率指纹
            
            # 拼接构成 7维 结构特征空间
            struct_feat = torch.cat([fingerprints, rwpe], dim=1)
            
            # 计算原始拓扑特征的欧氏距离矩阵
            dist_struct = torch.cdist(struct_feat, struct_feat, p=2)
            
            # 局部均值自适应归一化：将绝对距离转化为相对距离，免疫不同图规模的尺度漂移
            norm_struct = dist_struct / (dist_struct.mean() + 1e-8)

        # ==========================================================
        # 2. 潜空间度量 (Latent Space Metric)
        # ==========================================================
        # 计算 64维 GINE 潜变量 Z_role 的欧氏距离矩阵
        dist_z = torch.cdist(Z_role, Z_role, p=2)
        norm_z = dist_z / (dist_z.mean() + 1e-8)

        # ==========================================================
        # 3. 几何同态对齐 (Geometric Homomorphism Alignment)
        # ==========================================================
        # 使用 MSE 强迫 Z_role 流形的相对距离严格复刻拓扑特征的相对距离
        metric_loss = F.mse_loss(norm_z, norm_struct)

        return metric_loss
        
class KLEnergyLoss(nn.Module):
    def __init__(self, eps=1e-8, energy_mode="edge_only"):
        super().__init__()
        self.eps = eps
        self.energy_mode = energy_mode  # "edge_only" or "full"

    def forward(self, pos, P_target, Z_role,
                edge_index, w_edge, l_edge, q_node,
                lambda_kl=1.0, lambda_energy=1.0):

        device = pos.device
        N = pos.shape[0]
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)

        # =====================================================
        # 1. Global KL（完全不动你原来的逻辑）
        # =====================================================
        posDiff = pos.unsqueeze(0) - pos.unsqueeze(1)
        distSq = (posDiff ** 2).sum(dim=-1)

        qUnnorm = 1.0 / (1.0 + distSq)
        qUnnorm = qUnnorm.masked_fill(eye_mask, 0.0)
        qPred = qUnnorm / qUnnorm.sum().clamp(min=self.eps)

        validMask = P_target > 0

        lossKL = torch.sum(
            P_target[validMask] *
            torch.log((P_target[validMask] + self.eps) /
                      (qPred[validMask] + self.eps))
        )

        # =====================================================
        # 2. Energy Loss（关键：阻止 collapse）
        # =====================================================
        row, col = edge_index

        diff = pos[col] - pos[row]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + self.eps)

        # -------- (A) 边的胡克能量 --------
        # 防止边长 -> 0
        energy_attr = w_edge * (dist - l_edge) ** 2
        energy_attr = energy_attr.mean()

        # -------- (B) 斥力能量（可选）--------
        if self.energy_mode == "full":
            diff_rep = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist_sq = (diff_rep ** 2).sum(dim=-1)

            dist_sq = dist_sq.masked_fill(eye_mask, float('inf'))

            qij = q_node.unsqueeze(1) * q_node.unsqueeze(0)
            energy_rep = qij / (dist_sq + 1e-4)

            energy_rep = energy_rep.mean()
        else:
            energy_rep = 0.0

        lossEnergy = energy_attr + energy_rep

        # =====================================================
        # 3. 总 loss
        # =====================================================
        totalLoss = lambda_kl * lossKL + lambda_energy * lossEnergy

        return totalLoss, {
            "kl": lossKL.item(),
            "energy": lossEnergy.item()
        }
class UnifiedOrbitalLayoutLoss(nn.Module):
    """
    [DeepCS V10.2] Unified Probabilistic & Orbital Layout Loss.
    Integrates Scale-Invariant KL Divergence, stripped-down Orbital Cohesion, 
    Log-Lorentzian Separation, and Physical Parameter Regularization.
    """
    def __init__(self, sigmaRatio=0.3, tauBase=0.0, eps=1e-8):
        super(UnifiedOrbitalLayoutLoss, self).__init__()
        self.sigmaRatio = sigmaRatio
        self.tauBase = tauBase
        self.eps = eps

    def forward(self, pos, P_target, w_edge, l_edge, q_node, Z_role, edge_index, 
                lambda_kl=1.0, lambda_cohesion=0.0, lambda_separation=0.0, lambda_reg=0.1):
        numNodes = pos.shape[0]
        computeDevice = pos.device
        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)

        # ==========================================
        # Module 1: 空间尺度强制归一化 (Scale Normalization)
        # ==========================================
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        pos_std = pos_centered.std(dim=0).mean() + self.eps
        # 核心：后续所有距离计算必须基于 pos_normalized
        pos_normalized = pos_centered / pos_std

        # 计算归一化后的基础距离矩阵
        posDiff = pos_normalized.unsqueeze(0) - pos_normalized.unsqueeze(1)
        distSqNorm = (posDiff ** 2).sum(dim=-1)
        
        # ==========================================
        # Module 2: 纯粹拓扑 KL 散度 (宏观引力场)
        # ==========================================
        validMask = P_target > 0
        
        qUnnorm = 1.0 / (1.0 + distSqNorm)
        qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
        qSum = qUnnorm.sum().clamp(min=self.eps)
        qPred = qUnnorm / qSum

        P_valid = P_target[validMask]
        Q_valid = qPred[validMask].clamp(min=self.eps)
        lossGlobalKl = torch.sum(P_valid * torch.log(P_valid / Q_valid))

        # ==========================================
        # Module 3: 物理健康正则化 (Physical Health Regularization)
        # ==========================================
        reg_w = torch.mean((w_edge - 1.0) ** 2)
        reg_q = torch.mean((q_node - 1.0) ** 2)
        reg_l = torch.mean((l_edge - 1.0) ** 2)
        lossReg = reg_q + reg_l + reg_w

        # ==========================================
        # Module 4: 语义轨道物理约束 (Semantic Orbital Field)
        # ==========================================
        lossCohesion = torch.tensor(0.0, device=computeDevice)
        lossSeparation = torch.tensor(0.0, device=computeDevice)

        zDetach = Z_role.detach() if Z_role is not None else None
        if zDetach is not None and (lambda_cohesion > 0 or lambda_separation > 0):
            # 特征相似度场 S_IJ
            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeMat).mean().clamp(min=self.eps)
            sigma = meanZDist * self.sigmaRatio
            S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
            S_IJ_exp = S_IJ.unsqueeze(0).expand(numNodes, numNodes, numNodes)

            # 拓扑基础矩阵
            adjMat = validMask.float()
            deg = adjMat.sum(dim=-1) 
            hubDegrees = deg.clamp(min=1.0)
            
            # Hub 三元组掩码定义
            maskTriadic = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
            hubMask = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
                      (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
            maskTriadic = maskTriadic * hubMask.float() * (~eyeMat.unsqueeze(0)).float()

            # 基于归一化空间的对数距离
            distMatNorm = torch.sqrt(distSqNorm + self.eps)
            logDistMatNorm = torch.log(distMatNorm)
            logDistKI = logDistMatNorm.unsqueeze(2)
            logDistKJ = logDistMatNorm.unsqueeze(1)
            deltaLogD = logDistKI - logDistKJ
            
            distKI = distMatNorm.unsqueeze(2)
            distKJ = distMatNorm.unsqueeze(1)
            deltaD = distKI - distKJ  # 纯线性距离差

            # 拓扑引力质量场 (Topological Mass Field)
            isValidHub = (maskTriadic.sum(dim=(1, 2)) > 0).float()
            validHubCount = isValidHub.sum().clamp(min=1.0)
            mass = deg + torch.matmul(adjMat, deg.unsqueeze(1)).squeeze(1)
            
            hubWeights = mass * isValidHub
            totalWeight = hubWeights.sum().clamp(min=self.eps)
            normHubWeights = hubWeights * (validHubCount / totalWeight)

            # ---------------------------------------------------------
            # 4.1 凝聚项：纯 2D 弹性局部宽容对齐 (已移除核盾)
            # ---------------------------------------------------------
            if lambda_cohesion > 0:
                peerCount = (maskTriadic * S_IJ_exp).sum(dim=2)
                violationCoh = F.relu(torch.abs(deltaLogD) - self.tauBase)
                errCohVar = maskTriadic * S_IJ_exp * (violationCoh ** 2)
                
                nodeCohVar = errCohVar.sum(dim=2) 
                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
                perHubCohVarO1 = nodeCohVarNorm.sum(dim=1) / hubDegrees
                
                weightedHubCoh = perHubCohVarO1 * normHubWeights
                lossCohesion = weightedHubCoh.sum() / validHubCount

            # ---------------------------------------------------------
            # 4.2 分离项：Log-Lorentzian 拓扑排斥力
            # ---------------------------------------------------------
            if lambda_separation > 0:
                massI = mass.view(1, -1, 1)
                massJ = mass.view(1, 1, -1)
                dirMask = (massI < massJ).float()
                
                margin = 0.1
                violationIInner = F.relu(deltaLogD + margin)
                violationJInner = F.relu(-deltaLogD + margin)
                directedViolation = dirMask * violationIInner + (1.0 - dirMask) * violationJInner
                
                errSep = maskTriadic * (1.0 - S_IJ_exp) * torch.log(1.0 + directedViolation ** 2)
                perHubSepO1 = errSep.sum(dim=(1, 2)) / hubDegrees
                
                weightedHubSep = perHubSepO1 * normHubWeights
                lossSeparation = weightedHubSep.sum() / validHubCount

        # ==========================================
        # Final Loss Fusion
        # ==========================================
#        totalLoss = (lambda_kl * lossGlobalKl) + \
#                    (lambda_cohesion * lossCohesion) + \
#                    (lambda_separation * lossSeparation) + \
#                    (lambda_reg * lossReg)
        totalLoss = (lambda_kl * lossGlobalKl) + \
                    (lambda_cohesion * lossCohesion) + \
                    (lambda_separation * lossSeparation) + \
                    (lambda_reg * lossReg)

        return totalLoss, {
            'kl_loss': lossGlobalKl.item(),
            'cohesion_loss': lossCohesion.item(),
            'separation_loss': lossSeparation.item(),
            'reg_loss': lossReg.item()
        }
        
        
class MetricsDrivenLayoutLoss(nn.Module):
    def __init__(self, num_negatives=5, margin_ar=0.05, eps=1e-8):
        super(MetricsDrivenLayoutLoss, self).__init__()
        self.num_negatives = num_negatives  # NP 负采样数量 K
        self.margin_ar = margin_ar          # AR 容忍裕度
        self.eps = eps

    def forward(self, pos, edge_index, lambda_np=1.0, lambda_ar=0.5, lambda_eu=0.1):
        N = pos.shape[0]
        device = pos.device
        row, col = edge_index
        num_edges = row.shape[0]

        # ==========================================
        # 0. 空间尺度强制归一化 (Scale Normalization)
        # ==========================================
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        pos_std = pos_centered.std(dim=0).mean() + self.eps
        pos_normalized = pos_centered / pos_std

        # [修复点]: 将共享基础变量 edge_diff 的计算提取到全局作用域
        if num_edges > 0:
            edge_diff = pos_normalized[row] - pos_normalized[col]
        else:
            edge_diff = None

        # ==========================================
        # Module 1: Edge Length Uniformity (EU) - 尺度不变均方差
        # ==========================================
        loss_eu = torch.tensor(0.0, device=device)
        if lambda_eu > 0 and num_edges > 0:
            edge_lengths = torch.norm(edge_diff, dim=-1)
            mu_L = edge_lengths.mean()
            loss_eu = torch.mean(((edge_lengths - mu_L) / (mu_L + self.eps)) ** 2)

        # ==========================================
        # Module 2: Neighborhood Preservation (NP) - InfoNCE 负采样
        # ==========================================
        loss_np = torch.tensor(0.0, device=device)
        if lambda_np > 0 and num_edges > 0:
            # 正样本亲和力 (直接安全使用提前算好的 edge_diff)
            edge_dist_sq = (edge_diff ** 2).sum(dim=-1)
            q_pos = 1.0 / (1.0 + edge_dist_sq)

            # 负采样生成：为每条有向边的源节点随机抽取 K 个非邻居节点
            neg_idx = torch.randint(0, N, (num_edges, self.num_negatives), device=device)
            pos_src_expand = pos_normalized[row].unsqueeze(1)  # [E, 1, 2]
            pos_neg = pos_normalized[neg_idx]                  # [E, K, 2]

            # 负样本亲和力
            neg_dist_sq = ((pos_src_expand - pos_neg) ** 2).sum(dim=-1) # [E, K]
            q_neg = 1.0 / (1.0 + neg_dist_sq)

            # 拓扑对比交叉熵
            loss_np = -torch.mean(torch.log(q_pos / (q_pos + q_neg.sum(dim=1) + self.eps)))

        # ==========================================
        # Module 3: Angular Resolution (AR) - 余弦软间隔排斥
        # ==========================================
        loss_ar = torch.tensor(0.0, device=device)
        if lambda_ar > 0:
            adj = torch.zeros((N, N), device=device)
            adj[row, col] = 1.0
            adj[col, row] = 1.0
            adj.fill_diagonal_(0.0)

            deg = adj.sum(dim=-1)
            hub_mask = (deg >= 2).float()
            
            D = pos_normalized.unsqueeze(0) - pos_normalized.unsqueeze(1) 
            dist_matrix = torch.norm(D, dim=-1) + self.eps
            V = D / dist_matrix.unsqueeze(-1) 

            S_cos = torch.einsum('uid, ujd -> uij', V, V)

            mask_triadic = adj.unsqueeze(2) * adj.unsqueeze(1) 
            eye_N = torch.eye(N, device=device, dtype=torch.bool)
            mask_triadic = mask_triadic * (~eye_N.unsqueeze(0)).float()

            theta_min = 2.0 * torch.pi / deg.clamp(min=1.0)
            C_max = torch.cos(theta_min)

            violation = F.relu(S_cos - C_max.view(N, 1, 1) + self.margin_ar)
            ar_penalty = mask_triadic * (violation ** 2)

            norm_factor = deg * (deg - 1.0)
            hub_penalty = ar_penalty.sum(dim=(1, 2)) / norm_factor.clamp(min=1.0)

            valid_hub_count = hub_mask.sum().clamp(min=1.0)
            loss_ar = (hub_penalty * hub_mask).sum() / valid_hub_count

        # ==========================================
        # Final Fusion
        # ==========================================
        totalLoss = (lambda_np * loss_np) + (lambda_ar * loss_ar) + (lambda_eu * loss_eu)

        return totalLoss, {
            'np_loss': loss_np.item(),
            'ar_loss': loss_ar.item(),
            'eu_loss': loss_eu.item()
        }
class MetricsDrivenLayoutLossV2(nn.Module):
    def __init__(self, num_negatives=5, margin_ar=0.05, sigmaRatio=0.3, tauBase=0.0, eps=1e-8):
        super(MetricsDrivenLayoutLossV2, self).__init__()
        self.num_negatives = num_negatives  
        self.margin_ar = margin_ar          
        self.sigmaRatio = sigmaRatio        # [新增] COH 语义高斯核缩放系数
        self.tauBase = tauBase              # [新增] COH 宽容对齐裕度
        self.eps = eps

    def forward(self, pos, edge_index, Z_role=None, lambda_np=1.0, lambda_ar=0.5, lambda_eu=0.1, lambda_coh=0.0):
        N = pos.shape[0]
        device = pos.device
        row, col = edge_index
        num_edges = row.shape[0]

        # ==========================================
        # 0. 空间尺度强制归一化 (Scale Normalization)
        # ==========================================
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        pos_std = pos_centered.std(dim=0).mean() + self.eps
        pos_normalized = pos_centered / pos_std

        if num_edges > 0:
            edge_diff = pos_normalized[row] - pos_normalized[col]
        else:
            edge_diff = None

        # ==========================================
        # Module 1: Edge Length Uniformity (EU) 
        # ==========================================
        loss_eu = torch.tensor(0.0, device=device)
        if lambda_eu > 0 and num_edges > 0:
            edge_lengths = torch.norm(edge_diff, dim=-1)
            mu_L = edge_lengths.mean()
            loss_eu = torch.mean(((edge_lengths - mu_L) / (mu_L + self.eps)) ** 2)

        # ==========================================
        # Module 2: Neighborhood Preservation (NP) 
        # ==========================================
        loss_np = torch.tensor(0.0, device=device)
        if lambda_np > 0 and num_edges > 0:
            edge_dist_sq = (edge_diff ** 2).sum(dim=-1)
            q_pos = 1.0 / (1.0 + edge_dist_sq)

            neg_idx = torch.randint(0, N, (num_edges, self.num_negatives), device=device)
            pos_src_expand = pos_normalized[row].unsqueeze(1)  
            pos_neg = pos_normalized[neg_idx]                  

            neg_dist_sq = ((pos_src_expand - pos_neg) ** 2).sum(dim=-1) 
            q_neg = 1.0 / (1.0 + neg_dist_sq)

            loss_np = -torch.mean(torch.log(q_pos / (q_pos + q_neg.sum(dim=1) + self.eps)))

        # ==========================================
        # 共享拓扑基底计算 (为 AR 和 COH 节省显存)
        # ==========================================
        loss_ar = torch.tensor(0.0, device=device)
        loss_coh = torch.tensor(0.0, device=device)

        if lambda_ar > 0 or lambda_coh > 0:
            adj = torch.zeros((N, N), device=device)
            adj[row, col] = 1.0
            adj[col, row] = 1.0
            adj.fill_diagonal_(0.0)

            deg = adj.sum(dim=-1)
            eye_N = torch.eye(N, device=device, dtype=torch.bool)
            
            # [共享张量] 基础三元组掩码
            mask_triadic_base = adj.unsqueeze(2) * adj.unsqueeze(1) 
            mask_triadic_base = mask_triadic_base * (~eye_N.unsqueeze(0)).float()

            D = pos_normalized.unsqueeze(0) - pos_normalized.unsqueeze(1) 
            dist_matrix = torch.norm(D, dim=-1) + self.eps

            # ------------------------------------------
            # Module 3: Angular Resolution (AR)
            # ------------------------------------------
            if lambda_ar > 0:
                hub_mask_ar = (deg >= 2).float()
                V = D / dist_matrix.unsqueeze(-1) 
                S_cos = torch.einsum('uid, ujd -> uij', V, V)

                theta_min = 2.0 * torch.pi / deg.clamp(min=1.0)
                C_max = torch.cos(theta_min)

                violation = F.relu(S_cos - C_max.view(N, 1, 1) + self.margin_ar)
                ar_penalty = mask_triadic_base * (violation ** 2)

                norm_factor = deg * (deg - 1.0)
                hub_penalty = ar_penalty.sum(dim=(1, 2)) / norm_factor.clamp(min=1.0)

                valid_hub_count_ar = hub_mask_ar.sum().clamp(min=1.0)
                loss_ar = (hub_penalty * hub_mask_ar).sum() / valid_hub_count_ar

            # ------------------------------------------
            # Module 4: Semantic Orbital Cohesion (COH)
            # ------------------------------------------
            if lambda_coh > 0 and Z_role is not None:
                # 计算语义相似度矩阵 S_IJ
                zDetach = Z_role.detach()
                zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
                meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eye_N).mean().clamp(min=self.eps)
                sigma = meanZDist * self.sigmaRatio
                S_IJ = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
                S_IJ_exp = S_IJ.unsqueeze(0).expand(N, N, N)

                # 提取严格的层级 Hub 掩码 (u 的度数必须大于 i 和 j)
                hubMask_coh = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
                              (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
                maskTriadic_coh = mask_triadic_base * hubMask_coh.float()

                # 计算归一化后的对数距离差
                logDistMatNorm = torch.log(dist_matrix)
                deltaLogD = logDistMatNorm.unsqueeze(2) - logDistMatNorm.unsqueeze(1)

                # 计算质量场 (Topological Mass)
                isValidHub_coh = (maskTriadic_coh.sum(dim=(1, 2)) > 0).float()
                validHubCount_coh = isValidHub_coh.sum().clamp(min=1.0)
                mass = deg + torch.matmul(adj, deg.unsqueeze(1)).squeeze(1)
                
                hubWeights = mass * isValidHub_coh
                totalWeight = hubWeights.sum().clamp(min=self.eps)
                normHubWeights = hubWeights * (validHubCount_coh / totalWeight)

                # 轨道对齐惩罚
                peerCount = (maskTriadic_coh * S_IJ_exp).sum(dim=2)
                violationCoh = F.relu(torch.abs(deltaLogD) - self.tauBase)
                errCohVar = maskTriadic_coh * S_IJ_exp * (violationCoh ** 2)
                
                nodeCohVar = errCohVar.sum(dim=2) 
                nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
                perHubCohVarO1 = nodeCohVarNorm.sum(dim=1) / deg.clamp(min=1.0)
                
                weightedHubCoh = perHubCohVarO1 * normHubWeights
                loss_coh = weightedHubCoh.sum() / validHubCount_coh

        # ==========================================
        # Final Fusion
        # ==========================================
        totalLoss = (lambda_np * loss_np) + (lambda_ar * loss_ar) + \
                    (lambda_eu * loss_eu) + (lambda_coh * loss_coh)

        return totalLoss, {
            'np_loss': loss_np.item(),
            'ar_loss': loss_ar.item(),
            'eu_loss': loss_eu.item(),
            'coh_loss': loss_coh.item()
        }
        
        
class MetricsDrivenLayoutLossV3(nn.Module):
    def __init__(self, num_negatives=5, margin_ar=0.05, sigmaRatio=0.3, tauBase=0.0, eps=1e-8):
        super(MetricsDrivenLayoutLossV3, self).__init__()
        # 基础参数
        self.numNegatives = num_negatives  
        self.marginAr = margin_ar           
        self.sigmaRatio = sigmaRatio        
        self.tauBase = tauBase              
        self.eps = eps
        
        # 物理修正参数
        self.softeningAlpha = 0.1       # LAR 防止梯度爆炸的物理软化因子
        self.larInteractionRadius = 2.0 # LAR 软衰减控制半径 (对应 90 度的弦长平方：2-2cos(pi/2)=2)

    def forward(self, pos, edge_index, Z_role=None, 
                lambda_np=1.0, lambda_ar=0.5, lambda_eu=0.1, lambda_coh=0.0,
                lambda_lnp=0.0, lambda_lar=0.0, lambda_leu=0.0):
        
        N = pos.shape[0]
        device = pos.device
        row, col = edge_index
        numEdges = row.shape[0]

        # ==========================================
        # 0. 空间尺度强制归一化 & 全局共享特征
        # ==========================================
        posCentered = pos - pos.mean(dim=0, keepdim=True)
        posStd = posCentered.std(dim=0).mean() + self.eps
        posNormalized = posCentered / posStd

        # 全局距离矩阵 (O(N^2)计算一次，供全局复用)
        D = posNormalized.unsqueeze(0) - posNormalized.unsqueeze(1) 
        distMatrix = torch.norm(D, dim=-1) + self.eps

        if numEdges > 0:
            edgeDiff = posNormalized[row] - posNormalized[col]
            edgeLengths = torch.norm(edgeDiff, dim=-1)
        else:
            edgeDiff = None
            edgeLengths = None

        # 基础拓扑特征
        adj = torch.zeros((N, N), device=device)
        if numEdges > 0:
            adj[row, col] = 1.0
            adj[col, row] = 1.0
        adj.fill_diagonal_(0.0)
        
        deg = adj.sum(dim=-1)
        eyeN = torch.eye(N, device=device, dtype=torch.bool)
        
        # [动态 Hub 计算] 针对局部指标
        if N > 1:
            meanDeg = deg.mean()
            stdDeg = deg.std(unbiased=False)
            threshold = meanDeg + stdDeg
            hubMaskGlobal = (deg > threshold).float()
            if hubMaskGlobal.sum() == 0:
                hubMaskGlobal = (deg == deg.max()).float()
        else:
            hubMaskGlobal = torch.zeros(N, device=device)

        # ==========================================
        # Module 1 & 1.1: EU & LEU (Edge Length)
        # ==========================================
        lossEu = torch.tensor(0.0, device=device)
        lossLeu = torch.tensor(0.0, device=device)
        
        if numEdges > 0:
            # Global EU: 均方相对误差
            if lambda_eu > 0:
                muL = edgeLengths.mean()
                lossEu = torch.mean(((edgeLengths - muL) / (muL + self.eps)) ** 2)
            
            # Local EU (LEU): 严格尺度不变的平方变异系数 (CV^2)
            if lambda_leu > 0:
                hubEdgeMask = (hubMaskGlobal[row] > 0) | (hubMaskGlobal[col] > 0)
                if hubEdgeMask.any():
                    hubEdgeLengths = edgeLengths[hubEdgeMask]
                    meanLenHub = hubEdgeLengths.mean()
                    if meanLenHub > self.eps:
                        varLenHub = torch.var(hubEdgeLengths, unbiased=False)
                        lossLeu = varLenHub / (meanLenHub ** 2 + self.eps)

        # ==========================================
        # Module 2 & 2.1: NP & LNP (Neighborhood)
        # ==========================================
        lossNp = torch.tensor(0.0, device=device)
        lossLnp = torch.tensor(0.0, device=device)
        
        if numEdges > 0:
            edgeDistSq = (edgeDiff ** 2).sum(dim=-1)
            
            negIdx = torch.randint(0, N, (numEdges, self.numNegatives), device=device)
            posSrcExpand = posNormalized[row].unsqueeze(1)  
            posNeg = posNormalized[negIdx]                  
            negDistSq = ((posSrcExpand - posNeg) ** 2).sum(dim=-1) 
            
            qPos = 1.0 / (1.0 + edgeDistSq)
            qNeg = 1.0 / (1.0 + negDistSq)
            
            # Global NP: 标准 InfoNCE
            if lambda_np > 0:
                lossNp = -torch.mean(torch.log(qPos / (qPos + qNeg.sum(dim=1) + self.eps)))
            
            # Local NP (LNP): InfoNCE 底座 + Top-k 截断推拉边界
            if lambda_lnp > 0:
                hubEdgeMask = (hubMaskGlobal[row] > 0) | (hubMaskGlobal[col] > 0)
                lossLnpContrast = torch.tensor(0.0, device=device)
                if hubEdgeMask.any():
                    qPosHub = qPos[hubEdgeMask]
                    qNegHub = qNeg[hubEdgeMask]
                    lossLnpContrast = -torch.mean(torch.log(qPosHub / (qPosHub + qNegHub.sum(dim=1) + self.eps)))

                lossLnpBoundary = torch.tensor(0.0, device=device)
                hubIndices = torch.nonzero(hubMaskGlobal, as_tuple=False).squeeze(1)
                numHubs = hubIndices.numel()
                
                if numHubs > 0:
                    dHubs = distMatrix[hubIndices]
                    adjHubs = adj[hubIndices] > 0
                    kVals = deg[hubIndices].long()
                    
                    dSorted, _ = torch.sort(dHubs.detach(), dim=1)
                    kSafe = torch.clamp(kVals, max=N-1)
                    tau = dSorted[torch.arange(numHubs, device=device), kSafe] 
                    
                    # 正样本拉力 (以 kSafe 归一化)
                    posViolations = F.relu(dHubs - tau.unsqueeze(1)) * adjHubs
                    lossPosPerHub = posViolations.sum(dim=1) / kSafe.clamp(min=1.0).float()
                    
                    # 负样本推力 (同样以 kSafe 归一化，确保场域对称，无需平衡常数)
                    nonAdjHubs = ~adjHubs
                    nonAdjHubs[torch.arange(numHubs, device=device), hubIndices] = False 
                    negViolations = F.relu(tau.unsqueeze(1) - dHubs) * nonAdjHubs
                    lossNegPerHub = negViolations.sum(dim=1) / kSafe.clamp(min=1.0).float()
                    
                    lossLnpBoundary = (lossPosPerHub + lossNegPerHub).mean()
                
                lossLnp = lossLnpContrast + lossLnpBoundary

        # ==========================================
        # Module 3 & 3.1: AR & LAR (Angular Resolution)
        # ==========================================
        lossAr = torch.tensor(0.0, device=device)
        lossLar = torch.tensor(0.0, device=device)

        if lambda_ar > 0 or lambda_lar > 0:
            maskTriadicBase = adj.unsqueeze(2) * adj.unsqueeze(1) 
            maskTriadicBase = maskTriadicBase * (~eyeN.unsqueeze(0)).float()

            unitVectors = D / distMatrix.unsqueeze(-1) 
            sCos = torch.einsum('uid, ujd -> uij', unitVectors, unitVectors)
            normFactor = deg * (deg - 1.0)
            
            # Global AR (Hard Margin 安全网：仅惩罚重叠)
            if lambda_ar > 0:
                arViolation = F.relu(sCos - (1.0 - self.marginAr)) 
                arPenalty = maskTriadicBase * (arViolation ** 2)
                
                summedArPenalty = arPenalty.sum(dim=(1, 2)) / normFactor.clamp(min=1.0)
                hubMaskAr = (deg >= 2).float()
                lossAr = (summedArPenalty * hubMaskAr).sum() / hubMaskAr.sum().clamp(min=1.0)

            # Local LAR (Thomson 静电势能：软衰减与软化防爆)
            if lambda_lar > 0:
                chordalDistSq = 2.0 - 2.0 * sCos
                chordalDistSqSoft = chordalDistSq + self.softeningAlpha
                
                # 指数衰减权重：避免硬截断导致不可导点，同时压制远端噪声
                decayWeight = torch.exp(-chordalDistSq / self.larInteractionRadius)
                
                repulsiveEnergy = (decayWeight / chordalDistSqSoft) * maskTriadicBase
                
                summedLarPenalty = repulsiveEnergy.sum(dim=(1, 2)) / normFactor.clamp(min=1.0)
                lossLar = (summedLarPenalty * hubMaskGlobal).sum() / hubMaskGlobal.sum().clamp(min=1.0)

        # ==========================================
        # Module 4: Semantic Orbital Cohesion (COH)
        # ==========================================
        lossCoh = torch.tensor(0.0, device=device)

        if lambda_coh > 0 and Z_role is not None:
            maskTriadicBase = adj.unsqueeze(2) * adj.unsqueeze(1) 
            maskTriadicBase = maskTriadicBase * (~eyeN.unsqueeze(0)).float()

            zDetach = Z_role.detach()
            zDistSq = ((zDetach.unsqueeze(0) - zDetach.unsqueeze(1))**2).sum(dim=-1)
            meanZDist = torch.sqrt(zDistSq + self.eps).masked_select(~eyeN).mean().clamp(min=self.eps)
            sigma = meanZDist * self.sigmaRatio
            sIj = torch.exp(-zDistSq / (2.0 * (sigma ** 2) + self.eps))
            sIjExp = sIj.unsqueeze(0).expand(N, N, N)

            hubMaskCoh = (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(2)) & \
                         (deg.unsqueeze(1).unsqueeze(2) > deg.unsqueeze(0).unsqueeze(1))
            maskTriadicCoh = maskTriadicBase * hubMaskCoh.float()

            logDistMatNorm = torch.log(distMatrix)
            deltaLogD = logDistMatNorm.unsqueeze(2) - logDistMatNorm.unsqueeze(1)

            isValidHubCoh = (maskTriadicCoh.sum(dim=(1, 2)) > 0).float()
            validHubCountCoh = isValidHubCoh.sum().clamp(min=1.0)
            mass = deg + torch.matmul(adj, deg.unsqueeze(1)).squeeze(1)
            
            hubWeights = mass * isValidHubCoh
            totalWeight = hubWeights.sum().clamp(min=self.eps)
            normHubWeights = hubWeights * (validHubCountCoh / totalWeight)

            peerCount = (maskTriadicCoh * sIjExp).sum(dim=2)
            violationCoh = F.relu(torch.abs(deltaLogD) - self.tauBase)
            errCohVar = maskTriadicCoh * sIjExp * (violationCoh ** 2)
            
            nodeCohVar = errCohVar.sum(dim=2) 
            nodeCohVarNorm = nodeCohVar / peerCount.clamp(min=1.0)
            perHubCohVarO1 = nodeCohVarNorm.sum(dim=1) / deg.clamp(min=1.0)
            
            weightedHubCoh = perHubCohVarO1 * normHubWeights
            lossCoh = weightedHubCoh.sum() / validHubCountCoh

        # ==========================================
        # Final Fusion
        # ==========================================
        totalLoss = (lambda_np * lossNp) + (lambda_ar * lossAr) + \
                    (lambda_eu * lossEu) + (lambda_coh * lossCoh) + \
                    (lambda_lnp * lossLnp) + (lambda_lar * lossLar) + (lambda_leu * lossLeu)

        return totalLoss, {
            'np_loss': lossNp.item(),
            'ar_loss': lossAr.item(),
            'eu_loss': lossEu.item(),
            'lnp_loss': lossLnp.item(),
            'lar_loss': lossLar.item(),
            'leu_loss': lossLeu.item(),
            'coh_loss': lossCoh.item()
        }
        
class MetricsDrivenLayoutLossV5(nn.Module):
    def __init__(self, num_negatives=5, margin_ar=0.05, tau_soft=0.5, eps=1e-8):
        """
        MetricsDrivenLayoutLossV5
        - Strict ASCII implementation matching V4 signature.
        - Fixes V4 mathematical logic error in implicit LNP.
        - Introduces scale-invariant local non-dimensionalization.
        - Replaces V4 linear hub weighting with log-smooth weighting.
        """
        super().__init__()
        self.numNegatives = num_negatives
        self.marginAr = margin_ar
        self.tauSoft = tau_soft
        self.eps = eps

    def forward(self, pos, edge_index,
                lambda_np=1.0, lambda_ar=0.5, lambda_eu=0.05,
                alpha_deg=0.5, beta_soft=0.2):

        N = pos.shape[0]
        device = pos.device
        row, col = edge_index
        numEdges = row.shape[0]

        if N <= 1:
            return torch.tensor(0.0, device=device), {
                'np_loss': 0.0, 'ar_loss': 0.0, 'eu_loss': 0.0
            }

        # ==========================================
        # 0. Normalization & Core Matrices
        # ==========================================
        posCentered = pos - pos.mean(dim=0, keepdim=True)
        posStd = posCentered.std(dim=0).mean() + self.eps
        posNormalized = posCentered / posStd

        # Global distance matrix O(N^2)
        D = posNormalized.unsqueeze(0) - posNormalized.unsqueeze(1)
        distMatrix = torch.norm(D, dim=-1) + self.eps

        # Topological adjacency matrix
        adj = torch.zeros((N, N), device=device)
        if numEdges > 0:
            adj[row, col] = 1.0
            adj[col, row] = 1.0
        adj.fill_diagonal_(0.0)

        deg = adj.sum(dim=-1)

        # ==========================================
        # 1. NP (Degree-weight + Fixed Implicit LNP)
        # ==========================================
        lossNp = torch.tensor(0.0, device=device)

        if numEdges > 0 and lambda_np > 0:
            # (1) Global InfoNCE Part
            edgeDiff = posNormalized[row] - posNormalized[col]
            edgeDistSq = (edgeDiff ** 2).sum(dim=-1)

            negIdx = torch.randint(0, N, (numEdges, self.numNegatives), device=device)
            posSrc = posNormalized[row].unsqueeze(1)
            negPos = posNormalized[negIdx]
            negDistSq = ((posSrc - negPos) ** 2).sum(dim=-1)

            qPos = 1.0 / (1.0 + edgeDistSq)
            qNeg = 1.0 / (1.0 + negDistSq)

            # Degree weighting following V4 alpha_deg
            degW = ((deg[row] + deg[col]) / 2.0) ** alpha_deg
            degW = degW / (degW.mean() + self.eps)

            lossNce = -torch.log(qPos / (qPos + qNeg.sum(dim=1) + self.eps))
            lossNce = (lossNce * degW).mean()

            # (2) Fixed Implicit LNP Part
            # Compute local 1-hop spatial baseline scale for each node
            localScale = (distMatrix * adj).sum(dim=1) / (deg + self.eps)
            
            # Non-dimensionalize distance matrix to decouple from total node count N
            scaledDist = distMatrix / (localScale.unsqueeze(1) + self.eps)
            
            logits = -scaledDist / self.tauSoft
            logProb = torch.log_softmax(logits, dim=1)

            # Maximize conditional log-likelihood of topological neighbors
            lossSoftPerNode = - (logProb * adj).sum(dim=1) / (deg + self.eps)
            lossSoft = lossSoftPerNode.mean()

            lossNp = lossNce + beta_soft * lossSoft

        # ==========================================
        # 2. AR (Hub-focused via Log-Smooth Weighting)
        # ==========================================
        lossAr = torch.tensor(0.0, device=device)

        if lambda_ar > 0:
            eyeN = torch.eye(N, device=device, dtype=torch.bool)
            maskTriadic = adj.unsqueeze(2) * adj.unsqueeze(1)
            maskTriadic = maskTriadic * (~eyeN.unsqueeze(0)).float()

            unitVectors = D / distMatrix.unsqueeze(-1)
            cosMatrix = torch.einsum('uid,ujd->uij', unitVectors, unitVectors)

            violation = F.relu(cosMatrix - (1.0 - self.marginAr))
            penalty = maskTriadic * (violation ** 2)

            normFactor = deg * (deg - 1.0)
            perNodeAr = penalty.sum(dim=(1, 2)) / normFactor.clamp(min=1.0)

            # Replaces V4 linear weighting to handle scale-free graphs robustly
            wRaw = torch.log(deg + 1.0)
            wSmooth = wRaw / (wRaw.mean() + self.eps)

            lossAr = (perNodeAr * wSmooth).mean()

        # ==========================================
        # 3. EU (Global Weak Regularization)
        # ==========================================
        lossEu = torch.tensor(0.0, device=device)

        if numEdges > 0 and lambda_eu > 0:
            edgeLengths = torch.norm(posNormalized[row] - posNormalized[col], dim=-1)
            mu = edgeLengths.mean()
            lossEu = torch.mean(((edgeLengths - mu) / (mu + self.eps)) ** 2)

        # ==========================================
        # Final Fusion
        # ==========================================
        totalLoss = lambda_np * lossNp + \
                    lambda_ar * lossAr + \
                    lambda_eu * lossEu

        return totalLoss, {
            'np_loss': lossNp.item(),
            'ar_loss': lossAr.item(),
            'eu_loss': lossEu.item()
        }
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicSmoothLayoutLossV6_5(nn.Module):
    def __init__(self, margin_ar=0.05, beta_soft=0.1, lambda_scale_gamma=0.2, tau_rank=0.3, eps=1e-8):
        """
        DeterministicSmoothLayoutLossV6_5
        - Eliminates local measure bias via dynamic self-normalizing neighbor mapping (degLocal).
        - Sharpens negative pushing using a localized tauRank-bounded continuous soft-min operator.
        - Guarantees topological connection survival via a robust KNN-Adjacency union mask.
        - Counteracts scale null-mode drift with an exogenous density-compensated tau_init anchor.
        - Strict ASCII implementation with camelCase internal variables.
        """
        super().__init__()
        self.marginAr = margin_ar
        self.betaSoft = beta_soft               # Weight for local pairwise ranking term
        self.lambdaScaleGamma = lambda_scale_gamma # Core stiffness coefficient for absolute anchor
        self.tauRank = tau_rank                 # Temperature to sharpen the continuous soft-min profile
        self.eps = eps

    def forward(self, pos, edge_index, tau_init,
                lambda_np=1.0, lambda_ar=0.5, lambda_eu=0.05,
                alpha_deg=0.5):
        """
        Args:
            pos: Tensor (N, 2), absolute node coordinates from the implicit solver.
            edge_index: Tensor (2, E), topological edge indices.
            tau_init: Scalar Tensor, the frozen exogenous reference scale extracted at Epoch 0.
        """
        N = pos.shape[0]
        device = pos.device
        row, col = edge_index
        numEdges = row.shape[0]

        if N <= 1:
            return torch.tensor(0.0, device=device), {
                'np_loss': 0.0, 'ar_loss': 0.0, 'eu_loss': 0.0
            }

        # ==========================================
        # 0. Translation Invariance & Distance Matrix
        # ==========================================
        posCentered = pos - pos.mean(dim=0, keepdim=True)

        # Pairwise absolute distance matrix calculation O(N^2)
        D = posCentered.unsqueeze(0) - posCentered.unsqueeze(1)
        distMatrix = torch.norm(D, dim=-1) + self.eps

        # Topological adjacency matrix reconstruction
        adj = torch.zeros((N, N), device=device)
        if numEdges > 0:
            adj[row, col] = 1.0
            adj[col, row] = 1.0
        adj.fill_diagonal_(0.0)

        deg = adj.sum(dim=-1)

        # Smooth degree-weighting vector to balance hub node gradients
        degW = (deg ** alpha_deg)
        degW = degW / (degW.mean() + self.eps)

        # ==========================================
        # 1. NP (Union Horizon Softmax + Sharp Soft-min Ranking)
        # ==========================================
        lossNp = torch.tensor(0.0, device=device)

        if numEdges > 0 and lambda_np > 0:
            # (1) Non-differentiable Adaptive Temperature Field Selection
            localScale = (distMatrix * adj).sum(dim=1) / (deg + self.eps)
            tau = localScale.mean().detach()
            tau = torch.clamp(tau, min=0.1)

            # (2) Dynamic Adaptive KNN Window Selection
            k = min(max(20, int(0.1 * N)), N)
            _, knnIdx = distMatrix.topk(k=k, largest=False, dim=1)
            
            # Core Fix 2: Construct a robust union mask to protect true topological neighbors
            knnMask = torch.zeros_like(adj).scatter_(1, knnIdx, 1.0).bool()
            finalActiveMask = knnMask | adj.bool()
            
            # Erase far-field RHS noise by forcing excluded node logits to -1e9
            allLogits = - (distMatrix ** 2) / (2.0 * (tau ** 2) + self.eps)
            localLogits = allLogits.masked_fill(~finalActiveMask, -1e9)
            localLogProb = torch.log_softmax(localLogits, dim=1)
            
            # Core Fix 1: Dynamically count localized neighbors to prevent gradient dilution
            degLocal = (finalActiveMask.float() * adj).sum(dim=1)
            lossProbPerNode = - (localLogProb * adj).sum(dim=1) / (degLocal + self.eps)
            lossProb = (lossProbPerNode * degW).mean()

            # (3) Dimensionless Pairwise Ranking via Sharpened Log-Sum-Exp Soft-min
            posDist = distMatrix[row, col] / tau # Dimensionless active edge distances
            
            with torch.no_grad():
                eyeN = torch.eye(N, device=device, dtype=torch.bool)
                forbiddenMask = adj[row].bool() | eyeN[row] # Shape (E, N)
            
            maskedNegDistNorm = (distMatrix[row].clone()) / tau
            # Overwrite topological neighbors with large boundary value
            maskedNegDistNorm.masked_fill_(forbiddenMask, 1e4)
            
            # Core Fix 1 (Review 2): Inject tauRank to focus strictly on the hardest violators
            negSoftNorm = - self.tauRank * torch.logsumexp(-maskedNegDistNorm / self.tauRank, dim=1)
            
            # Smooth bounding envelope eliminates non-differentiable hard topk jitter
            negHard = torch.clamp(negSoftNorm, max=3.0)

            marginRank = 0.1
            lossRank = F.relu(posDist - negHard + marginRank).mean()

            # Balanced Neighborhood Preservation Loss Combination
            lossNp = lossProb + self.betaSoft * lossRank

        # ==========================================
        # 2. AR (Stable Hub-Focused Linear Barrier)
        # ==========================================
        lossAr = torch.tensor(0.0, device=device)

        if lambda_ar > 0:
            eyeN = torch.eye(N, device=device, dtype=torch.bool)
            maskTriadic = adj.unsqueeze(2) * adj.unsqueeze(1)
            maskTriadic = maskTriadic * (~eyeN.unsqueeze(0)).float()

            unitVectors = D / distMatrix.unsqueeze(-1)
            cosMatrix = torch.einsum('uid,ujd->uij', unitVectors, unitVectors)

            violation = F.relu(cosMatrix - (1.0 - self.marginAr))
            penalty = maskTriadic * violation

            normFactor = deg * (deg - 1.0)
            perNodeAr = penalty.sum(dim=(1, 2)) / normFactor.clamp(min=1.0)

            wRaw = torch.log(deg + 1.0)
            wSmooth = wRaw / (wRaw.mean() + self.eps)

            lossAr = (perNodeAr * wSmooth).mean()

        # ==========================================
        # 3. EU (Dual-Mode Scaling with Density Compensation)
        # ==========================================
        lossEu = torch.tensor(0.0, device=device)

        if numEdges > 0 and lambda_eu > 0:
            edgeLengths = torch.norm(posCentered[row] - posCentered[col], dim=-1)
            mu = edgeLengths.mean()
            
            # Sub-Loss A: Dimensionless relative variance constraints
            lossVar = torch.mean(((edgeLengths - mu) / (mu + self.eps)) ** 2)
            
            # Sub-Loss B: High-curvature absolute scale anchor bound to frozen tau_init
            lossScaleBase = ((mu / (tau_init + self.eps)) - 1.0) ** 2
            
            # Balance energy magnitude mismatch against O(N^2) global forces
            lossScale = lossScaleBase * deg.mean()
            
            lossEu = lossVar + self.lambdaScaleGamma * lossScale

        # ==========================================
        # Final Multitask Fusion
        # ==========================================
        totalLoss = lambda_np * lossNp + \
                    lambda_ar * lossAr + \
                    lambda_eu * lossEu

        return totalLoss, {
            'np_loss': lossNp.item(),
            'ar_loss': lossAr.item(),
            'eu_loss': lossEu.item()
        }


class DeterministicCanonicalEnergyLossV6_18(nn.Module):
    def __init__(self, margin_ar=0.05, lambda_scale_gamma=0.2, margin_np=0.2, 
                 margin_contrast=0.15, tau_soft=0.5, weight_src=0.5, eps=1e-8):
        """
        DeterministicCanonicalEnergyLossV6_18
        - Production-ready Canonical Ensemble Energy Model for high-stiffness graph visualization.
        - Core Fix 1: Eliminates dynamic reflection conditional branches to lock a single path Autograd graph.
        - Core Fix 2: Implements rigid integer-domain .clamp(min=1) to prevent log(eps) down-flow explosions.
        - Core Fix 3: Exposes direction-adaptive weight_src parameter to empower asymmetrical directional HPO search.
        - Core Fix 4: Strips deg.mean() from lossEu to achieve pure topology-isolated geometric scaling.
        """
        super().__init__()
        self.marginAr = margin_ar
        self.lambdaScaleGamma = lambda_scale_gamma 
        self.marginNp = margin_np               
        self.marginContrast = margin_contrast   
        self.tauSoft = tau_soft                 # Sole temperature scaling parameter for the canonical field
        self.weightSrc = weight_src             # Parametric source-direction focus factor (HPO Optimization Ready)
        self.eps = eps

    def forward(self, pos, edge_index, tau_init,
                lambda_np=1.0, lambda_ar=0.5, lambda_eu=0.05,
                alpha_deg=0.5):
        N = pos.shape[0]
        device = pos.device
        row, col = edge_index
        numEdges = row.shape[0]

        if N <= 1 or numEdges == 0:
            return torch.tensor(0.0, device=device), {
                'np_loss': 0.0, 'ar_loss': 0.0, 'eu_loss': 0.0
            }

        # 0. Translation Invariance & Pairwise Geometric Distance Discovery
        posCentered = pos - pos.mean(dim=0, keepdim=True)
        D = posCentered.unsqueeze(0) - posCentered.unsqueeze(1)
        distMatrix = torch.norm(D, dim=-1) + self.eps

        adj = torch.zeros((N, N), device=device)
        adj[row, col] = 1.0
        adj[col, row] = 1.0
        adj.fill_diagonal_(0.0)

        deg = adj.sum(dim=-1)

        lossNp = torch.tensor(0.0, device=device)
        if lambda_np > 0:
            # Anchor scale factor securely on GPU memory registers
            tau = tau_init.detach().to(device)
            tau = torch.clamp(tau, min=0.1)

            # Dynamic Adaptive KNN Window Discovery
            k = min(max(20, int(0.1 * N)), N)
            _, knnIdx = distMatrix.topk(k=k, largest=False, dim=1)
            
            knnMask = torch.zeros_like(adj).scatter_(1, knnIdx, 1.0).bool()
            finalActiveMask = knnMask | adj.bool()
            
            # Pristine continuous Gaussian potential fields
            baseLogits = - (distMatrix ** 2) / (2.0 * (tau ** 2) + self.eps)
            allLogits = baseLogits + adj * self.marginNp

            # Symmetrical localized negative horizon masking
            negMask = finalActiveMask & (~adj.bool())
            
            # Stable Log-Mean-Exp executing max-subtraction under the hood
            # [FIXED TYPO HERE]: Changed 'allLog logits' to 'allLogits'
            scaledNegLogits = allLogits / self.tauSoft
            scaledNegLogits = scaledNegLogits.masked_fill(~negMask, -1e9)
            
            rawLogSumExpNeg = torch.logsumexp(scaledNegLogits, dim=1, keepdim=True)
            
            # Core Fix 2: Clamp integer negative counts to 1 to eradicate log(eps) singularity traps
            negCount = negMask.sum(dim=1, keepdim=True).clamp(min=1)
            logMeanExpNeg = self.tauSoft * (rawLogSumExpNeg - torch.log(negCount.float()))

            # Core Fix 1: Linearized single-path variable index lookups (No dynamic fallbacks)
            posEdgeLogits = allLogits[row, col]                 # (E,)
            negLseSource = logMeanExpNeg[row].squeeze(-1)        # (E,)
            negLseTarget = logMeanExpNeg[col].squeeze(-1)        # (E,)
            
            # Direction A: Source -> Target Energy Delta
            deltaSrc = negLseSource - posEdgeLogits + self.marginContrast
            lossSrcToTgt = self.tauSoft * F.softplus(deltaSrc / self.tauSoft)
            
            # Direction B: Target -> Source Energy Delta
            deltaTgt = negLseTarget - posEdgeLogits + self.marginContrast
            lossTgtToSrc = self.tauSoft * F.softplus(deltaTgt / self.tauSoft)
            
            # Core Fix 3: Exposed parametric weighting balances bidirectional force importance
            lossNp = (self.weightSrc * lossSrcToTgt + (1.0 - self.weightSrc) * lossTgtToSrc).mean()

        # 2. AR Stable Barrier Component
        lossAr = torch.tensor(0.0, device=device)
        if lambda_ar > 0:
            eyeN = torch.eye(N, device=device, dtype=torch.bool)
            maskTriadic = adj.unsqueeze(2) * adj.unsqueeze(1)
            maskTriadic = maskTriadic * (~eyeN.unsqueeze(0)).float()

            unitVectors = D / distMatrix.unsqueeze(-1)
            cosMatrix = torch.einsum('uid,ujd->uij', unitVectors, unitVectors)

            violation = F.relu(cosMatrix - (1.0 - self.marginAr))
            penalty = maskTriadic * violation

            normFactor = deg * (deg - 1.0)
            perNodeAr = penalty.sum(dim=(1, 2)) / normFactor.clamp(min=1.0)

            wRaw = torch.log(deg + 1.0)
            wSmooth = wRaw / (wRaw.mean() + self.eps)
            lossAr = (perNodeAr * wSmooth).mean()

        # 3. EU Structural Scaling Component
        lossEu = torch.tensor(0.0, device=device)
        if lambda_eu > 0:
            edgeLengths = torch.norm(posCentered[row] - posCentered[col], dim=-1)
            mu = edgeLengths.mean()
            
            lossVar = torch.mean(((edgeLengths - mu) / (mu + self.eps)) ** 2)
            
            # Core Fix 4: Stripped deg.mean() to isolate topology from pure geometric volume metrics
            lossScale = ((mu / (tau + self.eps)) - 1.0) ** 2
            
            lossEu = lossVar + self.lambdaScaleGamma * lossScale

        totalLoss = lambda_np * lossNp + \
                    lambda_ar * lossAr + \
                    lambda_eu * lossEu

        return totalLoss, {
            'np_loss': lossNp.item(),
            'ar_loss': lossAr.item(),
            'eu_loss': lossEu.item()
        }

class UnifiedKlOrbitalLayoutLossV23(nn.Module):
    """
    [DeepCS V23 - 100% Pure Standard KL State]
    Unified Differentiable Graph Layout Loss with Rigorous Probabilistic Alignment.
    
    Rigid Corrections:
    1. Removes all topological mass weights, boosters, and external potential fields.
    2. Executes 100% textbook Standard KL Divergence over the dense P_target measurement.
    3. Retains 100% Pure ASCII characters, strict camelCase variables, and zero LaTeX symbols.
    """
    def __init__(self, lambdaNeighborBoost=0.0, lambdaRepulsion=0.0, eps=1e-8):
        """
        All boosters set to 0.0 to strictly enforce non-modified probability balance.
        """
        super(UnifiedKlOrbitalLayoutLossV23, self).__init__()
        self.eps = eps

    def forward(self, pos, P_target, edge_index, w_edge=None, l_edge=None, q_node=None,
                lambda_np_kl=1.0, lambda_ar=0.5, lambda_eu=0.1):
        """
        Args:
            pos (Tensor): Solved physical coordinates from ASGD engine, Shape [N, 2]
            P_target (Tensor): Rigorously normalized target probability matrix, Shape [N, N]
            edge_index (Tensor): Local sparse adjacency index of current graph, Shape [2, E]
        """
        numNodes = pos.shape[0]
        computeDevice = pos.device
        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
        
        row, col = edge_index
        numEdges = row.shape[0]

        if numNodes <= 1 or numEdges == 0:
            return torch.tensor(0.0, device=computeDevice), {
                "np_loss": 0.0, "ar_loss": 0.0, "eu_loss": 0.0
            }

        # =========================================================================
        # 0. Isotropic Scale Normalization
        # =========================================================================
        posCentered = pos - pos.mean(dim=0, keepdim=True)
        posStd = posCentered.std(dim=0).mean() + self.eps
        posNormalized = posCentered / posStd

        posDiff = posNormalized.unsqueeze(0) - posNormalized.unsqueeze(1)
        distSqNorm = (posDiff ** 2).sum(dim=-1)
        distMatNorm = torch.sqrt(distSqNorm + self.eps)

        adjMat = (P_target > 0).float()
        deg = adjMat.sum(dim=-1)

        # =========================================================================
        # 1. Textbook Standard KL Divergence Module (100% 纯净概率对齐)
        # =========================================================================
        lossNpKl = torch.tensor(0.0, device=computeDevice)
        
        if lambda_np_kl > 0:
            # 1.1 Predict geometric probability density matrix over full N x N space
            qUnnorm = 1.0 / (1.0 + distSqNorm)
            qUnnorm = qUnnorm.masked_fill(eyeMat, 0.0)
            qSum = qUnnorm.sum().clamp(min=self.eps)
            qPred = qUnnorm / qSum

            # 1.2 点对点无损香农熵对齐
            # 严格剥离任何度数加权，直接在 P_target > 0 的可微掩码上顺次执行经典求和
            validMask = P_target > 0
            P_valid = P_target[validMask]
            Q_valid = qPred[validMask].clamp(min=self.eps)
            
            # 标准教科书定义：无任何乘法系数与加法偏移
            lossNpKl = torch.sum(P_valid * torch.log(P_valid / Q_valid))

        # =========================================================================
        # 2. V23 Native AR Triadic Angular Resolution Component
        # =========================================================================
        lossAr = torch.tensor(0.0, device=computeDevice)
        if lambda_ar > 0:
            tri = adjMat.unsqueeze(2) * adjMat.unsqueeze(1)
            tri = tri * (~eyeMat.unsqueeze(0)).float()

            unit = posDiff / distMatNorm.unsqueeze(-1)
            cos = torch.einsum("aid,ajd->aij", unit, unit)

            violationAr = F.relu(cos - (1.0 - 0.05))
            penaltyAr = tri * violationAr

            normAr = deg * (deg - 1.0)
            perNodeAr = penaltyAr.sum(dim=(1, 2)) / normAr.clamp(min=1.0)

            wAr = torch.log(deg + 1.0)
            wAr = wAr / (wAr.mean() + self.eps)
            lossAr = (perNodeAr * wAr).mean()

        # =========================================================================
        # 3. V23 Corrected EU Variance Component
        # =========================================================================
        lossEu = torch.tensor(0.0, device=computeDevice)
        if lambda_eu > 0:
            posCenteredRaw = pos - pos.mean(dim=0, keepdim=True)
            edgeLenRaw = torch.norm(posCenteredRaw[row] - posCenteredRaw[col], dim=-1)
            muRaw = edgeLenRaw.mean()

            varLoss = ((edgeLenRaw - muRaw) / (muRaw + self.eps)) ** 2
            lossEu = varLoss.mean()

        # =========================================================================
        # 4. Total Pure Fusion
        # =========================================================================
        totalLoss = (lambda_np_kl * lossNpKl) + (lambda_ar * lossAr) + (lambda_eu * lossEu)

        return totalLoss, {
            "np_loss": lossNpKl.item(),
            "ar_loss": lossAr.item(),
            "eu_loss": lossEu.item()
        }


class UnifiedKlOrbitalLayoutLossV25(nn.Module):
    """
    [DeepCS V25.4 - Rigid Viewport Pairwise Rank Loss]
    Strictly aligns with the hard-truncation nature of NP and LNP metrics.
    
    Ultimate Architectural Adjustments:
    1. Rigid Isotropic Denominator Detachment: Isolates the scale variance from 
       the adjoint chain, leaving zero bypass for trivial scale-fitting.
    2. Adaptive Pairwise Gap: Scale-locks the margin parameter to the current 
       viewport radius, securing a stable rank gradient across all epochs.
    """
    def __init__(self, marginMargin=0.2, temperature=0.05, eps=1e-8):
        super(UnifiedKlOrbitalLayoutLossV25, self).__init__()
        self.eps = eps
        self.marginMargin = marginMargin  # 标准化视口下的相对偏序间距 gamma
        self.temperature = temperature    # 标准化视口下的 Softplus 平滑温控因子 tau

    def forward(self, pos, P_target, edge_index, lambda_np_kl=1.0, lambda_ar=0.5, lambda_eu=0.1):
        numNodes = pos.shape[0]
        computeDevice = pos.device
        eyeMat = torch.eye(numNodes, dtype=torch.bool, device=computeDevice)
        
        row, col = edge_index
        numEdges = row.shape[0]

        if numNodes <= 1 or numEdges == 0:
            return torch.tensor(0.0, device=computeDevice), {
                "np_loss": 0.0, "ar_loss": 0.0, "eu_loss": 0.0
            }

        # =========================================================================
        # 0. Rigid Viewport Standardizer (原位斩断尺度逃逸链路)
        # =========================================================================
        posCentered = pos - pos.mean(dim=0, keepdim=True)
        
        # 精准提取当前点阵的各向同性物理标准差，并实施显式可微脱离 (.detach())
        # 这宣布该尺度标量在伴随矩阵中被视为常数，扼杀了模型通过等比例缩放欺骗梯度的空间
        currentScale = (posCentered.std(dim=0).mean() + self.eps).detach()
        
        # 建立绝对刚性的标准化视口流形
        posViewport = posCentered / currentScale

        # 全局几何距离场矩阵计算（完全基于刚性视口）
        posDiff = posViewport.unsqueeze(0) - posViewport.unsqueeze(1) # Shape [N, N, 2]
        distMat = torch.sqrt((posDiff ** 2).sum(dim=-1) + self.eps) # Shape [N, N]

        # 提取直连邻居与非邻居二元遮罩
        neighborMask = P_target > 0 # Shape [N, N]
        nonNeighborMask = (~neighborMask) & (~eyeMat) # Shape [N, N]

        deg = neighborMask.float().sum(dim=-1)
        nodeWeights = torch.log(deg / (deg.mean() + self.eps) + 1.0)

        # =========================================================================
        # 1. Scale-Invariant Pairwise Rank Softplus Branch
        # =========================================================================
        lossNpKl = torch.tensor(0.0, device=computeDevice)
        
        if lambda_np_kl > 0:
            pairwiseLossSum = torch.zeros(numNodes, device=computeDevice)
            
            # 行级偏序对齐核验
            for i in range(numNodes):
                neighborIndices = torch.where(neighborMask[i])[0]
                nonNeighborIndices = torch.where(nonNeighborMask[i])[0]
                
                if neighborIndices.shape[0] == 0 or nonNeighborIndices.shape[0] == 0:
                    continue
                
                # 提取标准化视口下的邻居距离矢量与非邻居距离矢量
                d_ij = distMat[i, neighborIndices].unsqueeze(1) # Shape [NumNeighbors, 1]
                d_ik = distMat[i, nonNeighborIndices].unsqueeze(0) # Shape [1, NumNonNeighbors]
                
                # 序对差分核心：邻居距离必须小于非邻居距离
                # 间距 gamma 和温度 tau 已天然在标准视口尺度下标定，无需再做尺度映射
                violationMat = d_ij + self.marginMargin - d_ik
                
                # 全连续、全鲁棒的连续排序 Softplus 函数，提供绝对平滑的排斥动力
                softRankCost = self.temperature * torch.log(1.0 + torch.exp(violationMat / self.temperature))
                
                # 行内序对求取期望，斩断高度数节点的梯度霸权
                pairwiseLossSum[i] = softRankCost.mean()

            # 融入度数权重，收拢全图偏序损失
            lossNpKl = torch.sum(nodeWeights * pairwiseLossSum) / float(numNodes)

        # =========================================================================
        # 2. Native AR Component (Calculated on Rigid Viewport)
        # =========================================================================
        lossAr = torch.tensor(0.0, device=computeDevice)
        if lambda_ar > 0:
            tri = neighborMask.float().unsqueeze(2) * neighborMask.float().unsqueeze(1)
            tri = tri * (~eyeMat.unsqueeze(0)).float()

            unit = posDiff / distMat.unsqueeze(-1)
            cos = torch.einsum("aid,ajd->aij", unit, unit)

            violationAr = F.relu(cos - 0.95)
            penaltyAr = tri * violationAr

            normAr = deg * (deg - 1.0)
            perNodeAr = penaltyAr.sum(dim=(1, 2)) / normAr.clamp(min=1.0)

            wAr = torch.log(deg + 1.0)
            wAr = wAr / (wAr.mean() + self.eps)
            lossAr = (perNodeAr * wAr).mean()

        # =========================================================================
        # 3. Corrected EU Variance Component (Viewport Aligned Standard Deviation)
        # =========================================================================
        lossEu = torch.tensor(0.0, device=computeDevice)
        if lambda_eu > 0:
            varLoss = torch.tensor(0.0, device=computeDevice)
            if numEdges > 0:
                edgeLenRaw = torch.norm(posViewport[row] - posViewport[col], dim=-1)
                muRaw = edgeLenRaw.mean()
                varLoss = ((edgeLenRaw - muRaw) / (muRaw + self.eps)) ** 2
            lossEu = varLoss.mean()

        totalLoss = (lambda_np_kl * lossNpKl) + (lambda_ar * lossAr) + (lambda_eu * lossEu)

        return totalLoss, {
            "np_loss": lossNpKl.item(),
            "ar_loss": lossAr.item(),
            "eu_loss": lossEu.item()
        }