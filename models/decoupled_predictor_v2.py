import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv
import math

#class DecoupledForcePredictor(nn.Module):
#    
#    def __init__(self, node_in_dim=16, edge_in_dim=4, hidden_dim=64, latent_dim=128, num_layers=3, heads=4, dropout=0.1):
#        super(DecoupledForcePredictor, self).__init__()
#        self.num_layers = num_layers
#        self.dropout = nn.Dropout(dropout)
#        
#        # ==========================================================
#        # Stream 1: Spatial (深层 GATv2) -> 需要大局观，寻找宏观星系
#        # ==========================================================
#        self.spatial_node_proj = nn.Linear(node_in_dim, hidden_dim)
#        self.spatial_edge_proj = nn.Linear(edge_in_dim, hidden_dim)
#        
#        self.spatial_convs = nn.ModuleList()
#        for _ in range(num_layers): # 保持 3 层深度
#            self.spatial_convs.append(
#                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout)
#            )
#        self.spatial_out_proj = nn.Linear(hidden_dim * (num_layers + 1), latent_dim // 2)
#
#        # ==========================================================
#        # Stream 2: Role (浅层 GINE) -> 拒绝过度混合，坚守结构指纹与边缘属性
#        # ==========================================================
#        self.role_node_proj = nn.Linear(node_in_dim, hidden_dim)
#        
#        # 为 GINE 添加专属的边特征投影层，将 4 维边特征映射到 64 维
#        self.role_edge_proj = nn.Linear(edge_in_dim, hidden_dim)
#        
#        self.role_convs = nn.ModuleList()
#        # 强制为 1 层，防止特征过度平滑化 (Over-smoothing)
#        for _ in range(1): 
#            gin_nn = nn.Sequential(
#                nn.Linear(hidden_dim, hidden_dim * 2),
#                nn.LayerNorm(hidden_dim * 2),
#                nn.ELU(),
#                nn.Linear(hidden_dim * 2, hidden_dim)
#            )
#            # 使用 GINEConv，设置 edge_dim 以支持边特征注入
#            self.role_convs.append(GINEConv(gin_nn, train_eps=True, edge_dim=hidden_dim))
#            
#        # 融合原始特征(16) + 投影特征(64) + 1层GINE特征(64) = 144维
#        fusion_dim = hidden_dim * 2 + node_in_dim 
#        self.role_fusion_norm = nn.LayerNorm(fusion_dim)
#        self.role_out_proj = nn.Linear(fusion_dim, latent_dim // 2)
#        
#        # ==========================================================
#        # Physics Decoder (物理参数解码引擎)
#        # ==========================================================
#        self.repulsion_head = LatentNodeRepulsionHead(in_dim=latent_dim, q_min=0.1, q_max=50.0)
#        self.stiffness_head = LatentEdgeAttributeHead(in_dim=latent_dim, out_min=0.1, out_max=5.0)
#        self.length_head = LatentEdgeAttributeHead(in_dim=latent_dim, out_min=0.5, out_max=15.0)
#
#    def forward(self, x, edge_index, edge_attr):
#        row, col = edge_index
#
#        # --- Stream 1: Spatial Pass (深层) ---
#        h_sp = F.elu(self.spatial_node_proj(x))
#        e_sp = F.elu(self.spatial_edge_proj(edge_attr))
#        sp_list = [h_sp]
#        for conv in self.spatial_convs:
#            # GATv2 支持边特征输入
#            h_sp = conv(h_sp, edge_index, edge_attr=e_sp)
#            h_sp = F.elu(h_sp)
#            h_sp = self.dropout(h_sp)
#            sp_list.append(h_sp)
#        h_sp_jk = torch.cat(sp_list, dim=-1)
#        Z_spatial = self.spatial_out_proj(h_sp_jk)
#        
#        # --- Stream 2: Role Pass (浅层 GINE) ---
#        h_ro = F.elu(self.role_node_proj(x))
#        # 前向传播中计算角色轨道的边嵌入
#        e_ro = F.elu(self.role_edge_proj(edge_attr)) 
#        ro_list = [h_ro]
#        
#        for conv in self.role_convs:
#            # GINEConv 强制灌入边特征 (e_ro)
#            h_ro = conv(h_ro, edge_index, edge_attr=e_ro)
#            h_ro = F.elu(h_ro)
#            h_ro = self.dropout(h_ro)
#            ro_list.append(h_ro)
#            
#        ro_list.append(x) # 坚守原始护盾 (包含 RWPE 和 f)
#        h_ro_jk = torch.cat(ro_list, dim=-1)
#        h_ro_jk = self.role_fusion_norm(h_ro_jk)
#        Z_role = self.role_out_proj(h_ro_jk)
#
#        # --- Fusion ---
#        Z_fused = torch.cat([Z_spatial, Z_role.detach()], dim=-1)
#        
#        # --- Physics Decoding ---
#        q_node = self.repulsion_head(Z_fused)
#        w_edge = self.stiffness_head(Z_fused[row], Z_fused[col])
#        l_edge = self.length_head(Z_fused[row], Z_fused[col])
#        
#        return Z_fused, Z_spatial, Z_role, q_node, w_edge, l_edge
#        
#        
#class LatentNodeRepulsionHead(nn.Module):
#    def __init__(self, in_dim=128, q_min=0.1, q_max=50.0): 
#        super().__init__()
#        self.q_min = q_min
#        self.q_max = q_max
#        self.mlp = nn.Sequential(
#            nn.Linear(in_dim, 64),
#            nn.ELU(),
#            nn.Linear(64, 1)
#        )
#        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
#        nn.init.zeros_(self.mlp[-1].bias)
#
#    def forward(self, z):
#        x = self.mlp(z).squeeze(-1)
#        return torch.sigmoid(x) * (self.q_max - self.q_min) + self.q_min
#
#
#class LatentEdgeAttributeHead(nn.Module):
#    def __init__(self, in_dim=128, out_min=0.1, out_max=50.0): 
#        super().__init__()
#        self.out_min = out_min
#        self.out_max = out_max
#        
#        self.mlp = nn.Sequential(
#            nn.Linear(in_dim * 2, 64),
#            nn.ELU(),
#            nn.Linear(64, 1)
#        )
#        
#        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
#        nn.init.zeros_(self.mlp[-1].bias)
#
#    def forward(self, z_u, z_v):
#        sim_feat = z_u * z_v
#        diff_feat = (z_u - z_v) ** 2
#        h_edge = torch.cat([sim_feat, diff_feat], dim=-1)
#        
#        x = self.mlp(h_edge).squeeze(-1)
#        return torch.sigmoid(x) * (self.out_max - self.out_min) + self.out_min

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv


class DecoupledForcePredictor(nn.Module):
    
    def __init__(self, node_in_dim=16, edge_in_dim=1, hidden_dim=64, latent_dim=128, num_layers=3, heads=4, dropout=0.1):
        super(DecoupledForcePredictor, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # ==========================================================
        # Stream 1: Spatial (深层 GATv2) -> 需要大局观，寻找宏观星系
        # ==========================================================
        self.spatial_node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.spatial_edge_proj = nn.Linear(edge_in_dim, hidden_dim)
        
        self.spatial_convs = nn.ModuleList()
        for _ in range(num_layers): 
            self.spatial_convs.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout)
            )
        self.spatial_out_proj = nn.Linear(hidden_dim * (num_layers + 1), latent_dim // 2)

        # ==========================================================
        # Stream 2: Role (浅层 GINE) -> 拒绝过度混合，坚守结构指纹与边缘属性
        # ==========================================================
        self.role_node_proj = nn.Linear(node_in_dim, hidden_dim)
        
        # 为 GINE 添加专属的边特征投影层
        self.role_edge_proj = nn.Linear(edge_in_dim, hidden_dim)
        
        self.role_convs = nn.ModuleList()
        for _ in range(1): 
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.role_convs.append(GINEConv(gin_nn, train_eps=True, edge_dim=hidden_dim))
            
        fusion_dim = hidden_dim * 2 + node_in_dim 
        self.role_fusion_norm = nn.LayerNorm(fusion_dim)
        self.role_out_proj = nn.Linear(fusion_dim, latent_dim // 2)
        
        # ==========================================================
        # Physics Decoder (物理参数解码引擎) - 启用 Softplus 无阻碍推断
        # ==========================================================
        # 释放 q_max 到更大的空间，同时保证 q_min
        self.repulsion_head = LatentNodeRepulsionHead(in_dim=latent_dim, q_min=0.01, q_max=15.0)
        self.stiffness_head = LatentEdgeAttributeHead(in_dim=latent_dim, out_min=0.001, out_max=15.0)
        # 解除 l_edge 的天花板限制，允许模型撑大空间
        self.length_head = LatentEdgeAttributeHead(in_dim=latent_dim, out_min=0.1, out_max=15.0)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        # --- Stream 1: Spatial Pass ---
        h_sp = F.elu(self.spatial_node_proj(x))
        e_sp = F.elu(self.spatial_edge_proj(edge_attr))
        sp_list = [h_sp]
        for conv in self.spatial_convs:
            h_sp = conv(h_sp, edge_index, edge_attr=e_sp)
            h_sp = F.elu(h_sp)
            h_sp = self.dropout(h_sp)
            sp_list.append(h_sp)
        h_sp_jk = torch.cat(sp_list, dim=-1)
        Z_spatial = self.spatial_out_proj(h_sp_jk)
        
        # --- Stream 2: Role Pass ---
        h_ro = F.elu(self.role_node_proj(x))
        e_ro = F.elu(self.role_edge_proj(edge_attr)) 
        ro_list = [h_ro]
        
        for conv in self.role_convs:
            h_ro = conv(h_ro, edge_index, edge_attr=e_ro)
            h_ro = F.elu(h_ro)
            h_ro = self.dropout(h_ro)
            ro_list.append(h_ro)
            
        ro_list.append(x) 
        h_ro_jk = torch.cat(ro_list, dim=-1)
        h_ro_jk = self.role_fusion_norm(h_ro_jk)
        Z_role = self.role_out_proj(h_ro_jk)

        # --- Fusion ---
        Z_fused = torch.cat([Z_spatial, Z_role.detach()], dim=-1)
        
        # --- Physics Decoding ---
        q_node = self.repulsion_head(Z_fused)
        w_edge = self.stiffness_head(Z_fused[row], Z_fused[col])
        l_edge = self.length_head(Z_fused[row], Z_fused[col])
        
        return Z_fused, Z_spatial, Z_role, q_node, w_edge, l_edge
        


class AblationForcePredictor(nn.Module):
    def __init__(self, node_in_dim=16, edge_in_dim=1, hidden_dim=64, latent_dim=128, 
                 num_layers=3, heads=4, dropout=0.1,
                 spatial_type="GATv2",   # 可选: "GATv2", "GCN", "GIN"
                 use_role_stream=True,   # 可选: True, False
                 isolate_gradient=True,  # 默认锁定为 True，与 Baseline 行为对齐
                 capacity_match=False    # 控制 Exp 3 容量匹配的拓扑开关
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.spatial_type = spatial_type
        self.use_role_stream = use_role_stream
        self.isolate_gradient = isolate_gradient
        self.capacity_match = capacity_match
        
        # 显式挂载全局特征维度，修复前向作用域缺陷
        self.latent_dim = latent_dim
        self.spatial_out_dim = latent_dim if ((not use_role_stream) and capacity_match) else (latent_dim // 2)
        self.role_out_dim = latent_dim // 2

        # ==========================================================
        # 1. Stream 1: Spatial Backbone 核心实例化
        # ==========================================================
        self.spatial_node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.spatial_edge_proj = nn.Linear(edge_in_dim, hidden_dim)
        
        self.spatial_convs = nn.ModuleList()
        for _ in range(num_layers):
            if spatial_type == "GATv2":
                self.spatial_convs.append(
                    GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout)
                )
            elif spatial_type == "GCN":
                self.spatial_convs.append(GCNConv(hidden_dim, hidden_dim)) 
            elif spatial_type == "GIN":
                gin_nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                self.spatial_convs.append(GINEConv(gin_nn, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f"Unsupported spatial backbone type: {spatial_type}")
            
        self.spatial_out_proj = nn.Linear(hidden_dim * (num_layers + 1), self.spatial_out_dim)

        # ==========================================================
        # 2. Physics Decoder (顺序大幅提至此处)
        # 确保下游解码器的初始权重在 6 组消融中绝对一致，彻底免疫由下方条件分支引发的 RNG 种子漂移
        # ==========================================================
        from models.decoupled_predictor_v2 import LatentNodeRepulsionHead, LatentEdgeAttributeHead
        self.repulsion_head = LatentNodeRepulsionHead(in_dim=latent_dim, q_min=0.01, q_max=15.0)
        self.stiffness_head = LatentEdgeAttributeHead(in_dim=latent_dim, out_min=0.001, out_max=15.0)
        self.length_head = LatentEdgeAttributeHead(in_dim=latent_dim, out_min=0.1, out_max=15.0)

        # ==========================================================
        # 3. Stream 2: Role Backbone 条件实例化 (移至尾部)
        # 彻底解决 Exp 2 / Exp 3 在关闭时参数空间非同构、优化器存在未使用参数污染的学术漏洞
        # ==========================================================
        if self.use_role_stream:
            self.role_node_proj = nn.Linear(node_in_dim, hidden_dim)
            self.role_edge_proj = nn.Linear(edge_in_dim, hidden_dim)
            
            self.role_convs = nn.ModuleList()
            for _ in range(1): 
                role_gin_nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                self.role_convs.append(GINEConv(role_gin_nn, train_eps=True, edge_dim=hidden_dim))
            
            fusion_dim = hidden_dim * 2 + node_in_dim 
            self.role_fusion_norm = nn.LayerNorm(fusion_dim)
            self.role_out_proj = nn.Linear(fusion_dim, self.role_out_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        # --- Stream 1: Spatial Pass ---
        h_sp = F.elu(self.spatial_node_proj(x))
        e_sp = F.elu(self.spatial_edge_proj(edge_attr))
        sp_list = [h_sp]
        
        for conv in self.spatial_convs:
            if self.spatial_type in ["GATv2", "GIN"]:
                h_sp = conv(h_sp, edge_index, edge_attr=e_sp)
            elif self.spatial_type == "GCN":
                h_sp = conv(h_sp, edge_index, edge_weight=edge_attr.mean(dim=-1))
            h_sp = F.elu(h_sp)
            h_sp = self.dropout(h_sp)
            sp_list.append(h_sp)
            
        h_sp_jk = torch.cat(sp_list, dim=-1)
        Z_spatial = self.spatial_out_proj(h_sp_jk)
        
        # --- Stream 2: Role Pass & Fusion Routing ---
        if self.use_role_stream:
            h_ro = F.elu(self.role_node_proj(x))
            e_ro = F.elu(self.role_edge_proj(edge_attr))
            ro_list = [h_ro]
            
            for conv in self.role_convs:
                h_ro = conv(h_ro, edge_index, edge_attr=e_ro)
                h_ro = F.elu(h_ro)
                h_ro = self.dropout(h_ro)
                ro_list.append(h_ro)
                
            ro_list.append(x)
            h_ro_jk = torch.cat(ro_list, dim=-1)
            h_ro_jk = self.role_fusion_norm(h_ro_jk)
            
            # 保持 Z_role 本身作为带全量计算图的活张量，用于前向返回
            Z_role = self.role_out_proj(h_ro_jk)
            
            # 【核心修复】：将隔离机制转交给专用局部变量，杜绝污染 Z_role 本身
            Z_role_for_fusion = Z_role.detach() if self.isolate_gradient else Z_role
            
            # 使用隔离后的张量执行拼接
            Z_fused = torch.cat([Z_spatial, Z_role_for_fusion], dim=-1)
        else:
            # 严格对齐“只切模块，不改结构”原则：生成纯粹的静态零张量执行等长占位
            Z_role = torch.zeros((x.shape[0], self.role_out_dim), device=x.device, dtype=x.dtype)
            
            if self.capacity_match:
                # Exp 3 (容量匹配): 空间轨直推 128 维，不拼接，防止引入零值特征对 Decoder 的偏置干扰
                Z_fused = Z_spatial
            else:
                # Exp 2 (零占位消融): 严格保持双轨 Concat 物理拓扑
                Z_fused = torch.cat([Z_spatial, Z_role], dim=-1)

        # --- Physics Decoding ---
        q_node = self.repulsion_head(Z_fused)
        w_edge = self.stiffness_head(Z_fused[row], Z_fused[col])
        l_edge = self.length_head(Z_fused[row], Z_fused[col])
        
        # 此时返回的 Z_role 与原 Baseline 语义位对齐，外界计算 latent loss 时图完全连通
        return Z_fused, Z_spatial, Z_role, q_node, w_edge, l_edge
#class LatentNodeRepulsionHead(nn.Module):
#    def __init__(self, in_dim=128, q_min=0.1, q_max=100.0): 
#        super().__init__()
#        self.q_min = q_min
#        self.q_max = q_max
#        self.mlp = nn.Sequential(
#            nn.Linear(in_dim, 64),
#            nn.ELU(),
#            nn.Linear(64, 1)
#        )
#        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
#        nn.init.zeros_(self.mlp[-1].bias)
#
#    def forward(self, z):
#        x = self.mlp(z).squeeze(-1)
#        # [核心重构] Softplus 保证输出为正，彻底消除高值区的梯度消失
#        val = F.softplus(x) + self.q_min
#        # 加上硬截断防止早期梯度爆炸冲垮求解器
#        return torch.clamp(val, min=self.q_min, max=self.q_max)
#
#
#class LatentEdgeAttributeHead(nn.Module):
#    def __init__(self, in_dim=128, out_min=0.1, out_max=50.0): 
#        super().__init__()
#        self.out_min = out_min
#        self.out_max = out_max
#        
#        self.mlp = nn.Sequential(
#            nn.Linear(in_dim * 2, 64),
#            nn.ELU(),
#            nn.Linear(64, 1)
#        )
#        
#        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
#        nn.init.zeros_(self.mlp[-1].bias)
#
#    def forward(self, z_u, z_v):
#        sim_feat = z_u * z_v
#        diff_feat = (z_u - z_v) ** 2
#        h_edge = torch.cat([sim_feat, diff_feat], dim=-1)
#        
#        x = self.mlp(h_edge).squeeze(-1)
#        # [核心重构] Softplus 无界限输出，赋予模型撑开空间的物理能力
#        val = F.softplus(x) + self.out_min
#        return torch.clamp(val, min=self.out_min, max=self.out_max)

#class LatentNodeRepulsionHead(nn.Module):
#    def __init__(self, in_dim=128, q_min=0.1, q_max=100.0): 
#        super().__init__()
#        self.q_min = q_min
#        self.q_max = q_max
#        self.mlp = nn.Sequential(
#            nn.Linear(in_dim, 64),
#            nn.LayerNorm(64), # 增加 LayerNorm 稳定极端图的特征漂移
#            nn.ELU(),
#            nn.Linear(64, 1)
#        )
#        # 使用更小的初始化增益，确保初始输出在 Sigmoid 的线性区（靠近 0）
#        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
#        nn.init.zeros_(self.mlp[-1].bias)
#
#    def forward(self, z):
#        x = self.mlp(z).squeeze(-1)
#        # [修改] 使用 Scaled Sigmoid 替代 Softplus + Clamp
#        # 初始时 x 接近 0，sigmoid(0) = 0.5，输出在 (q_min + q_max)/2 左右
#        val = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(x)
#        return val
#
#
#class LatentEdgeAttributeHead(nn.Module):
#    def __init__(self, in_dim=128, out_min=0.001, out_max=50.0): 
#        super().__init__()
#        self.out_min = out_min
#        self.out_max = out_max
#        
#        self.mlp = nn.Sequential(
#            nn.Linear(in_dim * 2, 64),
#            nn.LayerNorm(64),
#            nn.ELU(),
#            nn.Linear(64, 1)
#        )
#        
#        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
#        nn.init.zeros_(self.mlp[-1].bias)
#
#    def forward(self, z_u, z_v):
#        # 保持原有的特征交叉方式
#        sim_feat = z_u * z_v
#        diff_feat = (z_u - z_v) ** 2
#        h_edge = torch.cat([sim_feat, diff_feat], dim=-1)
#        
#        x = self.mlp(h_edge).squeeze(-1)
#        # [修改] 使用 Scaled Sigmoid 替代 Softplus + Clamp
#        val = self.out_min + (self.out_max - self.out_min) * torch.sigmoid(x)
#        return val
        
class LatentNodeRepulsionHead(nn.Module):
    def __init__(self, in_dim=128, q_min=0.1, q_max=100.0, init_val=1.0): 
        super().__init__()
        self.q_min = q_min
        self.q_max = q_max
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        
        # 【核心修改】斥力头同步引入逆向偏置推导，迫使初始输出等于 init_val (1.0)
        targetRatio = (init_val - q_min) / (q_max - q_min + 1e-6)
        targetRatio = max(0.001, min(0.999, targetRatio))
        initBias = math.log(targetRatio / (1.0 - targetRatio))
        
        nn.init.constant_(self.mlp[-1].bias, initBias)

    def forward(self, z):
        x = self.mlp(z).squeeze(-1)
        # 保持 Scaled Sigmoid 有界安全映射
        val = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(x)
        return val
        
        
class LatentEdgeAttributeHead(nn.Module):
    def __init__(self, in_dim=128, out_min=0.001, out_max=50.0, init_val=1.0): 
        super().__init__()
        self.out_min = out_min
        self.out_max = out_max
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        
        # 【核心修改】计算逆向偏置，迫使前向初始输出精准等于 init_val (1.0)
        targetRatio = (init_val - out_min) / (out_max - out_min + 1e-6)
        targetRatio = max(0.001, min(0.999, targetRatio)) 
        initBias = math.log(targetRatio / (1.0 - targetRatio))
        
        nn.init.constant_(self.mlp[-1].bias, initBias)

    def forward(self, z_u, z_v):
        sim_feat = z_u * z_v
        diff_feat = (z_u - z_v) ** 2
        h_edge = torch.cat([sim_feat, diff_feat], dim=-1)
        
        x = self.mlp(h_edge).squeeze(-1)
        val = self.out_min + (self.out_max - self.out_min) * torch.sigmoid(x)
        return val