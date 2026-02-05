import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class PureStressLoss(nn.Module):
    def forward(self, final_pos, d_sp, weights, mask, batch_data=None):
        mask_mat = mask.float().unsqueeze(2) * mask.float().unsqueeze(1)
        dist_curr = torch.cdist(final_pos, final_pos, p=2).clamp(min=1e-8)
        
        #s
        num = (weights * d_sp * dist_curr * mask_mat).sum(dim=(1, 2), keepdim=True)
        den = (weights * (dist_curr**2) * mask_mat).sum(dim=(1, 2), keepdim=True)
        s = num / (den + 1e-12)
        
        # Normalized Stress
        loss_stress = (weights * (s * dist_curr - d_sp)**2 * mask_mat).sum() / \
                      ((weights * d_sp**2 * mask_mat).sum() + 1e-12)
        
        return loss_stress, loss_stress, torch.tensor(0.0, device=final_pos.device)

class CognitiveRankLoss(nn.Module):
    def __init__(self, t=0.5, margin=0.5, alpha=1.0, lambda_stress=5.0, lambda_rank=1.0):
        """
        RankLoss V3.0 (Dual-Space & Log-Ratio)
        Args:
            t: Viewport Scale (Gaussian width). Fixed ruler in standardized viewport.
            margin: Target Log-Ratio threshold. Target area ratio in log space.
            alpha: Weight power for status difference.
            lambda_stress: Weight for physical stress loss.
            lambda_rank: Weight for cognitive rank loss.
        """
        super(CognitiveRankLoss, self).__init__()
        self.t = t
        self.margin = margin
        self.alpha = alpha
        self.lambda_stress = lambda_stress
        self.lambda_rank = lambda_rank

    def viewport_projection(self, pos, mask):
        """
        Step 1: Viewport Projection (Masked Instance Normalization)
        Map physical coordinates to standardized Viewport Space N(0, 1) 
        to eliminate scale sensitivity.
        """
        # pos: [B, N, 2]
        # mask: [B, N]
        
        mask_f = mask.unsqueeze(-1).float() # [B, N, 1]
        num_nodes = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0) # [B, 1, 1]
        
        # 1. Compute Mean (excluding padding)
        mu = (pos * mask_f).sum(dim=1, keepdim=True) / num_nodes
        
        # 2. Compute Std (excluding padding)
        centered = (pos - mu) * mask_f
        var = (centered ** 2).sum(dim=1, keepdim=True) / num_nodes
        sigma = torch.sqrt(var + 1e-8)
        
        # Use mean std of x and y to preserve Aspect Ratio
        sigma_global = sigma.mean(dim=-1, keepdim=True) # [B, 1, 1]
        
        # 3. Project
        z_view = centered / (sigma_global + 1e-8)
        
        return z_view * mask_f # Padding remains 0

    def compute_visual_proxy(self, z_view, adj_mask, mask):
        """
        Step 2: Geometric Visual Proxy
        Phi = -ln( Density )
        Density = sum( Gaussian(dist) ) for neighbors
        """
        # z_view: [B, N, 2]
        
        # 1. Compute Pairwise Distances in Viewport Space
        sq_norm = (z_view ** 2).sum(dim=2, keepdim=True)
        dist_sq = sq_norm + sq_norm.transpose(1, 2) - 2 * (z_view @ z_view.transpose(1, 2))
        dist_sq = dist_sq.clamp(min=0) # [B, N, N]
        
        # 2. Gaussian Kernel Density Estimation
        # Fixed scale t=0.5
        gaussian = torch.exp(-dist_sq / (2 * self.t ** 2))
        
        # 3. Count only real neighbors (adj_mask)
        density = (gaussian * adj_mask).sum(dim=2) # [B, N]
        
        # 4. Visual Area Proxy
        # Phi = -ln(density)
        # Higher density (crowded neighbors) -> Lower Phi
        # Lower density (far neighbors) -> Higher Phi
        phi = -torch.log(density + 1e-8)
        
        # Mask padding
        phi = phi * mask.float()
        
        return phi

    def forward(self, final_pos, d_target, weights, mask, batch_data):
        """
        final_pos: [B, N, 2] (Z_world)
        d_target: [B, N, N]
        weights: [B, N, N] (Stress weights)
        mask: [B, N] (Bool)
        batch_data: PyG batch object (contains .status)
        """
        mask_float = mask.float()
        mask_mat = mask_float.unsqueeze(2) * mask_float.unsqueeze(1)
        
        # --- L_stress (Physical Space) ---
        sq_norm = (final_pos ** 2).sum(dim=2, keepdim=True)
        dist_sq = sq_norm + sq_norm.transpose(1, 2) - 2 * (final_pos @ final_pos.transpose(1, 2))
        dist_curr = torch.sqrt(dist_sq.clamp(min=1e-8))
        
        diff = dist_curr - d_target
        stress_sq = weights * (diff ** 2)
        loss_stress = (stress_sq * mask_mat).sum() / (mask_mat.sum() + 1e-8)
        
        # --- L_rank (Viewport Space) ---
        
        # 0. Prepare Status
        status_dense, _ = to_dense_batch(batch_data.status, batch_data.batch, max_num_nodes=mask.shape[1])
        status = status_dense.squeeze(-1) * mask_float # [B, N]
        
        # 1. Project to Viewport (Normalization)
        z_view = self.viewport_projection(final_pos, mask)
        
        # 2. Compute Visual Proxy (Phi)
        # Neighbor check: weights > 0 (for Rome dataset d_sp=1 implies weights=1)
        adj_mask = (weights > 0.001).float() * mask_mat
        phi = self.compute_visual_proxy(z_view, adj_mask, mask)
        
        # 3. All-Pairs Broadcasting for Loss
        phi_i = phi.unsqueeze(2)     # [B, N, 1]
        phi_j = phi.unsqueeze(1)     # [B, 1, N]
        
        status_i = status.unsqueeze(2)
        status_j = status.unsqueeze(1)
        
        # Filtering Condition: Status_i > Status_j
        # We only penalize if High Status (i) vs Low Status (j) relationship is violated
        pair_mask = (status_i > status_j).float() * mask_mat
        
        # Dynamic Weighting: (S_i - S_j)^alpha
        w_ij = (status_i - status_j).pow(self.alpha) * pair_mask
        
        # Log-Ratio Difference: Delta = Phi_i - Phi_j
        # Goal: Phi_i > Phi_j + m
        delta_phi = phi_i - phi_j
        
        # Hinge Loss: ReLU( m - (Phi_i - Phi_j) )
        loss_matrix = w_ij * F.relu(self.margin - delta_phi)
        
        # Normalize Loss
        num_pairs = pair_mask.sum().clamp(min=1.0)
        loss_rank = loss_matrix.sum() / num_pairs
        
        # --- Total Loss ---
        # Note: Scale Loss is removed, handled by Hard Constraint in Train Loop
        total_loss = self.lambda_stress * loss_stress + self.lambda_rank * loss_rank
        
        return total_loss, loss_stress, loss_rank