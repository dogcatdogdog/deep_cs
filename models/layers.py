import torch
import torch.nn as nn
import numpy as np

class DifferentiableAPSP(nn.Module):
    """
    Differentiable All-Pairs Shortest Path (APSP) layer using the Doubling Algorithm.
    Approximates the Min-Plus matrix multiplication in log-space to compute global 
    graph distances from local edge weights (Beta).
    """
    def __init__(self, temperature=0.1):
        super(DifferentiableAPSP, self).__init__()
        # temperature (tau): 
        # Low (0.1) provides sharp, accurate distances but sparser gradients.
        # High (1.0) provides smoother gradients but can lead to distance 'drift'.
        self.temperature = temperature 

    def forward(self, beta_1hop, adj_mask):
        """
        Args:
            beta_1hop: [B, N, N] Predicted weights for direct edges.
            adj_mask:  [B, N, N] Boolean mask where True indicates a physical edge.
        Returns:
            dist_final: [B, N, N] Reconstructed global distance matrix.
        """
        B, N, _ = beta_1hop.shape
        device = beta_1hop.device
        
        # Use a moderate INF value to prevent float32 overflow during exp() operations.
        # 100.0 is sufficient for graphs with diameter up to ~50.
        INF_VAL = 100.0 
        
        # 1. Initialize distance matrix
        dist = torch.full_like(beta_1hop, INF_VAL)
        dist = torch.where(adj_mask.bool(), beta_1hop, dist)
        
        # Identity distance: Force diagonals to 0.0
        eye_mask = torch.eye(N, device=device).bool().unsqueeze(0).expand(B, -1, -1)
        dist = dist.masked_fill(eye_mask, 0.0)
        
        # 2. Map to Log-Space (Soft-Min mapping: x -> -x / T)
        # This converts the Min-Plus algebra into Log-Sum-Exp algebra.
        dist_neg = -dist / self.temperature
        
        # Number of doubling iterations required: ceil(log2(N))
        max_steps = int(torch.ceil(torch.log2(torch.tensor(N))))
        
        for _ in range(max_steps):
            # Construct path combinations: D_ij = min_k (D_ik + D_kj)
            # Use broadcasting to create an [i, k, j] tensor volume
            # d1 (i, k, 1): Distance from start node to intermediate node
            # d2 (1, k, j): Distance from intermediate node to end node
            d1 = dist_neg.unsqueeze(3) # [B, N, N, 1]
            d2 = dist_neg.unsqueeze(1) # [B, 1, N, N]
            
            # combined[b, i, k, j] = dist[b, i, k] + dist[b, k, j]
            combined = d1 + d2 
            
            # LogSumExp over the intermediate node dimension 'k' (dim=2)
            # This approximates the minimum path across all possible middle nodes.
            dist_new_neg = torch.logsumexp(combined, dim=2) 
            
            # Residual update: dist = min(old_dist, new_dist)
            # Ensures paths only get shorter over iterations.
            dist_neg = torch.logsumexp(torch.stack([dist_neg, dist_new_neg], dim=-1), dim=-1)
            
            # Reset diagonals in every step to prevent negative drift accumulation.
            dist_neg = dist_neg.masked_fill(eye_mask, 0.0)
            
        # 3. Map back to Linear Space
        dist_final = -dist_neg * self.temperature
        
        # 4. Safety Guardrails
        # min=1e-4: Avoid division by zero in Stress Solver.
        # max=50.0: Cap disconnected components to prevent gradient explosion.
        return dist_final.clamp(min=1e-4, max=50.0)