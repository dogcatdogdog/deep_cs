import torch
import torch.nn as nn
import torch.linalg

class StressMajorizationSolver(nn.Module):
    def __init__(self, iterations=5, epsilon=1e-6):
        super(StressMajorizationSolver, self).__init__()
        self.iterations = iterations
        self.epsilon = epsilon

    def forward(self, init_pos, weights, d_target, mask=None):
        """
        Robust Stress Majorization Solver (English Version)
        Args:
            init_pos: [B, N, 2] Initial positions
            weights:  [B, N, N] Edge weights
            d_target: [B, N, N] Target distances (Must be clamped)
            mask:     [B, N]    Node mask
        """
        # 1. Force Double Precision
        # Matrix inversion is very sensitive to precision. Float32 often leads to explosion.
        init_pos = init_pos.double()
        weights = weights.double()
        d_target = d_target.double()

        # 2. Apply Mask to Weights
        if mask is not None:
            mask_float = mask.unsqueeze(-1).double() # [B, N, 1]
            # Weights are valid only if both nodes are valid
            w_mask = mask_float @ mask_float.transpose(1, 2) # [B, N, N]
            weights = weights * w_mask

        # 3. Construct Laplacian Matrix L_w
        # L_w = D - A
        degree = weights.sum(dim=2) # [B, N]
        L_w = torch.diag_embed(degree) - weights # [B, N, N]
        
        # 4. Regularization and Inversion (Anti-Explosion)
        B, N, _ = L_w.shape
        eye = torch.eye(N, device=L_w.device, dtype=torch.double).unsqueeze(0)
        
        # [Robust Fix] Increase regularization factor
        # 1e-4 -> 1e-2. This significantly improves condition number stability.
        L_reg = L_w + 1e-2 * eye 
        
        try:
            # Try standard inversion first
            L_inv = torch.linalg.inv(L_reg)
        except RuntimeError:
            # Fallback to pseudo-inverse if singular (rare)
            L_inv = torch.linalg.pinv(L_reg)

        X = init_pos

        # 5. Iterative Optimization
        for i in range(self.iterations):
            X = self._step(X, d_target, weights, L_inv, mask)
            
        return X.float()

    def _step(self, X, d_target, w, L_inv, mask):
        # Calculate current Euclidean distances
        sq_norm = (X ** 2).sum(dim=2, keepdim=True)
        dist_sq = sq_norm + sq_norm.transpose(1, 2) - 2 * (X @ X.transpose(1, 2))
        
        # [Safety] Prevent sqrt(0) -> NaN gradients
        dist_curr = torch.sqrt(dist_sq.clamp(min=1e-8))
        
        # Calculate reciprocal distances
        # [Safety] Prevent division by zero
        inv_dist = torch.reciprocal(dist_curr.clamp(min=1e-5))
        
        # Apply mask
        if mask is not None:
            mask_float = mask.unsqueeze(-1).double()
            w_mask = mask_float @ mask_float.transpose(1, 2)
            inv_dist = inv_dist * w_mask
        
        # Stress Majorization Core Construction
        # scaling_matrix[i,j] = w[i,j] * d_target[i,j] / dist_curr[i,j]
        scaling_matrix = w * d_target * inv_dist
        
        # Diagonals must be zero
        scaling_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
        
        # Construct L_X
        degree_scaling = scaling_matrix.sum(dim=2)
        L_X = torch.diag_embed(degree_scaling) - scaling_matrix
        
        # Update positions: X_new = L_inv * L_X * X
        Y = L_X @ X
        X_new = L_inv @ Y
        
        return X_new