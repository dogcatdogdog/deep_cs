# utils.py

import os
import random
import yaml
import torch
import math
import numpy as np
import scipy.linalg

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f)

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        if np.isnan(val_loss) or np.isinf(val_loss):
            return  # 忽略异常 Loss，防止破坏最佳权重
        
        # score = -val_loss。Loss 越小，score 越大。
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        # 核心逻辑：判断是否有实质性改善
        # 如果新 score (s1) <= 旧 score (s0) + delta，说明改善不足 delta
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'[EarlyStopping] Counter: {self.counter}/{self.patience} (Best: {self.val_loss_min:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 发生了实质性改善 (> delta)
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'[EarlyStopping] Significant improvement detected ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving...')
        folder = os.path.dirname(self.path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def compute_topological_status(edge_index, num_nodes):
    """
    Step 1 Preprocessing: Calculate Static Topological Status (S)
    S = Norm(Degree) * (1.0 - Norm(HKS))
    """
    # 1. Adjacency & Degree
    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[rows, cols] = 1.0
    
    degrees = np.sum(adj, axis=1) # [N]
    
    # 2. Heat Kernel Signature (HKS)
    # L = D - A (Combinatorial Laplacian)
    L = np.diag(degrees) - adj
    
    try:
        # Eigen-decomposition
        evals, evecs = scipy.linalg.eigh(L)
        # Heat Kernel at t=1.0: H = U * exp(-evals) * U.T
        exp_evals = np.exp(-1.0 * evals)
        # We only need the diagonal (HKS)
        # H_ii = sum(U_ik^2 * exp(-lambda_k))
        hks = np.sum((evecs ** 2) * exp_evals, axis=1)
    except:
        # Fallback if decomposition fails
        hks = np.zeros(num_nodes)

    # 3. Robust Normalization [0.1, 1.0]
    # Prevents status from being 0, giving every node a chance.
    def robust_norm(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-6:
            return np.ones_like(x) * 0.1 # Flat distribution
        # Scale to [0, 0.9] then add 0.1
        return ((x - xmin) / (xmax - xmin)) * 0.9 + 0.1

    norm_deg = robust_norm(degrees)
    norm_hks = robust_norm(hks)
    
    # 4. Synthesize Status
    # High Degree + Low Heat Retention (Hubs) = High Status
    status = norm_deg * (1.0 - norm_hks + 0.1) # +0.1 to avoid strict 0
    
    # Final check to ensure range [0.1, 1.0] approx
    status = robust_norm(status)
    
    return torch.from_numpy(status).float().unsqueeze(-1) # [N, 1]
    

class StarVectorGenerator:
    """
    Implements the Rotation-Aligned Star Constraint generator.
    Strictly follows Zero-Bug Policy:
    1. Detaches Rotation Matrices (no gradients flow through SVD).
    2. Handles variable degrees via Bucketing.
    3. Prevents Reflection (enforces proper Rotation).
    """
    def __init__(self, device):
        self.device = device
        self.template_cache = {}

    def _get_star_template(self, k):
        """Generates a normalized k-pointed star template centered at (0,0)."""
        if k in self.template_cache:
            return self.template_cache[k]
        
        angles = torch.linspace(0, 2 * math.pi, k + 1, device=self.device)[:-1]
        # [k, 2]
        x = torch.cos(angles)
        y = torch.sin(angles)
        template = torch.stack([x, y], dim=1).double() # Force Double for precision
        self.template_cache[k] = template
        return template

    def generate_targets(self, current_pos, adj_matrix, degree, beta_map, d_gt, hub_mask):
        """
        Args:
            current_pos: [B, N, 2] (Double)
            adj_matrix:  [B, N, N] (Binary or float, used for connectivity)
            degree:      [B, N] Node degrees
            beta_map:    [B, N, N] Predicted scaling factors
            d_gt:        [B, N, N] Ground Truth distances
            hub_mask:    [B, N] Boolean mask identifying Hubs
            
        Returns:
            target_vectors: [B, N, N, 2] Dense tensor of relative target vectors
            vector_mask:    [B, N, N] Boolean mask (1 where target is valid)
        """
        B, N, _ = current_pos.shape
        target_vectors = torch.zeros((B, N, N, 2), dtype=torch.float64, device=self.device)
        vector_mask = torch.zeros((B, N, N), dtype=torch.bool, device=self.device)
        
        # We process each batch item separately for safety, or we can try advanced batching.
        # Given the complexity of graph structures, iterating B is acceptable, 
        # but inside B we MUST vectorize by degree.
        
        for b in range(B):
            b_pos = current_pos[b]     # [N, 2]
            b_deg = degree[b]          # [N]
            b_hub = hub_mask[b]        # [N]
            b_adj = adj_matrix[b]      # [N, N]
            
            # Identify unique degrees among active hubs to bucketize
            # Only consider hubs that have neighbors (degree > 1)
            active_hubs = torch.nonzero(b_hub & (b_deg > 1)).squeeze(-1)
            if len(active_hubs) == 0:
                continue
                
            unique_degrees = torch.unique(b_deg[active_hubs])
            
            for k_tensor in unique_degrees:
                k = int(k_tensor.item())
                if k < 2: continue # Cannot form a shape with 1 neighbor
                
                # 1. Gather Hubs of degree k
                # mask for hubs with degree k
                k_hub_indices = active_hubs[b_deg[active_hubs] == k] 
                num_k_hubs = len(k_hub_indices)
                
                # 2. Gather Neighbors
                # We need to extract neighbors for these hubs.
                # Since k is constant here, we can stack them.
                # This requires finding the indices of neighbors.
                # b_adj[u] gives the row.
                
                # [num_k_hubs, k] indices of neighbors
                # Note: This relies on b_adj being binary/valid. 
                # We use topk to get indices if it's strictly k neighbors.
                _, neighbor_indices = torch.topk(b_adj[k_hub_indices], k, dim=1)
                
                # 3. Construct Relative Vectors
                # Hub pos: [num_k_hubs, 1, 2]
                centers = b_pos[k_hub_indices].unsqueeze(1)
                # Neighbor pos: [num_k_hubs, k, 2]
                neighbors = b_pos[neighbor_indices]
                
                # V_rel: Vector FROM Hub TO Neighbor
                V_rel = neighbors - centers # [M, k, 2]
                
                # 4. Procrustes Alignment (The Core)
                template = self._get_star_template(k) # [k, 2]
                S = template.unsqueeze(0).expand(num_k_hubs, -1, -1) # [M, k, 2]
                
                # Compute Covariance: H = V_rel^T @ S
                # transpose V_rel to [M, 2, k]
                H = torch.bmm(V_rel.transpose(1, 2), S) # [M, 2, 2]
                
                # SVD
                U, _, V_T = torch.linalg.svd(H)
                
                # R = V @ U.T
                R = torch.bmm(V_T.transpose(1, 2), U.transpose(1, 2)) # [M, 2, 2]
                
                # Determinant Check (Fix Reflections)
                det = torch.det(R) # [M]
                # If det < 0, we have a reflection. We need to flip.
                # Standard fix: S = diag(1, ..., 1, det)
                # Updated R = V @ S @ U.T
                # Efficiently: If det < 0, flip the last column of V_T (before transpose) or U.
                
                # Create correction matrix
                corr = torch.eye(2, device=self.device, dtype=torch.double).unsqueeze(0).repeat(num_k_hubs, 1, 1)
                corr[:, 1, 1] = det # will be 1 or -1 roughly
                
                # Recompute R with correction
                # Strict rotation matrix
                R = torch.bmm(torch.bmm(V_T.transpose(1, 2), corr), U.transpose(1, 2))
                
                # Stop Gradient! The target shape shouldn't pull the rotation itself.
                R = R.detach()
                
                # 5. Generate Aligned Unit Vectors
                # U_star = S @ R_transposed?
                # Formula: We want V_rel \approx S @ R. 
                # So target direction is S @ R.
                U_star = torch.bmm(S, R) # [M, k, 2]
                
                # 6. Scale by Predictions (Beta * D_gt)
                # Get Beta for these edges: [M, k]
                # gather from beta_map[b] using (hub, neighbor) indices
                # indices: k_hub_indices (rows), neighbor_indices (cols)
                row_idx = k_hub_indices.unsqueeze(1).expand(-1, k)
                
                curr_beta = beta_map[b].gather(0, row_idx).gather(1, neighbor_indices)
                curr_dgt = d_gt[b].gather(0, row_idx).gather(1, neighbor_indices)
                
                # Target Magnitude
                target_mag = curr_beta * curr_dgt # [M, k]
                
                # Final Vector: [M, k, 2]
                final_vecs = U_star * target_mag.unsqueeze(-1)
                
                # 7. Scatter back to Global Tensor
                # We need to fill target_vectors[b, u, v]
                # This requires flat indices or careful scattering.
                
                # Flat indices for scatter
                flat_hubs = row_idx.flatten()
                flat_neighbors = neighbor_indices.flatten()
                flat_vecs = final_vecs.reshape(-1, 2)
                
                # Assign
                target_vectors[b].index_put_((flat_hubs, flat_neighbors), flat_vecs)
                vector_mask[b].index_put_((flat_hubs, flat_neighbors), torch.tensor(True, device=self.device))
                
                # Note: This sets the target for edge u->v. 
                # StressMajorization solver expects 'target_vectors' to represent (x_i - x_j).
                # Here we calculated (neighbor - hub).
                # If solver loop is over i, and j is neighbor. (x_i - x_j) is (Hub - Neighbor).
                # Our V_rel was (Neighbor - Hub).
                # So we must negate the vectors to match the (x_i - x_j) convention in Solver?
                # Let's check Solver logic carefully. 
                # Standard force is (x_i - x_j).
                # If we want x_j to be at x_i + v, then x_i - x_j = -v.
                # Yes. The vectors we calculated are (Neighbor - Hub). 
                # If i=Hub, j=Neighbor, we want x_i - x_j = -(Neighbor - Hub).
                # So we negate.
                target_vectors[b].index_put_((flat_hubs, flat_neighbors), -flat_vecs)

        return target_vectors, vector_mask