# utils.py

import os
import random
import yaml
import torch
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
        self.val_loss_min = np.Inf
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