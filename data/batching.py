import torch
from torch_geometric.data import Batch

def custom_collate_fn(data_list):
    # 1. Standard PyG Batch for sparse graphs
    # EXCLUDE 'd_sp' and 'weights' so PyG doesn't crash trying to cat matrices of different sizes.
    batch = Batch.from_data_list(data_list, exclude_keys=['d_sp', 'weights'])
    
    # 2. Dense Batch for the Solver
    num_nodes_list = [d.num_nodes for d in data_list]
    max_num_nodes = max(num_nodes_list)
    B = len(data_list)
    
    # Initialize dense tensors [B, Max_N, Max_N]
    d_sp_dense = torch.zeros((B, max_num_nodes, max_num_nodes), dtype=torch.float)
    weights_dense = torch.zeros((B, max_num_nodes, max_num_nodes), dtype=torch.float)
    # mask: [B, Max_N], True for real nodes, False for padding
    mask = torch.zeros((B, max_num_nodes), dtype=torch.bool)
    
    for i, data in enumerate(data_list):
        N = data.num_nodes
        # Fill in the real data
        d_sp_dense[i, :N, :N] = data.d_sp
        weights_dense[i, :N, :N] = data.weights
        mask[i, :N] = True
        
    return batch, d_sp_dense, weights_dense, mask