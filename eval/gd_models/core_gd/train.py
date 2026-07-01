import torch
import numpy as np
from tqdm import tqdm

# We only keep the imports actually needed for loading the model and doing inference
from .model import get_model

import scipy
if not hasattr(scipy, 'newaxis'):
    scipy.newaxis = np.newaxis

try:
    import pygsp as gsp
    from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
    from torch_geometric.data import Data
    
    # 导入粗化核心算法（既然你已经装了 graph_coarsening）
    from graph_coarsening.coarsening_utils import coarsen
    
    # 导入 Core-GD 自己的预处理库
    from . import preprocessing
    from .transforms import convert_for_stress
except ImportError as e:
    print(f"[Warning] Failed to import one or more coarsening dependencies: {e}")

# -----------------------------------------------------------------------------
# Inference Helpers (Retained & Cleaned for layout.py usage)
# -----------------------------------------------------------------------------
def create_coarsened_dataset(config, dataset):
    dataset_gsp = [gsp.graphs.Graph(to_scipy_sparse_matrix(G.edge_index)) for G in dataset]

    method = config.coarsen_algo
    r    = config.coarsen_r
    k    = config.coarsen_k 
        
    coarsened_pyg = []
    coarsening_matrices = []
    for i in tqdm(range(len(dataset))):
        pyg_graphs = [convert_for_stress(dataset[i])]
        matrices = []
        while pyg_graphs[-1].x.shape[0] > config.coarsen_min_size:
            C, Gc, Call, Gall = coarsen(gsp.graphs.Graph(to_scipy_sparse_matrix(pyg_graphs[-1].edge_index)), K=k, r=r, method=method, max_levels=1)
            
            if len(Gall) < 2 or len(Call) == 0:
                print(f"[Debug] Coarsening stopped early at {pyg_graphs[-1].x.shape[0]} nodes due to rigid graph topology.")
                break
            pyg_graphs.append(convert_for_stress(Data(edge_index=from_scipy_sparse_matrix(Gall[1].W)[0])))
            matrices.append(Call[0])
        coarsened_pyg.append(pyg_graphs)

        for i in range(len(matrices)):
            matrices[i][matrices[i] > 0] = 1.0
            matrices[i] = matrices[i].tocoo()
            matrices[i] = torch.sparse.LongTensor(torch.LongTensor([matrices[i].row.tolist(), matrices[i].col.tolist()]),
                            torch.FloatTensor(matrices[i].data))
        coarsening_matrices.append(matrices)
    
    preprocessed_dataset = preprocessing.preprocess_dataset([coarsened[-1] for coarsened in coarsened_pyg], config)
    for i in range(len(preprocessed_dataset)):
        preprocessed_dataset[i].index = i
        preprocessed_dataset[i].coarsening_level = 0
        
    return preprocessed_dataset, coarsened_pyg, coarsening_matrices

def propagate_max_degrees(current_degrees, P_matrix):
    """
    Max Pooling for node degrees during coarsening
    """
    import torch_scatter
    
    indices = P_matrix._indices()
    coarse_idx = indices[0]
    fine_idx = indices[1]
    
    fine_degrees = current_degrees[fine_idx]
    
    num_coarse = P_matrix.size(0)
    
    max_degrees, _ = torch_scatter.scatter_max(
        fine_degrees, 
        coarse_idx, 
        dim=0, 
        dim_size=num_coarse
    )
    
    max_degrees[max_degrees == 0] = 1.0 
    
    return max_degrees

def go_to_coarser_graph(graph, last_embeddings, device, batch, coarsened_graphs, coarsening_matrices, noise):
    """
    Unpools embeddings to the finer graph during the inference pass
    """
    new_level = graph.coarsening_level+1
    
    # Sparse multiplication to unpool embeddings
    embeddings_finer = torch.transpose(
        torch.sparse.mm(
            torch.transpose(last_embeddings, 0, 1), 
            coarsening_matrices[graph.index][-new_level].to(device)
        ), 
        0, 1
    )
    
    graph.edge_index = coarsened_graphs[graph.index][-new_level-1].edge_index.to(device)
    
    if hasattr(coarsened_graphs[graph.index][-new_level-1], 'full_edge_index'):
        graph.full_edge_index = coarsened_graphs[graph.index][-new_level-1].full_edge_index.to(device)
    if hasattr(coarsened_graphs[graph.index][-new_level-1], 'full_edge_attr'):
        graph.full_edge_attr = coarsened_graphs[graph.index][-new_level-1].full_edge_attr.to(device)
        
    graph.x = embeddings_finer
    
    if batch:
        graph.batch = torch.zeros(embeddings_finer.shape[0], device=device, dtype=torch.int64)
        
    # add noise to embeddings
    if noise > 0:
        mean = 0
        std = noise
        noise_tensor = torch.tensor(np.random.normal(mean, std, graph.x.size()), dtype=torch.float, device=device)
        graph.x = graph.x + noise_tensor
        
    graph.coarsening_level = new_level
    return graph

# -----------------------------------------------------------------------------
# Forward Pass Wrapper (Simulating the test function behavior for inference)
# -----------------------------------------------------------------------------
@torch.no_grad()
def test(model, device, loader, loss_fun, layer_num, coarsened_graphs=None, coarsening_matrices=None, coarsen=False, noise=0.01):
    """
    Modified to just perform the forward pass. 
    (Returns the coordinates instead of the loss for LayoutGenerator)
    """
    model.eval()
    
    # Just take the first batch (since we only pass one graph at a time)
    for batch in loader:
        batch = batch.to(device)
        
        # Initial forward pass
        pred, states = model(batch, layer_num, return_layers=True)
        
        # If the model uses a U-Net style coarsening architecture
        if coarsen and coarsening_matrices is not None:
            for i in range(1, len(coarsening_matrices[batch.index])+1):
                batch = go_to_coarser_graph(batch, states[-1], device, True, coarsened_graphs, coarsening_matrices, noise=noise)
                # Subsequent forward passes on finer graphs
                pred, states = model(batch, layer_num, encode=False, return_layers=True)
                
        # Return the final node coordinates
        return pred