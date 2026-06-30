# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import networkx as nx
import scipy.linalg
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, to_networkx, to_undirected, remove_self_loops
from torch_scatter import scatter_add

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fetch_suitesparse import fetch_and_parse_suitesparse


def compute_topological_status(edge_index, num_nodes):

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
        # Eigen-decomposition (针对实对称拉普拉斯矩阵的高效分解)
        evals, evecs = scipy.linalg.eigh(L)
        # Heat Kernel at t=1.0: H = U * exp(-evals) * U.T
        exp_evals = np.exp(-1.0 * evals)
        # We only need the diagonal (HKS)
        # H_ii = sum(U_ik^2 * exp(-lambda_k))
        hks = np.sum((evecs ** 2) * exp_evals, axis=1)
    except Exception:
        # Fallback if decomposition fails
        hks = np.zeros(num_nodes)

    # 3. Robust Normalization [0.1, 1.0]
    def robust_norm(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-6:
            return np.ones_like(x) * 0.1 
        return ((x - xmin) / (xmax - xmin)) * 0.9 + 0.1

    norm_deg = robust_norm(degrees)
    norm_hks = robust_norm(hks)
    
    # 4. Synthesize Status
    status = norm_deg * (1.0 - norm_hks + 0.1) 
    status = robust_norm(status)
    
    return torch.from_numpy(status).float().unsqueeze(-1) # [N, 1]


def extract_global_hierarchy(G, N):
    pr = nx.pagerank(G, alpha=0.85)
    pr_tensor = torch.tensor([pr[i] for i in range(N)], dtype=torch.float32).view(N, 1)
    pr_norm = pr_tensor / (pr_tensor.max() + 1e-8)
    core_dict = nx.core_number(G)
    core_tensor = torch.tensor([core_dict[i] for i in range(N)], dtype=torch.float32).view(N, 1)
    core_norm = core_tensor / (core_tensor.max() + 1e-8)
    return torch.cat([pr_norm, core_norm], dim=1)


def compute_probabilistic_targets(G, N, pyg_edge_index):
    degrees = np.array([G.degree(i) for i in range(N)], dtype=float)
    degrees[degrees == 0] = 1.0 
    
    row, col = pyg_edge_index.numpy()
    p_cond_ij = 1.0 / degrees[row]
    p_cond_ji = 1.0 / degrees[col]
    
    p_joint = (p_cond_ij + p_cond_ji) / (2.0 * N)
    return torch.from_numpy(p_joint).float()


def compute_pos_init(G, N):
    L_norm = nx.normalized_laplacian_matrix(G).toarray().astype(np.float32)
    try:
        _, eig_vecs = np.linalg.eigh(L_norm)
        pos_init = torch.from_numpy(eig_vecs[:, 1:3]).float()
        pos_init = pos_init + torch.randn(N, 2) * 0.01 
    except Exception:
        pos_init = torch.randn(N, 2) * 0.1

    return pos_init


def extract_structural_fingerprints(G, N, pyg_edge_index):
    row, col = pyg_edge_index[0], pyg_edge_index[1]
    
    deg_np = np.array([G.degree(i) for i in range(N)], dtype=float)
    deg = torch.from_numpy(deg_np).float()
    
    f1 = torch.log(deg + 1.0)
    
    # 建立在严格双向边表基础上的邻域聚合，保证结构特征空间极度稳定
    arith_mass = scatter_add(deg[col], row, dim=0, dim_size=N)
    f2 = torch.log(arith_mass + 1.0)

    inv_deg = 1.0 / (deg + 1e-6)
    sum_inv_deg = scatter_add(inv_deg[col], row, dim=0, dim_size=N)
    harmonic_mass = deg / (sum_inv_deg + 1e-6)
    f3 = torch.log(harmonic_mass + 1.0)

    fingerprints = torch.stack([f1, f2, f3], dim=1)
    return fingerprints


def compute_rwpe(pyg_edge_index, N, walk_length=4):
    adj = torch.zeros((N, N), dtype=torch.float32)
    adj[pyg_edge_index[0], pyg_edge_index[1]] = 1.0
    
    deg = adj.sum(dim=-1, keepdim=True)
    deg[deg == 0] = 1.0
    P = adj / deg
    
    pe_list = []
    P_curr = P.clone()
    for _ in range(walk_length):
        pe_list.append(torch.diag(P_curr))
        P_curr = torch.matmul(P_curr, P)
        
    rwpe = torch.stack(pe_list, dim=1)
    return rwpe


def extract_features(G, N, pyg_edge_index):
    fingerprints = extract_structural_fingerprints(G, N, pyg_edge_index)
    
    deg_np = np.array([G.degree(i) for i in range(N)], dtype=float)
    rel_deg = torch.from_numpy(deg_np / float(N)).float().view(N, 1)
    
    global_hier = extract_global_hierarchy(G, N)
    status = compute_topological_status(pyg_edge_index, N)
    rwpe = compute_rwpe(pyg_edge_index, N, walk_length=4)
        
    x = torch.cat([fingerprints, rel_deg, global_hier, rwpe, status], dim=1) 
    edge_attr = torch.ones((pyg_edge_index.shape[1], 1), dtype=torch.float32) 
    
    return x, edge_attr

# ===================================================================
# 特种拓扑发生器集群
# ===================================================================
def generate_fractal_snowflake(N):
    G = nx.Graph()
    G.add_node(0) 
    current_node = 1
    num_secondary_hubs = np.random.randint(3, max(5, N // 15))
    secondary_hubs = []
    
    for _ in range(num_secondary_hubs):
        G.add_edge(0, current_node)
        secondary_hubs.append(current_node)
        current_node += 1
        
    while current_node < N:
        if np.random.rand() < 0.2:
            G.add_edge(0, current_node) 
        else:
            target_hub = np.random.choice(secondary_hubs)
            G.add_edge(target_hub, current_node) 
        current_node += 1
    return G


def generate_extreme_barabasi(N):
    G = nx.barabasi_albert_graph(N, 1) 
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_hubs = [n for n, d in degrees[:3]]
    nodes = list(G.nodes())
    num_extra_edges = int(N * 0.15)
    for _ in range(num_extra_edges):
        u = np.random.choice(top_hubs)
        v = np.random.choice(nodes)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    return G


def generate_heavy_core_umbrella(N):
    core_size = min(30, max(5, int(N * 0.15))) 
    G_core = nx.erdos_renyi_graph(core_size, p=0.4)
    if not nx.is_connected(G_core):
        largest_cc = max(nx.connected_components(G_core), key=len)
        G_core = G_core.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G_core)
    core_size = G.number_of_nodes()
    
    super_hub = core_size
    G.add_node(super_hub)
    num_anchors = np.random.randint(1, min(4, core_size))
    anchors = np.random.choice(range(core_size), num_anchors, replace=False)
    for anchor in anchors:
        G.add_edge(super_hub, anchor)
        
    current_node = super_hub + 1
    while current_node < N:
        G.add_node(current_node)
        G.add_edge(super_hub, current_node)
        current_node += 1
    return G


class ProbForceDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['rome']

    @property
    def processed_file_names(self):
        # 显式升级为 v10 核心缓存文件标识，熔断旧有单向边表引发的表示层冲突
        return ['dataset_decoupled_v10.pt']

    def process(self):
        final_data_list = []

        def run_pipeline(G, step_sims, topo_label):
            G = G.to_undirected()
            G.remove_edges_from(list(nx.selfloop_edges(G)))
            
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            G = nx.convert_node_labels_to_integers(G)
            N = G.number_of_nodes()
            if N < 10 or N > 500: return None
            
            raw_edge_index = from_networkx(G).edge_index
            
            # ==========================================================
            # 严格双向对称化洗净重构
            # ==========================================================
            clean_edge_index, _ = remove_self_loops(raw_edge_index)
            pyg_edge_index = to_undirected(clean_edge_index, num_nodes=N)
            
            # ==========================================================
            # 刚性防御性断言：在线性时间内校验当前边表是否具备严格双向闭包语义
            # ==========================================================
            assert pyg_edge_index.shape[1] == to_undirected(pyg_edge_index).shape[1], \
                f"Data Pipeline Leak: pyg_edge_index with {pyg_edge_index.shape[1]} edges fails bidirectional closure checking!"
            
            pos_init = compute_pos_init(G, N)
            P_target = compute_probabilistic_targets(G, N, pyg_edge_index)
            x, edge_attr = extract_features(G, N, pyg_edge_index)
            
            row, col = pyg_edge_index
            deg = torch.bincount(row, minlength=N)
            if len(deg) > 0:
                hub_idx = torch.argmax(deg).item()
                neighbors = col[row == hub_idx]
                
                if len(neighbors) >= 2:
                    core_feats = x[neighbors, :7]
                    core_norm = F.normalize(core_feats, p=2, dim=-1)
                    sim_matrix = torch.matmul(core_norm, core_norm.T)
                    mask = ~torch.eye(len(neighbors), dtype=torch.bool, device=x.device)
                    valid_sims = sim_matrix[mask]
                    step_sims.append(valid_sims)

            return Data(
                x=x, 
                edge_index=pyg_edge_index, 
                edge_attr=edge_attr, 
                P_target=P_target,        
                pos_init=pos_init, 
                num_nodes=N,
                y=torch.tensor([topo_label], dtype=torch.long)
            )

        def print_probe_report(step_name, sims_list):
            if sims_list:
                all_sims = torch.cat(sims_list)
                print(f"  --> [{step_name} Probe Report] Raw Feature Local Similarity (Hub Neighbors)")
                print(f"  --> Min: {all_sims.min():.4f} | Mean: {all_sims.mean():.4f} | Max: {all_sims.max():.4f}")
            else:
                print(f"  --> [{step_name} Probe Report] Not enough hubs/neighbors to aggregate.")

        # [STEP 1] Rome
        print("\n[STEP 1] Processing Rome Graphs...")
        rome_dir = os.path.join(self.raw_dir, 'rome')
        rome_sims = []
        if os.path.exists(rome_dir):
            files = sorted([f for f in os.listdir(rome_dir) if f.endswith('.graphml')])[:500]
            for f in tqdm(files):
                try:
                    res = run_pipeline(nx.read_graphml(os.path.join(rome_dir, f)), rome_sims, topo_label=0)
                    if res: final_data_list.append(res)
                except Exception: continue
        print_probe_report("Rome", rome_sims)

        # [STEP 2] SuiteSparse
        print("\n[STEP 2] Processing SuiteSparse...")
        ss_sims = []
        try:
            ss_data = fetch_and_parse_suitesparse(max_graphs=100)
            for d in tqdm(ss_data):
                try:
                    res = run_pipeline(to_networkx(d, to_undirected=True), ss_sims, topo_label=1)
                    if res: final_data_list.append(res)
                except Exception: continue
        except Exception: pass
        print_probe_report("SuiteSparse", ss_sims)

        # [STEP 3] Extreme Topology Graphs (The Crucible)
        print("\n[STEP 3] Generating 500 Extreme Topology Graphs (The Crucible)...")
        syn_sims = []
        for i in tqdm(range(500)):
            rand_scale = np.random.rand()
            if rand_scale < 0.3: n = np.random.randint(20, 80)     
            elif rand_scale < 0.7: n = np.random.randint(80, 250)    
            else: n = np.random.randint(250, 450)   
                
            rand_topo = np.random.rand()
            try:
                if rand_topo < 0.3:
                    G = generate_fractal_snowflake(n)
                    current_label = 2
                elif rand_topo < 0.6:
                    G = generate_extreme_barabasi(n)
                    current_label = 3
                elif rand_topo < 0.8:
                    G = generate_heavy_core_umbrella(n)
                    current_label = 5
                else:
                    G = nx.planted_partition_graph(n // 5, 5, 0.25, 0.02)
                    current_label = 4
                    
                res = run_pipeline(G, syn_sims, topo_label=current_label)
                if res: final_data_list.append(res)
            except Exception:
                import traceback
                traceback.print_exc()
                sys.exit(1)
                
        print_probe_report("Synthetic", syn_sims)

        print(f"\n[DONE] Saved {len(final_data_list)} enriched symmetric graphs.")
        data, slices = self.collate(final_data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = ProbForceDataset(root='./data')