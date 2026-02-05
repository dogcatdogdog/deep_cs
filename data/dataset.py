# -*- coding: utf-8 -*-
import os
import os.path as osp
import shutil
import tarfile
import re
import sys
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.utils import from_networkx
from tqdm import tqdm

# Path compatibility
current_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import compute_topological_status

class DeepCSDataset(Dataset):
    url = 'http://www.graphdrawing.org/download/rome-graphml.tgz'
    filename = 'rome-graphml.tgz'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        if osp.exists(self.processed_paths[0]):
            try:
                self.data_list = torch.load(self.processed_paths[0], weights_only=False)
            except TypeError:
                self.data_list = torch.load(self.processed_paths[0])
        else:
            self.process() 

    @property
    def raw_file_names(self):
        return ['rome']

    @property
    def processed_file_names(self):
        return ['deepcs_rome_data_v2.pt']

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def download(self):
        tgz_path = osp.join(self.raw_dir, self.filename)
        if not osp.exists(tgz_path):
            download_url(self.url, self.raw_dir)
        with tarfile.open(tgz_path, 'r:gz') as archive:
            archive.extractall(self.raw_dir)

    def process(self):
        print("[INFO] Starting V2.0 Preprocessing (Topological Status)...")
        rome_path = osp.join(self.raw_dir, 'rome')
        
        if not osp.exists(rome_path):
             raise FileNotFoundError(f"Rome directory not found at {rome_path}.")

        files = [f for f in os.listdir(rome_path) if f.endswith('.graphml')]
        files = sorted(files, key=lambda x: int(re.search(r'grafo(\d+)', x).group(1)))
        
        data_list = []
        skip_count = 0

        for filename in tqdm(files, desc="Processing Graphs"):
            try:
                G = nx.read_graphml(osp.join(rome_path, filename))
            except:
                continue

            G = G.to_undirected()
            G.remove_edges_from(nx.selfloop_edges(G))
            if not nx.is_connected(G):
                skip_count += 1
                continue
            
            G = nx.convert_node_labels_to_integers(G)
            N = G.number_of_nodes()
            
            if N < 10 or N > 500: 
                skip_count += 1
                continue

            # 1. Ground Truth & Weights
            try:
                d_sp_np = nx.floyd_warshall_numpy(G)
            except:
                skip_count += 1
                continue
                
            d_sp = torch.from_numpy(d_sp_np).float()
            weights = 1.0 / (d_sp ** 2)
            weights[torch.isinf(weights)] = 0 
            weights.fill_diagonal_(0)

            # 2. Node Features
            # Degree
            degrees = np.array([d for _, d in G.degree()])
            log_deg = np.log(degrees + 1.0)
            feat_deg = torch.from_numpy((log_deg - log_deg.min()) / (log_deg.max() - log_deg.min() + 1e-6)).float().unsqueeze(1)
            # Noise
            feat_noise = torch.randn(N, 2)
            # PE (Laplacian)
            L_norm = nx.normalized_laplacian_matrix(G).todense().astype(np.float32)
            try:
                _, eig_vecs = np.linalg.eigh(L_norm)
                pe = eig_vecs[:, 1:9] if N > 8 else np.concatenate([eig_vecs[:, 1:], np.zeros((N, 8-(N-1)))], axis=1)
                feat_pe = torch.from_numpy(pe).float()
            except:
                skip_count += 1
                continue

            # 3. [V2.0 Core] Topological Status
            pyg_edge_index = from_networkx(G).edge_index
            status = compute_topological_status(pyg_edge_index, N) # [N, 1]

            # Concat Features: Deg(1) + Noise(2) + PE(8) + Status(1) = 12
            x = torch.cat([feat_deg, feat_noise, feat_pe, status], dim=1)
            
            data = Data(
                x=x,
                edge_index=pyg_edge_index,
                d_sp=d_sp,
                weights=weights,
                num_nodes=N,
                status=status # Save status explicitly
            )
            data_list.append(data)

        print(f"[INFO] Processed {len(data_list)} graphs. Skipped {skip_count}.")
        torch.save(data_list, self.processed_paths[0])