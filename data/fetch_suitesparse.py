# -*- coding: utf-8 -*-
import os
import glob
import torch
import scipy.io as sio
import ssgetpy
from torch_geometric.data import Data

def fetch_and_parse_suitesparse(download_dir='./raw/suitesparse', max_graphs=100):
    os.makedirs(download_dir, exist_ok=True)
    print("=" * 60)
    print("[INFO] Connecting to SuiteSparse Matrix Collection...")
    
    try:
        # 1. 基础拉取：获取所有节点数在 50 到 1500 之间的矩阵
        results = ssgetpy.search(rowbounds=(50, 1500), limit=2000)
    except Exception as e:
        print(f"[ERROR] API Connection failed: {e}")
        return []

    print(f"[INFO] Initial API fetch returned {len(results)} matrices.")
    if len(results) == 0:
        return []

    # 2. 物理意义过滤 (基于 kind 字符串)
    filtered_matrices = []
    
    for m in results:
        # 安全获取 kind 属性并转为小写，防止报错
        kind_str = getattr(m, 'kind', '').lower()
        
        # 核心过滤逻辑：
        # 1. 必须是图 ('graph' 包含 undirected graph, directed graph 等)
        # 2. 绝对不能是物理/几何网格 ('2d', '3d' 通常代表流体力学或结构力学网格)
        if 'graph' in kind_str and '2d' not in kind_str and '3d' not in kind_str:
            filtered_matrices.append(m)

    print(f"[INFO] Filtered down to {len(filtered_matrices)} pure topology graphs.")
    
    filtered_matrices = filtered_matrices[:max_graphs]
    if len(filtered_matrices) == 0:
        print("[WARN] Still 0 matrices. The database might not have graphs in this size range right now.")
        return []
        
    print(f"[INFO] Starting download and parsing for top {len(filtered_matrices)} matrices...")
    
    pyg_data_list = []
    
    for matrix in filtered_matrices:
        print(f"  -> Processing: [{matrix.group}] {matrix.name} (Nodes: {matrix.rows}, Kind: {matrix.kind})")
        try:
            # 下载并解压
            matrix.download(destpath=download_dir, extract=True)
            
            # 定位 .mtx 文件
            matrix_dir = os.path.join(download_dir, matrix.name)
            mtx_files = glob.glob(os.path.join(matrix_dir, '*.mtx'))
            
            if not mtx_files:
                print(f"     [WARN] No .mtx file found. Skipping.")
                continue
                
            mtx_file = mtx_files[0]
            
            # 解析稀疏矩阵
            sparse_mat = sio.mmread(mtx_file)
            
            # 强制对称化 (适配无向图架构)
            sparse_mat = sparse_mat.maximum(sparse_mat.transpose())
            
            # [核心修复：将矩阵强制转回 COO 坐标格式]
            sparse_mat = sparse_mat.tocoo()
            
            # 提取坐标
            row = torch.from_numpy(sparse_mat.row).to(torch.long)
            col = torch.from_numpy(sparse_mat.col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            
            # 去除自环
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            data = Data(
                edge_index=edge_index, 
                num_nodes=sparse_mat.shape[0],
                dataset_source=f"SuiteSparse_{matrix.group}_{matrix.name}"
            )
            pyg_data_list.append(data)
            
        except Exception as e:
            print(f"     [ERROR] Failed to process {matrix.name}: {e}")

    print("=" * 60)
    print(f"[SUCCESS] Downloaded and parsed {len(pyg_data_list)} SuiteSparse graphs.")
    return pyg_data_list

if __name__ == "__main__":
    data_list = fetch_and_parse_suitesparse(max_graphs=5)
    if data_list:
        print(f"\n[TEST RESULT] Sample Graph 0 Details:")
        print(data_list[0])