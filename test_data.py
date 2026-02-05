# -*- coding: utf-8 -*-
import os
import shutil
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from data.dataset import DeepCSDataset

def test_pipeline():
    print("="*40)
    print("[TEST] DeepCS Data Pipeline Test")
    print("="*40)

    # 路径设置
    ROOT_DIR = "data" 
    
    # 1. 实例化数据集
    print("[1/4] Loading Dataset from data...")
    dataset = DeepCSDataset(root=ROOT_DIR)
    
    print(f"    [OK] Dataset Loaded.")
    print(f"    Total Graphs: {len(dataset)}")
    print(f"    Input Features (Channels): {dataset.num_features}")
    
    if len(dataset) == 0:
        print("    [ERROR] Dataset is empty! Check raw files.")
        return

    # 2. 检查单个样本详情
    print("\n[2/4] Inspecting Sample Graph (Index 0)...")
    data = dataset[0]
    
    N = data.num_nodes
    print(f"    - Num Nodes (N): {N}")
    print(f"    - Input X:          {data.x.shape}      (Expected: [{N}, 11])")
    print(f"    - Edge Index:       {data.edge_index.shape} (Expected: [2, E])")
    print(f"    - Ground Truth D:   {data.d_sp.shape}   (Expected: [{N}, {N}])")
    print(f"    - Stress Weights W: {data.weights.shape} (Expected: [{N}, {N}])")

    # 3. 验证数值逻辑
    print("\n[3/4] Verifying Data Logic...")
    
    # A. 检查 X 是否包含 NaN
    if torch.isnan(data.x).any():
        print("    [FAIL] Feature X contains NaN!")
    else:
        print("    [OK] Feature X is clean (no NaNs).")
        
    # B. 检查 Log-Degree (第0维) 是否归一化到了 [0, 1]
    deg_feat = data.x[:, 0]
    if deg_feat.min() >= -1e-5 and deg_feat.max() <= 1.0 + 1e-5:
        print(f"    [OK] Log-Degree feature in range [0, 1] (Max: {deg_feat.max():.4f})")
    else:
        print(f"    [WARN] Log-Degree out of range? (Min: {deg_feat.min()}, Max: {deg_feat.max()})")

    # C. 检查权重矩阵计算逻辑
    u, v = 0, 1
    dist = data.d_sp[u, v].item()
    weight = data.weights[u, v].item()
    
    print(f"    - Checking pair ({u}, {v}): Distance={dist}, Weight={weight}")
    
    if dist > 0:
        expected_weight = 1.0 / (dist ** 2)
        if abs(weight - expected_weight) < 1e-5:
            print("    [OK] Weight calculation verified (w = 1/d^2).")
        else:
            print(f"    [FAIL] Weight calculation mismatch! Expected {expected_weight}, got {weight}")
    else:
        print("    (Skipped pair check due to zero distance)")

    # 4. DataLoader 测试
    print("\n[4/4] Testing DataLoader (Batch Size = 1)...")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    try:
        batch = next(iter(loader))
        print("    [OK] Batch loading successful.")
        print(f"    - Batch X shape: {batch.x.shape}")
        print(f"    - Batch D_sp shape: {batch.d_sp.shape}")
    except Exception as e:
        print(f"    [FAIL] Batch loading failed: {e}")

    print("\n" + "="*40)
    print("[DONE] Test Complete. You are ready to train!")
    print("="*40)

if __name__ == "__main__":
    if os.path.exists("data/processed"):
        print("Found existing processed data.")
    test_pipeline()