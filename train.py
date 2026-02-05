# -*- coding: utf-8 -*-
import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch, to_dense_adj

from data.dataset import DeepCSDataset
from data.batching import custom_collate_fn
from models.predictor import GraphPredictor
from models.solver import StressMajorizationSolver
from models.layers import DifferentiableAPSP 
from losses.loss import CognitiveRankLoss, PureStressLoss
from utils import seed_everything, load_config, save_config, EarlyStopping

def train_one_epoch(model, apsp_layer, solver, criterion, optimizer, loader, device, cfg):
    model.train()
    total_loss = 0
    log_meters = {'stress': 0, 'rank': 0}
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    count = 0
    
    for batch_data, d_sp_dense, weights_dense, mask in pbar:
        batch_data = batch_data.to(device)
        d_sp_dense = d_sp_dense.to(device)
        weights_dense = weights_dense.to(device)
        mask = mask.to(device)
        
        B, N_max = mask.shape
        init_pos = torch.randn(B, N_max, 2, device=device)
        
        # --- [1. 拓扑感知的邻接掩码 & 剔除自环] ---
        adj_mask = to_dense_adj(
            batch_data.edge_index, 
            batch=batch_data.batch, 
            max_num_nodes=N_max
        ).bool() # [B, N, N]
        
        # 显式剔除自环：确保对角线不参与均值计算
        eye = torch.eye(N_max, device=device).bool().unsqueeze(0).expand(B, -1, -1)
        valid_edge_mask = adj_mask & (~eye) 
        edge_mask_float = valid_edge_mask.float()
        
        # 1. 模型预测 Beta
        h_sparse = model(batch_data.x, batch_data.edge_index)
        h_dense, _ = to_dense_batch(h_sparse, batch_data.batch, max_num_nodes=N_max)
        beta_raw = model.predict_beta(h_dense, mask)
        
        # --- [2. 局部 1-Hop 归一化 (核心修改)] ---
        # 只针对真实的物理边计算均值，彻底封死“数值逃逸”漏洞
        beta_1hop_sum = (beta_raw * edge_mask_float).sum(dim=(1, 2), keepdim=True)
        edge_count = edge_mask_float.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        beta_mean = beta_1hop_sum / edge_count
        
        # 归一化后的 beta_norm 在物理边上的均值强制为 1.0
        beta_norm = (beta_raw / (beta_mean + 1e-8)).clamp(min=0.1, max=5.0)
        
        # 3. 可微分 APSP 层
        beta_1hop = beta_norm * edge_mask_float
        d_target = apsp_layer(beta_1hop, adj_mask)
        
        # [Debug] 监控：此时 Ratio 应该非常接近 1.0，因为 beta 均值锁死在了 1.0
        ratio = 1.0
        if count % 20 == 0:
            with torch.no_grad():
                mask_mat = mask.unsqueeze(2) * mask.unsqueeze(1)
                valid_pairs = (d_sp_dense > 0.5) & mask_mat.bool()
                actual_avg = d_target[valid_pairs].mean().item()
                # 理论均值：d_sp * 1.0 (因为 beta_norm 均值已强制为 1)
                theoretical_avg = d_sp_dense[valid_pairs].mean().item()
                ratio = actual_avg / (theoretical_avg + 1e-6)
        
        # 4. Stress Majorization Solver
        final_pos = solver(init_pos, weights_dense, d_target, mask)
        
        # 5. Loss 计算 (监督目标为 d_sp_dense)
        total_batch_loss, l_stress, l_rank = criterion(
            final_pos, d_sp_dense, weights_dense, mask, batch_data
        )
        
        # 反向传播
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['clip_grad'])
        optimizer.step()
        optimizer.zero_grad()
            
        log_meters['stress'] += l_stress.item()
        log_meters['rank'] += l_rank.item()
        total_loss += total_batch_loss.item()
        count += 1
        
        if count % 10 == 0:
            # 监控 1-hop beta 的实际均值，应稳定在 1.00
            real_b_mean = beta_norm[valid_edge_mask].mean().item()
            pbar.set_postfix({
                'S': f'{l_stress.item():.3f}', 
                'B_Mean': f'{real_b_mean:.2f}', 
                'Ratio': f'{ratio:.2f}'
            })

    avg_loss = total_loss / count
    avg_meters = {k: v / count for k, v in log_meters.items()}
    return avg_loss, avg_meters

def evaluate(model, apsp_layer, solver, criterion, loader, device, cfg):
    model.eval()
    total_loss = 0
    solver.iterations = cfg['solver']['eval_iters']
    
    with torch.no_grad():
        for batch_data, d_sp_dense, weights_dense, mask in loader:
            batch_data = batch_data.to(device)
            d_sp_dense = d_sp_dense.to(device)
            weights_dense = weights_dense.to(device)
            mask = mask.to(device)
            
            B, N_max = mask.shape
            init_pos = torch.randn(B, N_max, 2, device=device)
            
            adj_mask = to_dense_adj(batch_data.edge_index, batch=batch_data.batch, max_num_nodes=N_max).bool()
            eye = torch.eye(N_max, device=device).bool().unsqueeze(0).expand(B, -1, -1)
            valid_edge_mask = adj_mask & (~eye)
            edge_mask_float = valid_edge_mask.float()
            
            h_sparse = model(batch_data.x, batch_data.edge_index)
            h_dense, _ = to_dense_batch(h_sparse, batch_data.batch, max_num_nodes=N_max)
            beta_raw = model.predict_beta(h_dense, mask)
            
            # --- 同步归一化逻辑 ---
            beta_1hop_sum = (beta_raw * edge_mask_float).sum(dim=(1, 2), keepdim=True)
            edge_count = edge_mask_float.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            beta_mean = beta_1hop_sum / edge_count
            beta_norm = (beta_raw / (beta_mean + 1e-8)).clamp(min=0.1, max=5.0)
            
            d_target = apsp_layer(beta_norm * edge_mask_float, adj_mask)
            final_pos = solver(init_pos, weights_dense, d_target, mask)
            
            # 评估时也应监督其相对于 d_sp_dense 的还原能力
            batch_loss, _, _ = criterion(final_pos, d_sp_dense, weights_dense, mask, batch_data)
            total_loss += batch_loss.item()
            
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--force_restart', action='store_true', help='Delete old weights and start over')
    args = parser.parse_args()
    
    if not os.path.exists(args.config): args.config = 'configs/default.yaml'
    cfg = load_config(args.config)
    seed_everything(cfg['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_dir = os.path.join(cfg['log']['save_dir'], cfg['log']['exp_name'])
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_config(cfg, os.path.join(save_dir, 'config.yaml'))
    
    print(f"[INFO] Device: {device} | DeepCS V3.6 (Fixed Topology Mask)")
    
    # Dataset Loading
    dataset = DeepCSDataset(root=cfg['data']['root'])
    train_len = int(len(dataset) * cfg['data']['split_ratio'])
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_set, batch_size=cfg['data']['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg['data']['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    
    # Model Setup
    model = GraphPredictor(
        in_channels=12,
        hidden_channels=cfg['model']['hidden_channels'],
        heads=cfg['model']['heads']
    ).to(device)
    
    # [Config] 使用稳定的温度计算
    apsp_layer = DifferentiableAPSP(temperature=0.1).to(device)
    
    solver = StressMajorizationSolver(iterations=cfg['solver']['train_iters']).to(device)
    
    # Loss Configuration
    loss_type = cfg.get('loss', {}).get('type', 'cognitive')
    print(f"[INFO] Initializing {loss_type} loss...")
    
    if loss_type == 'pure_stress':
        criterion = PureStressLoss().to(device)
    else:
        criterion = CognitiveRankLoss(
            t=cfg['loss']['t'], margin=cfg['loss']['margin'], 
            alpha=cfg['loss']['alpha'],
            lambda_stress=cfg['loss']['lambda_stress'], 
            lambda_rank=cfg['loss']['lambda_rank']
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=float(cfg['train']['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_ckpt_path = os.path.join(save_dir, 'last.pth')
    
    # 从 config 读取阈值
    patience = cfg['train'].get('patience', 15)
    min_delta = cfg['train'].get('min_delta', 0.0001)  # 默认 0.0001
    early_stopping = EarlyStopping(
        patience=patience, 
        delta=min_delta,                 # 传递阈值
        path=best_model_path,
        verbose=True                    # 建议开启 verbose 观察 delta 是否生效
    )
    
    start_epoch = 0
    if os.path.exists(last_ckpt_path) and not args.force_restart:
        print(f"[INFO] Found checkpoint. Resuming...")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print(f"[INFO] Starting fresh training loop.")

    for epoch in range(start_epoch, cfg['train']['epochs']):
        start_time = time.time()
        train_loss, meters = train_one_epoch(model, apsp_layer, solver, criterion, optimizer, train_loader, device, cfg)
        val_loss = evaluate(model, apsp_layer, solver, criterion, val_loader, device, cfg)
        
        scheduler.step(val_loss)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} (Stress: {meters['stress']:.4f}) | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")
        
        early_stopping(val_loss, model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, last_ckpt_path)
        
        if early_stopping.early_stop:
            print("[INFO] Early stopping triggered.")
            break

if __name__ == "__main__":
    main()