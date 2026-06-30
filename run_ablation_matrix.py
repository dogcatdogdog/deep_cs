# -*- coding: utf-8 -*-
import os
import sys
import copy
import traceback
from utils import load_config
from train_ablation import run_training_pipeline

def run_ablation_matrix(base_config_path='configs/ablation_config.yaml', force_restart=False):
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
        
    # 1. 显式定义 6 组消融实验的正交开关矩阵
    ablation_matrix = [
        {
            "exp_name": "ablation_exp0_baseline",
            "spatial_type": "GATv2",
            "use_role_stream": True,
            "isolate_gradient": True,
            "capacity_match": False
        },
        {
            "exp_name": "ablation_exp1_nodetach",
            "spatial_type": "GATv2",
            "use_role_stream": True,
            "isolate_gradient": False,
            "capacity_match": False
        },
        {
            "exp_name": "ablation_exp2_zeropadding",
            "spatial_type": "GATv2",
            "use_role_stream": False,
            "isolate_gradient": True,
            "capacity_match": False
        },
        {
            "exp_name": "ablation_exp3_widening",
            "spatial_type": "GATv2",
            "use_role_stream": False,
            "isolate_gradient": True,
            "capacity_match": True
        },
        {
            "exp_name": "ablation_exp4_gcn",
            "spatial_type": "GCN",
            "use_role_stream": True,
            "isolate_gradient": True,
            "capacity_match": False
        },
        {
            "exp_name": "ablation_exp5_gin",
            "spatial_type": "GIN",
            "use_role_stream": True,
            "isolate_gradient": True,
            "capacity_match": False
        }
    ]

    print("=" * 80)
    print(f"[MATRIX RUNNER] Loaded Matrix. Total Experiments to Run: {len(ablation_matrix)}")
    print("=" * 80)

    # 2. 串行迭代执行计算图流水线
    for idx, task in enumerate(ablation_matrix):
        print(f"\n[TASK {idx+1}/{len(ablation_matrix)}] Initializing: {task['exp_name']}")
        
        # 每次循环重新读取/加载基础配置，确保起点绝对纯净
        base_cfg = load_config(base_config_path)
        
        # 刚性防御：确保 HPO 最优静态权重不被残留的 curriculum 动态污染
        base_cfg['train']['use_curriculum'] = True
        
        # 检查是否已有完备的训练产物，支持断点无感跳过
        save_dir = base_cfg['log'].get('save_dir', './logs')
        target_best_model = os.path.join(save_dir, task['exp_name'], 'best_model.pth')
        
        if os.path.exists(target_best_model) and not force_restart:
            print(f"  --> [SKIP] Detection hit: '{target_best_model}' already exists. Shifting to next task.")
            continue
            
        # 3. 内存深拷贝，动态注入当前组的消融控制算子
        current_cfg = copy.deepcopy(base_cfg)
        current_cfg['log']['exp_name'] = task['exp_name']
        
        # 覆写消融控制域
        if 'ablation' not in current_cfg:
            current_cfg['ablation'] = {}
            
        current_cfg['ablation']['spatial_type'] = task['spatial_type']
        current_cfg['ablation']['use_role_stream'] = task['use_role_stream']
        current_cfg['ablation']['isolate_gradient'] = task['isolate_gradient']
        current_cfg['ablation']['capacity_match'] = task['capacity_match']

        # 4. 执行单组完整训练管线
        print(f"  --> [EXECUTE] Launching pipeline for {task['exp_name']}...")
        try:
            metrics = run_training_pipeline(current_cfg, force_restart=force_restart)
            print(f"[SUCCESS] Finished {task['exp_name']}.")
            print(f"          Final HPO Fitness: {metrics.get('hpo_fitness', -1.0):.4f}")
        except Exception as e:
            print(f"[CRITICAL ERROR] Task {task['exp_name']} collapsed mid-way!")
            traceback.print_exc()
            print("Terminating matrix execution loop to safeguard remaining resources.")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("[MATRIX RUNNER] All 6 ablation experiments have been successfully processed completed.")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Ablation Matrix Executive Engine")
    parser.add_argument('--config', type=str, default='configs/ablation_config.yaml', help='Path to the base config file')
    parser.add_argument('--force_restart', action='store_true', help='Force wipe logs and restart from scratch')
    args = parser.parse_args()

    run_ablation_matrix(base_config_path=args.config, force_restart=args.force_restart)