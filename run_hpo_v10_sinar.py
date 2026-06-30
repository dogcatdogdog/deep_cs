# -*- coding: utf-8 -*-
import os
import sys
import csv
import copy
import traceback
import optuna
from utils import load_config

#  Debug  UnifiedRankingOrbitalField 
from train_hpo_v10_sinar import run_training_pipeline

def objective(trial):
    # 1. 
    baseCfg = load_config("configs/prob_force_decoupled_v18_batched_kl_rangeplus.yaml")
    cfg = copy.deepcopy(baseCfg)
    cfg['log']['save_dir'] = "./logs/hpo_v10_sinar"  # separate from old HPO runs

    # 2. UnifiedRankingOrbitalField
    #  (6 )
    lambdaKl = trial.suggest_float("lambda_kl", 0.5, 4.0)
    lambdaLnp = trial.suggest_float("lambda_lnp", 0.5, 4.0)
    lambdaAr = trial.suggest_float("lambda_ar", 0.1, 2.0)
    lambdaEu = trial.suggest_float("lambda_eu", 0.01, 0.5)
    lambdaScaleLock = trial.suggest_float("lambda_scale_lock", 0.1, 1.0)
    lambdaLatent = trial.suggest_float("lambda_latent", 0.1, 2.0)
    
    #  (3 )
    marginRank = trial.suggest_float("margin_rank", 0.01, 0.15)
    sigmaRatio = trial.suggest_float("sigma_ratio", 0.5, 2.0)
    tauSharp = trial.suggest_float("tau_sharp", 0.02, 0.12)
    
    # 3. 
    cfg['loss'].update({
        'lambda_kl': lambdaKl,
        'lambda_lnp': lambdaLnp,
        'lambda_ar': lambdaAr,
        'lambda_eu': lambdaEu,
        'lambda_scale_lock': lambdaScaleLock,
        'lambda_latent': lambdaLatent,
        'margin_rank': marginRank,
        'sigma_ratio': sigmaRatio,
        'tau_sharp': tauSharp,
        'neg_sample_multiplier': 2.0,  # 
        'lambda_reg': 0.01,             # 
    })
    
    #  LNP/AR/EU 
    cfg['train']['use_curriculum'] = True

    # 
    trialName = f"hpo_v10_sinar_trial_{trial.number:03d}"
    cfg['log']['exp_name'] = trialName
    
    print(f"\n{'='*85}")
    print(f"[-] [Trial {trial.number:03d}] Launching Enriched Rank-Field Joint Search")
    print(f"    Macro Weights: KL={lambdaKl:.4f} | LNP={lambdaLnp:.4f} | AR={lambdaAr:.4f} | EU={lambdaEu:.4f} | Latent={lambdaLatent:.4f}")
    print(f"    Micro Fields : MarginRank={marginRank:.4f} | SigmaRatio={sigmaRatio:.4f} | TauSharp={tauSharp:.4f}")
    print(f"{'='*85}")

    # 4. 
    try:
        bestRealMetrics = run_training_pipeline(cfg, trial=trial)
        
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()
        
    except Exception as e:
        print(f"\n{'!'*85}", file=sys.stderr)
        print(f"[FATAL EXCEPTION DETECTED IN TRIAL {trial.number:03d}] Runtime Physics Collapse Rescue Launched.", file=sys.stderr)
        print(f"{'!'*85}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"{'!'*85}\n", file=sys.stderr)
        
        #  SQLite 
        trial.set_user_attr("NP", 0.0)
        trial.set_user_attr("LNP", 0.0)
        trial.set_user_attr("AR", 0.0)
        trial.set_user_attr("LAR", 0.0)
        trial.set_user_attr("EU", 999.0)
        trial.set_user_attr("LEU", 999.0)
        trial.set_user_attr("Obj1_NP_LNP", 0.0)
        trial.set_user_attr("Obj2_AR_LAR", 0.0)
        trial.set_user_attr("Obj3_EU_LEU", 1998.0)
        return 0.0, 0.0, 1998.0
        
    # 5. 
    valNp = bestRealMetrics.get('np_metric', 0.0)
    valLnp = bestRealMetrics.get('lnp_metric', 0.0)
    valAr = bestRealMetrics.get('ar_metric', 0.0)
    valLar = bestRealMetrics.get('lar_metric', 0.0)
    valEu = bestRealMetrics.get('eu_metric', 999.0)
    valLeu = bestRealMetrics.get('leu_metric', 999.0)
    
    # 6. 
    obj1_np_lnp = valNp + valLnp    # Target: Maximize
    obj2_ar_lar = valAr + valLar    # Target: Maximize
    obj3_eu_leu = valEu + valLeu    # Target: Minimize
    
    #  Optuna 
    trial.set_user_attr("NP", valNp)
    trial.set_user_attr("LNP", valLnp)
    trial.set_user_attr("AR", valAr)
    trial.set_user_attr("LAR", valLar)
    trial.set_user_attr("EU", valEu)
    trial.set_user_attr("LEU", valLeu)
    trial.set_user_attr("Obj1_NP_LNP", obj1_np_lnp)
    trial.set_user_attr("Obj2_AR_LAR", obj2_ar_lar)
    trial.set_user_attr("Obj3_EU_LEU", obj3_eu_leu)
    
    return obj1_np_lnp, obj2_ar_lar, obj3_eu_leu


if __name__ == "__main__":
    os.makedirs("hpo_results", exist_ok=True)
    
    dbFile = "hpo_results/deepcs_v10_sin_ar_hpo.db"
    dbPath = f"sqlite:///{dbFile}"
    studyName = "deepcs_v10_sin_ar_degree_weighted"
    totalTargetTrials = 50
    csvFile = "hpo_results/pareto_front_v10_sin_ar_solutions.csv"
    
    # 
    study = optuna.create_study(
        study_name=studyName,
        storage=dbPath,
        load_if_exists=True,  
        directions=["maximize", "maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    historicalTrials = study.trials
    completedTrials = [t for t in historicalTrials if t.state.is_finished()]
    
    if len(historicalTrials) == 0:
        print("[INFO] Fresh database initialized. Enqueuing baseline calibrated seed...")
        study.enqueue_trial({
            "lambda_kl": 1.0000,
            "lambda_lnp": 2.0000,
            "lambda_ar": 0.5000,
            "lambda_eu": 0.1000,
            "lambda_scale_lock": 0.3000,
            "lambda_latent": 1.0000,
            "margin_rank": 0.0500,
            "sigma_ratio": 1.0000,
            "tau_sharp": 0.0500
        })
    else:
        print(f"[INFO] Existing database loaded. Found {len(historicalTrials)} historical attempts.")
        print(f"[INFO] Completed/Pruned/Failed trials count: {len(completedTrials)}")
        
    remainingTrials = totalTargetTrials - len(completedTrials)
    
    if remainingTrials <= 0:
        print(f"[SUCCESS] Total target budget of {totalTargetTrials} trials already fulfilled. Exiting.")
        sys.exit(0)
        
    print(f"[INFO] Starting joint multi-objective search. Remaining budget: {remainingTrials} trials.")
    print("-" * 85)
    
    study.optimize(objective, n_trials=remainingTrials)
    
    # =========================================================
    # Pareto Front
    # =========================================================
    print("\n[SUCCESS] Multi-Objective Optimization Target Fulfilled!")
    paretoFrontTrials = study.best_trials
    
    if not paretoFrontTrials:
        print("[WARNING] Optimization completed, but no non-dominated Pareto front variants recovered.")
        sys.exit(0)
        
    with open(csvFile, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Trial_ID", 
            "lambda_kl", "lambda_lnp", "lambda_ar", "lambda_eu", "lambda_scale_lock", "lambda_latent",
            "margin_rank", "sigma_ratio", "tau_sharp",
            "Obj1_NP_LNP_Sum", "Obj2_AR_LAR_Sum", "Obj3_EU_LEU_Sum",
            "Val_Global_NP", "Val_Local_LNP", "Val_Global_AR", "Val_Local_LAR", "Val_Global_EU", "Val_Local_LEU"
        ])
        
        for trial in paretoFrontTrials:
            writer.writerow([
                trial.number,
                f"{trial.params.get('lambda_kl', 0.0):.4f}",
                f"{trial.params.get('lambda_lnp', 0.0):.4f}",
                f"{trial.params.get('lambda_ar', 0.0):.4f}",
                f"{trial.params.get('lambda_eu', 0.0):.4f}",
                f"{trial.params.get('lambda_scale_lock', 0.0):.4f}",
                f"{trial.params.get('lambda_latent', 0.0):.4f}",
                f"{trial.params.get('margin_rank', 0.0):.4f}",
                f"{trial.params.get('sigma_ratio', 0.0):.4f}",
                f"{trial.params.get('tau_sharp', 0.0):.4f}",
                f"{trial.user_attrs.get('Obj1_NP_LNP', 0.0):.4f}",
                f"{trial.user_attrs.get('Obj2_AR_LAR', 0.0):.4f}",
                f"{trial.user_attrs.get('Obj3_EU_LEU', 0.0):.4f}",
                f"{trial.user_attrs.get('NP', 0.0):.4f}",  
                f"{trial.user_attrs.get('LNP', 0.0):.4f}", 
                f"{trial.user_attrs.get('AR', 0.0):.4f}",  
                f"{trial.user_attrs.get('LAR', 0.0):.4f}", 
                f"{trial.user_attrs.get('EU', 0.0):.4f}",  
                f"{trial.user_attrs.get('LEU', 0.0):.4f}"  
            ])
            
    print(f"[SUCCESS] Exported {len(paretoFrontTrials)} non-dominated Solutions to Pareto front register: {csvFile}")