# -*- coding: utf-8 -*-
import os, sys, csv
import optuna

DB_FILE = "hpo_results/deepcs_v10_sin_ar_hpo.db"
STUDY_NAME = "deepcs_v10_sin_ar_degree_weighted"
CSV_ALL = "hpo_results/all_trials_v10_sinar.csv"
CSV_BEST = "hpo_results/pareto_front_v10_sinar.csv"

def main():
    db_path = f"sqlite:///{DB_FILE}"

    if not os.path.exists(DB_FILE):
        print(f"[FATAL] Database not found: {DB_FILE}")
        sys.exit(1)

    study = optuna.load_study(study_name=STUDY_NAME, storage=db_path)
    trials = [t for t in study.trials if t.state.is_finished()]

    if not trials:
        print("[INFO] No completed trials found.")
        return

    # === All solutions ===
    with open(CSV_ALL, 'w', newline='') as f:
        w = csv.writer(f)
        header = ["Trial", "State", "lambda_kl", "lambda_lnp", "lambda_ar", "lambda_eu",
                  "lambda_scale_lock", "lambda_latent", "margin_rank", "sigma_ratio", "tau_sharp",
                  "NP", "LNP", "AR", "LAR", "EU", "LEU",
                  "Obj_NP_LNP", "Obj_AR_LAR", "Obj_EU_LEU"]
        w.writerow(header)

        for t in trials:
            ua = t.user_attrs
            w.writerow([
                t.number, "FINISHED" if t.state.is_finished() else str(t.state),
                f"{t.params.get('lambda_kl', 0):.4f}",
                f"{t.params.get('lambda_lnp', 0):.4f}",
                f"{t.params.get('lambda_ar', 0):.4f}",
                f"{t.params.get('lambda_eu', 0):.4f}",
                f"{t.params.get('lambda_scale_lock', 0):.4f}",
                f"{t.params.get('lambda_latent', 0):.4f}",
                f"{t.params.get('margin_rank', 0):.4f}",
                f"{t.params.get('sigma_ratio', 0):.4f}",
                f"{t.params.get('tau_sharp', 0):.4f}",
                f"{ua.get('NP', 0):.4f}", f"{ua.get('LNP', 0):.4f}",
                f"{ua.get('AR', 0):.4f}", f"{ua.get('LAR', 0):.4f}",
                f"{ua.get('EU', 0):.4f}", f"{ua.get('LEU', 0):.4f}",
                f"{ua.get('Obj1_NP_LNP', 0):.4f}",
                f"{ua.get('Obj2_AR_LAR', 0):.4f}",
                f"{ua.get('Obj3_EU_LEU', 0):.4f}",
            ])

    print(f"[DONE] All {len(trials)} trials -> {CSV_ALL}")

    # === Pareto front (best solutions) ===
    pareto = study.best_trials
    if not pareto:
        print("[WARNING] No Pareto front solutions found.")
        return

    with open(CSV_BEST, 'w', newline='') as f:
        w = csv.writer(f)
        header = ["Trial", "lambda_kl", "lambda_lnp", "lambda_ar", "lambda_eu",
                  "lambda_scale_lock", "lambda_latent", "margin_rank", "sigma_ratio", "tau_sharp",
                  "NP", "LNP", "AR", "LAR", "EU", "LEU",
                  "Obj_NP_LNP", "Obj_AR_LAR", "Obj_EU_LEU"]
        w.writerow(header)

        for t in pareto:
            ua = t.user_attrs
            w.writerow([
                t.number,
                f"{t.params.get('lambda_kl', 0):.4f}",
                f"{t.params.get('lambda_lnp', 0):.4f}",
                f"{t.params.get('lambda_ar', 0):.4f}",
                f"{t.params.get('lambda_eu', 0):.4f}",
                f"{t.params.get('lambda_scale_lock', 0):.4f}",
                f"{t.params.get('lambda_latent', 0):.4f}",
                f"{t.params.get('margin_rank', 0):.4f}",
                f"{t.params.get('sigma_ratio', 0):.4f}",
                f"{t.params.get('tau_sharp', 0):.4f}",
                f"{ua.get('NP', 0):.4f}", f"{ua.get('LNP', 0):.4f}",
                f"{ua.get('AR', 0):.4f}", f"{ua.get('LAR', 0):.4f}",
                f"{ua.get('EU', 0):.4f}", f"{ua.get('LEU', 0):.4f}",
                f"{ua.get('Obj1_NP_LNP', 0):.4f}",
                f"{ua.get('Obj2_AR_LAR', 0):.4f}",
                f"{ua.get('Obj3_EU_LEU', 0):.4f}",
            ])

    print(f"[DONE] Pareto front ({len(pareto)} solutions) -> {CSV_BEST}")
    print()
    print("=" * 80)
    print("BEST SOLUTIONS (Pareto Front)")
    print("=" * 80)
    for i, t in enumerate(pareto):
        ua = t.user_attrs
        print(f"\n  [{i+1}] Trial {t.number}:")
        print(f"      KL={t.params['lambda_kl']:.3f}  LNP={t.params['lambda_lnp']:.3f}  AR={t.params['lambda_ar']:.3f}  EU={t.params['lambda_eu']:.3f}")
        print(f"      margin_rank={t.params['margin_rank']:.4f}  sigma={t.params['sigma_ratio']:.3f}  tau={t.params['tau_sharp']:.4f}")
        print(f"      NP={ua.get('NP',0):.4f}  LNP={ua.get('LNP',0):.4f}  AR={ua.get('AR',0):.4f}  LAR={ua.get('LAR',0):.4f}  EU={ua.get('EU',0):.4f}  LEU={ua.get('LEU',0):.4f}")
    print()

if __name__ == "__main__":
    os.makedirs("hpo_results", exist_ok=True)
    main()
