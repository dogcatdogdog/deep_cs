# -*- coding: utf-8 -*-
"""
Master script: train KL, LNP, EU loss terms sequentially.
AR (ar_config.yaml) is already trained; included but commented out.
Each training saves to its own exp_name in ./logs/.
"""
import os
import sys
import subprocess

SCRIPT = "train_single_loss.py"

tasks = [
    ("ar", "configs/ar_config.yaml"),   # already trained, enable if re-training
    ("kl",  "configs/kl_config.yaml"),
    ("lnp", "configs/lnp_config.yaml"),
    ("eu",  "configs/eu_config.yaml"),
]


def run_one(label, config):
    print(f"\n{'='*60}")
    print(f"  Starting: {label} ({config})")
    print(f"{'='*60}")
    cmd = [sys.executable, SCRIPT, "--config", config]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}")
    else:
        print(f"\n[DONE] {label} completed successfully")


if __name__ == "__main__":
    for label, config in tasks:
        run_one(label, config)
    print("\n" + "=" * 60)
    print("  All tasks finished.")
    print("=" * 60)
