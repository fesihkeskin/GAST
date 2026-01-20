#!/usr/bin/env python3
# src/training/gast_hyperparameter_tuning.py

"""
GAST Hyperparameter Tuning Script
Performs a two‐phase (coarse + fine) Optuna search with Hyperband pruning
and budget‐aware epoch scheduling.
"""

# ─────────────────────────── paths / imports ───────────────────────────
from __future__ import annotations
import os
import sys
import time
from datetime import datetime, timedelta
from argparse import Namespace
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned
import joblib
import optuna.visualization as vis

import torch
import gc
import warnings

# CUDA / cuDNN settings
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

gc.collect()
warnings.filterwarnings("ignore", category=FutureWarning)

# project‐root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_with_oom_handling
from src.data.data_loader import load_dataset, DATASET_PATHS
from src.utils.utils import set_seed

# ───────────────────────────── constants ──────────────────────────────
MODEL_TYPE    = "gast"
COARSE_TRIALS = 50
FINE_TRIALS   = 20
MAX_EPOCHS    = 500
EARLY_STOP    = 50
FIXED_SEED    = 242
NUM_WORKERS   = 4
TRAIN_RATIO   = 0.05
VAL_RATIO     = 0.05

# Unified output paths
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "reports" / "results" / TS  # timestamped results
DB_DIR  = PROJECT_ROOT / "reports" / "optuna_db"    # permanent storage

# Create directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Summary file
SUMMARY = OUT_DIR / f"opt_summary_{MODEL_TYPE}.txt"


with open(SUMMARY, "w") as f:
    f.write("GAST Hyperparameter Search (Coarse → Fine)\n" + "="*60 + "\n\n")
    f.write(f"Timestamp: {TS}\n")
# ───────────────────────────── save CLI helper ──────────────────────────────
def _save_cli(dataset: str, args: Namespace, phase: str):
    """
    Write out a shell script to reproduce the best train+test commands.
    """
    base_out = PROJECT_ROOT / "models" / "final" / MODEL_TYPE / dataset
    checkpoint = base_out / f"gast_best_{dataset}.pth"
    script = OUT_DIR / f"{MODEL_TYPE}_{phase}_best_{dataset}.sh"

    with open(script, "w") as out:
        # build train cmd
        parts = ["python main.py --mode train"]
        for k, v in vars(args).items():
            if k in {"raw_gt","cube_path","gt_path","mode"} or v is None:
                continue
            if k == "output_dir":
                parts.append(f"--output_dir {base_out}")
            elif isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        train_cmd = " ".join(parts)

        # build test cmd
        test_parts = train_cmd.replace("--mode train","--mode test")
        test_cmd   = test_parts + f" --checkpoint {checkpoint}"

        out.write(train_cmd + "\n\n" + test_cmd + "\n")

    print(f"✅ Saved {phase}-phase CLI → {script}")

# ───────────────────────────── optimizer ──────────────────────────────
def _run_optuna(
    dataset: str,
    n_trials: int,
    param_space: dict[str, dict],
    study_name: str
) -> optuna.Study:
    """
    Generic Optuna runner that respects a given param_space dictionary.
    param_space[name] = {
      "type": "float"|"int"|"categorical",
      "low": …, "high": …, "log": bool,
      "choices": […], "step": …  
    }
    """
    # load once
    _, raw_gt = load_dataset(dataset)
    cube    = PROJECT_ROOT / DATASET_PATHS[dataset]["image"]
    gt      = PROJECT_ROOT / DATASET_PATHS[dataset]["ground_truth"]
    best_val = 0.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_val
        set_seed(FIXED_SEED)
        torch.cuda.empty_cache()

        # sample all hyperparams from param_space
        sampled = {}
        for name, spec in param_space.items():
            if spec["type"] == "float":
                sampled[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                    step=spec.get("step", None),
                )
            elif spec["type"] == "int":
                sampled[name] = trial.suggest_int(
                    name, spec["low"], spec["high"],
                    step=spec.get("step", 1)
                )
            else:  # categorical
                sampled[name] = trial.suggest_categorical(name, spec["choices"])

        # assemble args
        args = Namespace(
            dataset      = dataset,
            cube_path    = str(cube),
            gt_path      = str(gt),
            raw_gt       = raw_gt,
            train_ratio  = TRAIN_RATIO,
            val_ratio    = VAL_RATIO,
            mode         = "train",
            epochs       = int(sampled.get("epochs", MAX_EPOCHS)),
            early_stop   = EARLY_STOP,
            batch_size   = int(sampled.get("batch_size")),
            patch_size   = int(sampled.get("patch_size")),
            stride       = int(sampled.get("stride")),
            lr           = sampled.get("lr"),
            weight_decay = sampled.get("weight_decay"),
            dropout      = sampled.get("dropout"),
            embed_dim        = int(sampled.get("embed_dim")),
            gat_hidden_dim   = int(sampled.get("gat_hidden_dim")),
            gat_heads        = int(sampled.get("gat_heads")),
            gat_depth        = int(sampled.get("gat_depth")),
            transformer_heads  = int(sampled.get("transformer_heads")),
            transformer_layers = int(sampled.get("transformer_layers")),
            disable_spectral = False,
            disable_spatial  = False,
            fusion_mode      = "gate",
            seed         = FIXED_SEED,
            num_workers  = NUM_WORKERS,
            output_dir   = PROJECT_ROOT / "models" / "checkpoints" / MODEL_TYPE / dataset / f"trial_{trial.number}",
        )
        os.makedirs(args.output_dir, exist_ok=True)

        # train + prune OOMs
        try:
            val_oa, _ = train_with_oom_handling(args, trial)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise TrialPruned()
            raise

        # record best and save CLI
        if val_oa > best_val:
            best_val = val_oa
            _save_cli(dataset, args, study_name)

        return val_oa

    # Use consistent paths
    storage = f"sqlite:///{DB_DIR}/study_{dataset}_{study_name}.db"
    db_path = DB_DIR / f"study_{dataset}_{study_name}.db"
    dump_path = OUT_DIR / f"study_{dataset}_{study_name}.pkl"

    print("\n=== Database Debug Info ===")
    print(f"Looking for database at: {db_path}")
    print(f"Using storage URL: {storage}")
    print(f"Study name: {study_name}")

    if db_path.exists():
        print(f"\n💾 Found existing database at {db_path}")
        print(f"   Size: {db_path.stat().st_size / 1024:.1f} KB")
        print(f"   Modified: {datetime.fromtimestamp(db_path.stat().st_mtime)}")
        
        try:
            # List all studies in the database
            print("\nStudies in database:")
            loaded_studies = optuna.study.get_all_study_summaries(storage)
            for s in loaded_studies:
                print(f"  - {s.study_name}: {s.n_trials} trials")
            
            # Try to load our specific study
            loaded_study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"\n✅ Successfully loaded study '{study_name}' with {len(loaded_study.trials)} trials")
            if len(loaded_study.trials) > 0:
                print(f"   Best value so far: {loaded_study.best_value:.4f}")
        except Exception as e:
            print(f"\n❌ Failed to load study: {e}")

    study = optuna.create_study(
        direction="maximize",
        pruner=HyperbandPruner(min_resource=1, max_resource=MAX_EPOCHS, reduction_factor=3),
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    # Check existing trials and compute remaining
    existing = len(study.trials)
    print(f"⚡️ Loaded study with {existing} completed trials")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Number of completed trials: {len(completed_trials)}")

    to_run = n_trials - existing
    if to_run <= 0:
        print(f"✅ Already ran {existing} trials; skipping optimization.")
    else:
        print(f"🚀 Running {to_run} more trials (to reach {n_trials} total)...")
        start = time.time()
        study.optimize(objective, n_trials=to_run, n_jobs=4)
        elapsed = timedelta(seconds=int(time.time() - start))

        # dump and log
        joblib.dump(study, dump_path)
        with open(SUMMARY, "a") as f:
            f.write(f"{study_name.upper()} PHASE — Dataset: {dataset}\n")
            f.write(f"  Best Val OA   : {study.best_value:.4f}\n")
            f.write(f"  Best Trial ID : {study.best_trial.number}\n")
            f.write(f"  Time elapsed  : {elapsed}\n\n")

    try:
        print(f"\n📈 {study_name.title()} done for {dataset}: best OA = {study.best_value:.4f}\n")
    except ValueError:
        print(f"\n⚠️ {study_name.title()} done for {dataset}: No successful trial found (all pruned or failed).\n")
    
    vis.plot_optimization_history(study).write_image(
        str(OUT_DIR / f"{dataset}_{study_name}_history.png")
    )
    try:
        vis.plot_param_importances(study).write_image(
            str(OUT_DIR / f"{dataset}_{study_name}_importance.png")
        )
    except ValueError as e:
        print(f"Skipping param importances plot: {e}")
    
    return study

# ───────────────────────────── main ──────────────────────
if __name__ == "__main__":
    from src.data.data_loader import DATASET_NAME_LIST
    # DATASET_NAME_LIST = ["Salinas","SalinasA"]  # or more datasets
    # DATASET_NAME_LIST = ["Indian_Pines", "Kennedy_Space_Center", "Pavia_University","Botswana"]  # or more datasets
    DATASET_NAME_LIST = ["Houston13"]  # or more datasets

    for ds in DATASET_NAME_LIST:
        if ds not in DATASET_PATHS:
            print(f"❌ Dataset {ds} not found in DATASET_PATHS. Skipping.")
            continue
        print(f"\n🔍 Starting hyperparameter tuning for {ds} dataset...")
        # Phase 1: Coarse search over wide ranges
        coarse_space = {
            "epochs":         {"type":"int",    "low":50,   "high":MAX_EPOCHS, "step":50},
            "batch_size":     {"type":"categorical","choices":[16,32,48,64]},
            "patch_size":     {"type":"categorical","choices":[5,7,9,11,13]},
            "stride":         {"type":"categorical","choices":[1,2,3,4,5,6,7]},
            "lr":             {"type":"float",  "low":1e-8,  "high":1e-2, "log":True},
            "weight_decay":   {"type":"float",  "low":1e-8,  "high":1e-2, "log":True},
            "dropout":        {"type":"float",  "low":0.0,   "high":0.5, "step":0.05},
            "embed_dim":      {"type":"categorical","choices":[64,128,256]},
            "gat_hidden_dim": {"type":"categorical","choices":[32,64,128]},
            "gat_heads":      {"type":"categorical","choices":[2,4,6,8,10]},
            "gat_depth":      {"type":"categorical","choices":[2,4,6,8,10]},
            "transformer_heads":   {"type":"categorical","choices":[2,4,8,16]},
            "transformer_layers":  {"type":"categorical","choices":[2,4,6,8,10]},
        }
        coarse_study = _run_optuna(ds, COARSE_TRIALS, coarse_space, "coarse")
        best = coarse_study.best_trial.params

        # Phase 2: Fine search around best params
        def make_range(val, low_mul, high_mul, min_val=None, max_val=None):
            lo = val * low_mul
            hi = val * high_mul
            if min_val is not None: lo = max(lo, min_val)
            if max_val is not None: hi = min(hi, max_val)
            return lo, hi

        lr_lo, lr_hi = make_range(best["lr"], 0.1, 10)
        wd_lo, wd_hi = make_range(best["weight_decay"], 0.1, 10)

        fine_space = {
            "lr": {
                "type": "float",
                "low": lr_lo,
                "high": lr_hi,
                "log": True
            },
            "weight_decay": {
                "type": "float",
                "low": wd_lo,
                "high": wd_hi,
                "log": True
            },
            "epochs": {
                "type": "int",
                "low": max(50, best["epochs"] - 50),
                "high": min(MAX_EPOCHS, best["epochs"] + 50),
                "step": 50
            },
            "stride": {
                "type": "int",
                "low": max(1, best["stride"] - 1),
                "high": best["stride"] + 1
            },
            "transformer_layers": {
                "type": "int",
                "low": max(1, best["transformer_layers"] - 1),
                "high": best["transformer_layers"] + 1
            },

            # keep everything else fixed as single‐choice categoricals
            "batch_size":     {"type":"categorical","choices":[best["batch_size"]]},
            "patch_size":     {"type":"categorical","choices":[best["patch_size"]]},
            "embed_dim":      {"type":"categorical","choices":[best["embed_dim"]]},
            "gat_hidden_dim": {"type":"categorical","choices":[best["gat_hidden_dim"]]},
            "gat_heads":      {"type":"categorical","choices":[best["gat_heads"]]},
            "gat_depth":      {"type":"categorical","choices":[best["gat_depth"]]},
            "transformer_heads": {"type":"categorical","choices":[best["transformer_heads"]]},
            "dropout":        {"type":"categorical","choices":[best["dropout"]]},
        }
        # FINE PHASE
        fine_study = _run_optuna(ds, FINE_TRIALS, fine_space, "fine")

    print(f"\n✅ All done! Summary in {SUMMARY}")
    print("📂 Results and studies in", OUT_DIR)
# os.system("sudo /sbin/shutdown -h now")