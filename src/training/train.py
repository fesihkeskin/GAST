#!/usr/bin/env python3
# src/training/train.py

"""
End-to-end training routine for GAST with stratified splitting.
"""

import os
import time
import gc
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from optuna.exceptions import TrialPruned
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedShuffleSplit

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from src.data.dataset import HyperspectralDataset
from src.models.model_architecture import GAST
from src.utils.utils import save_checkpoint, OA
from src.utils.utils import stratified_min_samples_split, FocalLoss
from src.data.dataset_info import get_imbalance_ratio

# Training start at date
TRAIN_START_DATE = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"Training started at: {TRAIN_START_DATE}")
train_start_time = time.time()  # Start time for training

# Load dataset paths from configuration
def _load_split(args: Namespace, stage: str, indices, mean, std):
    return HyperspectralDataset(
        args.cube_path,
        args.gt_path,
        patch_size=args.patch_size,
        stride=args.stride, 
        mode=stage,
        indices=indices,
        mean=mean,
        std=std,
        augment=(stage == "train"),
    )

# Inverse frequency weights for CrossEntropyLoss
# This function computes weights inversely proportional to the class frequencies.
# It helps to balance the loss function, especially for imbalanced datasets.
# It returns a tensor of weights for each class, which can be used in the loss function.
# It uses log1p to avoid division by zero and to smooth the weights.
def _inv_freq_weights(labels, n_cls, device):
    cnt = np.bincount(labels, minlength=n_cls)
    weights = 1.0 / np.log1p(np.maximum(cnt, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(args: Namespace, trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print("🚀 device:", device)

    # ── 1. Print hyperparameters ──────────────────────────────────────────
    print("🔧 Hyperparameter settings:")
    for key, value in sorted(vars(args).items()):
        if key in {"raw_gt", "cube_path", "gt_path"}:
            continue
        print(f"   {key}: {value}")

    # ── 2. Stratified split ───────────────────────────────────────────────
    coords = np.argwhere(args.raw_gt > 0)
    labels = args.raw_gt[coords[:, 0], coords[:, 1]] - 1  

    # Calculate max allowed floors for each class
    class_counts = np.bincount(labels)
    max_allowed_train_floor = int(min(class_counts) * 0.8)  # 80% of smallest class
    train_floor = min(10, max_allowed_train_floor)
    max_allowed_val_floor = int(min(class_counts) * 0.8)  # 80% of smallest class
    val_floor = min(2, max_allowed_val_floor)
    train_idx, val_idx, test_idx = stratified_min_samples_split(
        coords, 
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_min_frac=args.train_ratio,  # minimum fraction for training
        val_min_frac=0.0,                 # no minimum fraction for validation
        train_floor=train_floor,          # minimum samples for training
        val_floor=val_floor,              # minimum samples for validation
        seed=args.seed
    )

    
    # --- Print and save class distribution ---
    def class_distribution(gt, indices):
        labs, counts = np.unique(gt[indices[:, 0], indices[:, 1]], return_counts=True)
        return dict(zip(labs, counts))

    train_dist = class_distribution(args.raw_gt, train_idx)
    val_dist   = class_distribution(args.raw_gt, val_idx)
    test_dist  = class_distribution(args.raw_gt, test_idx)

    all_classes = sorted(set(train_dist) | set(val_dist) | set(test_dist))
    header = f"{args.dataset}\n{'Class':>5} | {'Train':>6} | {'Val':>6} | {'Test':>6}"
    lines = [header, "-"*len(header)]
    for cls in all_classes:
        lines.append(f"{cls:5d} | {train_dist.get(cls,0):6d} | {val_dist.get(cls,0):6d} | {test_dist.get(cls,0):6d}")
    dist_report = "\n".join(lines)
    print("\nClass distribution (after split):\n" + dist_report)

    # Save to file
    split_dir = Path(args.output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"class_distribution_seed_{args.seed}.txt", "w") as f:
        f.write(dist_report)
        print(f"📊 Saved class distribution report → {split_dir / f'class_distribution_seed_{args.seed}.txt'}")

    # save split indices
    split_dir = Path(args.output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"train_idx_seed_{args.seed}.npy", train_idx)
    np.save(split_dir / f"val_idx_seed_{args.seed}.npy",   val_idx)
    np.save(split_dir / f"test_idx_seed_{args.seed}.npy",  test_idx)

    # ── 3. build datasets and loaders ────────────────────────────────────
    train_ds = _load_split(args, "train", train_idx, None, None)
    val_ds   = _load_split(args, "val",   val_idx,   train_ds.mean, train_ds.std)
    test_ds  = _load_split(args, "test",  test_idx,  train_ds.mean, train_ds.std)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # ── 4. model / loss / optimizer ───────────────────────────────────────
    n_cls = len(np.unique(args.raw_gt))
    model = GAST(
        in_channels=train_ds.B,
        n_classes=n_cls,
        patch_size=args.patch_size,
        spec_dim=args.embed_dim,
        spat_dim=args.gat_hidden_dim,
        n_heads=args.gat_heads,
        n_layers=args.gat_depth,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        disable_spectral=getattr(args, "disable_spectral", False),
        disable_spatial=getattr(args, "disable_spatial", False),
        fusion_mode=getattr(args, "fusion_mode", "gate"),
    ).to(device)

    # Calculate class distribution for train set (ignore background)
    # and determine if FocalLoss is needed
    train_labels = train_ds.labels[train_ds.labels >= 0]
    imbalance_ratio = get_imbalance_ratio(train_labels)
    print(f"\n⚖️ Imbalance ratio: {imbalance_ratio:.2f}")

    # Modified loss selection code
    if hasattr(args, 'force_wce') and args.force_wce:
        print("✅ Using weighted CrossEntropyLoss (forced)")
        crit = nn.CrossEntropyLoss(
            weight=_inv_freq_weights(train_ds.labels, n_cls, device),
            label_smoothing=0.05,
            ignore_index=-1
        ).to(device)
    elif hasattr(args, 'force_focal') and args.force_focal:
        print("⚠️ Using FocalLoss (forced)")
        class_weights = _inv_freq_weights(train_ds.labels, n_cls, device)
        crit = FocalLoss(
            gamma=2.0,
            alpha=class_weights,
            ignore_index=-1,
            reduction="mean",
        ).to(device)
    else:
        # Use existing adaptive logic with configurable threshold
        threshold = args.imbalance_threshold if hasattr(args, 'imbalance_threshold') else 5
        if imbalance_ratio > threshold:
            print(f"⚠️ Dataset is imbalanced (ρ={imbalance_ratio:.1f} > {threshold}). Using FocalLoss.")
            class_weights = _inv_freq_weights(train_ds.labels, n_cls, device)
            crit = FocalLoss(
                gamma=2.0,
                alpha=class_weights,
                ignore_index=-1,
                reduction="mean",
            ).to(device)
        else:
            print(f"✅ Dataset is balanced (ρ={imbalance_ratio:.1f} ≤ {threshold}). Using CrossEntropyLoss.")
            crit = nn.CrossEntropyLoss(
                weight=_inv_freq_weights(train_ds.labels, n_cls, device),
                label_smoothing=0.05,
                ignore_index=-1
            ).to(device)

    optimiser = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=10, factor=0.3
    )

    print(f"▶ train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_val, patience = 0.0, 0
    t0 = time.time()
    epoch_history = []

    # ── 5. epoch loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        tr_loss = tr_hit = 0
        scaler = GradScaler("cuda")
        for batch in tqdm(train_loader, desc=f"{epoch:03d}|train", leave=False, bar_format="{l_bar}{percentage:3.0f}%"):
            optimiser.zero_grad(set_to_none=True)
            x, y = batch["patch"].to(device), batch["label"].to(device)
            with autocast("cuda"):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
            tr_hit  += (logits.argmax(1) == y).sum().item()

        tr_loss /= len(train_ds)
        tr_acc = tr_hit / len(train_ds)

        # val
        model.eval()
        v_loss = v_hit = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{epoch:03d}|val  ", leave=False, bar_format="{l_bar}{percentage:3.0f}%"):
                x, y = batch["patch"].to(device), batch["label"].to(device)
                logits = model(x)
                v_loss += crit(logits, y).item() * x.size(0)
                v_hit  += (logits.argmax(1) == y).sum().item()

        v_loss /= len(val_ds)
        v_acc = v_hit / len(val_ds)
        scheduler.step(v_acc)

        print(f"epoch {epoch:03d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} "
              f"| val_loss {v_loss:.4f} val_acc {v_acc:.4f}")

        epoch_history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": v_loss,
            "val_acc": v_acc,
            "lr": optimiser.param_groups[0]["lr"]
        })
    
        # checkpoint
        if v_acc > best_val or epoch == 1:
            best_val, patience = v_acc, 0
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimiser.state_dict(),
                "val_acc": best_val,
                "args": vars(args)
            }, args.output_dir, f"gast_best_{args.dataset}.pth")
            print(f"  ✔ new best val OA {best_val:.4f} (checkpoint saved)")
        else:
            patience += 1
            if patience >= args.early_stop:
                print("⏹ early stop at epoch", epoch)
                break
            
        # optuna pruning         
        if trial is not None:
            trial.report(v_acc, epoch)
            if trial.should_prune():
                print(f"⏹ Trial pruned at epoch {epoch}")
                raise TrialPruned()

    # ── 6. final test ------------------------------------------------------
    ckpt = torch.load(Path(args.output_dir)/f"gast_best_{args.dataset}.pth",
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_oa = OA(model, test_loader, device)
    print(f"🏁 finished. best val OA={best_val:.4f} | test OA={test_oa:.4f} "
          f"| time={(time.time()-t0)/60:.1f} min")

    # ── 7. save history JSON ---------------------------------------------
    model_name = ("no_spectral" if args.disable_spectral else
                  "no_spatial"   if args.disable_spatial else
                  "concat_fusion"if args.fusion_mode=="concat" else
                  "full_gast")

    # Only include JSON‐serializable hyperparameters
    hp = {}
    for key, value in vars(args).items():
        if key in {"raw_gt", "cube_path", "gt_path"}:
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            hp[key] = value

    # Calculate model size in MB
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    # Calculate number of parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate training time in minutes
    train_time_minutes = round((time.time() - train_start_time) / 60, 2)
    print(f"Model size: {get_model_size(model):.2f} MB")
    print(f"Training time: {train_time_minutes} minutes")
    print(f"Number of parameters: {num_parameters}")

    history = {
        "dataset": args.dataset,
        "model_name": model_name,
        "hyperparameters": hp,
        "epoch_metrics": epoch_history,
        "best_val_OA": best_val,
        "test_OA": test_oa,
        "Model_size_MB": f"{get_model_size(model):.2f} MB",
        "training_time_minutes": train_time_minutes,
        "num_parameters": num_parameters
    }


    with open(Path(args.output_dir)/f"train_history_{args.dataset}_{model_name}.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"📊 Saved training history → {args.output_dir}")

    # ── 8. cleanup --------------------------------------------------------
    del model, optimiser, crit
    gc.collect()
    torch.cuda.empty_cache()
    return best_val, test_oa


def train_with_oom_handling(args, trial):
    try:
        return train(args, trial)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("⚠️ OOM: pruning trial / reduce batch size.")
            torch.cuda.empty_cache()
            raise TrialPruned()
        else:
            raise

if __name__ == "__main__":
    import argparse
    from src.data.data_loader import load_dataset, DATASET_PATHS
    from src.utils.utils import set_seed

    def resolve_dataset_paths(name):
        cfg = DATASET_PATHS[name]
        return str(PROJECT_ROOT/cfg["image"]), str(PROJECT_ROOT/cfg["ground_truth"])

    parser = argparse.ArgumentParser("GAST Ablation Training")

    # data
    parser.add_argument("--dataset", choices=list(DATASET_PATHS), default="SalinasA",
                        help="Dataset name (e.g., Indian_Pines, Botswana, SalinasA, etc.)")
    parser.add_argument("--train_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--patch_size",  type=int,   default=11)
    parser.add_argument("--stride",      type=int,   default=5)
    
    # model
    parser.add_argument("--embed_dim",      type=int, default=64)
    parser.add_argument("--gat_hidden_dim", type=int, default=64)
    parser.add_argument("--gat_heads",      type=int, default=8)
    parser.add_argument("--gat_depth",      type=int, default=4)
    parser.add_argument("--dropout",        type=float, default=0.2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    
    # ablation options
    parser.add_argument("--disable_spectral", action="store_true")
    parser.add_argument("--disable_spatial", action="store_true")
    parser.add_argument("--fusion_mode", choices=["gate", "concat", "spatial_only", "spectral_only"], default="gate")

    # training
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--epochs",       type=int, default=200)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--early_stop",   type=int,   default=20)
    

    # system
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--output_dir",   type=str, default="./models/checkpoints")
    parser.add_argument("--seed",         type=int, default=242)

    args = parser.parse_args()

    # reproducibility
    set_seed(args.seed)

    # load dataset
    img_arr, gt_arr = load_dataset(args.dataset)
    args.raw_gt = gt_arr
    args.cube_path, args.gt_path = resolve_dataset_paths(args.dataset)

    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # run training
    print("🚀 Starting training with the following settings:")
    train(args)
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    print("Training time:", (time.time() - train_start_time) / 60, "minutes")
    print("✅ Training completed successfully.")