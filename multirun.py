import subprocess
import os

import sys
from pathlib import Path

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration for each dataset (matches with optimum CLI exactly)
configs = [
    {
        "dataset": "Botswana",
        "batch_size": 64,
        "patch_size": 13,
        "stride": 3,
        "lr": 0.0004224382351548547,
        "weight_decay": 0.00011150299441886956,
        "dropout": 0.25,
        "embed_dim": 128,
        "gat_hidden_dim": 64,
        "gat_heads": 4,
        "gat_depth": 2,
        "transformer_heads": 8,
        "transformer_layers": 9,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "Houston13",
        "batch_size": 16,
        "patch_size": 13,
        "stride": 2,
        "lr": 0.00024092273071567313,
        "weight_decay": 8.500203317456351e-08,
        "dropout": 0.15,
        "embed_dim": 128,
        "gat_hidden_dim": 128,
        "gat_heads": 4,
        "gat_depth": 2,
        "transformer_heads": 8,
        "transformer_layers": 4,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "Indian_Pines",
        "batch_size": 48,
        "patch_size": 7,
        "stride": 3,
        "lr": 0.0001906390094523524,
        "weight_decay": 0.009021540956272492,
        "dropout": 0.1,
        "embed_dim": 128,
        "gat_hidden_dim": 32,
        "gat_heads": 2,
        "gat_depth": 8,
        "transformer_heads": 8,
        "transformer_layers": 6,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "Kennedy_Space_Center",
        "batch_size": 64,
        "patch_size": 9,
        "stride": 8,
        "lr": 0.0005940732289188326,
        "weight_decay": 0.0008803995798938349,
        "dropout": 0.25,
        "embed_dim": 256,
        "gat_hidden_dim": 64,
        "gat_heads": 10,
        "gat_depth": 6,
        "transformer_heads": 2,
        "transformer_layers": 4,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "Pavia_Centre",
        "batch_size": 64,
        "patch_size": 13,
        "stride": 4,
        "lr": 0.0001236480816653455,
        "weight_decay": 4.102919947676974e-07,
        "dropout": 0.45,
        "embed_dim": 256,
        "gat_hidden_dim": 64,
        "gat_heads": 4,
        "gat_depth": 4,
        "transformer_heads": 16,
        "transformer_layers": 3,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "Pavia_University",
        "batch_size": 64,
        "patch_size": 11,
        "stride": 4,
        "lr": 0.00015871160049527858,
        "weight_decay": 0.0009538211768148025,
        "dropout": 0.2,
        "embed_dim": 64,
        "gat_hidden_dim": 32,
        "gat_heads": 4,
        "gat_depth": 4,
        "transformer_heads": 16,
        "transformer_layers": 10,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "Salinas",
        "batch_size": 32,
        "patch_size": 13,
        "stride": 4,
        "lr": 0.00023614474992419862,
        "weight_decay": 0.0007723031252522047,
        "dropout": 0.15,
        "embed_dim": 128,
        "gat_hidden_dim": 32,
        "gat_heads": 10,
        "gat_depth": 4,
        "transformer_heads": 2,
        "transformer_layers": 2,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
    {
        "dataset": "SalinasA",
        "batch_size": 48,
        "patch_size": 11,
        "stride": 6,
        "lr": 0.0003376670734565324,
        "weight_decay": 8.662401208300589e-08,
        "dropout": 0.0,
        "embed_dim": 256,
        "gat_hidden_dim": 32,
        "gat_heads": 4,
        "gat_depth": 8,
        "transformer_heads": 16,
        "transformer_layers": 10,
        "epochs": 500,
        "early_stop": 50,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "fusion_mode": "gate",
    },
]

# # Configuration for each dataset (matches with optimum CLI exactly)
# configs = [
#     {
#         "dataset": "Pavia_University",
#         "batch_size": 64,
#         "patch_size": 11,
#         "stride": 4,
#         "lr": 0.00015871160049527858,
#         "weight_decay": 0.0009538211768148025,
#         "dropout": 0.2,
#         "embed_dim": 64,
#         "gat_hidden_dim": 32,
#         "gat_heads": 4,
#         "gat_depth": 4,
#         "transformer_heads": 16,
#         "transformer_layers": 10,
#         "epochs": 500,
#         "early_stop": 50,
#         "train_ratio": 0.05,
#         "val_ratio": 0.05,
#         "fusion_mode": "gate",
#     }
# ]



# Seeds for multiple runs
seeds = list(range(252,302,10))

for config in configs:
    dataset = config["dataset"]
    print(f"\n=== Dataset: {dataset} ===")
    out_dir = os.path.join(PROJECT_ROOT, "models", "final", "gast", dataset)
    os.makedirs(out_dir, exist_ok=True)
    test_results_dir = os.path.join(out_dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)

    for seed in seeds:
        result_file = os.path.join(test_results_dir, f"metrics_seed_{seed}.json")
        print(f"Result file: {result_file}")
        if os.path.exists(result_file):
            print(f" → Result already exists, skipped for seed={seed}.")
            continue

        # 1) TRAIN
        train_cmd = [
            "python", "main.py",
            "--mode", "train",
            "--dataset", dataset,
            "--train_ratio", str(config["train_ratio"]),
            "--val_ratio", str(config["val_ratio"]),
            "--stride", str(config["stride"]),
            "--epochs", str(config["epochs"]),
            "--early_stop", str(config["early_stop"]),
            "--batch_size", str(config["batch_size"]),
            "--patch_size", str(config["patch_size"]),
            "--lr", str(config["lr"]),
            "--weight_decay", str(config["weight_decay"]),
            "--dropout", str(config["dropout"]),
            "--embed_dim", str(config["embed_dim"]),
            "--gat_hidden_dim", str(config["gat_hidden_dim"]),
            "--gat_heads", str(config["gat_heads"]),
            "--gat_depth", str(config["gat_depth"]),
            "--transformer_heads", str(config["transformer_heads"]),
            "--transformer_layers", str(config["transformer_layers"]),
            "--seed", str(seed),
            "--num_workers", "4",
            "--fusion_mode", config["fusion_mode"],
            "--output_dir", out_dir,
        ]

        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {dataset}, seed={seed}. Skipping test. Error: {e}")
            continue

        test_cmd = train_cmd.copy()
        mode_idx = test_cmd.index("--mode") + 1
        test_cmd[mode_idx] = "test"

        checkpoint = os.path.join(out_dir, f"gast_best_{dataset}.pth")
        if not os.path.exists(checkpoint):
            print(f"Checkpoint {checkpoint} not found for {dataset}, seed={seed}. Skipping test.")
            continue
        test_cmd += ["--checkpoint", checkpoint]

        try:
            subprocess.run(test_cmd, check=True)
            print(f"    → Test completed for seed={seed}")
        except subprocess.CalledProcessError as e:
            print(f"Test failed for {dataset}, seed={seed}. Error: {e}")

# os.system("sudo /sbin/shutdown -h now")