import os
import subprocess
from src.data.data_loader import DATASET_NAME_LIST

# If you want to run t-SNE for specific datasets, uncomment the line below
DATASET_NAME_LIST = ["Pavia_University"]


for dname in DATASET_NAME_LIST:
    checkpoint = f"models/final/gast/{dname}/gast_best_{dname}.pth"
    print(f"\n=== Running t-SNE for {dname} ===")
    subprocess.run([
        "python", "src/training/tsne_plot.py",
        "--dataset", dname,
        "--checkpoint", checkpoint
    ], check=True)

# os.system("sudo /sbin/shutdown -h now")