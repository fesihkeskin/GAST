# src/data/downloader.py

import os
import requests
import gdown

# List of dataset URLs
urls = [
    # Indian Pines
    "https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
    "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
    "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    
    # Salinas
    "https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
    "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
    "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
    
    # SalinasA
    "https://www.ehu.eus/ccwintco/uploads/d/df/SalinasA.mat",
    "https://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat",
    "https://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat",
    
    # Pavia Centre
    "https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
    "https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
    
    # Pavia University
    "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
    "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
    
    # Kennedy_Space_Center
    "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
    "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
    
    # Botswana
    "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
    "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
    
    # Houston (UH)
    "http://hyperspectral.ee.uh.edu/2egc4t4jd76fd32/Houston.mat",
    "http://hyperspectral.ee.uh.edu/2egc4t4jd76fd32/Houston_gt.mat"
]

# Google Drive download for Kennedy_Space_Center_corrected.mat
GDRIVE_Kennedy_Space_Center_CORRECTED = "https://drive.google.com/uc?id=1RJWBqMV2ApBaGgQtqmQ4OYGDRFVIliZI"

# Mapping: dataset → list of expected file names
dataset_folders = {
    "Indian_Pines": [
        "Indian_pines.mat",
        "Indian_pines_corrected.mat",
        "Indian_pines_gt.mat"
    ],
    "Salinas": [
        "Salinas.mat",
        "Salinas_corrected.mat",
        "Salinas_gt.mat"
    ],
    "SalinasA": [
        "SalinasA.mat",
        "SalinasA_corrected.mat",
        "SalinasA_gt.mat"
    ],
    "Pavia_Centre": [
        "Pavia_Centre.mat",
        "Pavia_Centre_gt.mat"
    ],
    "Pavia_University": [
        "Pavia_University.mat",
        "Pavia_University_gt.mat"
    ],
    "Kennedy_Space_Center": [
        "Kennedy_Space_Center.mat",
        "Kennedy_Space_Center_corrected.mat",
        "Kennedy_Space_Center_gt.mat"
    ],
    "Botswana": [
        "Botswana.mat",
        "Botswana_gt.mat"
    ],
    "Houston": [
        "Houston.mat",
        "Houston_gt.mat"
    ]
}

# Original filename to target filename mapping (for normalization)
rename_map = {
    "Pavia.mat": "Pavia_Centre.mat",
    "Pavia_gt.mat": "Pavia_Centre_gt.mat",
    "PaviaU.mat": "Pavia_University.mat",
    "PaviaU_gt.mat": "Pavia_University_gt.mat",
    "KSC.mat": "Kennedy_Space_Center.mat",
    "KSC_gt.mat": "Kennedy_Space_Center_gt.mat",
    # For the one from Google Drive
    "KSC_corrected.mat": "Kennedy_Space_Center_corrected.mat",
}

def get_dataset_dir() -> str:
    """Return the absolute path to the Dataset directory."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "Dataset")

def download_file(url: str, save_path: str):
    """Download file via HTTP and save to the specified path."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[✅] Downloaded: {os.path.basename(save_path)}")
    except requests.RequestException as e:
        print(f"[❌] Failed to download {url}\n    Reason: {e}")

def download_gdrive_file(url: str, save_path: str):
    """Download a file from Google Drive using gdown."""
    try:
        gdown.download(url, save_path, quiet=False)
        print(f"[✅] Downloaded (GDrive): {os.path.basename(save_path)}")
    except Exception as e:
        print(f"[❌] GDrive download failed for {url}\n    Reason: {e}")

def prepare_directories(base_dir: str):
    """Create necessary directories for each dataset."""
    for folder in dataset_folders.keys():
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def match_folder(filename: str) -> str:
    """Find the correct folder name given a filename."""
    for folder, files in dataset_folders.items():
        if filename in files:
            return folder
    return None

def download_datasets():
    """Main function to download and organize all dataset files."""
    base_dir = get_dataset_dir()
    prepare_directories(base_dir)

    for url in urls:
        orig_name = os.path.basename(url)
        final_name = rename_map.get(orig_name, orig_name)
        folder = match_folder(final_name)

        if folder:
            save_path = os.path.join(base_dir, folder, final_name)
            if not os.path.exists(save_path):
                download_file(url, save_path)
            else:
                print(f"[=] Already exists: {final_name}")
        else:
            print(f"[!] Warning: No match found for {orig_name}")

    # Special case: Google Drive file
    gdrive_path = os.path.join(base_dir, "Kennedy_Space_Center", "Kennedy_Space_Center_corrected.mat")
    if not os.path.exists(gdrive_path):
        download_gdrive_file(GDRIVE_Kennedy_Space_Center_CORRECTED, gdrive_path)
    else:
        print(f"[=] Already exists: Kennedy_Space_Center_corrected.mat")

    print("\n✅ All downloads complete. ✅")

if __name__ == "__main__":
    download_datasets()