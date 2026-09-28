import json
import platform
import sys
import torch
import hashlib
import os

def get_hash(path):
    if not os.path.exists(path):
        return "N/A"
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

manifest = {
    "git_commit": "N/A",  # Not a git repo as checked earlier
    "repository_version": "2.0-research",
    "python_version": sys.version.split()[0],
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "os": platform.platform(),
    "config_hash": get_hash("configs/base.yaml"),
    "environment_hash": get_hash("environment/world.py"),
    "experiment_hash": get_hash("backend/experiments/configs/flagship.json")
}

with open("backend_manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)
    
print("Manifest generated successfully.")
