import sys
import os

def main():
    print("========================================")
    print("EVOLVE ENVIRONMENT VERIFICATION")
    print("========================================")

    all_passed = True

    # 1. Basic System & Interpreter Info
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")

    # 2. PyTorch & CUDA Verification
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else "N/A"
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    except Exception as e:
        torch_ver = f"FAIL ({e})"
        cuda_avail = False
        cuda_ver = "N/A"
        gpu_name = "N/A"
        all_passed = False

    print(f"PyTorch version: {torch_ver}")
    print(f"CUDA available: {cuda_avail}")
    print(f"CUDA version: {cuda_ver}")
    print(f"GPU: {gpu_name}")

    # 3. Core Scientific & Machine Learning Stack
    deps = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("SciPy", "scipy"),
        ("Scikit-learn", "sklearn"),
        ("FastAPI", "fastapi"),
        ("Gymnasium", "gymnasium"),
    ]

    for label, mod_name in deps:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "OK")
            print(f"{label}: {ver}")
        except Exception as e:
            print(f"{label}: FAIL ({e})")
            all_passed = False

    # 4. EVOLVE Packages Verification
    print("\nEVOLVE packages:")
    evolve_pkgs = ["brain", "evolution", "environment", "training", "backend", "storage"]

    # Ensure project root is in sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for pkg in evolve_pkgs:
        try:
            __import__(pkg)
            print(f"{pkg}: OK")
        except Exception as e:
            print(f"{pkg}: FAIL ({e})")
            all_passed = False

    print("\nFinal status:")
    if all_passed:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
