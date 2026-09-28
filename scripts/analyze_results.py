import os
import json
import csv
import glob
import numpy as np

EXPERIMENT_DIRS = [
    "forgetting",
    "ood",
    "plasticity",
    "learned_policy",
    "consequence",
    "ablation_no_memory",
    "flagship"
]

def load_data():
    data = {}
    base_dir = "storage/experiments"
    for exp_dir in EXPERIMENT_DIRS:
        path = os.path.join(base_dir, exp_dir)
        if not os.path.exists(path):
            continue
            
        data[exp_dir] = {}
        for f in glob.glob(os.path.join(path, "*.json")):
            with open(f, "r") as fp:
                content = json.load(fp)
                
            cond = content.get("condition")
            seed = content.get("seed")
            
            if cond not in data[exp_dir]:
                data[exp_dir][cond] = []
                
            data[exp_dir][cond].append(content)
            
    return data

def safe_mean(vals):
    return float(np.mean(vals)) if len(vals) > 0 else 0.0

def safe_std(vals):
    return float(np.std(vals)) if len(vals) > 0 else 0.0

def analyze():
    data = load_data()
    results = {}
    
    for exp, conds in data.items():
        results[exp] = {}
        for cond, runs in conds.items():
            # Aggregate tasks
            # Assuming all runs have same task lengths
            num_runs = len(runs)
            if num_runs == 0:
                continue
                
            num_tasks = len(runs[0]["tasks"])
            
            agg = {
                "num_seeds": num_runs,
                "seeds": [r["seed"] for r in runs],
                "avg_final_performance": 0.0,
                "avg_forgetting": 0.0,
                "avg_retention": 0.0,
                "avg_plasticity_auc": 0.0,
                "task_breakdown": []
            }
            
            all_final_perfs = []
            all_forgetting = []
            all_retention = []
            all_plasticity = []
            
            for t_idx in range(num_tasks):
                task_finals = [r["tasks"][t_idx]["final_performance"] for r in runs if len(r["tasks"]) > t_idx]
                task_forgets = [r["tasks"][t_idx]["forgetting"] for r in runs if len(r["tasks"]) > t_idx]
                task_retents = [r["tasks"][t_idx]["retention"] for r in runs if len(r["tasks"]) > t_idx]
                
                agg["task_breakdown"].append({
                    "task_id": runs[0]["tasks"][t_idx]["task_id"],
                    "final_performance_mean": safe_mean(task_finals),
                    "final_performance_std": safe_std(task_finals),
                    "forgetting_mean": safe_mean(task_forgets),
                    "forgetting_std": safe_std(task_forgets),
                    "retention_mean": safe_mean(task_retents),
                    "retention_std": safe_std(task_retents)
                })
                
                all_final_perfs.extend(task_finals)
                all_forgetting.extend(task_forgets)
                all_retention.extend(task_retents)
                
            for r in runs:
                if "plasticity_probes" in r and r["plasticity_probes"]:
                    all_plasticity.extend([p["auc"] for p in r["plasticity_probes"]])
                    
            agg["avg_final_performance"] = safe_mean(all_final_perfs)
            agg["avg_forgetting"] = safe_mean(all_forgetting)
            agg["avg_retention"] = safe_mean(all_retention)
            agg["avg_plasticity_auc"] = safe_mean(all_plasticity) if all_plasticity else 0.0
            
            results[exp][cond] = agg
            
    return results

def main():
    results = analyze()
    
    os.makedirs("storage", exist_ok=True)
    with open("storage/final_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Write CSV summary
    with open("storage/final_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment", 
            "Condition", 
            "Seeds", 
            "Avg Final Performance", 
            "Avg Forgetting", 
            "Avg Retention", 
            "Avg Plasticity AUC"
        ])
        
        for exp, conds in results.items():
            for cond, agg in conds.items():
                writer.writerow([
                    exp,
                    cond,
                    str(agg["seeds"]),
                    f'{agg["avg_final_performance"]:.4f}',
                    f'{agg["avg_forgetting"]:.4f}',
                    f'{agg["avg_retention"]:.4f}',
                    f'{agg["avg_plasticity_auc"]:.4f}'
                ])
                
if __name__ == "__main__":
    main()
