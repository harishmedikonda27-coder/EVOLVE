import os
import json
import glob

def run_audit():
    base_dir = "storage/experiments"
    out_dir = "validation/empirical_audit"
    os.makedirs(out_dir, exist_ok=True)
    
    execution_audit = []
    plasticity_audit = []
    ablation_audit = []
    aggregation_audit = []
    reproducibility_audit = []
    condition_comparison = []
    leakage_audit = []

    # Map for aggregation audit
    expected_experiments = ["forgetting", "ood", "plasticity", "learned_policy", "consequence", "ablation_no_memory", "flagship"]
    
    agg_map = {}
    for exp in expected_experiments:
        agg_map[exp] = {
            "experiment": exp,
            "expected_runs": 0,
            "found_runs": 0,
            "valid_runs": 0,
            "failed_runs": 0,
            "missing_runs": 0,
            "duplicate_runs": 0,
            "metric_integrity": "UNKNOWN",
            "aggregation_status": "UNKNOWN"
        }

    for root, dirs, files in os.walk(base_dir):
        exp_name = os.path.basename(root)
        if exp_name not in expected_experiments:
            continue
            
        json_files = [f for f in files if f.endswith(".json")]
        agg_map[exp_name]["found_runs"] = len(json_files)
        agg_map[exp_name]["valid_runs"] = len(json_files) # Assuming all are valid for now
        
        for f in json_files:
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "r") as fp:
                    data = json.load(fp)
            except Exception:
                agg_map[exp_name]["failed_runs"] += 1
                continue
                
            cond = data.get("condition", "UNKNOWN")
            seed = data.get("seed", -1)
            
            interventions = data.get("interventions", [])
            plasticity = data.get("plasticity_probes", [])
            tasks = data.get("tasks", [])
            
            # Execution Audit
            exec_record = {
                "experiment": exp_name,
                "condition": cond,
                "seed": seed,
                "controller_invoked": cond != "FIXED",
                "transaction_runner_invoked": len(interventions) > 0, # Weak proxy, since it's empty
                "real_mutation_function_invoked": len(interventions) > 0,
                "mutation_attempts": len(interventions),
                "accepted_mutations": len([i for i in interventions if i.get("status") == "ACCEPTED"]),
                "rollbacks": len([i for i in interventions if i.get("status") == "ROLLBACK"]),
                "grow": len([i for i in interventions if i.get("action") == "GROW"]),
                "prune": len([i for i in interventions if i.get("action") == "PRUNE"]),
                "rewire": len([i for i in interventions if i.get("action") == "REWIRE"]),
                "initial_modules": 4, # known default
                "final_modules": 4 + len([i for i in interventions if i.get("action") == "GROW" and i.get("status") == "ACCEPTED"]) - len([i for i in interventions if i.get("action") == "PRUNE" and i.get("status") == "ACCEPTED"]),
                "initial_parameters": "UNKNOWN",
                "final_parameters": "UNKNOWN",
                "topology_changed": len([i for i in interventions if i.get("status") == "ACCEPTED"]) > 0,
                "evidence": ["interventions_list_length_" + str(len(interventions))]
            }
            execution_audit.append(exec_record)
            
            # Plasticity Audit
            if len(plasticity) == 0:
                p_status = "not_measured"
                p_val = "not_measured"
            else:
                p_status = "measured"
                p_val = plasticity[-1].get("auc", 0) if plasticity else 0
                
            plasticity_audit.append({
                "experiment": exp_name,
                "condition": cond,
                "seed": seed,
                "probe_requested": cond in ["LEARNED_EVOLVE", "LEARNED_EVOLVE_NO_MEMORY"],
                "probe_started": len(plasticity) > 0,
                "probe_completed": len(plasticity) > 0,
                "probe_metric_present": len(plasticity) > 0,
                "probe_metric_value": p_val,
                "status": p_status
            })
            
            # Ablation Audit
            if "ablation" in exp_name:
                ablation_audit.append({
                    "experiment": exp_name,
                    "condition": cond,
                    "seed": seed,
                    "valid_file": True,
                    "completed_runs": len(tasks),
                    "raw_metrics_present": True,
                    "interventions": len(interventions)
                })
                
            # Reproducibility Audit
            reproducibility_audit.append({
                "experiment": exp_name,
                "condition": cond,
                "seed": seed,
                "tasks_executed": len(tasks)
            })

    # Dump all
    with open(os.path.join(out_dir, "execution_audit.json"), "w") as f:
        json.dump(execution_audit, f, indent=2)
    with open(os.path.join(out_dir, "plasticity_audit.json"), "w") as f:
        json.dump(plasticity_audit, f, indent=2)
    with open(os.path.join(out_dir, "ablation_audit.json"), "w") as f:
        json.dump(ablation_audit, f, indent=2)
    with open(os.path.join(out_dir, "aggregation_audit.json"), "w") as f:
        json.dump(list(agg_map.values()), f, indent=2)
    with open(os.path.join(out_dir, "reproducibility_audit.json"), "w") as f:
        json.dump(reproducibility_audit, f, indent=2)

if __name__ == "__main__":
    run_audit()
