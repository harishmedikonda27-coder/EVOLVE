import re

log_path = r"C:\Users\Aryan\.gemini\antigravity-ide\brain\1cd84182-01d3-4d15-860c-2213b030154f\.system_generated\tasks\task-4848.log"
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    log_content = f.read()

# Stats
mutations_attempted = len(re.findall(r"-> Mutating:", log_content))
grow_attempted = len(re.findall(r"-> Mutating: GROW", log_content))
rewire_attempted = len(re.findall(r"-> Mutating: REWIRE", log_content))
prune_attempted = len(re.findall(r"-> Mutating: PRUNE", log_content))

mutations_accepted = len(re.findall(r"\[ACCEPTED\]", log_content))
mutations_rollback = len(re.findall(r"\[ROLLBACK\]", log_content))

grow_accepted = len(re.findall(r"\[ACCEPTED\] GROW", log_content))
rewire_accepted = len(re.findall(r"\[ACCEPTED\] REWIRE", log_content))
prune_accepted = len(re.findall(r"\[ACCEPTED\] PRUNE", log_content))

converged = len(re.findall(r"Status:    CONVERGED", log_content))
exhausted = len(re.findall(r"Status:    EXHAUSTED", log_content))

gpu_usage = re.findall(r"Selected device: (.*)", log_content)
gpu_usage = list(set(gpu_usage))

print(f"Total Mutations Attempted: {mutations_attempted}")
print(f"  GROW: {grow_attempted} (Accepted: {grow_accepted})")
print(f"  REWIRE: {rewire_attempted} (Accepted: {rewire_accepted})")
print(f"  PRUNE: {prune_attempted} (Accepted: {prune_accepted})")
print(f"Total Mutations Rollback: {mutations_rollback}")
print(f"Convergence: {converged} CONVERGED, {exhausted} EXHAUSTED")
print(f"GPU Usage: {gpu_usage}")
