with open("E:/EVOLVE/backend/experiments/run.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i >= 181: # line 182 is index 181
        if line.startswith("            "):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open("E:/EVOLVE/backend/experiments/run.py", "w") as f:
    f.writelines(new_lines)
