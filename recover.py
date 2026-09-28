import json
import re

transcript_path = r"C:\Users\Aryan\.gemini\antigravity-ide\brain\1cd84182-01d3-4d15-860c-2213b030154f\.system_generated\logs\transcript_full.jsonl"
output_path = r"E:\EVOLVE\frontend\components\environment\EnvironmentCanvas.tsx.recovered"

lines = {}

with open(transcript_path, 'r', encoding='utf-8') as f:
    for row in f:
        try:
            data = json.loads(row)
        except:
            continue
        if data.get('type') == 'VIEW_FILE' and 'EnvironmentCanvas.tsx' in data.get('content', ''):
            content = data['content']
            for line in content.split('\n'):
                m = re.match(r'^(\d+):\s(.*)$', line)
                if m:
                    lines[int(m.group(1))] = m.group(2)

with open(output_path, 'w', encoding='utf-8') as f:
    for i in range(1, max(lines.keys()) + 1):
        f.write(lines.get(i, f"// MISSING LINE {i}") + '\n')
print(f"Recovered {len(lines)} lines to {output_path}")
