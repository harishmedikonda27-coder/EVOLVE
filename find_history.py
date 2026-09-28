import json

transcript_path = r"C:\Users\Aryan\.gemini\antigravity-ide\brain\1cd84182-01d3-4d15-860c-2213b030154f\.system_generated\logs\transcript_full.jsonl"
output_path = r"E:\EVOLVE\frontend\components\environment\EnvironmentCanvas.tsx.history"

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

found_count = 0
for row in reversed(lines):
    try:
        data = json.loads(row)
    except:
        continue
    
    if data.get('type') == 'PLANNER_RESPONSE':
        tool_calls = data.get('tool_calls', [])
        for call in tool_calls:
            if call.get('name') == 'write_to_file':
                args = call.get('args', {})
                if 'EnvironmentCanvas.tsx' in args.get('TargetFile', ''):
                    found_count += 1
                    if found_count == 2:  # Get the second to last write
                        print(f"Found write_to_file! Count: {found_count}")
                        with open(output_path, 'w', encoding='utf-8') as out:
                            out.write(args.get('CodeContent', ''))
                        exit(0)
