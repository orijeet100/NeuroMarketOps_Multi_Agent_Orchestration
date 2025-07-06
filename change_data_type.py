import json

with open('train_final.jsonl', 'r') as f:
    with open('train_fixed.jsonl', 'w') as out:
        for line in f:
            data = json.loads(line)
            fixed = {
                "instruction": data.get("instruction", ""),
                "context": data.get("input", ""),  # or empty string
                "response": data.get("output", data.get("response", ""))
            }
            out.write(json.dumps(fixed) + '\n')