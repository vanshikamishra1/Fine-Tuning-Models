# Format dataset for fine-tuning
import json

input_file = "dataset.jsonl"
output_file = "formatted_dataset.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        prompt = f"User: {data['user']}\nAssistant: {data['assistant']}"
        json.dump({"text": prompt}, fout)
        fout.write("\n")
print("Dataset formatted successfully.")