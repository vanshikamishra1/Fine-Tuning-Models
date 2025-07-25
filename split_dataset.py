import json
from sklearn.model_selection import train_test_split

with open("formatted_dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

train, eval = train_test_split(data, test_size=0.2, random_state=42)

with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with open("eval.jsonl", "w", encoding="utf-8") as f:
    for item in eval:
        f.write(json.dumps(item) + "\n")

print("Train/Eval split complete.")