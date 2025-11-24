import os
import json
from datasets import load_dataset

# 1. Ensure output directory exists
os.makedirs("data/train", exist_ok=True)

# 2. Load dataset from HuggingFace (automatically handles all parquet shards)
ds = load_dataset("Fate-Zero/Archer2.0-Code-1.5B", split="train")

# 3. Output file
output_path = "data/train/archer2.0-code-1.5b-train.json"

print(f"Dataset length: {len(ds)}")
print(f"Writing to: {output_path}")

# 4. Stream to JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done.")
