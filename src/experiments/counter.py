import numpy as np
from collections import Counter
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
file_path = repo_root / "data" / "ecg" / "processed" / "v1" / "labels" / "label_shard.npz"
label_key = "data"   # change if your npz uses a different key

data = np.load(file_path, allow_pickle=True)

print("Available keys:")
for k in data.files:
    print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")

labels = data[label_key].reshape(-1)

counts = Counter(labels.tolist())
total = len(labels)

print(f"\nUsing label key: {label_key}")
print(f"Total samples: {total}")
print("\nClass frequency:")

for cls, cnt in sorted(counts.items(), key=lambda x: x[0]):
    print(f"Class {cls}: {cnt} ({100 * cnt / total:.2f}%)")