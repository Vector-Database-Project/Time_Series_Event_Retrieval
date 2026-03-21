from pathlib import Path
from test_train_splitter import TestTrainSplitter

# Get the repo root relative to this runner file.
# This assumes the runner lives at repo_root/src/data_processing/
repo_root = Path(__file__).resolve().parents[2]

# Build dataset paths relative to the repo root.
processed_root = repo_root / "data" / "ecg" / "processed" / "v1"
output_root = repo_root / "data" / "ecg" / "processed" / "v1tts"

test_size = 0.2
random_state = 42
max_shard_mb = 40

splitter = TestTrainSplitter(
    processed_root=processed_root,
    output_root=output_root,
    test_size=test_size,
    random_state=random_state,
    max_shard_mb=max_shard_mb,
)

splitter.split_and_save()

print("Train-test split complete.")
print(f"Repo root: {repo_root}")
print(f"Input:     {processed_root}")
print(f"Output:    {output_root}")