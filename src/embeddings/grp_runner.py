from pathlib import Path

from gaussian_random_projection import GaussianRandomProjectionEmbedder

time = True # True for frequency domain, False for time domain
n_components = 128
random_state = 42

# Get repo root relative to this runner file.
# Assumes this runner lives at repo_root/src/embedding/
repo_root = Path(__file__).resolve().parents[2]

# Upstream split roots
train_root = repo_root / "data" / "ecg" / "processed" / "v1tts" / "train"
test_root = repo_root / "data" / "ecg" / "processed" / "v1tts" / "test"

# Output root for saved embeddings and metadata
output_root = repo_root / "results" / "embeddings" / "ecg"

# Current run config
if time:
    representation = "time"
    extract_fn_name = "TD_extractor"
else:
    representation = "frequency"
    extract_fn_name = "FD_extractor"

run_name = f"grp_{representation}_{n_components:03d}"


embedder = GaussianRandomProjectionEmbedder(
    train_root=train_root,
    test_root=test_root,
    output_root=output_root,
    n_components=n_components,
    random_state=random_state,
)

X_train, y_train = embedder.load_train_split(extract_fn_name)
X_test, y_test = embedder.load_test_split(extract_fn_name)

Z_train = embedder.fit_transform(X_train)
Z_test = embedder.transform(X_test)

run_root = embedder.save_outputs(
    Z_train=Z_train,
    Z_test=Z_test,
    y_train=y_train,
    y_test=y_test,
    run_name=run_name,
    extra_config={
        "representation": representation,
        "extract_fn_name": extract_fn_name,
    },
)

print("GaussianRandomProjection run complete.")
print(f"Repo root:       {repo_root}")
print(f"Train root:      {train_root}")
print(f"Test root:       {test_root}")
print(f"Representation:  {representation}")
print(f"Train X shape:   {X_train.shape}")
print(f"Test X shape:    {X_test.shape}")
print(f"Train Z shape:   {Z_train.shape}")
print(f"Test Z shape:    {Z_test.shape}")
print(f"Saved to:        {run_root}")