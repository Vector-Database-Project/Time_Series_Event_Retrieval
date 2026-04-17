from pathlib import Path

from supervised_umap_embedder import SupervisedUMAPEmbedder

input_mode = "time"
n_components = 128
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"
target_metric = "categorical"
low_memory = True
verbose = True
n_jobs = -1

repo_root = Path(__file__).resolve().parents[2]

train_root = repo_root / "data" / "ecg" / "processed" / "v1tts" / "train"
test_root = repo_root / "data" / "ecg" / "processed" / "v1tts" / "test"

output_root = repo_root / "results" / "embeddings" / "ecg"

run_name = f"sup_umap_{input_mode}_nn{n_neighbors}_dim{n_components}"

embedder = SupervisedUMAPEmbedder(
    train_root=train_root,
    test_root=test_root,
    output_root=output_root,
    input_mode=input_mode,
    n_components=n_components,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric=metric,
    target_metric=target_metric,
    low_memory=low_memory,
    verbose=verbose,
    n_jobs=n_jobs,
)

X_train, y_train = embedder.load_train_split()
X_test, y_test = embedder.load_test_split()

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "Feature dimension mismatch between train and test after loading: "
        f"train has {X_train.shape[1]} columns, "
        f"test has {X_test.shape[1]} columns."
    )

Z_train = embedder.fit_transform(X_train, y_train)
Z_test = embedder.transform(X_test)

run_root = embedder.save_outputs(
    Z_train=Z_train,
    Z_test=Z_test,
    y_train=y_train,
    y_test=y_test,
    run_name=run_name,
    extra_config={
        "retrieval_pool": "full_train",
        "query_split": "full_test",
        "fit_policy": "Supervised_UMAP_fit_on_full_train_transform_full_train_test",
    },
)

print("Supervised UMAP run complete.")
print(f"Repo root:      {repo_root}")
print(f"Train root:     {train_root}")
print(f"Test root:      {test_root}")
print(f"Input mode:     {input_mode}")
print(f"Train X shape:  {X_train.shape}")
print(f"Test X shape:   {X_test.shape}")
print(f"Train Z shape:  {Z_train.shape}")
print(f"Test Z shape:   {Z_test.shape}")
print(f"Saved to:       {run_root}")
