from pathlib import Path

from NCA_pure import NeighborhoodComponentsAnalysisEmbedder

input_mode = "time"   # "time", "frequency", or "mixed"
nca_fit_samples = 60000
n_components = 32
init = "identity"
max_iter = 10
tol = 1e-5
random_state = 42
verbose = 1

# Folder containing this runner file
repo_root = Path(__file__).resolve().parent

# Upstream split roots
train_root = repo_root / "data" / "ecg" / "processed" / "v1tts" / "train"
test_root = repo_root / "data" / "ecg" / "processed" / "v1tts" / "test"

# Output root for saved embeddings and metadata
output_root = repo_root / "results" / "embeddings" / "ecg"

run_name = f"nca_{input_mode}_fit{nca_fit_samples}_dim{n_components}"

embedder = NeighborhoodComponentsAnalysisEmbedder(
    train_root=train_root,
    test_root=test_root,
    output_root=output_root,
    input_mode=input_mode,
    nca_fit_samples=nca_fit_samples,
    n_components=n_components,
    init=init,
    max_iter=max_iter,
    tol=tol,
    random_state=random_state,
    verbose=verbose,
)

X_train, y_train = embedder.load_train_split()
X_test, y_test = embedder.load_test_split()

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "Feature dimension mismatch between train and test after loading: "
        f"train has {X_train.shape[1]} columns, "
        f"test has {X_test.shape[1]} columns."
    )

X_train_scaled, X_test_scaled = embedder.standardize_train_test(X_train, X_test)

if n_components > X_train_scaled.shape[1]:
    raise ValueError(
        f"n_components={n_components} exceeds input feature dimension "
        f"{X_train_scaled.shape[1]}."
    )

subset_idx = embedder._make_subset_indices(y_train)
X_train_fit = X_train_scaled[subset_idx]
y_train_fit = y_train[subset_idx]

embedder.fit(X_train_fit, y_train_fit)

Z_train = embedder.transform(X_train_scaled)
Z_test = embedder.transform(X_test_scaled)

run_root = embedder.save_outputs(
    Z_train=Z_train,
    Z_test=Z_test,
    y_train=y_train,
    y_test=y_test,
    run_name=run_name,
    extra_config={
        "input_mode": input_mode,
        "nca_fit_samples": nca_fit_samples,
        "fit_subset_size": int(len(subset_idx)),
        "retrieval_pool": "full_train",
        "query_split": "full_test",
        "fit_policy": "NCA_on_standardized_train_subset_only",
    },
)

print("NeighborhoodComponentsAnalysis run complete.")
print(f"Repo root:        {repo_root}")
print(f"Train root:       {train_root}")
print(f"Test root:        {test_root}")
print(f"Input mode:       {input_mode}")
print(f"Train X shape:    {X_train.shape}")
print(f"Test X shape:     {X_test.shape}")
print(f"Train scaled:     {X_train_scaled.shape}")
print(f"Test scaled:      {X_test_scaled.shape}")
print(f"Fit subset shape: {X_train_fit.shape}")
print(f"Train Z shape:    {Z_train.shape}")
print(f"Test Z shape:     {Z_test.shape}")
print(f"Saved to:         {run_root}")