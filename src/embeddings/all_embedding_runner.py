from pathlib import Path
import json
import traceback

import numpy as np
from tqdm.auto import tqdm

from data_extractor import DataExtractor
from gaussian_random_projection import GaussianRandomProjectionEmbedder
from umap_embedder import UMAPEmbedder
from supervised_umap_embedder import SupervisedUMAPEmbedder


# ============================================================
# Global run configuration
# ============================================================
EMBEDDING_DIMS = [64, 128]
INPUT_MODES = ["time", "frequency", "mixed"]
OVERWRITE_EXISTING = False

# GRP params
GRP_RANDOM_STATE = 42

# UMAP params
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
UMAP_LOW_MEMORY = False
UMAP_VERBOSE = True
UMAP_N_JOBS = -1
UMAP_TARGET_METRIC = "categorical"


# ============================================================
# Repo paths
# Assumes this runner lives at repo_root/src/embedding/
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "data" / "ecg" / "processed" / "v2" / "train"
TEST_ROOT = REPO_ROOT / "data" / "ecg" / "processed" / "v2" / "test"
OUTPUT_ROOT = REPO_ROOT / "results" / "embeddings" / "ecg_v2"


# ============================================================
# Shared loading helpers
# ============================================================
def discover_shard_indices(split_root: Path):
    label_dir = split_root / "labels"

    if not label_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {label_dir}")

    shard_files = sorted(label_dir.glob("label_shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No label shards found in: {label_dir}")

    shard_indices = []
    for path in shard_files:
        shard_str = path.stem.split("_")[-1]
        shard_indices.append(int(shard_str))

    return shard_indices


def validate_X_y(X_shard, y_shard, split_name, shard_idx, source_name):
    if X_shard.shape[0] != y_shard.shape[0]:
        raise ValueError(
            f"Mismatch in {split_name} shard {shard_idx:03d} ({source_name}): "
            f"X has {X_shard.shape[0]} samples, y has {y_shard.shape[0]} labels."
        )


def load_single_mode_split(extractor, extract_fn_name, split_root, split_name):
    if not hasattr(extractor, extract_fn_name):
        raise AttributeError(
            f"{extractor.__class__.__name__} has no method '{extract_fn_name}'."
        )

    extract_fn = getattr(extractor, extract_fn_name)
    shard_indices = discover_shard_indices(split_root)

    X_parts = []
    y_parts = []

    for shard_idx in tqdm(
        shard_indices,
        total=len(shard_indices),
        desc=f"Loading {split_name} shards ({extract_fn_name})",
        unit="shard",
    ):
        X_shard, y_shard = extract_fn(shard_idx)
        validate_X_y(X_shard, y_shard, split_name, shard_idx, extract_fn_name)
        X_parts.append(X_shard)
        y_parts.append(y_shard)

    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch after concatenation for {split_name}: "
            f"X has {X.shape[0]} samples, y has {y.shape[0]} labels."
        )

    return X, y


def load_mixed_mode_split(extractor, split_root, split_name):
    if not hasattr(extractor, "TD_extractor"):
        raise AttributeError(
            f"{extractor.__class__.__name__} has no method 'TD_extractor'."
        )
    if not hasattr(extractor, "FD_extractor"):
        raise AttributeError(
            f"{extractor.__class__.__name__} has no method 'FD_extractor'."
        )

    shard_indices = discover_shard_indices(split_root)
    X_parts = []
    y_parts = []

    for shard_idx in tqdm(
        shard_indices,
        total=len(shard_indices),
        desc=f"Loading {split_name} shards (mixed)",
        unit="shard",
    ):
        X_td, y_td = extractor.TD_extractor(shard_idx)
        X_fd, y_fd = extractor.FD_extractor(shard_idx)

        validate_X_y(X_td, y_td, split_name, shard_idx, "TD_extractor")
        validate_X_y(X_fd, y_fd, split_name, shard_idx, "FD_extractor")

        if X_td.shape[0] != X_fd.shape[0]:
            raise ValueError(
                f"Mixed-mode row mismatch in {split_name} shard {shard_idx:03d}: "
                f"TD has {X_td.shape[0]} rows, FD has {X_fd.shape[0]} rows."
            )

        if not np.array_equal(y_td, y_fd):
            raise ValueError(
                f"Mixed-mode alignment failure in {split_name} shard {shard_idx:03d}: "
                f"TD labels and FD labels are not identical."
            )

        X_mixed = np.concatenate([X_td, X_fd], axis=1)
        X_parts.append(X_mixed)
        y_parts.append(y_td)

    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch after concatenation for {split_name}: "
            f"X has {X.shape[0]} samples, y has {y.shape[0]} labels."
        )

    return X, y


def load_split_for_mode(split_root: Path, input_mode: str, split_name: str):
    extractor = DataExtractor(split_root)

    if input_mode == "time":
        return load_single_mode_split(
            extractor=extractor,
            extract_fn_name="TD_extractor",
            split_root=split_root,
            split_name=split_name,
        )

    if input_mode == "frequency":
        return load_single_mode_split(
            extractor=extractor,
            extract_fn_name="FD_extractor",
            split_root=split_root,
            split_name=split_name,
        )

    if input_mode == "mixed":
        return load_mixed_mode_split(
            extractor=extractor,
            split_root=split_root,
            split_name=split_name,
        )

    raise ValueError(f"Unsupported input_mode: {input_mode}")


# ============================================================
# Run-name helpers
# ============================================================
def get_run_name(method_name: str, input_mode: str, n_components: int):
    if method_name == "grp":
        return f"grp_{input_mode}_{n_components:03d}"
    if method_name == "umap":
        return f"umap_{input_mode}_nn{UMAP_N_NEIGHBORS}_dim{n_components}"
    if method_name == "sup_umap":
        return f"sup_umap_{input_mode}_nn{UMAP_N_NEIGHBORS}_dim{n_components}"
    raise ValueError(f"Unsupported method_name: {method_name}")


def run_already_exists(run_root: Path):
    required = [
        run_root / "train_embeddings.npz",
        run_root / "test_embeddings.npz",
        run_root / "train_labels.npz",
        run_root / "test_labels.npz",
        run_root / "run_config.json",
    ]
    return all(path.exists() for path in required)


# ============================================================
# Method runners
# ============================================================
def run_grp(input_mode: str, n_components: int):
    run_name = get_run_name("grp", input_mode, n_components)
    run_root = OUTPUT_ROOT / run_name

    if run_already_exists(run_root) and not OVERWRITE_EXISTING:
        print(f"[SKIP] {run_name} already exists.")
        return str(run_root)

    embedder = GaussianRandomProjectionEmbedder(
        train_root=TRAIN_ROOT,
        test_root=TEST_ROOT,
        output_root=OUTPUT_ROOT,
        n_components=n_components,
        random_state=GRP_RANDOM_STATE,
    )

    X_train, y_train = load_split_for_mode(TRAIN_ROOT, input_mode, "train")
    X_test, y_test = load_split_for_mode(TEST_ROOT, input_mode, "test")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Feature dimension mismatch between train and test after loading: "
            f"train has {X_train.shape[1]} columns, test has {X_test.shape[1]} columns."
        )

    Z_train = embedder.fit_transform(X_train)
    Z_test = embedder.transform(X_test)

    saved_root = embedder.save_outputs(
        Z_train=Z_train,
        Z_test=Z_test,
        y_train=y_train,
        y_test=y_test,
        run_name=run_name,
        extra_config={
            "representation": input_mode,
            "extract_policy": "custom_runner_shared_loader",
            "retrieval_pool": "full_train",
            "query_split": "full_test",
            "fit_policy": "GRP_fit_on_full_train_transform_full_train_test",
        },
    )

    return str(saved_root)


def run_umap(input_mode: str, n_components: int):
    run_name = get_run_name("umap", input_mode, n_components)
    run_root = OUTPUT_ROOT / run_name

    if run_already_exists(run_root) and not OVERWRITE_EXISTING:
        print(f"[SKIP] {run_name} already exists.")
        return str(run_root)

    embedder = UMAPEmbedder(
        train_root=TRAIN_ROOT,
        test_root=TEST_ROOT,
        output_root=OUTPUT_ROOT,
        input_mode=input_mode,
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        low_memory=UMAP_LOW_MEMORY,
        verbose=UMAP_VERBOSE,
        n_jobs=UMAP_N_JOBS,
    )

    X_train, y_train = embedder.load_train_split()
    X_test, y_test = embedder.load_test_split()

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Feature dimension mismatch between train and test after loading: "
            f"train has {X_train.shape[1]} columns, test has {X_test.shape[1]} columns."
        )

    Z_train = embedder.fit_transform(X_train)
    Z_test = embedder.transform(X_test)

    saved_root = embedder.save_outputs(
        Z_train=Z_train,
        Z_test=Z_test,
        y_train=y_train,
        y_test=y_test,
        run_name=run_name,
        extra_config={
            "retrieval_pool": "full_train",
            "query_split": "full_test",
            "fit_policy": "UMAP_fit_on_full_train_transform_full_train_test",
        },
    )

    return str(saved_root)


def run_supervised_umap(input_mode: str, n_components: int):
    run_name = get_run_name("sup_umap", input_mode, n_components)
    run_root = OUTPUT_ROOT / run_name

    if run_already_exists(run_root) and not OVERWRITE_EXISTING:
        print(f"[SKIP] {run_name} already exists.")
        return str(run_root)

    embedder = SupervisedUMAPEmbedder(
        train_root=TRAIN_ROOT,
        test_root=TEST_ROOT,
        output_root=OUTPUT_ROOT,
        input_mode=input_mode,
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        target_metric=UMAP_TARGET_METRIC,
        low_memory=UMAP_LOW_MEMORY,
        verbose=UMAP_VERBOSE,
        n_jobs=UMAP_N_JOBS,
    )

    X_train, y_train = embedder.load_train_split()
    X_test, y_test = embedder.load_test_split()

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Feature dimension mismatch between train and test after loading: "
            f"train has {X_train.shape[1]} columns, test has {X_test.shape[1]} columns."
        )

    Z_train = embedder.fit_transform(X_train, y_train)
    Z_test = embedder.transform(X_test)

    saved_root = embedder.save_outputs(
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

    return str(saved_root)


# ============================================================
# Main matrix execution
# ============================================================
def build_run_matrix():
    matrix = []
    for n_components in EMBEDDING_DIMS:
        for input_mode in INPUT_MODES:
            matrix.append(("grp", input_mode, n_components))
            matrix.append(("umap", input_mode, n_components))
            matrix.append(("sup_umap", input_mode, n_components))
    return matrix


def execute_one(method_name: str, input_mode: str, n_components: int):
    print("=" * 80)
    print(f"Running method={method_name}, input_mode={input_mode}, dim={n_components}")

    if method_name == "grp":
        return run_grp(input_mode, n_components)
    if method_name == "umap":
        return run_umap(input_mode, n_components)
    if method_name == "sup_umap":
        return run_supervised_umap(input_mode, n_components)

    raise ValueError(f"Unsupported method_name: {method_name}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "repo_root": str(REPO_ROOT),
        "train_root": str(TRAIN_ROOT),
        "test_root": str(TEST_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "embedding_dims": EMBEDDING_DIMS,
        "input_modes": INPUT_MODES,
        "overwrite_existing": OVERWRITE_EXISTING,
        "runs": [],
    }

    run_matrix = build_run_matrix()

    for method_name, input_mode, n_components in run_matrix:
        run_name = get_run_name(method_name, input_mode, n_components)
        try:
            saved_root = execute_one(method_name, input_mode, n_components)
            summary["runs"].append(
                {
                    "run_name": run_name,
                    "method": method_name,
                    "input_mode": input_mode,
                    "n_components": n_components,
                    "status": "success",
                    "saved_root": saved_root,
                }
            )
            print(f"[OK] {run_name} -> {saved_root}")
        except Exception as exc:
            summary["runs"].append(
                {
                    "run_name": run_name,
                    "method": method_name,
                    "input_mode": input_mode,
                    "n_components": n_components,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[FAIL] {run_name}: {exc}")

    summary_path = OUTPUT_ROOT / "all_embedding_runner_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("All requested runs processed.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
