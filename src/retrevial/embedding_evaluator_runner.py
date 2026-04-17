from pathlib import Path
import json
import traceback

from retrieval_evaluator import EmbeddingRetrievalEvaluator


# ============================================================
# Global evaluation configuration
# ============================================================
DATASET_NAME = "ecg_v2"
TOP_K = (1, 5, 10)
METRIC = "l2"
BATCH_SIZE = 512
OVERWRITE_EXISTING = False


# ============================================================
# Repo paths
# Assumes this runner lives at repo_root/src/representation/
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_ROOT = REPO_ROOT / "results" / "embeddings" / DATASET_NAME
EVALUATION_ROOT = REPO_ROOT / "results" / "evaluations" / DATASET_NAME


# ============================================================
# Helpers
# ============================================================
def discover_run_dirs(embeddings_root: Path):
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings root not found: {embeddings_root}")

    run_dirs = []
    for path in sorted(embeddings_root.iterdir()):
        if not path.is_dir():
            continue

        required = [
            path / "train_embeddings.npz",
            path / "test_embeddings.npz",
            path / "train_labels.npz",
            path / "test_labels.npz",
        ]

        if all(p.exists() for p in required):
            run_dirs.append(path)

    return run_dirs


def eval_outputs_exist(eval_run_root: Path):
    required = [
        eval_run_root / "metrics_at_k.json",
        eval_run_root / "classification_report_at_k.json",
        eval_run_root / "eval_config.json",
        eval_run_root / "predictions_by_k.npz",
        eval_run_root / "topk_neighbor_indices.npz",
        eval_run_root / "topk_neighbor_labels.npz",
    ]
    return all(path.exists() for path in required)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Main execution
# ============================================================
def evaluate_one_run(run_name: str):
    evaluator = EmbeddingRetrievalEvaluator(
        repo_root=REPO_ROOT,
        dataset_name=DATASET_NAME,
        run_name=run_name,
        embeddings_root=EMBEDDINGS_ROOT,
        evaluation_root=EVALUATION_ROOT,
        top_k=TOP_K,
        metric=METRIC,
        batch_size=BATCH_SIZE,
    )
    metrics = evaluator.evaluate()
    return evaluator.assets_root, metrics


def main():
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(EMBEDDINGS_ROOT)
    if not run_dirs:
        raise FileNotFoundError(
            f"No valid embedding runs found under: {EMBEDDINGS_ROOT}"
        )

    summary = {
        "dataset_name": DATASET_NAME,
        "repo_root": str(REPO_ROOT),
        "embeddings_root": str(EMBEDDINGS_ROOT),
        "evaluation_root": str(EVALUATION_ROOT),
        "top_k": list(TOP_K),
        "metric": METRIC,
        "batch_size": BATCH_SIZE,
        "overwrite_existing": OVERWRITE_EXISTING,
        "runs": [],
    }

    for run_dir in run_dirs:
        run_name = run_dir.name
        eval_run_root = EVALUATION_ROOT / run_name

        print("=" * 80)
        print(f"Evaluating: {run_name}")

        if eval_outputs_exist(eval_run_root) and not OVERWRITE_EXISTING:
            metrics_path = eval_run_root / "metrics_at_k.json"
            metrics = load_json(metrics_path)

            summary["runs"].append(
                {
                    "run_name": run_name,
                    "status": "skipped_existing",
                    "run_root": str(run_dir),
                    "assets_root": str(eval_run_root),
                    "metrics_at_k": metrics,
                }
            )
            print(f"[SKIP] Existing evaluation found for {run_name}")
            continue

        try:
            assets_root, metrics = evaluate_one_run(run_name)
            summary["runs"].append(
                {
                    "run_name": run_name,
                    "status": "success",
                    "run_root": str(run_dir),
                    "assets_root": str(assets_root),
                    "metrics_at_k": metrics,
                }
            )
            print(f"[OK] {run_name} -> {assets_root}")

        except Exception as exc:
            summary["runs"].append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "run_root": str(run_dir),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[FAIL] {run_name}: {exc}")

    summary_path = EVALUATION_ROOT / "all_embedding_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("All requested evaluations processed.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
