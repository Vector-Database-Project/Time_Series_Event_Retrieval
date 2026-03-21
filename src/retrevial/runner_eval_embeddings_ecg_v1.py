from pathlib import Path

from retrieval_evaluator import EmbeddingRetrievalEvaluator


# Assumes this runner lives at repo_root/src/representation/
repo_root = Path(__file__).resolve().parents[2]

dataset_name = "ecg"

# Set to a specific run folder name if you want one exact run.
# Leave as None to evaluate the latest saved run under results/embeddings/ecg/
run_name = "grp_frequency_128"

top_k = (1, 5, 10)
metric = "l2"
batch_size = 512


evaluator = EmbeddingRetrievalEvaluator(
    repo_root=repo_root,
    dataset_name=dataset_name,
    run_name=run_name,
    top_k=top_k,
    metric=metric,
    batch_size=batch_size,
)

metrics = evaluator.evaluate()

print("Evaluation complete.")
print(f"Run root:    {evaluator.run_root}")
print(f"Assets root: {evaluator.assets_root}")

for k, vals in metrics.items():
    print(
        f"{k} | "
        f"accuracy={vals['accuracy']:.4f} | "
        f"recall={vals['macro_recall']:.4f} | "
        f"f1={vals['macro_f1']:.4f}"
    )