from pathlib import Path
import json

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class EmbeddingRetrievalEvaluator:
    def __init__(
        self,
        repo_root,
        dataset_name="ecg",
        run_name=None,
        embeddings_root=None,
        evaluation_root=None,
        top_k=(1, 5, 10),
        metric="l2",
        batch_size=512,
    ):
        """
        Common evaluator for saved embedding runs.

        Parameters
        ----------
        repo_root : str or Path
            Project repo root.

        dataset_name : str
            Dataset folder name.

        run_name : str or None
            Name of the saved run folder.
            If None, the most recently modified valid run under embeddings_root is used.

        embeddings_root : str or Path or None
            Root folder containing saved embedding run directories.
            Defaults to: repo_root/results/embeddings/{dataset_name}

        evaluation_root : str or Path or None
            Root folder where evaluation artifacts should be saved.
            Defaults to: repo_root/results/evaluation/{dataset_name}

        top_k : tuple[int]
            K values to evaluate.

        metric : str
            Retrieval distance metric.
            Supported: 'l2', 'cosine'

        batch_size : int
            Query batch size for brute-force search.
        """
        self.repo_root = Path(repo_root)
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.top_k = tuple(sorted(set(int(k) for k in top_k)))
        self.metric = metric
        self.batch_size = int(batch_size)

        if len(self.top_k) == 0:
            raise ValueError("top_k cannot be empty.")

        if self.metric not in {"l2", "cosine"}:
            raise ValueError("metric must be either 'l2' or 'cosine'.")

        if embeddings_root is None:
            self.embeddings_root = (
                self.repo_root / "results" / "embeddings" / self.dataset_name
            )
        else:
            self.embeddings_root = Path(embeddings_root)

        self.run_root = self._resolve_run_root()

        if evaluation_root is None:
            base_eval_root = self.repo_root / "results" / "evaluation" / self.dataset_name
        else:
            base_eval_root = Path(evaluation_root)

        self.assets_root = base_eval_root / self.run_root.name

        self.Z_train = None
        self.Z_test = None
        self.y_train = None
        self.y_test = None
        self.run_config = None
        self.class_labels = None

    def _resolve_run_root(self):
        if not self.embeddings_root.exists():
            raise FileNotFoundError(f"Embeddings root not found: {self.embeddings_root}")

        if self.run_name is not None:
            run_root = self.embeddings_root / self.run_name
            if not run_root.exists():
                raise FileNotFoundError(f"Run folder not found: {run_root}")
            return run_root

        candidate_dirs = []
        for path in self.embeddings_root.iterdir():
            if not path.is_dir():
                continue

            required = [
                path / "train_embeddings.npz",
                path / "test_embeddings.npz",
                path / "train_labels.npz",
                path / "test_labels.npz",
            ]

            if all(p.exists() for p in required):
                candidate_dirs.append(path)

        if not candidate_dirs:
            raise FileNotFoundError(
                f"No valid embedding run folders found under: {self.embeddings_root}"
            )

        candidate_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidate_dirs[0]

    def _load_npz_array(self, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(
                    f"Expected exactly one array in {path}, found {len(data.files)}: {data.files}"
                )
            array = data[data.files[0]]

        return array

    def load_artifacts(self):
        file_map = {
            "Z_train": self.run_root / "train_embeddings.npz",
            "Z_test": self.run_root / "test_embeddings.npz",
            "y_train": self.run_root / "train_labels.npz",
            "y_test": self.run_root / "test_labels.npz",
        }

        loaded = {}
        for name, path in tqdm(
            file_map.items(),
            total=len(file_map),
            desc="Loading evaluation artifacts",
            unit="file",
        ):
            loaded[name] = self._load_npz_array(path)

        self.Z_train = loaded["Z_train"].astype(np.float32, copy=False)
        self.Z_test = loaded["Z_test"].astype(np.float32, copy=False)
        self.y_train = loaded["y_train"].astype(np.int64, copy=False)
        self.y_test = loaded["y_test"].astype(np.int64, copy=False)

        if self.Z_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(
                f"Train mismatch: {self.Z_train.shape[0]} embeddings vs {self.y_train.shape[0]} labels."
            )

        if self.Z_test.shape[0] != self.y_test.shape[0]:
            raise ValueError(
                f"Test mismatch: {self.Z_test.shape[0]} embeddings vs {self.y_test.shape[0]} labels."
            )

        max_k = self.top_k[-1]
        if max_k > self.Z_train.shape[0]:
            raise ValueError(
                f"Requested max K={max_k}, but only {self.Z_train.shape[0]} reference samples exist."
            )

        config_path = self.run_root / "run_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.run_config = json.load(f)

        self.class_labels = np.unique(np.concatenate([self.y_train, self.y_test]))

    def _compute_topk_indices(self):
        max_k = self.top_k[-1]
        num_queries = self.Z_test.shape[0]
        num_refs = self.Z_train.shape[0]

        topk_chunks = []

        if self.metric == "cosine":
            ref_norms = np.linalg.norm(self.Z_train, axis=1, keepdims=True)
            ref_norms = np.clip(ref_norms, a_min=1e-12, a_max=None)
            ref_matrix = self.Z_train / ref_norms
        else:
            ref_matrix = self.Z_train

        for start in tqdm(
            range(0, num_queries, self.batch_size),
            desc="Searching reference pool",
            unit="batch",
        ):
            end = min(start + self.batch_size, num_queries)
            query_batch = self.Z_test[start:end]

            if self.metric == "l2":
                q_sq = np.sum(query_batch ** 2, axis=1, keepdims=True)
                r_sq = np.sum(ref_matrix ** 2, axis=1)
                distances = q_sq + r_sq[None, :] - 2.0 * (query_batch @ ref_matrix.T)

                idx_part = np.argpartition(distances, kth=max_k - 1, axis=1)[:, :max_k]
                row_ids = np.arange(idx_part.shape[0])[:, None]
                part_dist = distances[row_ids, idx_part]
                order = np.argsort(part_dist, axis=1)
                idx_sorted = idx_part[row_ids, order]
            else:
                query_norms = np.linalg.norm(query_batch, axis=1, keepdims=True)
                query_norms = np.clip(query_norms, a_min=1e-12, a_max=None)
                query_matrix = query_batch / query_norms

                similarities = query_matrix @ ref_matrix.T

                idx_part = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
                row_ids = np.arange(idx_part.shape[0])[:, None]
                part_sim = similarities[row_ids, idx_part]
                order = np.argsort(-part_sim, axis=1)
                idx_sorted = idx_part[row_ids, order]

            if idx_sorted.shape != (end - start, max_k):
                raise ValueError(
                    f"Unexpected top-k index shape: {idx_sorted.shape}, expected {(end - start, max_k)}"
                )

            topk_chunks.append(idx_sorted)

        topk_indices = np.concatenate(topk_chunks, axis=0)

        if topk_indices.shape != (num_queries, max_k):
            raise ValueError(
                f"Unexpected full top-k shape: {topk_indices.shape}, expected {(num_queries, max_k)}"
            )

        if np.any(topk_indices < 0) or np.any(topk_indices >= num_refs):
            raise ValueError("Top-k indices contain out-of-range reference indices.")

        return topk_indices

    def _majority_vote_ranked(self, ranked_labels):
        counts = {}
        best_label = None
        best_count = -1

        for label in ranked_labels:
            counts[label] = counts.get(label, 0) + 1
            if counts[label] > best_count:
                best_label = label
                best_count = counts[label]

        return int(best_label)

    def _predict_labels_from_topk(self, topk_neighbor_labels, k):
        preds = []

        for row in tqdm(
            topk_neighbor_labels,
            total=topk_neighbor_labels.shape[0],
            desc=f"Voting labels at K={k}",
            unit="query",
        ):
            pred = self._majority_vote_ranked(row[:k])
            preds.append(pred)

        return np.asarray(preds, dtype=np.int64)

    def _save_npz_map(self, path, data_map):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data_map)

    def _save_confusion_matrix(self, cm, labels, k):
        self.assets_root.mkdir(parents=True, exist_ok=True)

        npy_path = self.assets_root / f"confusion_matrix_k_{k:03d}.npz"
        png_path = self.assets_root / f"confusion_matrix_k_{k:03d}.png"

        self._save_npz_map(
            npy_path,
            {
                "confusion_matrix": cm.astype(np.int64),
                "labels": labels.astype(np.int64),
            },
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
        ax.set_title(f"Confusion Matrix @ K={k}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _build_per_class_metrics(self, y_true, y_pred):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=self.class_labels,
            average=None,
            zero_division=0,
        )

        per_class = {}
        for idx, label in enumerate(self.class_labels):
            per_class[str(int(label))] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }

        return per_class

    def evaluate(self):
        """
        Run full evaluation and save outputs.

        Saved outputs
        -------------
        - metrics_at_k.json
        - classification_report_at_k.json
        - eval_config.json
        - topk_neighbor_indices.npz
        - topk_neighbor_labels.npz
        - predictions_by_k.npz
        - confusion_matrix_k_XXX.npz
        - confusion_matrix_k_XXX.png
        """
        self.load_artifacts()
        self.assets_root.mkdir(parents=True, exist_ok=True)

        topk_indices = self._compute_topk_indices()
        topk_neighbor_labels = self.y_train[topk_indices]

        self._save_npz_map(
            self.assets_root / "topk_neighbor_indices.npz",
            {"indices": topk_indices.astype(np.int64)},
        )

        self._save_npz_map(
            self.assets_root / "topk_neighbor_labels.npz",
            {"labels": topk_neighbor_labels.astype(np.int64)},
        )

        predictions_by_k = {}
        metrics_at_k = {}
        classification_report_at_k = {}

        for k in self.top_k:
            y_pred = self._predict_labels_from_topk(topk_neighbor_labels, k)
            predictions_by_k[f"k_{k:03d}"] = y_pred

            acc = float(accuracy_score(self.y_test, y_pred))
            prec = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))

            cm = confusion_matrix(self.y_test, y_pred, labels=self.class_labels)
            self._save_confusion_matrix(cm, self.class_labels, k)

            metrics_at_k[f"k_{k}"] = {
                "accuracy": acc,
                "macro_precision": prec,
                "macro_recall": rec,
                "macro_f1": f1,
            }

            classification_report_at_k[f"k_{k}"] = {
                "macro": {
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                },
                "per_class": self._build_per_class_metrics(self.y_test, y_pred),
            }

        self._save_npz_map(
            self.assets_root / "predictions_by_k.npz",
            predictions_by_k,
        )

        with open(self.assets_root / "metrics_at_k.json", "w", encoding="utf-8") as f:
            json.dump(metrics_at_k, f, indent=4)

        with open(self.assets_root / "classification_report_at_k.json", "w", encoding="utf-8") as f:
            json.dump(classification_report_at_k, f, indent=4)

        eval_config = {
            "dataset_name": self.dataset_name,
            "run_name": self.run_root.name,
            "run_root": str(self.run_root),
            "assets_root": str(self.assets_root),
            "metric": self.metric,
            "top_k": list(self.top_k),
            "batch_size": self.batch_size,
            "class_labels": [int(x) for x in self.class_labels.tolist()],
        }

        if self.run_config is not None:
            eval_config["run_config"] = self.run_config

        with open(self.assets_root / "eval_config.json", "w", encoding="utf-8") as f:
            json.dump(eval_config, f, indent=4)

        return metrics_at_k
