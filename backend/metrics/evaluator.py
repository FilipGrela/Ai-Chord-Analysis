# backend/metrics/evaluator.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)
from typing import List, Optional, Dict


class MetricsEvaluator:
    def __init__(self, class_names: Optional[List[str]] = None, output_dir: str = "out/metrics"):
        self.class_names = list(class_names) if class_names is not None else None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None, top_k: tuple = (1, 3)) -> Dict:
        # normalize inputs to numpy ints
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        acc = float(accuracy_score(y_true, y_pred))

        labels = range(len(self.class_names)) if self.class_names else None
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        n_classes = cm.shape[0]

        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(n_classes), zero_division=0
        )
        # ensure arrays for safe indexing
        p = np.asarray(p)
        r = np.asarray(r)
        f1 = np.asarray(f1)
        support = np.asarray(support)

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        per_class = {}
        if self.class_names:
            names = self.class_names
        else:
            names = [str(i) for i in range(n_classes)]

        for i in range(n_classes):
            name = names[i] if i < len(names) else str(i)
            per_class[name] = {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        # additional musical metrics
        def _split_chord(label: str) -> tuple[str, str]:
            # label examples: 'C', 'Cm', 'C#', 'C#m', 'C7', 'Cm7', 'N'
            if label == 'N':
                return ('N', 'N')
            # prefer matching 'm7' then '7' then 'm' suffixes
            if label.endswith('m7') and len(label) > 2:
                root = label[:-2]
                quality = 'm7'
            elif label.endswith('7') and len(label) > 1:
                root = label[:-1]
                quality = '7'
            elif label.endswith('m') and len(label) > 1:
                root = label[:-1]
                quality = 'm'
            else:
                root = label
                quality = 'maj'
            return (root, quality)

        # map indices -> names
        idx_to_name = {i: (names[i] if i < len(names) else str(i)) for i in range(n_classes)}

        # compute root/quality accuracy
        roots_true = []
        roots_pred = []
        qual_true = []
        qual_pred = []
        for t_idx, p_idx in zip(y_true, y_pred):
            t_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
            p_name = idx_to_name.get(int(p_idx), str(int(p_idx)))
            rt, qt = _split_chord(t_name)
            rp, qp = _split_chord(p_name)
            roots_true.append(rt)
            roots_pred.append(rp)
            qual_true.append(qt)
            qual_pred.append(qp)

        root_acc = float(np.mean([1 if a == b else 0 for a, b in zip(roots_true, roots_pred)]))
        quality_acc = float(np.mean([1 if a == b else 0 for a, b in zip(qual_true, qual_pred)]))

        # chord error rate (sequence edit distance on collapsed sequences)
        def _collapse(seq):
            if len(seq) == 0:
                return []
            out = [seq[0]]
            for s in seq[1:]:
                if s != out[-1]:
                    out.append(s)
            return out

        def _levenshtein(a, b):
            # simple Levenshtein distance for sequences
            la, lb = len(a), len(b)
            if la == 0:
                return lb
            if lb == 0:
                return la
            dp = [[0] * (lb + 1) for _ in range(la + 1)]
            for i in range(la + 1):
                dp[i][0] = i
            for j in range(lb + 1):
                dp[0][j] = j
            for i in range(1, la + 1):
                for j in range(1, lb + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            return dp[la][lb]

        collapsed_ref = _collapse([idx_to_name.get(int(i), str(int(i))) for i in y_true.tolist()])
        collapsed_pred = _collapse([idx_to_name.get(int(i), str(int(i))) for i in y_pred.tolist()])
        edit_dist = _levenshtein(collapsed_ref, collapsed_pred)
        cer = float(edit_dist / max(1, len(collapsed_ref)))

        # compute top-K accuracies if probabilities provided
        topk_res = {}
        if y_probs is not None:
            try:
                y_probs = np.asarray(y_probs)
                ks = tuple(top_k) if isinstance(top_k, (list, tuple)) else (int(top_k),)
                for k in ks:
                    # get top-k indices per sample
                    topk_idx = np.argsort(y_probs, axis=1)[:, -k:]
                    # check membership
                    hits = 0
                    for i, true_idx in enumerate(y_true.astype(int)):
                        if true_idx in topk_idx[i]:
                            hits += 1
                    topk_res[f"top_{k}"] = float(hits / len(y_true)) if len(y_true) > 0 else 0.0
            except Exception:
                topk_res = {}

        return {
            "accuracy": acc,
            "cm": cm,
            "per_class": per_class,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "root_accuracy": root_acc,
            "quality_accuracy": quality_acc,
            "cer": cer,
            "top_k": topk_res,
            "timestamp": datetime.now().isoformat(),
        }

    def append_history(self, row: Dict, csv_name: str = "history.csv"):
        csv_path = os.path.join(self.output_dir, csv_name)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return csv_path


class MetricsVisualizer:
    def __init__(self, output_dir: str = "out/metrics"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], out_path: Optional[str] = None, normalize: bool = True):
        if normalize:
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum[cm_sum == 0] = 1
            cm_plot = cm.astype(float) / cm_sum
        else:
            cm_plot = cm

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fmt = ".2f" if normalize else "d"
        thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.5
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                text = format(cm_plot[i, j], fmt)
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if cm_plot[i, j] > thresh else "black", fontsize=5)
        plt.tight_layout()
        out_path = out_path or os.path.join(self.output_dir, f"cm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_per_class_metrics(self, per_class: Dict, out_path: Optional[str] = None):
        names = list(per_class.keys())
        precision = [per_class[n]["precision"] for n in names]
        recall = [per_class[n]["recall"] for n in names]
        f1 = [per_class[n]["f1"] for n in names]

        x = np.arange(len(names))
        width = 0.28
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.15), 6))
        ax.bar(x - width, precision, width, label="precision")
        ax.bar(x, recall, width, label="recall")
        ax.bar(x + width, f1, width, label="f1")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.legend()
        plt.tight_layout()
        out_path = out_path or os.path.join(self.output_dir, f"per_class_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_history(self, csv_path: str, out_path: Optional[str] = None):
        import csv as _csv
        epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                epochs.append(int(row.get("epoch", len(epochs) + 1)))
                train_loss.append(float(row.get("train_loss", np.nan)))
                val_loss.append(float(row.get("val_loss", np.nan)))
                train_acc.append(float(row.get("train_acc", np.nan)))
                val_acc.append(float(row.get("val_acc", np.nan)))

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue")
        ax1.plot(epochs, val_loss, label="val_loss", color="tab:orange")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(epochs, train_acc, label="train_acc", color="tab:green", linestyle="--")
        ax2.plot(epochs, val_acc, label="val_acc", color="tab:red", linestyle="--")
        ax2.set_ylabel("accuracy (%)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.tight_layout()
        out_path = out_path or os.path.join(self.output_dir, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_confusion_grid(
        self,
        cm: np.ndarray,
        class_names: List[str],
        out_path: Optional[str] = None,
        normalize: bool = True,
        top_k: int = 5,
        ncols: int | None = None,
        class_color_map: Optional[Dict[str, str]] = None,
    ):
        """
        Draw a grid of small plots — one per true class — showing the top-K predicted classes
        that the true class is confused with. Saves a single PNG with all subplots arranged in a grid.
        """
        n = len(class_names)
        if n == 0:
            raise ValueError("class_names must be a non-empty list")

        # normalize per-row (true -> predicted)
        if normalize:
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum[cm_sum == 0] = 1
            cm_norm = cm.astype(float) / cm_sum
        else:
            cm_norm = cm.astype(float)

        # determine grid shape
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        # prepare per-class color mapping (deterministic default if not provided)
        if class_color_map is None:
            cmap = plt.get_cmap("tab20")
            # generate n distinct colors
            colors = [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]
            class_color_map = {class_names[i]: colors[i] for i in range(n)}

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 1.8), constrained_layout=True)
        # use a plain Python list to avoid confusing numpy/list __getitem__ typings
        axes_list = list(np.array(axes).reshape(-1))

        for idx in range(n):
            ax = axes_list[idx]
            row = cm_norm[idx]

            # exclude self (perfect predictions) to emphasize confusions
            candidates = [(j, row[j]) for j in range(n) if j != idx]
            # sort descending by confusion probability
            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[:top_k]
            if len(top) == 0:
                labels = []
                values = []
            else:
                labels = [class_names[j] for j, _ in top]
                values = [v for _, v in top]

            # reverse for horizontal bar chart (largest at top)
            labels = labels[::-1]
            values = values[::-1]

            y = np.arange(len(values))
            # map labels to colors (keep order)
            bar_colors = [class_color_map.get(lbl, "#3b82c4") for lbl in labels]
            ax.barh(y, values, color=bar_colors)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlim(0, 1 if normalize else cm_norm.max())
            ax.set_xlabel("prob")
            ax.set_title(class_names[idx], fontsize=7)

        # hide any unused subplots
        for j in range(n, len(axes_list)):
            axes_list[j].axis("off")

        out_path = out_path or os.path.join(self.output_dir, f"confusion_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_metrics_summary(self, results: Dict, out_path: Optional[str] = None):
        """Plot a small summary bar chart with accuracy, root_accuracy, quality_accuracy and CER."""
        labels = ["accuracy", "root_acc", "quality_acc", "CER"]
        acc = results.get("accuracy", 0.0)
        root = results.get("root_accuracy", 0.0)
        qual = results.get("quality_accuracy", 0.0)
        cer = results.get("cer", 0.0)

        # For plotting, invert CER so higher is better (1 - cer)
        values = [acc, root, qual, 1.0 - cer]

        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(labels, values, color=["#2b8cbe", "#7bccc4", "#a6bddb", "#f03b20"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title("Metrics Summary (1-CER shown)")
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2.0, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

        out_path = out_path or os.path.join(self.output_dir, f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path