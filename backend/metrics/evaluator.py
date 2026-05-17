# backend/metrics/evaluator.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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

        return {
            "accuracy": acc,
            "cm": cm,
            "per_class": per_class,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
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