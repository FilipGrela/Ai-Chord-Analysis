# backend/metrics/evaluator.py
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
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

    def compute_support_vs_perf(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, class_names: Optional[List[str]] = None, min_support: int = 1, out_name: str | None = None) -> dict:
        """
        Compute per-class support and basic performance (accuracy, precision, recall, f1).

        Returns a dict with:
        - records: list of per-class dicts {class, index, support, accuracy, precision, recall, f1, mean_confidence}
        - stats: summary stats including spearman_rho and p-value
        - csv_path: path to written CSV

        The CSV fields: class,index,support,accuracy,precision,recall,f1,mean_confidence
        """
        from collections import defaultdict
        try:
            from scipy.stats import spearmanr
        except Exception:
            spearmanr = None

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_prob is not None:
            y_prob = np.asarray(y_prob)

        # determine classes
        if class_names is not None:
            names = list(class_names)
            indices = list(range(len(names)))
        else:
            indices = np.unique(np.concatenate([y_true, y_pred])).tolist()
            names = [str(i) for i in indices]

        records = []
        supports = []
        accs = []

        for idx, cname in zip(indices, names):
            mask = (y_true == idx)
            supp = int(mask.sum())
            if supp > 0:
                correct = int((y_pred[mask] == y_true[mask]).sum())
                acc = float(correct / supp)
                # precision/recall/f1: compute via contingency
                # precision = TP / (TP + FP)
                tp = correct
                fp = int(((y_pred == idx) & (y_true != idx)).sum())
                fn = int(((y_true == idx) & (y_pred != idx)).sum())
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
                f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float('nan')
            else:
                acc = float('nan')
                precision = float('nan')
                recall = float('nan')
                f1 = float('nan')

            mean_conf = float(np.nan)
            if y_prob is not None and supp > 0:
                # mean max-softmax confidence for samples of this class
                probs = y_prob[mask]
                if probs.size:
                    mean_conf = float(np.nanmean(np.max(probs, axis=1)))

            records.append({
                "class": str(cname),
                "index": int(idx),
                "support": supp,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_confidence": mean_conf,
            })

            if supp > 0:
                supports.append(supp)
                accs.append(acc)

        # correlation
        stats = {}
        if len(supports) >= 2:
            try:
                if spearmanr is not None:
                    rho, p = spearmanr(supports, accs)
                else:
                    rho = float(np.corrcoef(supports, accs)[0, 1])
                    p = float('nan')
                stats["spearman_rho"] = float(rho)
                stats["spearman_p"] = float(p)
            except Exception:
                stats["spearman_rho"] = float('nan')
                stats["spearman_p"] = float('nan')
        else:
            stats["spearman_rho"] = float('nan')
            stats["spearman_p"] = float('nan')

        # write CSV
        out_name = out_name or f"support_vs_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(self.output_dir, out_name)
        fieldnames = ["class", "index", "support", "accuracy", "precision", "recall", "f1", "mean_confidence"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        return {"records": records, "stats": stats, "csv_path": csv_path}


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
        # Jedna linijka zamiast ręcznego rysowania siatki i pętli z tekstem
        sns.heatmap(cm_plot, annot=True, fmt=".2f" if normalize else "g", 
                    cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax,
                    annot_kws={"size": 6}) # Automatycznie dostosuje kolor tekstu (czarny/biały)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
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
        train_ce_hard, train_kl_soft, val_ce_hard, val_kl_soft = [], [], [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                epochs.append(int(row.get("epoch", len(epochs) + 1)))
                train_loss.append(float(row.get("train_loss", np.nan)))
                val_loss.append(float(row.get("val_loss", np.nan)))
                train_acc.append(float(row.get("train_acc", np.nan)))
                val_acc.append(float(row.get("val_acc", np.nan)))
                train_ce_hard.append(float(row.get("train_ce_hard", np.nan)))
                train_kl_soft.append(float(row.get("train_kl_soft", np.nan)))
                val_ce_hard.append(float(row.get("val_ce_hard", np.nan)))
                val_kl_soft.append(float(row.get("val_kl_soft", np.nan)))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue")
        ax1.plot(epochs, val_loss, label="val_loss", color="tab:orange")
        ax1.plot(epochs, train_ce_hard, label="train_ce_hard", color="tab:blue", linestyle="--")
        ax1.plot(epochs, val_ce_hard, label="val_ce_hard", color="tab:orange", linestyle="--")
        ax1.plot(epochs, train_kl_soft, label="train_kl_soft", color="tab:purple", linestyle=":")
        ax1.plot(epochs, val_kl_soft, label="val_kl_soft", color="tab:brown", linestyle=":")
        ax1.set_ylabel("loss")
        ax1.legend(loc="upper right", fontsize=8, ncol=2)

        ax2.plot(epochs, train_acc, label="train_acc", color="tab:green", linestyle="--")
        ax2.plot(epochs, val_acc, label="val_acc", color="tab:red", linestyle="--")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy (%)")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        out_path = out_path or os.path.join(self.output_dir, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_support_vs_perf(self, csv_path: str | None = None, data: Optional[list] = None, out_path: Optional[str] = None):
        """
        Plot a dual-axis chart showing support (as bars) and accuracy (as a line).
        Classes are sorted descending by support.
        
        Either provide `csv_path` (CSV produced by `compute_support_vs_perf`) or `data` as list of records.
        """
        # load data
        if data is None:
            if csv_path is None:
                raise ValueError("Provide csv_path or data")
            import csv as _csv
            data = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    data.append({
                        "class": row.get("class"),
                        "index": int(row.get("index", -1)),
                        "support": int(row.get("support", 0)),
                        "accuracy": float(row.get("accuracy", float('nan'))),
                        "mean_confidence": float(row.get("mean_confidence", float('nan'))),
                    })

        # basic filter & sort descending by support
        valid_data = [d for d in data if not np.isnan(d.get("accuracy", float('nan')))]
        valid_data.sort(key=lambda x: x["support"], reverse=True)

        if not valid_data:
            return None

        classes = [str(d["class"]) for d in valid_data]
        supports = [d["support"] for d in valid_data]
        accs = [d["accuracy"] for d in valid_data]

        fig, ax1 = plt.subplots(figsize=(max(10, len(classes) * 0.3), 6))

        # Oś główna (lewa) - Support jako słupki
        color_support = '#b0c4de'  # Jasnoniebiesko-szary, stanowi dobre tło
        ax1.bar(classes, supports, color=color_support, label='Support')
        ax1.set_xlabel("Klasa (posortowane po support)")
        ax1.set_ylabel("Support (liczba przykładów)", color='#4682b4')
        ax1.tick_params(axis='y', labelcolor='#4682b4')
        ax1.tick_params(axis='x', rotation=90, labelsize=8)

        # Oś pomocnicza (prawa) - Accuracy jako linia
        ax2 = ax1.twinx()
        color_acc = '#d62728'  # Czerwony dla wyraźnego kontrastu
        ax2.plot(classes, accs, color=color_acc, marker='o', linestyle='-', linewidth=2, markersize=5, label='Accuracy')
        ax2.set_ylabel("Accuracy", color=color_acc)
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.set_ylim(0.0, 1.05)  # 1.05 daje lekki bufor na górze wykresu

        plt.title("Support and Accuracy per class (Dual-Axis)")

        # Połączenie legend z obu osi w jednej ramce
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.tight_layout()
        out_path = out_path or os.path.join(self.output_dir, f"support_vs_perf_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
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
        """Plot a compact summary of the main global and musical metrics."""
        main_labels = ["accuracy", "macro_f1", "weighted_f1", "root_acc", "quality_acc", "1-CER"]
        acc = float(results.get("accuracy", 0.0))
        macro_f1 = float(results.get("macro_f1", 0.0))
        weighted_f1 = float(results.get("weighted_f1", 0.0))
        root = float(results.get("root_accuracy", 0.0))
        qual = float(results.get("quality_accuracy", 0.0))
        cer = float(results.get("cer", 0.0))

        main_values = [acc, macro_f1, weighted_f1, root, qual, 1.0 - cer]

        topk = results.get("top_k", {}) or {}
        topk_items = []
        for key, value in topk.items():
            try:
                suffix = int(str(key).split("_")[-1])
            except Exception:
                suffix = 0
            topk_items.append((suffix, key, float(value)))
        topk_items.sort(key=lambda item: item[0])
        topk_labels = [item[1] for item in topk_items]
        topk_values = [item[2] for item in topk_items]

        has_topk = len(topk_labels) > 0
        if has_topk:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
        else:
            fig, ax1 = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
            ax2 = None

        main_bars = ax1.bar(
            main_labels,
            main_values,
            color=["#2b8cbe", "#4c78a8", "#72b7b2", "#7bccc4", "#a6bddb", "#f03b20"],
        )
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("score")
        ax1.set_title("Global / musical metrics")
        ax1.tick_params(axis="x", rotation=25)
        for rect, v in zip(main_bars, main_values):
            ax1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

        if has_topk and ax2 is not None:
            topk_bars = ax2.bar(topk_labels, topk_values, color="#59a14f")
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("score")
            ax2.set_title("Top-K accuracy")
            for rect, v in zip(topk_bars, topk_values):
                ax2.text(rect.get_x() + rect.get_width() / 2.0, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

        out_path = out_path or os.path.join(self.output_dir, f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path