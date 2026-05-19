import os
import argparse
import csv
from datetime import datetime

import torch
import numpy as np

from backend.config import cfg_paths, cfg_train, cfg_model
from backend.data.loader import DataLoaderFactory
from backend.models.crnn import ChordCRNN
from backend.logger.logger import Logger
from backend.metrics.evaluator import MetricsEvaluator, MetricsVisualizer
from backend.data.builder import DatasetBuilder

logger = Logger(__name__)


def load_model(checkpoint_path: str, device: torch.device):
    model = ChordCRNN()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        metadata = state.get("metadata", {})
        config_snapshot = metadata.get("config", {})
        if config_snapshot:
            logger.info(f"Checkpoint config: {config_snapshot}")
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_test(checkpoint: str, batch_size: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading dataloaders from {cfg_paths.PROCESSED_DATA} (batch_size={batch_size})")
    _, val_loader = DataLoaderFactory.create_dataloaders(cfg_paths.PROCESSED_DATA, batch_size=batch_size)

    logger.info(f"Loading model from {checkpoint} on {device}")
    model = load_model(checkpoint, device)

    vocab = DatasetBuilder.VOCAB
    ev = MetricsEvaluator(class_names=vocab, output_dir=out_dir)
    viz = MetricsVisualizer(output_dir=out_dir)

    all_preds = []
    all_trues = []
    all_probs = []
    all_probs_full = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_trues.extend(labels.cpu().numpy().tolist())
            # store top1 prob
            all_probs.extend(probs.cpu().numpy().max(axis=1).tolist())
            all_probs_full.extend(probs.cpu().numpy())

    # save raw predictions CSV
    csv_path = os.path.join(out_dir, f"preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_idx", "pred_idx", "true_label", "pred_label", "prob"])
        for i, (t, p, pr) in enumerate(zip(all_trues, all_preds, all_probs)):
            writer.writerow([i, int(t), int(p), vocab[int(t)] if int(t) < len(vocab) else str(t), vocab[int(p)] if int(p) < len(vocab) else str(p), float(pr)])

    # compute metrics and plots
    y_true = np.array(all_trues, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.vstack(all_probs_full) if all_probs_full else None
    results = ev.evaluate(y_true, y_pred, y_probs=y_probs, top_k=(1, 3))

    # save metrics JSON-like CSV summary
    summary_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"accuracy: {results['accuracy']:.4f}\n")
        f.write(f"macro_f1: {results['macro_f1']:.4f}\n")
        f.write(f"weighted_f1: {results['weighted_f1']:.4f}\n")

    viz.plot_confusion_matrix(results['cm'], vocab, out_path=os.path.join(out_dir, 'cm.png'))
    viz.plot_per_class_metrics(results['per_class'], out_path=os.path.join(out_dir, 'per_class.png'))
    try:
        viz.plot_confusion_grid(results['cm'], vocab, out_path=os.path.join(out_dir, 'confusion_grid.png'), normalize=True, top_k=6)
    except Exception as exc:
        logger.warning(f"Failed to generate confusion grid: {exc}")
    try:
        viz.plot_metrics_summary(results, out_path=os.path.join(out_dir, 'metrics_summary.png'))
    except Exception as exc:
        logger.warning(f"Failed to generate metrics summary plot: {exc}")

    logger.info(f"Test finished. Outputs saved to: {out_dir}")
    logger.info(f"Raw preds CSV: {csv_path}")
    logger.info(f"Metrics summary: {summary_path}")


def cli():
    parser = argparse.ArgumentParser(description="Test a saved model on validation set and generate metrics/plots")
    parser.add_argument('--checkpoint', '-c', default=cfg_paths.MODEL_SAVE_PATH, help='Path to model checkpoint')
    parser.add_argument('--batch-size', '-b', type=int, default=cfg_train.BATCH_SIZE, help='Batch size for dataloader')
    parser.add_argument('--out', '-o', default=os.path.join(os.getcwd(), 'out', 'metrics', 'test_run'), help='Output directory')
    args = parser.parse_args()

    run_test(args.checkpoint, args.batch_size, args.out)


if __name__ == '__main__':
    cli()
