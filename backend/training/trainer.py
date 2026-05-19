from pathlib import Path
import torch
import torch.optim as optim
import shutil
from datetime import datetime
from tqdm import tqdm
from backend.config import cfg_train, cfg_paths, get_config_snapshot
from backend.logger.logger import Logger
import os
import numpy as np
from backend.metrics.evaluator import MetricsEvaluator, MetricsVisualizer
from backend.data.builder import DatasetBuilder

logger = Logger(__name__)

class Trainer:
    """Silnik zarządzający cyklem życia modelu (Trening, Walidacja, Checkpointing)."""

    def __init__(self, model, train_loader, val_loader, criterion, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
        self.config = cfg_train
        self.paths = cfg_paths
        
        # Inicjalizacja optymalizatora z L2 Regularization (Weight Decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=5e-4
        )
        
        # Scheduler zmniejszający LR, gdy Val Loss przestaje spadać
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )

        # Metrics / visualization
        self.metrics_dir = os.path.join(os.getcwd(), "out", "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.evaluator = MetricsEvaluator(class_names=DatasetBuilder.VOCAB, output_dir=self.metrics_dir)
        self.visualizer = MetricsVisualizer(output_dir=self.metrics_dir)
        self.history_csv = os.path.join(self.metrics_dir, "history.csv")

    def _log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """MIEJSCE NA MLOps: W przyszłości tutaj podpinamy np. wandb.log()"""
        logger.info(
            f"Epoka {epoch}/{self.config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)"
        )

    def _build_checkpoint_payload(self, epoch: int, val_acc: float, val_loss: float) -> dict:
        return {
            "state_dict": self.model.state_dict(),
            "metadata": {
                "epoch": epoch,
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "config": get_config_snapshot(),
            },
        }

    @staticmethod
    def _is_better_checkpoint(
        candidate_val_acc: float,
        candidate_val_loss: float,
        best_val_acc: float,
        best_val_loss: float,
    ) -> bool:
        """
        Określa, czy nowy checkpoint jest lepszy od aktualnie najlepszego.

        Priorytetem jest skuteczność (val_acc). Dopiero przy remisie patrzymy na val_loss.
        Dzięki temu model z niższą skutecznością nie nadpisze lepszego modelu tylko dlatego,
        że ma minimalnie lepszy loss.
        """
        if candidate_val_acc > best_val_acc:
            return True

        if candidate_val_acc == best_val_acc and candidate_val_loss < best_val_loss:
            return True

        return False

    def train(self):
        logger.info(f"Rozpoczęcie treningu na urządzeniu: {self.device}")
        best_val_loss = float('inf')
        best_val_acc = float('-inf')
        epochs_no_improve = 0
        best_model_path: Path | None = None

        for epoch in range(1, self.config.EPOCHS + 1):
            # ================= FAZA TRENINGU =================
            self.model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0
            
            pbar = tqdm(self.train_loader, desc=f"Trenowanie Epoki {epoch}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct_train / total_train
            
            # ================= FAZA WALIDACJI =================
            self.model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            
            # collect predictions for epoch-level metrics
            val_preds_all = []
            val_labels_all = []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)

                    # collect for evaluator
                    val_preds_all.extend(predicted.cpu().numpy().tolist())
                    val_labels_all.extend(labels.cpu().numpy().tolist())

                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
            val_loss /= len(self.val_loader)
            val_acc = 100 * correct_val / total_val
            
            # ================= RAPORTOWANIE I STEROWANIE =================
            self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            if self.optimizer.param_groups[0]['lr'] < current_lr:
                logger.info(f"-> Scheduler zmniejszył Learning Rate do: {self.optimizer.param_groups[0]['lr']}")
                
            # Model Checkpointing i Early Stopping
            if self._is_better_checkpoint(val_acc, val_loss, best_val_acc, best_val_loss):
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_no_improve = 0

                # Generowanie nazwy z datą i skutecznością
                date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                base_path = Path(self.paths.MODEL_SAVE_PATH)
                dir_path = base_path.parent
                extension = base_path.suffix
                filename = base_path.stem

                model_name = f"{filename}_{date_str}_acc{val_acc:.2f}{extension}"
                model_path = dir_path / model_name

                checkpoint_payload = self._build_checkpoint_payload(epoch, val_acc, val_loss)
                torch.save(checkpoint_payload, str(model_path))
                shutil.copy2(model_path, base_path)

                if best_model_path and best_model_path != model_path and best_model_path.exists():
                    try:
                        best_model_path.unlink()
                        logger.info(f"-> Usunięto poprzedni model: {best_model_path.name}")
                    except OSError as exc:
                        logger.warning(f"-> Nie udało się usunąć poprzedniego modelu {best_model_path}: {exc}")

                best_model_path = model_path
                logger.info(
                    f"-> Zapisano nowy, lepszy model: {model_name} | "
                    f"Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}"
                )
            else:
                epochs_no_improve += 1
                logger.warning(
                    f"-> Brak poprawy od {epochs_no_improve} epok. "
                    f"Aktualny model nie został nadpisany (Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f})."
                )
                if epochs_no_improve >= self.config.PATIENCE:
                    logger.warning("Early Stopping: Przerwano trening by uniknąć przeuczenia.")
                    break

            # --- append epoch history and generate epoch-level metrics/plots ---
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            csv_path = self.evaluator.append_history(row, csv_name=os.path.basename(self.history_csv))

            # compute per-epoch metrics on full validation set
            y_true = np.array(val_labels_all, dtype=int)
            y_pred = np.array(val_preds_all, dtype=int)
            try:
                results = self.evaluator.evaluate(y_true, y_pred)

                # save confusion matrix + per-class metrics for this epoch
                self.visualizer.plot_confusion_matrix(results["cm"], DatasetBuilder.VOCAB,
                                                      out_path=os.path.join(self.metrics_dir, f"cm_epoch{epoch}.png"))
                self.visualizer.plot_per_class_metrics(results["per_class"],
                                                       out_path=os.path.join(self.metrics_dir, f"per_class_epoch{epoch}.png"))

                # update rolling history plot
                self.visualizer.plot_history(csv_path, out_path=os.path.join(self.metrics_dir, "history.png"))
            except Exception as exc:
                logger.warning(f"Metrics generation failed for epoch {epoch}: {exc}")