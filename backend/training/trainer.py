from pathlib import Path
import torch
import torch.optim as optim
import shutil
from datetime import datetime
from tqdm import tqdm
from backend.config import cfg_train, cfg_paths
from backend.logger.logger import Logger

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

    def _log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """MIEJSCE NA MLOps: W przyszłości tutaj podpinamy np. wandb.log()"""
        logger.info(
            f"Epoka {epoch}/{self.config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)"
        )

    def train(self):
        logger.info(f"Rozpoczęcie treningu na urządzeniu: {self.device}")
        best_val_loss = float('inf')
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
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                # Generowanie nazwy z datą i skutecznością
                date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                base_path = Path(self.paths.MODEL_SAVE_PATH)
                dir_path = base_path.parent
                extension = base_path.suffix
                filename = base_path.stem

                model_name = f"{filename}_{date_str}_acc{val_acc:.2f}{extension}"
                model_path = dir_path / model_name

                torch.save(self.model.state_dict(), str(model_path))
                shutil.copy2(model_path, base_path)

                if best_model_path and best_model_path != model_path and best_model_path.exists():
                    try:
                        best_model_path.unlink()
                        logger.info(f"-> Usunięto poprzedni model: {best_model_path.name}")
                    except OSError as exc:
                        logger.warning(f"-> Nie udało się usunąć poprzedniego modelu {best_model_path}: {exc}")

                best_model_path = model_path
                logger.info(f"-> Zapisano nowy, lepszy model: {model_name}")
            else:
                epochs_no_improve += 1
                logger.warning(f"-> Brak poprawy od {epochs_no_improve} epok.")
                if epochs_no_improve >= self.config.PATIENCE:
                    logger.warning("Early Stopping: Przerwano trening by uniknąć przeuczenia.")
                    break