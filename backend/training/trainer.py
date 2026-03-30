import os
import torch
import torch.optim as optim
from tqdm import tqdm
from backend.config import cfg_train, cfg_paths

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
            weight_decay=1e-4
        )
        
        # Scheduler zmniejszający LR, gdy Val Loss przestaje spadać
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )

    def _log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """MIEJSCE NA MLOps: W przyszłości tutaj podpinamy np. wandb.log()"""
        print(f"Epoka {epoch}/{self.config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")

    def train(self):
        print(f"\nRozpoczęcie treningu na urządzeniu: {self.device}")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
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
                print(f"-> Scheduler zmniejszył Learning Rate do: {self.optimizer.param_groups[0]['lr']}")
                
            # Model Checkpointing i Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.paths.MODEL_SAVE_PATH)
                print(f"-> Zapisano nowy, lepszy model: {os.path.basename(self.paths.MODEL_SAVE_PATH)}\n")
            else:
                epochs_no_improve += 1
                print(f"-> Brak poprawy od {epochs_no_improve} epok.\n")
                if epochs_no_improve >= self.config.PATIENCE:
                    print("Early Stopping: Przerwano trening by uniknąć przeuczenia.")
                    break