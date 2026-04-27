import logging
import colorlog
import sys
from backend.config import cfg_logger
from typing import Any
from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Handler, który wypisuje logi przez tqdm.write, żeby nie psuć pasków postępu."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)

class Logger:
    """
    Centralna klasa do zarządzania logami w całej aplikacji.
    """
    def __init__(self, nazwa_klasy: str):
        self.logger = logging.getLogger(nazwa_klasy)
        self.logger.setLevel(logging.DEBUG) # Nasłuchujemy wszystkich zdarzeń od poziomu DEBUG wzwyż
        self.logger.propagate = False

        if not self.logger.handlers:
            # Ustawienie wysyłania logów do konsoli (standardowe wyjście)
            console_handler = TqdmLoggingHandler()
            formatter = colorlog.ColoredFormatter(
                fmt="%(log_color)s%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            
            # Przypisanie formatu do handlera i handlera do loggera
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, wiadomosc: str):
        self.logger.info(wiadomosc)

    def warning(self, wiadomosc: str):
        self.logger.warning(wiadomosc)

    def error(self, wiadomosc: str):
        self.logger.error(wiadomosc)
        
    def debug(self, wiadomosc: str):
        if not cfg_logger.DEBUG:
            return

        self.logger.debug(wiadomosc)

    def infoModelTraining(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        best_val_loss: float,
        best_val_acc: float,
        epochs_no_improve: int,
        patience: int,
        epoch_time_s: float,
        samples_per_sec: float,
        avg_grad_norm: float | None = None,
    ):
        message = (
            f"Epoka {epoch}/{total_epochs} | \n"
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | \n"
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%) | \n"
            f"LR: {lr:.8f} | \n"
            f"Best Val Loss: {best_val_loss:.4f} | \n"
            f"Best Val Acc: {best_val_acc:.2f}% | \n"
            f"No Improve: {epochs_no_improve}/{patience} | \n"
            f"Epoch Time: {epoch_time_s:.2f}s | \n"
            f"Throughput: {samples_per_sec:.2f} samples/s"
        )

        if avg_grad_norm is not None:
            message += f" | Avg Grad Norm: {avg_grad_norm:.4f}"

        self.logger.info(message)

    def infoDataSummary(self, train_samples: int, val_samples: int, train_batches: int, val_batches: int):
        self.logger.info(
            f"Data Summary | Train: {train_samples} próbek ({train_batches} batchy) | "
            f"Val: {val_samples} próbek ({val_batches} batchy)"
        )
    
    def infoModelConfig(self, config: Any):
        self.logger.info(f"Model Configuration: {config}")

    def infoCheckpointSaved(self, path: str, val_loss: float, val_acc: float):
        self.logger.info(
            f"Checkpoint saved: {path} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    def warningNoImprovement(self, epochs_no_improve: int, patience: int):
        self.logger.warning(f"Brak poprawy od {epochs_no_improve} epok ({epochs_no_improve}/{patience}).")

    def infoEarlyStopping(self, epoch: int, best_epoch: int, best_val_loss: float, best_val_acc: float):
        self.logger.info(
            f"Early Stopping na epoce {epoch}. "
            f"Najlepsza epoka: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%"
        )

    def infoTrainingSummary(self, total_time_s: float, best_epoch: int, best_val_loss: float, best_val_acc: float):
        self.logger.info(
            f"Training Summary | Total Time: {total_time_s:.2f}s | "
            f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%"
        )