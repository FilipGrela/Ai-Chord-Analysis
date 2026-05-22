import torch

from backend.config import cfg_paths, cfg_train
from backend.data.loader import DataLoaderFactory
from backend.models.crnn import ChordCRNN
from backend.training.loss import LossFactory
from backend.training.trainer import Trainer
from backend.logger.logger import Logger

logger = Logger(__name__)

def main():
    logger.info("--- Inicjalizacja Architektury Treningowej AI-Chord-Analysis ---")
    
    # 1. Konfiguracja sprzętowa (GPU/CPU)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")

    
    # 2. Załadowanie danych z dysku do RAM-u (on-demand)
    logger.info(f"Tworzę DataLoadery z katalogu: {cfg_paths.PROCESSED_DATA} (batch_size={cfg_train.BATCH_SIZE})")
    train_loader, val_loader = DataLoaderFactory.create_dataloaders(
        data_dir=cfg_paths.PROCESSED_DATA, 
        batch_size=cfg_train.BATCH_SIZE
        
    )
    
    # 3. Budowa grafu obliczeniowego
    logger.info("Inicjalizacja modelu sieci (ChordCRNN)")
    model = ChordCRNN()
    logger.info("Model utworzony")
    
    # 4. Inteligentna funkcja straty (z wagami klas)
    logger.info("Tworzę funkcję straty (LossFactory)")
    criterion = LossFactory.create_loss_function(train_loader, device)
    logger.info("Funkcja straty gotowa")
    
    # 5. Uruchomienie pętli uczącej
    logger.info("Konfiguruję Trainer i rozpoczynam trening")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()