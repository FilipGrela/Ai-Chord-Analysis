import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from backend.logger.logger import Logger
from backend.data.augment.pipeline import build_train_augment_pipeline
from backend.config import cfg_train

logger = Logger(__name__)

class ChordDataset(Dataset):
    """Memory-efficient dataset that loads spectrogram files on-demand.

    Zamiast wczytywać wszystkie dane do RAM, ładuje każdy plik spektrogramu
    przy dostępie do próbki. Wersja rozszerzona o opcjonalny pipeline augmentacji,
    który działa online na spektrogramach przed zwróceniem próbki (dla train).
    """

    def __init__(self, x_files: list[str], y_files: list[str], augment_pipeline=None):
        """
        Args:
            x_files: lista ścieżek do plików _X.npy
            y_files: lista ścieżek do plików _y.npy (muszą być w tej samej kolejności)
            augment_pipeline: opcjonalny pipeline augmentacji
        """
        if len(x_files) != len(y_files):
            raise ValueError("Liczba plików X i y musi być równa!")
        
        self.x_files = x_files
        self.y_files = y_files
        self.augment_pipeline = augment_pipeline
        
        # Pre-compute indices: dla każdego pliku wiemy, ile ma próbek
        self.file_indices = []  # [(file_idx, sample_idx_in_file), ...]
        self.file_lengths = []  # ile próbek w każdym pliku
        
        for i, y_file in enumerate(y_files):
            try:
                y_data = np.load(y_file, allow_pickle=False)
                num_samples = len(y_data)
                self.file_lengths.append(num_samples)
                for sample_idx in range(num_samples):
                    self.file_indices.append((i, sample_idx))
            except Exception as e:
                logger.error(f"Błąd wczytywania {y_file}: {e}")
                raise
    
    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        
        # Wczytaj spektrogram i etykietę z pliku (cached w OS buffer)
        try:
            X_file = np.load(self.x_files[file_idx], allow_pickle=False)
            y_file = np.load(self.y_files[file_idx], allow_pickle=False)
            
            x = X_file[sample_idx]  # (seq_len, num_bins)
            y = y_file[sample_idx]  # skalar
            
            # Dodaj wymiar kanału dla Conv2d: (1, seq_len, 84)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.long)
            
            if self.augment_pipeline is not None:
                try:
                    x = self.augment_pipeline(x)
                except Exception as e:
                    logger.error(f"Błąd podczas augmentacji: {e}")
                    # Zwróć bez augmentacji
            
            return x, y
        except Exception as e:
            logger.error(f"Błąd wczytywania próbki {idx} (file {file_idx}, sample {sample_idx}): {e}")
            raise

class DataLoaderFactory:
    """Fabryka produkująca gotowe paczki danych (Batches) dla modelu.
    
    Używa on-demand ładowania danych zamiast wczytywania wszystkiego do RAM.
    """

    @staticmethod
    def _is_offline_transposed(path: str) -> bool:
        name = os.path.basename(path)
        return "_T+" in name or "_T-" in name

    @staticmethod
    def create_dataloaders(data_dir: str, batch_size: int, test_size: float = 0.15) -> tuple[DataLoader, DataLoader]:
        x_all = sorted(glob.glob(os.path.join(data_dir, "*_X.npy")))
        y_all = sorted(glob.glob(os.path.join(data_dir, "*_y.npy")))

        if cfg_train.USE_OFFLINE_TRANSPOSE:
            x_files = x_all
            y_files = y_all
            logger.info("USE_OFFLINE_TRANSPOSE=True -> używam bazowych i transponowanych próbek offline.")
        else:
            x_files = [p for p in x_all if not DataLoaderFactory._is_offline_transposed(p)]
            y_files = [p for p in y_all if not DataLoaderFactory._is_offline_transposed(p)]
            logger.info("USE_OFFLINE_TRANSPOSE=False -> pomijam pliki offline transpose (_T+N/_T-N).")
        
        if not x_files or len(x_files) != len(y_files):
            raise ValueError(f"Błąd danych w {data_dir}. Zgodność plików X i y została naruszona.")
        
        # Dzielenie losowe, ale z ziarnem (random_state=42), aby przy każdym 
        # odpaleniu skryptu zbiór walidacyjny zawierał dokładnie te same utwory
        x_train_files, x_val_files, y_train_files, y_val_files = train_test_split(
            x_files, y_files, test_size=test_size, random_state=42
        )
        
        logger.info("--- Przygotowanie Danych ---")
        logger.info(f"Ładowanie {len(x_train_files)} utworów do zbioru Treningowego (on-demand)...")
        logger.info(f"Ładowanie {len(x_val_files)} utworów do zbioru Walidacyjnego (on-demand)...")
        
        # Zbuduj pipeline augmentacji (online) tylko dla train (jeśli włączone w cfg)
        train_pipeline = build_train_augment_pipeline(train_cfg=cfg_train)

        # Użyj memory-efficient datasetu, który ładuje pliki na bieżąco
        train_dataset = ChordDataset(x_train_files, y_train_files, augment_pipeline=train_pipeline)
        val_dataset = ChordDataset(x_val_files, y_val_files, augment_pipeline=None)

        # shuffle=True tylko dla treningu, aby sieć nie uczyła się piosenek "po kolei"
        # num_workers=2 pozwala na paralelne ładowanie danych
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        return train_loader, val_loader