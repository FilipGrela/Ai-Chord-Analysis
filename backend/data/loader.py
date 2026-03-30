import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ChordDataset(Dataset):
    """Pojedynczy pojemnik PyTorch na dane ułożone w pamięci RAM."""
    
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray):
        # Dodajemy pusty wymiar kanału (C=1) wymagany przez Conv2d: (Batch, 1, seq_len, 84)
        self.X = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y_data, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataLoaderFactory:
    """Fabryka produkująca gotowe paczki danych (Batches) dla modelu."""

    @staticmethod
    def create_dataloaders(data_dir: str, batch_size: int, test_size: float = 0.15) -> tuple[DataLoader, DataLoader]:
        x_files = sorted(glob.glob(os.path.join(data_dir, "*_X.npy")))
        y_files = sorted(glob.glob(os.path.join(data_dir, "*_y.npy")))
        
        if not x_files or len(x_files) != len(y_files):
            raise ValueError(f"Błąd danych w {data_dir}. Zgodność plików X i y została naruszona.")
            
        # Dzielenie losowe, ale z ziarnem (random_state=42), aby przy każdym 
        # odpaleniu skryptu zbiór walidacyjny zawierał dokładnie te same utwory
        x_train_files, x_val_files, y_train_files, y_val_files = train_test_split(
            x_files, y_files, test_size=test_size, random_state=42
        )
        
        print(f"--- Przygotowanie Danych ---")
        print(f"Ładowanie {len(x_train_files)} utworów do zbioru Treningowego...")
        X_train = np.vstack([np.load(f) for f in x_train_files])
        y_train = np.concatenate([np.load(f) for f in y_train_files])
        
        print(f"Ładowanie {len(x_val_files)} utworów do zbioru Walidacyjnego...")
        X_val = np.vstack([np.load(f) for f in x_val_files])
        y_val = np.concatenate([np.load(f) for f in y_val_files])
        
        train_dataset = ChordDataset(X_train, y_train)
        val_dataset = ChordDataset(X_val, y_val)
        
        # shuffle=True tylko dla treningu, aby sieć nie uczyła się piosenek "po kolei"
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader