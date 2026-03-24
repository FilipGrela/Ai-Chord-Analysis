import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MultiFileDataset(Dataset):
    """
    Klasa wczytująca listę plików .npy do pamięci RAM i tworząca z nich jeden tensor.
    Zakładamy, że 200 piosenek zmieści się w RAMie (zazwyczaj to ok. 2-4 GB).
    """
    def __init__(self, x_files, y_files):
        print(f"Ładowanie {len(x_files)} utworów do pamięci RAM...")
        X_list = []
        y_list = []

        # Wczytujemy pliki jeden po drugim
        for x_path, y_path in zip(x_files, y_files):
            X_list.append(np.load(x_path))
            y_list.append(np.load(y_path))

        # Łączymy wszystkie piosenki w jedną wielką macierz
        # X ma kształt (N, 40, 84), y ma (N,)
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)

        # Konwersja na tensory PyTorch
        # unsqueeze(1) dodaje wymiar "kanału" dla warstw konwolucyjnych -> (Batch, 1, 40, 84)
        self.X = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y_full, dtype=torch.long)

        print(f"Gotowe! Zbudowano zbiór składający się z {len(self.X)} okien treningowych.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders_from_folder(folder_path, batch_size=64):
    """
    Skanuje folder z plikami _X.npy i _y.npy, dzieli je na train/val/test
    i zwraca gotowe obiekty DataLoader.
    """
    # 1. Pobieramy i sortujemy ścieżki (sortowanie gwarantuje, że X i y będą pasować)
    x_files = sorted(glob.glob(os.path.join(folder_path, "*_X.npy")))
    y_files = sorted(glob.glob(os.path.join(folder_path, "*_y.npy")))

    if len(x_files) != len(y_files) or len(x_files) == 0:
        raise ValueError("Błąd: Brak plików w folderze lub liczba plików X i y się nie zgadza!")

    print(f"Znaleziono {len(x_files)} par plików (X, y) w folderze '{folder_path}'.")

    # 2. Dzielimy PLIKI (utwory), a nie ramki! 
    # 80% piosenek do nauki, 20% do sprawdzania
    x_train_files, x_temp, y_train_files, y_temp = train_test_split(
        x_files, y_files, test_size=0.2, random_state=42
    )
    
    # Z tych 20% robimy połowę na walidację (w trakcie treningu) i połowę na ostateczny test
    x_val_files, x_test_files, y_val_files, y_test_files = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    # 3. Tworzymy zbiory
    print("\n--- Przygotowanie zbioru Treningowego ---")
    train_dataset = MultiFileDataset(x_train_files, y_train_files)

    print("\n--- Przygotowanie zbioru Walidacyjnego ---")
    val_dataset = MultiFileDataset(x_val_files, y_val_files)

    # 4. Pakujemy w DataLoader (który będzie podawał sieci np. po 64 ramki na raz)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader