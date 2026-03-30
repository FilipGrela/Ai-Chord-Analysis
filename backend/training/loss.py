import torch
import torch.nn as nn
import numpy as np

class LossFactory:
    """Klasa odpowiedzialna za produkcję zbalansowanych funkcji straty."""

    @staticmethod
    def get_smoothed_weights(train_loader, num_classes: int = 25) -> torch.Tensor:
        print("Obliczanie wygładzonych wag klas (Label Smoothing)...")
        class_counts = np.zeros(num_classes)
        
        # Zliczanie wystąpień klas
        for _, labels in train_loader:
            counts = np.bincount(labels.numpy(), minlength=num_classes)
            class_counts += counts
        
        # Ochrona przed dzieleniem przez zero dla pustych klas
        class_counts[class_counts == 0] = 1 
        total_samples = np.sum(class_counts)
        
        # Złagodzenie wag pierwiastkiem kwadratowym (chroni przed eksplozją błędu)
        raw_weights = np.sqrt(total_samples / class_counts)
        
        # Normalizacja wag, aby ich średnia wynosiła 1.0 (stabilizuje to Learning Rate)
        normalized_weights = raw_weights / np.mean(raw_weights)
        
        return torch.tensor(normalized_weights, dtype=torch.float)

    @classmethod
    def create_loss_function(cls, train_loader, device: torch.device, num_classes: int = 25) -> nn.Module:
        """Tworzy i zwraca gotową do użycia funkcję CrossEntropyLoss z wagami."""
        weights = cls.get_smoothed_weights(train_loader, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=weights)