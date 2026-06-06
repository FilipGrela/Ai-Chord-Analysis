import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from backend.logger.logger import Logger
from backend.config import cfg_model

logger = Logger(__name__)

class LossFactory:
    """Klasa odpowiedzialna za produkcję zbalansowanych funkcji straty."""

    @staticmethod
    def _count_labels_from_y_files(y_files: list[str], num_classes: int) -> np.ndarray:
        """Count labels directly from saved *_y.npy files without touching spectrogram X files."""
        class_counts = np.zeros(num_classes, dtype=np.int64)

        for y_file in tqdm(y_files, desc="Liczenie klas z plików etykiet"):
            y_data = np.load(y_file, allow_pickle=False)
            y_arr = np.asarray(y_data, dtype=np.int64).ravel()
            if y_arr.size == 0:
                continue
            counts = np.bincount(y_arr, minlength=num_classes)
            class_counts += counts

        return class_counts

    @staticmethod
    def get_smoothed_weights(train_loader, num_classes: int | None = None) -> torch.Tensor:
        if num_classes is None:
            num_classes = cfg_model.NUM_CLASSES
        logger.info("Obliczanie wygładzonych wag klas (Label Smoothing)...")

        dataset = getattr(train_loader, "dataset", None)
        y_files = getattr(dataset, "y_files", None)

        # Najszybsza ścieżka: licz bezpośrednio z plików etykiet, bez iterowania po DataLoaderze.
        if y_files:
            class_counts = LossFactory._count_labels_from_y_files(y_files, num_classes).astype(np.float64)
        else:
            # Fallback dla nietypowych loaderów/datasetów.
            class_counts = np.zeros(num_classes, dtype=np.float64)
            for _, labels in tqdm(train_loader, desc="Liczenie klas w danych treningowych"):
                labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
                counts = np.bincount(labels_np.astype(np.int64), minlength=num_classes)
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
    def create_loss_function(cls, train_loader, device: torch.device, num_classes: int | None = None) -> nn.Module:
        """Tworzy i zwraca gotową do użycia funkcję CrossEntropyLoss z wagami."""
        if num_classes is None:
            num_classes = cfg_model.NUM_CLASSES
        weights = cls.get_smoothed_weights(train_loader, num_classes).to(device)
        from backend.config import cfg_train, cfg_analysis
        # If music-aware loss enabled, return a combined loss module
        if getattr(cfg_train, "MUSIC_AWARE_LOSS_ENABLED", False):
            # build similarity matrix based on chord vocabulary
            try:
                from backend.data.builder import DatasetBuilder
                class_names = DatasetBuilder.VOCAB
            except Exception:
                # fallback to uniform names
                class_names = [str(i) for i in range(num_classes)]

            sim_matrix = cls._build_similarity_matrix(class_names, cfg_analysis)
            return SoftLabelLoss(weight=weights, similarity_matrix=sim_matrix.to(device), alpha=cfg_train.MUSIC_LOSS_ALPHA, temperature=cfg_train.MUSIC_LOSS_TEMPERATURE, topk=cfg_train.MUSIC_LOSS_TOPK)

        return nn.CrossEntropyLoss(weight=weights)

    @staticmethod
    def _build_similarity_matrix(class_names: list, cfg_analysis) -> torch.Tensor:
        """Create a class-by-class similarity matrix in [0,1]."""
        n = len(class_names)
        S = np.zeros((n, n), dtype=float)

        # Normalizacja bemoli do krzyżyków
        ENHARMONIC_MAP = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
        }

        def split_chord(label: str):
            if label == 'N':
                return ('N', 'N', 'none')
            
            # Ekstrakcja rozszerzenia (siódemki)
            extension = '7' if label.endswith('7') else 'none'
            
            # Usunięcie '7' z końca do analizy triady
            base_label = label[:-1] if extension == '7' else label
            
            # Ekstrakcja triady i root'a
            if base_label.endswith('m'):
                root = base_label[:-1]
                triad = 'm'
            else:
                root = base_label
                triad = 'maj'
                
            # Normalizacja enharmoniczna roota
            root = ENHARMONIC_MAP.get(root, root)
                
            return (root, triad, extension)

        root_w = getattr(cfg_analysis, 'CHORD_SIMILARITY_ROOT_WEIGHT', 0.45)
        triad_w = getattr(cfg_analysis, 'CHORD_SIMILARITY_QUALITY_WEIGHT', 0.30)  # Dawne 
        ext_w = getattr(cfg_analysis, 'CHORD_SIMILARITY_EXTENSION_WEIGHT', 0.10)
        key_w = getattr(cfg_analysis, 'CHORD_SIMILARITY_KEY_WEIGHT', 0.15)

        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for i, a in enumerate(class_names):
            ra, ta, ea = split_chord(a)
            for j, b in enumerate(class_names):
                rb, tb, eb = split_chord(b)
                score = 0.0
                
                if a == b:
                    score = 1.0
                elif a == 'N' or b == 'N':
                    # Jeśli jeden to N (cisza/brak akordu), a drugi nie, podobieństwo to zawsze 0
                    score = 0.0
                else:
                    if ra == rb:
                        score += root_w
                    
                    # 1. Porównanie bazy (Triady maj/m) - Cmaj i C7 mają tę samą bazę 'maj'
                    if ta == tb:
                        score += triad_w
                        
                    # 2. Porównanie rozszerzeń (none/7)
                    if ea == eb:
                        score += ext_w
                        
                    # 3. Dystans tonalny (koło kwintowe / odległość w półtonach)
                    try:
                        if ra in NOTES and rb in NOTES:
                            idx_a = NOTES.index(ra)
                            idx_b = NOTES.index(rb)
                            dist = min((idx_a - idx_b) % 12, (idx_b - idx_a) % 12)
                            score += (1.0 - dist / 6.0) * key_w
                    except Exception:
                        pass
                        
                S[i, j] = max(0.0, min(1.0, score))

        for i in range(n):
            if S[i, i] == 0:
                S[i, i] = 1.0

        return torch.tensor(S, dtype=torch.float)


class SoftLabelLoss(nn.Module):
    """Combined hard CE loss with soft-label KL loss based on similarity matrix.

    loss = (1-alpha) * CE(hard) + alpha * KL(soft || pred)
    """
    def __init__(self, weight: torch.Tensor, similarity_matrix: torch.Tensor, alpha: float = 0.3, temperature: float = 0.3, topk: int | None = 12):
        super().__init__()
        self.register_buffer('weight', weight)
        self.sim = similarity_matrix  # (C, C)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.topk = int(topk) if topk is not None else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        # hard cross-entropy
        hard_loss = self.ce(logits, targets)

        # build soft targets from similarity matrix
        with torch.no_grad():
            S = self.sim  # (C,C)
            device = logits.device
            S = S.to(device)
            t = targets.long()
            # gather rows for each target: (B, C)
            soft = S[t]  # (B, C)
            if self.topk is not None and self.topk < soft.size(1):
                # zero out all but topk entries per row
                topk_vals, topk_idx = torch.topk(soft, k=self.topk, dim=1)
                mask = torch.zeros_like(soft)
                mask.scatter_(1, topk_idx, 1.0)
                soft = soft * mask
            # apply temperature-like sharpening via softmax over (soft / T)
            soft = torch.softmax(soft / max(1e-6, self.temperature), dim=1)

        logp = torch.log_softmax(logits, dim=1)
        # KL divergence batch mean: sum q * (log q - log p)
        kld = (soft * (torch.log(soft + 1e-12) - logp)).sum(dim=1).mean()

        total_loss = (1.0 - self.alpha) * hard_loss + self.alpha * kld

        return {
            "loss": total_loss,
            "ce_hard": hard_loss,
            "kl_soft": kld
        }