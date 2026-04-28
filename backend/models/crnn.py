import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.config import cfg_model


class AttentionBlock(nn.Module):
    """Mechanizm uwagi (Dot-product Attention), który zastępuje global pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 192):
        super(AttentionBlock, self).__init__()
        # Warstwa mapująca ukryte stany GRU na wagi ważności (skalarny wynik dla każdej ramki)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Hidden_Dim)

        # 1. Obliczamy nieunormowane wagi uwagi
        weights = self.attention(x)  # (Batch, Time, 1)

        # 2. Normalizacja wag przez Softmax wzdłuż wymiaru czasu
        # Dzięki temu suma wag dla całego okna czasowego wynosi 1.0
        attn_weights = F.softmax(weights, dim=1)

        # 3. Ważona suma: mnożymy każdą ramkę przez jej wagę i sumujemy czas
        # (Batch, Time, Hidden_Dim) * (Batch, Time, 1) -> sumujemy po Time
        context_vector = torch.sum(x * attn_weights, dim=1)

        return context_vector, attn_weights


class ChordCRNN(nn.Module):
    """Architektura CRNN z mechanizmem Attention dla lepszej reprezentacji akordów."""

    def __init__(
        self,
        num_classes: int = cfg_model.NUM_CLASSES,
        dropout_rate: float = cfg_model.DROPOUT,
        rnn_num_layers: int = cfg_model.RNN_NUM_LAYERS,
        rnn_hidden_size: int = cfg_model.RNN_HIDDEN_SIZE,
        cnn_channels: tuple[int, int, int] = cfg_model.CNN_CHANNELS,
        n_bins: int = cfg_model.N_BINS,
    ):
        super(ChordCRNN, self).__init__()

        c1, c2, c3 = cnn_channels

        # BLOK CNN (bez zmian)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(p=dropout_rate / 2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(p=dropout_rate / 2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(p=dropout_rate)
        )

        # BLOK RNN (bez zmian)
        gru_dropout = dropout_rate if rnn_num_layers > 1 else 0.0
        pooled_freq_bins = n_bins // 8
        self.rnn = nn.GRU(
            input_size=c3 * pooled_freq_bins,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout
        )

        # --- NOWOŚĆ: Blok Attention ---
        # Wejście: 2 * hidden_size (wynik bidirectional GRU)
        self.attention_head = AttentionBlock(2 * rnn_hidden_size, hidden_dim=rnn_hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(2 * rnn_hidden_size, num_classes)

    def forward(self, x, return_attention=False):
        # 1. CNN
        x = self.cnn(x)

        # 2. Reshape: (B, C, T, F) -> (B, T, C*F)
        B, C, T, F = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, T, C * F)

        # 3. GRU`
        x, _ = self.rnn(x)  # x: (B, T, 256)

        # 4. ZAMIANA: Zamiast torch.mean(x, dim=1), używamy Attention
        # x_weighted staje się "podsumowaniem" okna o stałym rozmiarze (B, 256)
        x_weighted, attn_weights = self.attention_head(x)

        # 5. Klasyfikacja
        x_out = self.dropout(x_weighted)
        logits = self.fc(x_out)

        if return_attention:
            return logits, attn_weights
        return logits