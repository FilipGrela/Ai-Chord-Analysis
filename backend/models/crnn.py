import torch
import torch.nn as nn

class ChordCRNN(nn.Module):
    """Architektura Convolutional Recurrent Neural Network dla rozpoznawania akordów."""
    
    def __init__(self, num_classes=25, dropout_rate=0.4, rnn_num_layers: int = 2):
        super(ChordCRNN, self).__init__()
        
        # BLOK CNN: Ekstrakcja cech przestrzennych z macierzy CQT
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(p=dropout_rate / 2), 
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(p=dropout_rate / 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # BLOK RNN: Zrozumienie kontekstu w czasie (sekwencje)
        # 64 filtry z CNN * 10 (wysokość wymiaru częstotliwości po poolingach) = 640
        # Dropout w GRU ma efekt tylko dla num_layers > 1.
        gru_dropout = dropout_rate if rnn_num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=640,
            hidden_size=128,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        # Z Bidirectional GRU wychodzi 128 * 2 = 256
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. Przejście przez konwolucje
        x = self.cnn(x) 
        
        # 2. Przekształcenie pod RNN: (Batch, Channels, Time, Freq) -> (Batch, Time, Channels * Freq)
        B, C, T, F = x.size()
        x = x.transpose(1, 2).contiguous() 
        x = x.view(B, T, C * F)             
        
        # 3. Przejście przez warstwy rekurencyjne
        x, _ = self.rnn(x) 
        
        # 4. Global Average Pooling (Uśrednienie wiedzy z całego dwusekundowego okna)
        x = torch.mean(x, dim=1) 
        
        # 5. Klasyfikacja
        x = self.dropout(x)
        return self.fc(x)