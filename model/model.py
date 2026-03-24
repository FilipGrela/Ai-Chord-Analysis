import torch
import torch.nn as nn

class ChordCRNN(nn.Module):
    # Dodaliśmy parametr dropout_rate (domyślnie 30% wyłączanych neuronów)
    def __init__(self, num_classes=25, dropout_rate=0.3):
        super(ChordCRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Lekki dropout dla wizji (wyłącza przestrzenne mapy)
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
        
        # PyTorch ma wbudowany dropout wewnątrz RNN
        self.rnn = nn.GRU(input_size=640, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # Mocny dropout przed ostateczną decyzją
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x) 
        
        B, C, T, F = x.size()
        x = x.transpose(1, 2).contiguous() 
        x = x.view(B, T, C * F)             
        
        x, _ = self.rnn(x) 
        x = x[:, -1, :] 
        
        # Przejście przez warstwę zapominania
        x = self.dropout(x)
        out = self.fc(x) 
        return out