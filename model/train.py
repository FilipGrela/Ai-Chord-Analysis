import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=30, learning_rate=0.001, patience=5):
    device = torch.device("cpu") # Zostajemy na stabilnym CPU!
    print(f"Rozpoczęcie  treningu na urządzeniu: {device}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Dodano Weight Decay (L2 Regularization) równe 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Zmienne do Early Stopping i Checkpointingu
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        train_accuracy = 100 * correct_train / total_train
        
        # --- WALIDACJA ---
        model.eval() 
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct_val / total_val
        val_loss /= len(val_loader)
        
        print(f"Epoka {epoch+1} Podsumowanie:")
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # --- MODEL CHECKPOINTING & EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Nadpisuje plik na dysku tylko wtedy, gdy model jest lepszy niż poprzednio!
            torch.save(model.state_dict(), "best_crnn_model.pth")
            print("Zapisano nowy, lepszy model na dysku!\n")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Brak poprawy od {epochs_no_improve} epok.\n")
            if epochs_no_improve >= patience:
                print("Early Stopping: Model przestał się uczyć reguł. Przerywamy trening!")
                break


if __name__ == "__main__":
    from data_loader import create_dataloaders_from_folder
    
    # Wskazujesz folder, w którym masz te 200 par plików X i y
    DATASET_FOLDER = r"D:\SI_Studia\Ai-Chord-Analysis\out\full_dataset" 
    
    # 1. Tworzymy Loadery
    train_loader, val_loader = create_dataloaders_from_folder(DATASET_FOLDER, batch_size=64)
    
    # 2. Inicjalizujemy model CRNN z pliku model.py
    from model import ChordCRNN
    crnn_model = ChordCRNN(num_classes=25)
    
    # 3. Odpalamy trening!
    train_model(crnn_model, train_loader, val_loader, epochs=30, learning_rate=0.001)