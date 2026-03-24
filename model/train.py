import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data_loader import create_dataloaders_from_folder
from model import ChordCRNN
import os

os.environ["CUDA_MODULE_LOADING"] = "LAZY"  

def calculate_class_weights(train_loader, num_classes=25):
    class_counts = np.zeros(num_classes)
    for _, labels in train_loader:
        counts = np.bincount(labels.numpy(), minlength=num_classes)
        class_counts += counts
        
    class_counts[class_counts == 0] = 1 
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(model, train_loader, val_loader, epochs=30, learning_rate=0.001, patience=5):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")

    props = torch.cuda.get_device_properties(device)
    print(f"Urządzenie: {props.name}")
    print(f"Architektura (Capability): {props.major}.{props.minor}")

        
    model.to(device)

    class_weights = calculate_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)   
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train() 
        running_loss, correct_train, total_train = 0.0, 0, 0
        
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
            
        # Walidacja
        model.eval() 
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        val_loss /= len(val_loader)
        
        print(f"Epoka {epoch+1}: Train Acc: {100*correct_train/total_train:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_crnn_model.pth")
            print("Zapisano checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early Stopping.")
                break

if __name__ == "__main__":
    DATASET_FOLDER = r"D:\SI_Studia\Ai-Chord-Analysis\out\full_dataset" 
    train_loader, val_loader = create_dataloaders_from_folder(DATASET_FOLDER, batch_size=64)
    model = ChordCRNN(num_classes=25)
    train_model(model, train_loader, val_loader)