import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("Karta:", torch.cuda.get_device_name(0))
print("Wysyłanie danych na GPU...")

try:
    x = torch.tensor([1.0, 2.0], device='cuda')
    y = torch.tensor([3.0, 4.0], device='cuda')
    
    wynik = x + y
    
    print("\n--- SUKCES ---")
    
except Exception as e:
    print("\n--- BŁĄD ---")
    print("Karta odmówiła współpracy. Błąd:", e)