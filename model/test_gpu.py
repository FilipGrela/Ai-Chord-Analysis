import torch
import os

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

def check_blackwell_support():
    print(f"Wersja PyTorch: {torch.__version__}")
    print(f"Wersja CUDA wewnątrz Torch: {torch.version.cuda}")
    
    if not torch.cuda.is_available():
        print("BŁĄD: CUDA nadal niedostępna.")
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"Urządzenie: {props.name}")
    print(f"Architektura (Capability): {props.major}.{props.minor}")

    try:
        # Test właściwych obliczeń na GPU
        a = torch.randn(1000, 1000).to(device)
        b = torch.randn(1000, 1000).to(device)
        c = torch.matmul(a, b)
        print("SUKCES: Obliczenia na RTX 5070 przebiegły pomyślnie.")
    except Exception as e:
        print(f"BŁĄD PODCZAS OBLICZEŃ: {e}")

if __name__ == "__main__":
    check_blackwell_support()