from glob import glob
import os
import torch
from backend.api.inference import ChordInferenceEngine
from backend.config import cfg_paths

# Optymalizacja ładowania kerneli Blackwell dla szybkiej predykcji
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

def main():
    print("--- Inicjalizacja Modułu AI-Chord-Analysis ---")
    
    try:
        # Silnik automatycznie załaduje best_crnn_model.pth z folderu 'out'
        engine = ChordInferenceEngine()
    except FileNotFoundError as e:
        print(f"BŁĄD: {e}")
        return

    # Używamy utworu z folderu testowego
    search_path = cfg_paths.SINGLE_TEST_DATA

    # Szukamy wszystkich plików mp3 i wav
    audio_files = glob(os.path.join(search_path, "*.mp3")) + \
                glob(os.path.join(search_path, "*.wav"))

    # Sprawdzamy, czy cokolwiek znaleziono
    if audio_files:
        test_song = audio_files[0]  # Bierze pierwszy znaleziony plik
        print(f"Znaleziono utwór: {test_song}")
    else:
        test_song = None
        raise Exception("Nie znaleziono plików .mp3 ani .wav w podanym folderze.")
    
    if not os.path.exists(test_song):
        print(f"Nie znaleziono pliku testowego: {test_song}")
        return

    print(f"Trwa analiza AI pliku: {os.path.basename(test_song)} ...\n")
    results = engine.predict(test_song)
    
    print("=== WYNIKI ANALIZY ===")
    for block in results:
        print(f"[{block['start']:05.2f}s - {block['end']:05.2f}s] : {block['chord']}")

if __name__ == "__main__":
    main()