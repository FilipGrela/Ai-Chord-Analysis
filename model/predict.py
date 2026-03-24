import os
import sys
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from spectograms.spectograms import read_audio_universal, generate_spectrogram
from model import ChordCRNN

# Słownik do tłumaczenia cyfr z powrotem na format tekstowy
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
VOCAB = NOTES + [n + 'm' for n in NOTES] + ['N']
INT_TO_CHORD = {idx: chord for idx, chord in enumerate(VOCAB)}

def predict_chords(audio_path, model_path, hop_size_ms=50, seq_len=40):
    print(f"Analiza pliku: {os.path.basename(audio_path)}")
    

    device = torch.device("cpu")
    model = ChordCRNN(num_classes=25)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Załadowano wagi modelu CRNN.")
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        return

    # Audip processing
    audio_data, sample_rate = read_audio_universal(audio_path)
    if audio_data is None:
        return

    print("Generowanie spektrogramu CQT...")
    cqt_matrix = generate_spectrogram(
        audio_data, 
        sample_rate, 
        method='cqt_fast', 
        hop_size_ms=hop_size_ms,
        apply_smoothing=True, 
        apply_whitening=False, 
        apply_denoise=True
    )
    
    num_bins, num_frames = cqt_matrix.shape
    
    print("Wycinanie okien czasowych...")
    X_sequences = []
    time_stamps = []
    
    # Przesuwamy okno co 1 ramkę, aby uzyskać maksymalną precyzję czasową
    for start_idx in range(0, num_frames - seq_len + 1, 1):
        end_idx = start_idx + seq_len
        patch = cqt_matrix[:, start_idx:end_idx].T 
        X_sequences.append(patch)
        
        # Obliczamy fizyczny czas w sekundach dla środka tego okna
        center_idx = start_idx + (seq_len // 2)
        time_sec = center_idx * (hop_size_ms / 1000.0)
        time_stamps.append(time_sec)
        
    # Konwersja do Tensora PyTorch: (Liczba_Okien, 1, 40, 84)
    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32).unsqueeze(1)
    
    # 4. Inferencja
    print("Trwa wnioskowanie sieci...")
    with torch.no_grad(): # Wyłączamy śledzenie gradientów
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    predictions_int = predicted.numpy()
    
    # 5. Grupowanie wyników w bloki czasowe
    print("\n" + "="*40)
    print("ROZPOZNANE AKORDY")
    print("="*40)
    
    current_chord = None
    start_time = 0.0
    
    for time_sec, pred_int in zip(time_stamps, predictions_int):
        chord_name = INT_TO_CHORD[pred_int]
        
        # Jeśli akord uległ zmianie, drukujemy podsumowanie poprzedniego bloku
        if chord_name != current_chord:
            if current_chord is not None and current_chord != 'N':
                print(f"[{start_time:05.2f}s - {time_sec:05.2f}s] : {current_chord}")
            current_chord = chord_name
            start_time = time_sec
            
    # Drukujemy ostatni trwający akord
    if current_chord is not None and current_chord != 'N':
        print(f"[{start_time:05.2f}s - KONIEC] : {current_chord}")
    print("="*40)

if __name__ == "__main__":
    MODEL_FILE = "best_crnn_model.pth"
    # Ścieżka do docelowego pliku audio
    TEST_SONG = r"D:\SI_Studia\Ai-Chord-Analysis\spectograms\samples\Dużo Ciebie mi (Live Akustycznie).wav"
    
    predict_chords(TEST_SONG, MODEL_FILE)