import torch
import numpy as np
import os
from backend.config import cfg_paths, cfg_audio
from backend.models.crnn import ChordCRNN
from backend.dsp.spectrograms import AudioProcessor

class ChordInferenceEngine:
    """Silnik analityczny wywołujący wytrenowany model na nowych plikach audio."""
    
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    VOCAB = NOTES + [n + 'm' for n in NOTES] + ['N']
    INT_TO_CHORD = {idx: chord for idx, chord in enumerate(VOCAB)}

    def __init__(self, model_path: str | None = None, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AudioProcessor()
        
        # Inicjalizacja modelu i wczytanie wag
        self.model = ChordCRNN(num_classes=len(self.VOCAB)).to(self.device)
        load_path = model_path or cfg_paths.MODEL_SAVE_PATH
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {load_path}. Wytrenuj go najpierw!")
            
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.eval() # Tryb ewaluacji (wyłącza Dropout)

    def _create_inference_sequences(self, cqt_matrix: np.ndarray):
        """Tnie spektrogram na okna przesuwne (Sliding Window)."""
        seq_len = cfg_audio.SEQ_LEN
        hop_seq = 10  # Krok przesuwu okna (definiuje rozdzielczość czasową)
        num_bins, num_frames = cqt_matrix.shape
        
        X_sequences, timestamps = [], []
        
        for start_idx in range(0, num_frames - seq_len + 1, hop_seq):
            end_idx = start_idx + seq_len
            patch_t = cqt_matrix[:, start_idx:end_idx].T 
            X_sequences.append(patch_t)
            
            # Dokładny czas na środku badanego okna
            center_idx = start_idx + (seq_len // 2)
            time_sec = center_idx * (cfg_audio.HOP_SIZE_MS / 1000.0)
            timestamps.append(time_sec)
            
        return np.array(X_sequences), timestamps

    def _merge_consecutive_chords(self, frame_results: list) -> list:
        """Grupuje powtarzające się akordy w czytelne przedziały czasowe."""
        if not frame_results:
            return []
            
        merged = []
        current_chord = frame_results[0]['chord']
        start_time = frame_results[0]['time']
        
        for i in range(1, len(frame_results)):
            if frame_results[i]['chord'] != current_chord:
                merged.append({
                    "start": start_time,
                    "end": frame_results[i]['time'],
                    "chord": current_chord
                })
                current_chord = frame_results[i]['chord']
                start_time = frame_results[i]['time']
                
        # Zapisanie ostatniego bloku
        merged.append({
            "start": start_time,
            "end": frame_results[-1]['time'] + (cfg_audio.HOP_SIZE_MS * 10 / 1000.0),
            "chord": current_chord
        })
        return merged

    def predict(self, audio_path: str) -> list[dict]:
        """Główna metoda dla API. Przyjmuje ścieżkę, oddaje listę JSON."""
        # 1. Przetworzenie audio -> CQT
        audio_data, sr = self.processor.read_audio_universal(audio_path)
        if audio_data is None:
            raise ValueError("Błąd dekodowania FFmpeg.")
            
        cqt_matrix = self.processor.generate_spectrogram(audio_data)
        
        # 2. Pocięcie na sekwencje
        X, timestamps = self._create_inference_sequences(cqt_matrix)
        
        # 3. Przygotowanie tensorów pod Conv2D: (Batch, 1, Time, Freq)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # 4. Szybkie wnioskowanie na GPU
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # 5. Tłumaczenie tensorów na słownik wyników
        predicted_classes = predicted.cpu().numpy()
        results = [
            {"time": t, "chord": self.INT_TO_CHORD[cls_idx]} 
            for t, cls_idx in zip(timestamps, predicted_classes)
        ]
            
        return self._merge_consecutive_chords(results)