import torch
import numpy as np
import os
import re
from pathlib import Path
from backend.config import cfg_paths, cfg_audio, cfg_builder
from backend.models.crnn import ChordCRNN
from backend.dsp.spectrograms import AudioProcessor
from backend.logger.logger import Logger
from backend.event_system.event_bus import event_bus, LogLevel

logger = Logger(__name__)


class ChordInferenceEngine:
    """Silnik analityczny wywołujący wytrenowany model na nowych plikach audio."""

    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    @staticmethod
    def _build_vocab(support_sevenths: bool = False) -> list[str]:
        """Buduje VOCAB na podstawie flagi SUPPORT_SEVENTHS.
        
        Args:
            support_sevenths: bool - czy uwzględniać akordy septymowe (C7, Cm7)
        
        Returns:
            list[str] - lista akordów w słowniku
        """
        NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        if support_sevenths:
            return (
                NOTES 
                + [n + "m" for n in NOTES]
                + [n + "7" for n in NOTES]
                + [n + "m7" for n in NOTES]
                + ["N"]
            )
        else:
            return NOTES + [n + "m" for n in NOTES] + ["N"]
    
    @staticmethod
    def _infer_support_sevenths_from_vocab_size(num_classes: int) -> bool:
        """Wnioskuje czy model wspiera akordy septymowe na podstawie liczby klas.
        
        Args:
            num_classes: int - liczba klas modelu
        
        Returns:
            bool - True jeśli model ma 49 klas (z septymowymi), False jeśli 25 (bez)
        """
        if num_classes == 49:
            return True
        elif num_classes == 25:
            return False
        else:
            logger.warning(f"Nieznana liczba klas: {num_classes}. Zakładam SUPPORT_SEVENTHS=False.")
            return False

    @staticmethod
    def _extract_state_dict(checkpoint: dict) -> dict:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    @staticmethod
    def _infer_rnn_layers_from_state_dict(state_dict: dict) -> int:
        layer_ids = []
        for key in state_dict.keys():
            match = re.match(r"^rnn\.weight_ih_l(\d+)(?:_reverse)?$", key)
            if match:
                layer_ids.append(int(match.group(1)))
        return (max(layer_ids) + 1) if layer_ids else 2

    @staticmethod
    def _resolve_model_path(load_path: str) -> str:
        if os.path.exists(load_path):
            return load_path

        candidate_dir = Path(load_path).parent
        pth_files = sorted(
            candidate_dir.glob("*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pth_files:
            logger.warning(
                f"Nie znaleziono dokładnego pliku modelu: {load_path}. Używam najnowszego checkpointu: {pth_files[0]}"
            )
            return str(pth_files[0])

        raise FileNotFoundError(
            f"Nie znaleziono pliku modelu: {load_path}. Wytrenuj go najpierw!"
        )

    def __init__(
        self, model_path: str | None = None, device: torch.device | None = None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AudioProcessor()
        load_path = self._resolve_model_path(model_path or cfg_paths.MODEL_SAVE_PATH)

        checkpoint = torch.load(load_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        
        # Wczytaj konfigurację z checkpointa
        support_sevenths = cfg_builder.SUPPORT_SEVENTHS
        metadata = {}
        if isinstance(checkpoint, dict):
            metadata = checkpoint.get("metadata", {})
            config_snapshot = metadata.get("config", {})
            if config_snapshot:
                logger.info(f"Checkpoint config: {config_snapshot}")
                # Jeśli dostępna jest informacja o SUPPORT_SEVENTHS w checkpoincie, użyj jej
                builder_config = config_snapshot.get("builder", {})
                if "SUPPORT_SEVENTHS" in builder_config:
                    support_sevenths = builder_config["SUPPORT_SEVENTHS"]
        
        rnn_num_layers = self._infer_rnn_layers_from_state_dict(state_dict)
        
        # Wnioskuj liczę klas z rozmiaru FC warstwy
        num_classes = None
        for key in state_dict.keys():
            if key.startswith("fc.weight"):
                num_classes = state_dict[key].shape[0]
                break
        
        if num_classes is None:
            # Fallback: spróbuj wnioskować z konfigu checkpointa
            model_config = metadata.get("config", {}).get("model", {})
            num_classes = model_config.get("NUM_CLASSES", 25)
            logger.warning(f"Nie mogę wnioskować num_classes z state_dict, używam: {num_classes}")
        
        # Wnioskuj czy jest SUPPORT_SEVENTHS na podstawie liczby klas jeśli nie mamy jawnie
        inferred_sevenths = self._infer_support_sevenths_from_vocab_size(num_classes)
        if inferred_sevenths and not support_sevenths:
            logger.warning(f"Model ma {num_classes} klas (z septymowymi), ale config ma SUPPORT_SEVENTHS=False. Korygując...")
            support_sevenths = True
        
        # Buduj VOCAB na podstawie wnioskowanej flagi
        self.VOCAB = self._build_vocab(support_sevenths)
        self.INT_TO_CHORD = {idx: chord for idx, chord in enumerate(self.VOCAB)}
        
        logger.info(f"Zbudowany VOCAB: {len(self.VOCAB)} akordów (SUPPORT_SEVENTHS={support_sevenths})")
        logger.info(f"Wczytywanie modelu z {rnn_num_layers} warstwami GRU, {num_classes} klasami z pliku: {load_path}")

        # Inicjalizacja modelu zgodnie z architekturą zapisaną w checkpoint.
        self.model = ChordCRNN(
            num_classes=num_classes, rnn_num_layers=rnn_num_layers
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Tryb ewaluacji (wyłącza Dropout)

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

        if not X_sequences:
            return np.empty((0, seq_len, num_bins), dtype=np.float32), timestamps

        return np.array(X_sequences), timestamps

    def _merge_consecutive_chords(self, frame_results: list) -> list:
        """Grupuje powtarzające się akordy w czytelne przedziały czasowe."""
        if not frame_results:
            return []

        merged = []
        current_chord = frame_results[0]["chord"]
        start_time = frame_results[0]["time"]

        for i in range(1, len(frame_results)):
            if frame_results[i]["chord"] != current_chord:
                merged.append(
                    {
                        "start": start_time,
                        "end": frame_results[i]["time"],
                        "chord": current_chord,
                    }
                )
                current_chord = frame_results[i]["chord"]
                start_time = frame_results[i]["time"]

        # Zapisanie ostatniego bloku
        merged.append(
            {
                "start": start_time,
                "end": frame_results[-1]["time"]
                + (cfg_audio.HOP_SIZE_MS * 10 / 1000.0),
                "chord": current_chord,
            }
        )
        return merged

    def predict(self, audio_path: str) -> list[dict]:
        """Główna metoda dla API. Przyjmuje ścieżkę, oddaje listę JSON."""
        logger.info(f"Predict: rozpoczynam analizę pliku {audio_path}")
        # 1. Przetworzenie audio -> CQT
        audio_data, sr = self.processor.read_audio_universal(audio_path)
        if audio_data is None:
            raise ValueError("Błąd dekodowania FFmpeg.")
        event_bus.log_message.emit(LogLevel.INFO, "Wczytano ścieżkę audio")
        event_bus.progress_updated.emit(5, "Wczytano ścieżkę audio")

        event_bus.log_message.emit(LogLevel.INFO, "Generowanie spektrogramu")
        cqt_matrix = self.processor.generate_spectrogram(audio_data)
        event_bus.log_message.emit(LogLevel.SUCCESS, "Utworzono spektrogram dla ścieżki audio")
        event_bus.progress_updated.emit(50, "Utworzono spektrogram")

        # 2. Pocięcie na sekwencje
        logger.info("Tworzę sekwencje wejściowe dla modelu (sliding windows)")
        event_bus.log_message.emit(LogLevel.INFO, "Cięcie ścieżki na sekwencje")
        X, timestamps = self._create_inference_sequences(cqt_matrix)

        if X.shape[0] == 0:
            logger.warning(
                "Plik audio jest zbyt krótki, aby utworzyć sekwencje wejściowe dla modelu. Zwracam pusty wynik."
            )
            return []
        event_bus.progress_updated.emit(60, "Pocięto ścieżkę na sekwencję")

        # 3. Przygotowanie tensorów pod Conv2D: (Batch, 1, Time, Freq)
        event_bus.log_message.emit(LogLevel.INFO, "Przygotowywanie tensorów")
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        event_bus.progress_updated.emit(70, "Przygotowano tensory")

        # 4. Szybkie wnioskowanie na GPU
        event_bus.log_message.emit(LogLevel.INFO, "Wnioskowanie")
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        event_bus.progress_updated.emit(80, "Zakończono wnioskowanie")

        # 5. Tłumaczenie tensorów na słownik wyników
        event_bus.log_message.emit(LogLevel.INFO, "Tłumaczenie tensorów na słownik wyników")
        predicted_classes = predicted.cpu().numpy()
        results = [
            {"time": t, "chord": self.INT_TO_CHORD[cls_idx]}
            for t, cls_idx in zip(timestamps, predicted_classes)
        ]
        event_bus.progress_updated.emit(90, "Utworzono słownik wyników")

        return self._merge_consecutive_chords(results)
