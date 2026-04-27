import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import cast
from tqdm import tqdm

from backend.config import cfg_paths, cfg_audio, cfg_builder
from backend.dsp.spectrograms import AudioProcessor
from backend.data.parser import ChordLabelParser
from backend.logger.logger import Logger

logger = Logger(__name__)

class DatasetBuilder:
    """Klasa orkiestrująca budowanie zbioru danych (Multiprocessing)."""

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    VOCAB = NOTES + [n + 'm' for n in NOTES] + ['N']
    CHORD_TO_INT = {chord: idx for idx, chord in enumerate(VOCAB)}

    def __init__(self, config_paths=cfg_paths, config_audio=cfg_audio):
        self.paths = config_paths
        self.audio_cfg = config_audio
        self.processor = AudioProcessor(config=self.audio_cfg)
        self.cqt_method = cfg_builder.CQT_METHOD

    @staticmethod
    def _output_paths(folder_name: str, output_dir: str) -> tuple[str, str]:
        return (
            os.path.join(output_dir, f"{folder_name}_X.npy"),
            os.path.join(output_dir, f"{folder_name}_y.npy"),
        )

    @classmethod
    def _is_folder_already_processed(cls, folder_name: str, output_dir: str) -> bool:
        x_path, y_path = cls._output_paths(folder_name, output_dir)
        return os.path.isfile(x_path) and os.path.isfile(y_path)

    @classmethod
    def align_frames_with_labels(cls, num_frames: int, parsed_labels: list, hop_size_ms: int) -> np.ndarray:
        """
        Funkcja wyrównuje klatki nagrań z powstałe podczas fragmentacji przez algorytm, z tymi widocznymi w datasecie.
        """
        frame_labels_int = np.full(num_frames, cls.CHORD_TO_INT['N'], dtype=np.int32) # Pusta tablica wypełniona ciszą ('N')
        for frame_idx in range(num_frames):
            time_sec = frame_idx * (hop_size_ms / 1000.0) # Aktualny moment w nagraniu w sekundach
            for start, end, chord in parsed_labels:
                if start <= time_sec < end: # Sprawdzamy, czy aktualny czas mieści się w zakresie etykiety
                    # Przypisanie nowej sygnatury czasowej do klatki.
                    safe_chord = chord if chord in cls.CHORD_TO_INT else 'N'
                    frame_labels_int[frame_idx] = cls.CHORD_TO_INT[safe_chord]
                    break
        return frame_labels_int

    @classmethod
    def create_sequences(cls, cqt_matrix: np.ndarray, frame_labels_int: np.ndarray, seq_len: int, hop_seq: int = 10):
        """
        Funkcja tworzy zbiór sekwencji (X) i odpowiadających im etykiet (y)
        na podstawie macierzy CQT i wyrównanych etykiet klatkowych.
        """
        num_bins, num_frames = cqt_matrix.shape
        X_sequences, y_labels = [], []
        for start_idx in range(0, num_frames - seq_len + 1, hop_seq): 
            end_idx = start_idx + seq_len
            # Ekstrakcja wycinka nagrania o długości hop_seq. Transpozycja
            # Tak, aby sekwencje były w formacie (seq_len, num_bins)
            patch_t = cqt_matrix[:, start_idx:end_idx].T

            # Wybranie akordu, który wybrzmiewa w środkowej części fragmentu
            center_idx = start_idx + (seq_len // 2) 
            label = frame_labels_int[center_idx]
            

            X_sequences.append(patch_t)
            y_labels.append(label)
        return np.array(X_sequences), np.array(y_labels)

    @staticmethod
    def _process_single_folder(folder_path: str, output_dir: str, hop_size_ms: int, seq_len: int) -> tuple[bool, str]:
        """
        Funkcja 
        """
        folder_name = os.path.basename(folder_path)
        x_path, y_path = DatasetBuilder._output_paths(folder_name, output_dir)

        # Jeśli wynik już istnieje, nie przeliczamy utworu ponownie.
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            return True, f"Pominięto {folder_name}: wynik już istnieje"

        # Wszystkie pliki audio i labels w konkretnym folderze. 
        # Zakładamy, że mają taką samą zawartość a folder zawiera dane tylko do jednego nagrania.
        audio_files = glob.glob(os.path.join(folder_path, '*.mp3')) + \
                        glob.glob(os.path.join(folder_path, '*.wav'))
        label_files = glob.glob(os.path.join(folder_path, '*.jams')) + \
                        glob.glob(os.path.join(folder_path, '*.csv')) + \
                        glob.glob(os.path.join(folder_path, '*.txt'))

        # Sprawdzamy czy foldery nie są puste.
        if not audio_files or not label_files:
            return False, f"Pominięto {folder_name}: brak wymaganych plików"

        try:
            processor = AudioProcessor() # Lokalna instancja dla procesu

            # Próbuj odczytu kolejnych plików audio, aż któryś się wczyta poprawnie.
            audio_data, sample_rate = None, None
            for audio_file in audio_files:
                audio_data, sample_rate = processor.read_audio_universal(audio_file)
                if audio_data is not None:
                    break
                
            if audio_data is None:
                return False, f"Błąd odczytu audio: {folder_name}"  

            audio_data = cast(np.ndarray, audio_data)

            # Próbuj kolejnych plików etykiet, aż któryś będzie niepusty i poprawnie się sparsuje.
            parsed_labels = None
            for label_file in label_files:
                if os.path.getsize(label_file) == 0:
                    continue
                try:
                    parsed_labels = ChordLabelParser.parse(label_file)
                    break
                except Exception:
                    continue

            if parsed_labels is None:
                return False, f"Pominięto {folder_name}: brak poprawnego pliku etykiet"
            cqt_matrix = processor.generate_spectrogram(method=cfg_builder.CQT_METHOD, audio_data=audio_data)

            num_frames = cqt_matrix.shape[1]
            frame_labels_int = DatasetBuilder.align_frames_with_labels(num_frames, parsed_labels, hop_size_ms)
            X, y = DatasetBuilder.create_sequences(cqt_matrix, frame_labels_int, seq_len)

            np.save(x_path, X.astype(np.float32))
            np.save(y_path, y.astype(np.int64))

            return True, folder_name
        except Exception as e:
            return False, f"Błąd krytyczny w {folder_name}: {str(e)}"

    def build_entire_dataset(self):
        # Tworzy folder na output, jeżeli wcześniej nie istnieje.
        if not os.path.exists(self.paths.PROCESSED_DATA):
            os.makedirs(self.paths.PROCESSED_DATA)

        #  Pobieranie listy albumów (katalogów wewnątrz RAW_DATA)
        albums = [f.path for f in os.scandir(self.paths.RAW_DATA) if f.is_dir()]
        total_albums = len(albums)
        logger.info(f"Rozpoczynam zrównoleglone przetwarzanie {total_albums} albumów...")

        # Obliczamy bezpieczną liczbę "robotników" (procesów)
        total_cores = os.cpu_count() or 4
        safe_workers = max(1, total_cores - 2) # Zawsze zostawiamy 2 wolne wątki, ale nie mniej niż 1 do pracy
        
        logger.info(f"Używam {safe_workers} procesów z {total_cores} dostępnych na Twoim CPU.")

        # Przygotowanie funkcji roboczej
        worker_func = partial(self._process_single_folder, output_dir=self.paths.PROCESSED_DATA, 
                              hop_size_ms=self.audio_cfg.HOP_SIZE_MS, seq_len=self.audio_cfg.SEQ_LEN)

        results, errors = [], []
        skipped = 0

        # Najpierw liczymy realną liczbę utworów do przetworzenia, aby globalny pasek był wiarygodny.
        total_pending_tracks = 0
        album_track_map: list[tuple[str, list[str], list[str]]] = []
        for album_path in albums:
            tracks = [f.path for f in os.scandir(album_path) if f.is_dir()]
            pending_tracks = [
                track for track in tracks
                if not self._is_folder_already_processed(os.path.basename(track), self.paths.PROCESSED_DATA)
            ]
            total_pending_tracks += len(pending_tracks)
            album_track_map.append((album_path, tracks, pending_tracks))

        if total_pending_tracks == 0:
            logger.info("Nie ma już nic do przeliczenia — wszystkie utwory mają gotowe pliki X/y.")
            return

        # Inicjalizacja puli procesów (robimy to raz dla całej operacji, aby oszczędzić zasoby)
        with ProcessPoolExecutor(max_workers=safe_workers) as executor, \
                tqdm(total=total_pending_tracks,
                     desc="Budowanie datasetu",
                     position=0,
                     leave=True,
                     dynamic_ncols=True,
                     mininterval=0.1,
                     smoothing=0.05) as overall_bar:

            for album_path, tracks, pending_tracks in album_track_map:
                album_name = os.path.basename(album_path)
                skipped += len(tracks) - len(pending_tracks)

                if not pending_tracks:
                    logger.info(f"Pominięto album {album_name}: wszystkie utwory są już gotowe.")
                    continue

                logger.info(f"Album {album_name}: do przetworzenia {len(pending_tracks)} utworów, pominięto {len(tracks) - len(pending_tracks)}.")
                overall_bar.set_description_str(f"Budowanie datasetu | {album_name}")
                overall_bar.set_postfix_str(f"pozostało {overall_bar.total - overall_bar.n} utworów")

                futures = [executor.submit(worker_func, track) for track in pending_tracks]
                processed_in_album = 0

                for future in as_completed(futures):
                    success, msg = future.result()
                    processed_in_album += 1
                    overall_bar.update(1)
                    overall_bar.set_postfix_str(
                        f"album {processed_in_album}/{len(pending_tracks)} | pozostało {overall_bar.total - overall_bar.n}"
                    )
                    if success:
                        results.append(msg)
                    else:
                        errors.append(msg)

        # Dodatkowe przełamanie linii (\n), aby oddzielić wynik od pasków tqdm
        logger.info(f"Ukończono! Sukcesy: {len(results)}, Pominięte: {skipped}, Błędy: {len(errors)}")
        if errors:
            logger.warning("Raport błędów:")
            for err in errors: 
                logger.error(f" - {err}")