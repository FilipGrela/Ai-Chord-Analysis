import numpy as np
import glob
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import cast
from tqdm import tqdm

from backend.config import cfg_paths, cfg_audio, cfg_builder
from backend.dsp.spectrograms import AudioProcessor
from backend.data.parser import ChordLabelParser
from backend.data.augment.label_ops import ChordTranspose
from backend.logger.logger import Logger

logger = Logger(__name__)

class DatasetBuilder:
    """Klasa orkiestrująca budowanie zbioru danych (Multiprocessing)."""

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Build VOCAB optionally including seventh chords (C7, Cm7) based on config
    if getattr(cfg_builder, 'SUPPORT_SEVENTHS', False):
        VOCAB = (
            NOTES
            + [n + 'm' for n in NOTES]
            + [n + '7' for n in NOTES]
            + [n + 'm7' for n in NOTES]
            + ['N']
        )
    else:
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
        
        Zoptymalizowana dla oszczędzania pamięci: zapisuje rezultaty w chunkami
        zamiast akumulować wszystko w listach.
        """
        num_bins, num_frames = cqt_matrix.shape
        
        # Wstępnie oblicz ile będzie sekwencji
        num_sequences = max(0, (num_frames - seq_len) // hop_seq + 1)
        
        if num_sequences == 0:
            return (
                np.empty((0, seq_len, num_bins), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        
        # Prealokuj tablice zamiast używać list (oszczędza pamięć)
        X_sequences = np.empty((num_sequences, seq_len, num_bins), dtype=np.float32)
        y_labels = np.empty((num_sequences,), dtype=np.int64)
        
        seq_idx = 0
        for start_idx in range(0, num_frames - seq_len + 1, hop_seq):
            end_idx = start_idx + seq_len
            # Ekstrakcja wycinka nagrania o długości seq_len. Transpozycja
            # Tak, aby sekwencje były w formacie (seq_len, num_bins)
            X_sequences[seq_idx] = cqt_matrix[:, start_idx:end_idx].T

            # Wybranie akordu, który wybrzmiewa w środkowej części fragmentu
            center_idx = start_idx + (seq_len // 2)
            y_labels[seq_idx] = frame_labels_int[center_idx]
            
            seq_idx += 1
        
        return X_sequences[:seq_idx], y_labels[:seq_idx]

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

            # Zwolnij audio zaraz po jego użyciu — oszczędzaj RAM
            del audio_data, processor
            
            num_frames = cqt_matrix.shape[1]
            frame_labels_int = DatasetBuilder.align_frames_with_labels(num_frames, parsed_labels, hop_size_ms)
            X, y = DatasetBuilder.create_sequences(cqt_matrix, frame_labels_int, seq_len)

            # Zwolnij spektrogram zaraz po utworzeniu sekwencji
            del cqt_matrix, frame_labels_int
            
            np.save(x_path, X.astype(np.float32))
            np.save(y_path, y.astype(np.int64))
            
            # Zwolnij sekwencje
            del X, y
            gc.collect()  # Wymuś garbage collection aby zwolnić RAM natychmiast

            return True, folder_name
        except Exception as e:
            return False, f"Błąd krytyczny w {folder_name}: {str(e)}"

    @staticmethod
    def _transpose_spectrogram(spec: np.ndarray, semitones: int) -> np.ndarray:
        """
        Transponuje spektrogram CQT poprzez przesunięcie osi częstotliwości.
        
        Args:
            spec: np.ndarray - spektrogram o wymiarach (num_bins, num_frames)
            semitones: int - liczba półtonów do transponowania (dodatnia lub ujemna)
        
        Returns:
            np.ndarray - transponowany spektrogram
        """
        if semitones == 0:
            return spec

        bins_to_shift = semitones % 12  # Zawijamy do zakresu ±12 półtonów (1 oktawa)

        # Przesunięcie wzdłuż osi binów (częstotliwości)
        shifted = np.roll(spec, bins_to_shift, axis=0)

        # Zerowanie przesunętych binów
        if bins_to_shift > 0:
            shifted[:bins_to_shift, :] = 0  # Zeruj początek
        elif bins_to_shift < 0:
            shifted[bins_to_shift:, :] = 0  # Zeruj koniec
        
        return shifted

    @classmethod
    def _transpose_sequences(cls, X: np.ndarray, y: np.ndarray, semitones: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Transpozycja sekwencji spektrogramu (X) i ich etykiet (y).
        
        Args:
            X: np.ndarray - sekwencje spektrogramu (num_sequences, seq_len, num_bins)
            y: np.ndarray - etykiety (num_sequences,) jako indeksy do CHORD_TO_INT
            semitones: int - liczba półtonów
        
        Returns:
            Tuple[np.ndarray, np.ndarray] - (transponowane X, transponowane y)
        """
        if semitones == 0:
            return X, y
        
        X_transposed = []
        for patch in X:
            # patch: (seq_len, num_bins)
            # Transponuj każdy patch poprzez transpozycję spektrogramu
            transposed_patch = cls._transpose_spectrogram(patch.T, semitones).T
            X_transposed.append(transposed_patch)
        
        X_transposed = np.array(X_transposed, dtype=np.float32)
        
        # Dla etykiet: transpozycja etykiet akordów
        int_to_chord = tuple(cls.CHORD_TO_INT.keys())
        chord_names = [int_to_chord[idx] for idx in y]
        transposed_chords = [ChordTranspose.transpose_chord_label(chord, semitones) for chord in chord_names]
        y_transposed = np.array([cls.CHORD_TO_INT[chord] for chord in transposed_chords], dtype=np.int64)
        
        return X_transposed, y_transposed

    def apply_offline_transposition(self, semitones_list: list[int] | None = None) -> None:
        """
        Aplikuj offline transpozycję do już wygenerowanego datasetu.
        Generuje dodatkowe pliki z transpozycjonowanymi danymi.
        
        Args:
            semitones_list: list[int] - lista półtonów, dla których generować dane (np. [-2, -1, 1, 2])
                                        Jeśli None, nic się nie robi.
        """
        if semitones_list is None or len(semitones_list) == 0:
            logger.info("Offline transpozycja wyłączona.")
            return
        
        # Filtruj 0 z listy (nie ma sensu)
        semitones_list = [s for s in semitones_list if s != 0]
        
        if len(semitones_list) == 0:
            logger.info("Offline transpozycja wyłączona (brak półtonów do zastosowania).")
            return
        
        logger.info(f"Rozpoczynam offline transpozycję dla {len(semitones_list)} wariantów: {semitones_list}")
        
        # Znajdź wszystkie pliki X/y w PROCESSED_DATA
        x_files = sorted(glob.glob(os.path.join(self.paths.PROCESSED_DATA, "*_X.npy")))
        total_files = len(x_files)
        
        if total_files == 0:
            logger.warning(f"Brak danych do transpozycji w {self.paths.PROCESSED_DATA}")
            return
        
        with tqdm(total=total_files * len(semitones_list), desc="Transpozycja offline", leave=True) as pbar:
            for x_path in x_files:
                y_path = x_path.replace("_X.npy", "_y.npy")
                
                if not os.path.exists(y_path):
                    pbar.update(len(semitones_list))
                    continue
                
                try:
                    X = np.load(x_path, allow_pickle=False)
                    y = np.load(y_path, allow_pickle=False)
                    
                    base_name = os.path.basename(x_path).replace("_X.npy", "")
                    
                    for semitones in semitones_list:
                        X_transposed, y_transposed = self._transpose_sequences(X, y, semitones)
                        
                        # Zapisz z prefiksem do nazwy
                        suffix = f"_T{semitones:+d}"  # Np. _T-2, _T+3
                        x_out = os.path.join(self.paths.PROCESSED_DATA, f"{base_name}{suffix}_X.npy")
                        y_out = os.path.join(self.paths.PROCESSED_DATA, f"{base_name}{suffix}_y.npy")
                        
                        np.save(x_out, X_transposed)
                        np.save(y_out, y_transposed)
                        pbar.update(1)
                
                except Exception as e:
                    logger.error(f"Błąd transpozycji {x_path}: {e}")
                    pbar.update(len(semitones_list))
        
        logger.info("Offline transpozycja ukończona.")

    def build_entire_dataset(self):
        # Tworzy folder na output, jeżeli wcześniej nie istnieje.
        if not os.path.exists(self.paths.PROCESSED_DATA):
            os.makedirs(self.paths.PROCESSED_DATA)

        #  Pobieranie listy albumów (katalogów wewnątrz RAW_DATA)
        albums = [f.path for f in os.scandir(self.paths.RAW_DATA) if f.is_dir()]
        total_albums = len(albums)
        logger.info(f"Rozpoczynam zrównoleglone przetwarzanie {total_albums} albumów...")

        # Obliczamy bezpieczną liczbę "robotników" (procesów)
        # Każdy proces ładuje audio i tworzy CQT, więc ograniczamy do oszczędzenia RAM
        total_cores = os.cpu_count() or 4
        # Używamy max 1/3 rdzeni, ale nie mniej niż 1, aby oszczędzić pamięć
        safe_workers = max(1, max(total_cores // 3, 1))
        
        if safe_workers > cfg_builder.MAX_WORKERS:
            safe_workers = cfg_builder.MAX_WORKERS
            logger.warning(f"Limitowanie liczby procesów do {safe_workers} (maksimum dozwolone przez konfigurację).")

        logger.warning(f"Twój system ma {total_cores} rdzeni CPU. Użyję {safe_workers} procesów do budowania datasetu.")
        
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
                futures = [executor.submit(worker_func, track) for track in pending_tracks]  # type: ignore[arg-type]
                processed_in_album = 0

                for future in as_completed(futures):
                    success, msg = future.result()
                    processed_in_album += 1
                    overall_bar.update(1)
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