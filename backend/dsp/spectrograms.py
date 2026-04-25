import subprocess
import shutil
import numpy as np
import librosa
from backend.config import cfg_audio
from backend.dsp.src.cqtTransform import CqtTransform
from backend.dsp.src.hpssFilter import HpssFilter
from backend.dsp.src.pipeline import Pipeline
from backend.logger.logger import Logger

logger = Logger(__name__)

class AudioProcessor:
    """Warstwa kompatybilnosci: publiczne API + DSP z backend/dsp/src."""

    def __init__(self, config=cfg_audio):
        self.sample_rate = config.SAMPLE_RATE
        self.hop_size_ms = config.HOP_SIZE_MS
        self.n_bins = config.N_BINS
        self.bins_per_octave = config.BINS_PER_OCTAVE
        self.f_min = config.F_MIN
        self.hpss_harmonic_margin = config.HPSS_HARMONIC_MARGIN
        self.hpss_percussive_margin = config.HPSS_PERCUSSIVE_MARGIN

        self.default_apply_denoise = config.APPLY_DENOISE
        self.default_apply_short_noises = config.APPLY_SHORT_NOISES
        self.default_apply_whitening = config.APPLY_WHITENING
        self.default_apply_smoothing = config.APPLY_SMOOTHING
        self.default_apply_hpss = config.APPLY_HPSS
        self.hop_length = int(self.sample_rate * (self.hop_size_ms / 1000.0))

        # To jest glowny silnik DSP, ktory ma byc uzywany w calym projekcie.
        self._pipeline = Pipeline(
            harmonicMargin=self.hpss_harmonic_margin,
            percussiveMargin=self.hpss_percussive_margin,
            fMin=self.f_min,
            fS=self.sample_rate,
            hopLength=self.hop_length,
        )

        # Niezalezny filtr HPSS, ktory mozna uruchomic przed dowolna metoda CQT.
        self._hpss = HpssFilter(
            harmonicMargin=self.hpss_harmonic_margin,
            percussiveMargin=self.hpss_percussive_margin,
        )

        self._cqt = CqtTransform(self.bins_per_octave, self.f_min, self.sample_rate, self.hop_length)

    @staticmethod
    def _prepare_audio_float32(audio_data: np.ndarray) -> np.ndarray:
        # Konwersja wejscia do float32, bo tego oczekuje nowy pipeline.
        if audio_data.dtype.kind == 'i':
            audio_data = audio_data.astype(np.float32) / (np.iinfo(audio_data.dtype).max + 1)
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        return audio_data

    @staticmethod
    def read_audio_universal(file_path: str, target_sr: int = 44100):
        ffmpeg_bin = shutil.which('ffmpeg')
        if ffmpeg_bin is None:
            raise RuntimeError("FFmpeg nie został znaleziony w systemie. Dodaj go do PATH i uruchom terminal ponownie.")

        command = [
            ffmpeg_bin, '-i', file_path, '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ac', '1', '-ar', str(target_sr), '-loglevel', 'quiet', '-'
        ]

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if process.returncode != 0:
                logger.error(f"Błąd FFmpeg: {err.decode('utf-8')}")
                return None, None
            signal = np.frombuffer(out, dtype=np.int16)
            return signal, target_sr
        except FileNotFoundError:
            raise RuntimeError("FFmpeg nie został znaleziony w systemie. Dodaj go do PATH i uruchom terminal ponownie.")

    def calculate_spectogram_cqt_fast(self, audio_data: np.ndarray) -> np.ndarray:
        # Szybka ścieżka oparta o librosa.cqt.
        audio_data = self._prepare_audio_float32(audio_data)
        cqt_result = np.abs(librosa.cqt(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.f_min,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        ))
        return librosa.amplitude_to_db(cqt_result, ref=np.max)

    def calculate_spectogram_cqt(self, audio_data: np.ndarray) -> np.ndarray:
        # Wlasna sciezka DSP z backend/dsp/src.
        audio_data = self._prepare_audio_float32(audio_data)
        return self._pipeline.processArrayForAI(audio_data, self.sample_rate)

    def apply_hpss_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Zwraca harmoniczna skladowa sygnalu przez HPSS."""
        audio_data = self._prepare_audio_float32(audio_data)
        self._hpss.loadAudioArray(audio_data, self.sample_rate)
        y_harm, _ = self._hpss.extractHarmonic()

        return y_harm

    def smooth_harmonics(self, spectrogram: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        # Utrzymujemy to samo zachowanie co wczesniej (moving average), ale bez tqdm.
        if kernel_size <= 1:
            return np.array(spectrogram, copy=True)

        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smoothed_cqt = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode='same'),
            axis=1,
            arr=spectrogram,
        )
        return smoothed_cqt.astype(spectrogram.dtype, copy=False)

    def denoise_normalize_audio(self, cqt_db: np.ndarray, dynamic_range: float = 35) -> np.ndarray:
        max_db = np.max(cqt_db)
        threshold_db = max_db - dynamic_range
        cqt_db = np.array(cqt_db, copy=True)
        cqt_db[cqt_db < threshold_db] = threshold_db
        return (cqt_db - threshold_db) / dynamic_range

    def spectral_whitening(self, spectrogram: np.ndarray) -> np.ndarray:
        whitened = np.zeros_like(spectrogram)
        for i in range(spectrogram.shape[0]):
            row_median = np.median(spectrogram[i, :])
            whitened[i, :] = spectrogram[i, :] - row_median
        return whitened

    def remove_short_noises(self, matrix: np.ndarray, min_duration_frames: int = 3) -> np.ndarray:
        _ = min_duration_frames
        cleaned = np.copy(matrix)
        max_rows = min(12, cleaned.shape[0])
        for i in range(max_rows):
            row = cleaned[i, :]
            for f in range(1, len(row) - 1):
                if row[f] > 0 and row[f - 1] == 0 and row[f + 1] == 0:
                    cleaned[i, f] = 0
        return cleaned

    def calculate_spectogram_rfft(self, audio_data: np.ndarray, window_size=100) -> np.ndarray:
        audio_data = self._prepare_audio_float32(audio_data)
        window_size_samples = int(self.sample_rate * (window_size / 1000.0))
        hop_length = self.hop_length

        if len(audio_data) < window_size_samples:
            return np.empty((window_size_samples // 2 + 1, 0), dtype=np.float32)

        total_sample_num = 1 + int(len(audio_data) - window_size_samples) // hop_length
        window = np.hanning(window_size_samples)
        spectrogram = []

        for i in range(total_sample_num):
            start = i * hop_length
            end = start + window_size_samples
            frame = audio_data[start:end]
            windowed_frame = frame * window
            fft_result = np.fft.rfft(windowed_frame)
            magnitude = np.abs(fft_result)
            db_magnitude = 20 * np.log10(magnitude + 1e-10)
            spectrogram.append(db_magnitude)

        return np.array(spectrogram, dtype=np.float32).T

    def create_chromagram(self, cqt_matrix: np.ndarray, threshold_percent=25) -> np.ndarray:
        _ = threshold_percent
        return self._cqt.toChromagram(cqt_matrix)

    def generate_spectrogram(self, audio_data: np.ndarray, method='cqt',
                             apply_denoise=None, apply_short_noises=None,
                             apply_whitening=None, apply_smoothing=None,
                             apply_hpss=None, **kwargs) -> np.ndarray:

        if apply_denoise is None:
            apply_denoise = self.default_apply_denoise
        if apply_short_noises is None:
            apply_short_noises = self.default_apply_short_noises
        if apply_whitening is None:
            apply_whitening = self.default_apply_whitening
        if apply_smoothing is None:
            apply_smoothing = self.default_apply_smoothing
        if apply_hpss is None:
            apply_hpss = self.default_apply_hpss

        audio_data = self._prepare_audio_float32(audio_data)

        # Dodane bo reczny CQT juz ma 
        if apply_hpss and method != 'pipeline' and method != 'cqt':
            audio_data = self.apply_hpss_filter(audio_data)

        if method == 'cqt':
            # Reczna implementacja CQT (wolniejsza ale lepsza)
            spectrogram = self.calculate_spectogram_cqt(audio_data, **kwargs)
        elif method == 'cqt_fast':
            # CQT z biblioteki (szybkie mniej dokladne)
            spectrogram = self.calculate_spectogram_cqt_fast(audio_data)
        elif method == 'pipeline':
            # Wersja z HPSS i CQT jako jeden Pipe
            spectrogram = self.calculate_spectogram_cqt(audio_data, **kwargs)
        elif method == 'rfft':
            spectrogram = self.calculate_spectogram_rfft(audio_data, **kwargs)
        else:
            raise ValueError(f"Nieznana metoda: {method}")

        # Opcjonalne filtry do nałożenia
        if apply_denoise:
            spectrogram = self.denoise_normalize_audio(spectrogram)
        if apply_short_noises:
            spectrogram = self.remove_short_noises(spectrogram)
        if apply_whitening:
            spectrogram = self.spectral_whitening(spectrogram)
        if apply_smoothing:
            spectrogram = self.smooth_harmonics(spectrogram)

        return spectrogram