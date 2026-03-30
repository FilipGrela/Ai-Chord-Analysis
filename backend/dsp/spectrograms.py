import subprocess
import numpy as np
import librosa
from tqdm import tqdm
from backend.config import cfg_audio

class AudioProcessor:
    """Kompletny silnik DSP zawierający wszystkie metody ekstrakcji cech."""

    def __init__(self, config=cfg_audio):
        self.sample_rate = config.SAMPLE_RATE
        self.hop_size_ms = config.HOP_SIZE_MS
        self.n_bins = config.N_BINS

    @staticmethod
    def read_audio_universal(file_path: str, target_sr: int = 44100):
        command = [
            'ffmpeg', '-i', file_path, '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ac', '1', '-ar', str(target_sr), '-loglevel', 'quiet', '-'
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if process.returncode != 0:
                print(f"Błąd FFmpeg: {err.decode('utf-8')}")
                return None, None
            signal = np.frombuffer(out, dtype=np.int16)
            return signal, target_sr
        except FileNotFoundError:
            raise RuntimeError("FFmpeg nie został znaleziony w systemie.")

    # ==========================================
    #            ENHANCEMENT PIPELINE
    # ==========================================

    def smooth_harmonics(self, spectrogram: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        smoothed_cqt = np.zeros_like(spectrogram)
        kernel = np.ones(kernel_size) / kernel_size
        for i in tqdm(range(spectrogram.shape[0]), desc="Smoothing Spectrogram", leave=False):
            smoothed_cqt[i, :] = np.convolve(spectrogram[i, :], kernel, mode='same')
        return smoothed_cqt

    def denoise_normalize_audio(self, cqt_db: np.ndarray, dynamic_range: float = 35) -> np.ndarray:
        max_db = np.max(cqt_db)
        threshold_db = max_db - dynamic_range
        cqt_db[cqt_db < threshold_db] = threshold_db
        return (cqt_db - threshold_db) / dynamic_range

    @staticmethod
    def generate_hann_window(window_size: int) -> np.ndarray:
        n = np.arange(window_size)
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (window_size - 1))

    def spectral_whitening(self, spectrogram: np.ndarray) -> np.ndarray:
        whitened_cqt = np.zeros_like(spectrogram)
        for i in tqdm(range(spectrogram.shape[0]), desc="Spectral Whitening", leave=False):
            row_median = np.median(spectrogram[i, :])
            whitened_cqt[i, :] = spectrogram[i, :] - row_median
        return whitened_cqt

    def remove_short_noises(self, chroma_matrix: np.ndarray, min_duration_frames: int = 3) -> np.ndarray:
        cleaned_chroma = np.copy(chroma_matrix)
        for i in range(12):
            row = cleaned_chroma[i, :]
            for f in range(1, len(row) - 1):
                if row[f] > 0 and row[f-1] == 0 and row[f+1] == 0:
                    cleaned_chroma[i, f] = 0
        return cleaned_chroma

    # ==========================================
    #                CORE MATH
    # ==========================================

    def calculate_spectogram_cqt_fast(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data.dtype.kind == 'i':
            audio_data = audio_data.astype(np.float32) / (np.iinfo(audio_data.dtype).max + 1)
        elif audio_data.dtype == np.float64:
            audio_data = audio_data.astype(np.float32)

        hop_length = int(self.sample_rate * (self.hop_size_ms / 1000.0))
        cqt_result = np.abs(librosa.cqt(
            y=audio_data, sr=self.sample_rate, hop_length=hop_length, 
            fmin=32.703, n_bins=self.n_bins, bins_per_octave=12
        ))
        return librosa.amplitude_to_db(cqt_result, ref=np.max)

    def calculate_spectogram_cqt(self, audio_data: np.ndarray, fmin=32.703, bins_per_octave=12) -> np.ndarray:
        Q = 1.0 / (2**(1.0 / bins_per_octave) - 1.0)
        freqs = fmin * (2.0 ** (np.arange(self.n_bins) / bins_per_octave))
        window_lengths = np.ceil(Q * self.sample_rate / freqs).astype(int)
        max_window = window_lengths[0] 

        filters = []
        for k in range(self.n_bins):
            N = window_lengths[k]
            window = self.generate_hann_window(N)
            complex_wave = np.exp(-2j * np.pi * freqs[k] * np.arange(N) / self.sample_rate)
            filters.append((window * complex_wave) / N)
        
        hop_length = int(self.sample_rate * (self.hop_size_ms / 1000.0))
        total_frames = 1 + (len(audio_data) - max_window) // hop_length
        cqt_result = np.zeros((self.n_bins, total_frames))
        
        for i in tqdm(range(total_frames), desc="Analyzing Audio Frames", leave=False):
            start_idx = i * hop_length
            for k in range(self.n_bins):
                N = window_lengths[k]
                frame = audio_data[start_idx : start_idx + N]
                cqt_result[k, i] = np.abs(np.dot(frame, filters[k]))
                
        return 20 * np.log10(cqt_result + 1e-10)

    def calculate_spectogram_rfft(self, audio_data: np.ndarray, window_size=100) -> np.ndarray:
        window_size_samples = int(self.sample_rate * (window_size / 1000.0))
        hop_length = int(self.sample_rate * (self.hop_size_ms / 1000.0))
        total_sample_num = 1 + int(len(audio_data) - window_size_samples) // hop_length
        window = self.generate_hann_window(window_size_samples)
        spectrogram = []

        for i in tqdm(range(total_sample_num), desc="Calculating RFFT Frames", leave=False):
            start = i * hop_length
            end = start + window_size_samples
            frame = audio_data[start:end]
            windowed_frame = frame * window
            fft_result = np.fft.rfft(windowed_frame)
            magnitude = np.abs(fft_result)
            db_magnitude = 20 * np.log10(magnitude + 1e-10)
            spectrogram.append(db_magnitude)

        return np.array(spectrogram).T

    def create_chromagram(self, cqt_matrix: np.ndarray, threshold_percent=25) -> np.ndarray:
        num_bins, num_frames = cqt_matrix.shape
        chroma = np.zeros((12, num_frames))
        cqt_enhanced = np.power(cqt_matrix, 3) 

        for i in range(num_bins):
            pitch_class = i % 12
            chroma[pitch_class, :] += cqt_enhanced[i, :]
            
        for f in range(num_frames):
            col = chroma[:, f]
            col_max = np.max(col)
            if col_max > 0:
                col /= col_max
                col[col < threshold_percent / 100.0] = 0
                final_max = np.max(col)
                if final_max > 0:
                    col /= final_max
            chroma[:, f] = col
                
        return chroma

    # ==========================================
    #                 PIPELINE
    # ==========================================

    def generate_spectrogram(self, audio_data: np.ndarray, method='cqt_fast', 
                             apply_denoise=True, apply_short_noises=True, 
                             apply_whitening=True, apply_smoothing=True, **kwargs) -> np.ndarray:
        if method == 'cqt':
            spectrogram = self.calculate_spectogram_cqt(audio_data, **kwargs)
        elif method == 'cqt_fast':
            spectrogram = self.calculate_spectogram_cqt_fast(audio_data)
        elif method == 'rfft':
            spectrogram = self.calculate_spectogram_rfft(audio_data, **kwargs)
        else:
            raise ValueError(f"Nieznana metoda: {method}")
        
        if apply_denoise:
            spectrogram = self.denoise_normalize_audio(spectrogram)
        if apply_short_noises:
            spectrogram = self.remove_short_noises(spectrogram)
        if apply_whitening:
            spectrogram = self.spectral_whitening(spectrogram)
        if apply_smoothing:
            spectrogram = self.smooth_harmonics(spectrogram)
            
        return spectrogram