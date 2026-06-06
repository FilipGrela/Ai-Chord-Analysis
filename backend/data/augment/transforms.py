from backend.logger import logger
from backend.config import cfg_audio, cfg_train
import numpy as np
import random
from backend.data.augment.label_ops import ChordTranspose

class RandomGain:
    def __init__(self, prob=cfg_train.AUGMENT_GAIN_PROB, min_gain_db=cfg_train.AUGMENT_GAIN_DB_MIN, max_gain_db=cfg_train.AUGMENT_GAIN_DB_MAX):
        self.logger = logger.Logger(__name__)

        if prob > 1 or prob < 0:
            self.logger.error(f"Nieprawidłowa wartość prawdopodobieństwa: {prob}. Powinna być w zakresie [0, 1]. Pomijam random gain.")
            raise ValueError()

        if min_gain_db > max_gain_db:
            self.logger.error(f"Min gain > Max gain [{min_gain_db} > {max_gain_db}]. Pomijam random gain.")
            raise ValueError()

        self.prob = prob
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def apply(self, audio):
        """Losowo wzmacnia lub osłabia sygnał audio, symulując różne poziomy głośności nagrań."""

        if self.prob > 0 and random.random() < self.prob:
            gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
            gain_factor = 10 ** (gain_db / 20)  # Konwersja z dB na współczynnik liniowy
            audio = audio * gain_factor
        return audio

class AdditiveGaussianNoise:
    def __init__(
            self,
            prob = cfg_train.AUGMENT_NOISE_PROB,
            snr_min_db=cfg_train.AUGMENT_NOISE_SNR_DB_MIN,
            snr_max_db=cfg_train.AUGMENT_NOISE_SNR_DB_MAX):

        self.logger = logger.Logger(__name__)


        if prob < 0 or prob > 1:
            self.logger.error(f"Nieprawidłowe prawdopodobieństwo: {prob}. Ustawiam 0.0.")
            prob = 0.0

        if snr_min_db > snr_max_db:
            self.logger.error(f"Min SNR > Max SNR [{snr_min_db} > {snr_max_db}]. Pomijam additive noise.")
            raise ValueError()

        self.prob = prob
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    def _add_noise_numpy_spec(self, spec: np.ndarray) -> np.ndarray:
        # spec: 2D (T,F) or 3D (C,T,F)
        target_snr_db = random.uniform(self.snr_min_db, self.snr_max_db)
        snr_linear = 10 ** (target_snr_db / 10.0)

        signal_power = np.mean(spec ** 2)
        if signal_power <= 0:
            return spec

        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), size=spec.shape)
        return spec + noise

    def _add_noise_torch_spec(self, spec: "torch.Tensor") -> "torch.Tensor":
        import torch as _torch

        target_snr_db = random.uniform(self.snr_min_db, self.snr_max_db)
        snr_linear = 10 ** (target_snr_db / 10.0)

        # oblicz moc sygnału
        signal_power = _torch.mean(spec.pow(2))
        if signal_power <= 0:
            return spec

        noise_power = signal_power / snr_linear
        noise = _torch.randn_like(spec) * _torch.sqrt(noise_power)
        return spec + noise

    def apply(self, x):
        """Obsługuje:
           - 1D numpy (audio) -> naturalne zachowanie
           - 2D/3D numpy (spec) -> dodaje noise do spektrogramu
           - torch.Tensor (spec) -> dodaje noise (zachowując typ tensor)
        """
        # losowość: stosuj zgodnie z prawdopodobieństwem
        if self.prob <= 0 or random.random() >= self.prob:
            return x

        # numpy audio (1D)
        if isinstance(x, np.ndarray) and x.ndim == 1:
            try:
                # TODO
                return self._add_noise_numpy_audio(x)
            except Exception:
                return x

        # numpy spectrogram (2D lub 3D)
        if isinstance(x, np.ndarray) and x.ndim >= 2:
            try:
                return self._add_noise_numpy_spec(x)
            except Exception:
                return x

        # torch Tensor (np. (1, T, F) lub (T, F))
        import torch
        if isinstance(x, torch.Tensor):
            try:
                # Operujemy na kopii, żeby nie modyfikować zerowej próbki w pamięci
                out = x.clone()
                return self._add_noise_torch_spec(out)
            except Exception:
                return x

        # fallback: nic nie robimy
        return x


class RandomSpecMask:
    def __init__(self, prob=cfg_train.AUGMENT_SPECMASK_PROB, max_time_masks=cfg_train.AUGMENT_SPECMASK_MAX_TIME_MASKS, max_freq_masks=cfg_train.AUGMENT_SPECMASK_MAX_FREQ_MASKS, max_time_width=cfg_train.AUGMENT_SPECMASK_MAX_TIME_WIDTH, max_freq_width=cfg_train.AUGMENT_SPECMASK_MAX_FREQ_WIDTH):
        self.logger = logger.Logger(__name__)
        self.prob = prob
        self.max_time_masks = max_time_masks
        self.max_freq_masks = max_freq_masks
        self.max_time_width = max_time_width
        self.max_freq_width = max_freq_width

    def apply(self, spec):
        """Losowo maskuje fragmenty spektrogramu w wymiarze czasu i częstotliwości. Wypełia je wartosćią średnią."""
        if random.random() >= self.prob:
            return spec

        spec = spec.copy() if isinstance(spec, np.ndarray) else spec.clone()

        if spec.ndim < 2:
            return spec

        # Wymiary: [C, T, F] lub [T, F]
        time_dim = spec.shape[-2] if spec.ndim == 3 else spec.shape[0]
        freq_dim = spec.shape[-1] if spec.ndim == 3 else spec.shape[1]

        mask_value = spec.mean()

        # Time masking
        num_time_masks = random.randint(0, self.max_time_masks)
        for _ in range(num_time_masks):
            if time_dim <= 1:
                break
            width = random.randint(1, min(self.max_time_width, time_dim))
            start = random.randint(0, time_dim - width)
            if spec.ndim == 3:
                spec[:, start:start + width, :] = mask_value
            else:
                spec[start:start + width, :] = mask_value

        # Freq masking
        num_freq_masks = random.randint(0, self.max_freq_masks)
        for _ in range(num_freq_masks):
            if freq_dim <= 1:
                break
            width = random.randint(1, min(self.max_freq_width, freq_dim))
            start = random.randint(0, freq_dim - width)
            if spec.ndim == 3:
                spec[:, :, start:start + width] = mask_value
            else:
                spec[:, start:start + width] = mask_value

        return spec


class RandomTranspose:
    """Losowo transpozycjonuje spektrogram CQT poprzez przesunięcie danych wzdłuż osi częstotliwości."""

    @staticmethod
    def _semitones_to_bins(semitones: int) -> int:
        bins_per_semitone = cfg_audio.BINS_PER_OCTAVE / 12
        return int(round(semitones * bins_per_semitone))

    def __init__(
        self,
        prob: float = getattr(cfg_train, "AUGMENT_TRANSPOSE_PROB", 0.0),
        min_semitones: int = getattr(cfg_train, "AUGMENT_TRANSPOSE_MIN", -6),
        max_semitones: int = getattr(cfg_train, "AUGMENT_TRANSPOSE_MAX", 6),
    ):
        self.logger = logger.Logger(__name__)

        if prob < 0 or prob > 1:
            self.logger.error(f"Nieprawidłowe prawdopodobieństwo: {prob}. Ustawiam 0.0.")
            prob = 0.0

        if min_semitones > max_semitones:
            self.logger.error(
                f"Min semitones > Max semitones [{min_semitones} > {max_semitones}]. Pomijam transpose."
            )
            raise ValueError()

        self.prob = prob
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def _transpose_numpy_spec(self, spec: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose spectral data (2D or 3D) by shifting frequency axis."""
        # spec: (T, F) or (C, T, F)
        # Przesunięcie w dziedzinie częstotliwości, ostatnia oś
        bins_to_shift = self._semitones_to_bins(semitones)

        if spec.ndim == 2:  # (T, F)
            shifted = np.roll(spec, bins_to_shift, axis=1)
            # Zeruj przetoczone dane (części, które weszły z drugiej strony)
            if bins_to_shift > 0:
                shifted[:, :bins_to_shift] = 0  # Wypełnij początek zerami
            elif bins_to_shift < 0:
                shifted[:, bins_to_shift:] = 0  # Wypełnij koniec zerami
            return shifted

        elif spec.ndim == 3:  # (C, T, F)
            shifted = np.roll(spec, bins_to_shift, axis=2)
            if bins_to_shift > 0:
                shifted[:, :, :bins_to_shift] = 0
            elif bins_to_shift < 0:
                shifted[:, :, bins_to_shift:] = 0
            return shifted

        return spec

    def _transpose_torch_spec(self, spec: "torch.Tensor", semitones: int) -> "torch.Tensor":
        """Transpose torch tensor spectral data."""
        import torch as _torch

        bins_to_shift = self._semitones_to_bins(semitones)

        if spec.ndim == 2:  # (T, F)
            shifted = _torch.roll(spec, bins_to_shift, dims=1)
            if bins_to_shift > 0:
                shifted[:, :bins_to_shift] = 0
            elif bins_to_shift < 0:
                shifted[:, bins_to_shift:] = 0
            return shifted

        elif spec.ndim == 3:  # (C, T, F)
            shifted = _torch.roll(spec, bins_to_shift, dims=2)
            if bins_to_shift > 0:
                shifted[:, :, :bins_to_shift] = 0
            elif bins_to_shift < 0:
                shifted[:, :, bins_to_shift:] = 0
            return shifted

        return spec

    def apply(self, x):
        """
        Transpose spectrogram by shifting frequency axis.
        Supports:
        - 2D numpy (T, F) -> transpose
        - 3D numpy (C, T, F) -> transpose
        - torch.Tensor -> transpose
        """
        if self.prob <= 0 or random.random() >= self.prob:
            return x

        semitones = random.randint(self.min_semitones, self.max_semitones)

        if semitones == 0:
            return x

        # numpy spectrogram
        if isinstance(x, np.ndarray) and x.ndim >= 2:
            try:
                return self._transpose_numpy_spec(x, semitones)
            except Exception as e:
                self.logger.error(f"Błąd transpozycji numpy: {e}")
                return x

        # torch Tensor
        import torch
        if isinstance(x, torch.Tensor):
            try:
                out = x.clone()
                return self._transpose_torch_spec(out, semitones)
            except Exception as e:
                self.logger.error(f"Błąd transpozycji torch: {e}")
                return x

        return x

