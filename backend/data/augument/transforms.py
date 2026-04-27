from backend.logger import logger
from backend.config import cfg_train
import numpy as np
import random

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