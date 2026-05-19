import os
from dataclasses import asdict, dataclass

# Dynamiczne ustalenie głównego folderu projektu (AI-CHORD-ANALYSIS),
# niezależnie od tego, z jakiego miejsca w terminalu odpalisz skrypt.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class LoggerConfig:
    DEBUG: bool = False

@dataclass
class PathsConfig:
    RAW_DATA: str = os.path.join(BASE_DIR, "raw_dataset")
    
    PROCESSED_DATA: str = os.path.join(BASE_DIR, "out", "full_dataset")
    MODEL_SAVE_PATH: str = os.path.join(BASE_DIR, "out", "model.pth")
    
    TEST_OUTPUT: str = os.path.join(BASE_DIR, "out", "dataset_output")

    SINGLE_TEST_DATA: str = os.path.join(BASE_DIR, "single_test_data")


@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 44100
    HOP_SIZE_MS: int = 50
    SEQ_LEN: int = 40
    N_BINS: int = 84
    BINS_PER_OCTAVE: int = 12
    F_MIN: float = 32.703

    HPSS_HARMONIC_MARGIN: float = 2.0
    HPSS_PERCUSSIVE_MARGIN: float = 1.0

    APPLY_DENOISE: bool = False
    APPLY_SHORT_NOISES: bool = False
    APPLY_WHITENING: bool = False
    APPLY_SMOOTHING: bool = False
    APPLY_HPSS: bool = False

@dataclass
class ModelConfig:
    NUM_CLASSES: int = -1  # Ustawiane dynamicznie na podstawie buildera
    DROPOUT_RATE: float = 0.25
    RNN_NUM_LAYERS: int = 4
    RNN_HIDDEN_SIZE: int = 192
    CNN_CHANNELS: tuple = (32, 64, 128)
    N_BINS: int = 84

@dataclass
class TrainConfig:
    BATCH_SIZE: int = 64
    EPOCHS: int = 4
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 5e-4
    PATIENCE: int = 2

    USE_OFFLINE_TRANSPOSE: bool = False # Czy używać dodatkowych próbek offline przez transponowanie (zmiana tonacji).

    # Augmentacja danych (MVP): działa tylko dla train dataset.
    AUGMENT_ENABLED: bool = True  # Glowny przelacznik augmentacji (True = wlaczona)

    # Random gain (zmiana glosnosci)
    AUGMENT_GAIN_ENABLED: bool = True  # Czy uzywac tej augmentacji
    AUGMENT_GAIN_PROB: float = 0.5  # Szansa na zastosowanie dla probki (0-1)
    AUGMENT_GAIN_DB_MIN: float = -6.0  # Minimalna zmiana glosnosci [dB]
    AUGMENT_GAIN_DB_MAX: float = 6.0  # Maksymalna zmiana glosnosci [dB]

    # Additive noise (dodawanie szumu)
    AUGMENT_NOISE_ENABLED: bool = False  # Czy uzywac tej augmentacji
    AUGMENT_NOISE_PROB: float = 0.25  # Szansa na zastosowanie dla probki (0-1)
    AUGMENT_NOISE_SNR_DB_MIN: float = 20.0  # Min SNR [dB], nizsze = wiecej szumu
    AUGMENT_NOISE_SNR_DB_MAX: float = 35.0  # Max SNR [dB], wyzsze = mniej szumu

    # SpecMask (maskowanie fragmentow spektrogramu)
    AUGMENT_SPECMASK_ENABLED: bool = True  # Czy uzywac tej augmentacji
    AUGMENT_SPECMASK_PROB: float = 0.20  # Szansa na zastosowanie dla probki (0-1)
    AUGMENT_SPECMASK_MAX_TIME_MASKS: int = 2  # Maks. liczba masek w osi czasu
    AUGMENT_SPECMASK_MAX_FREQ_MASKS: int = 2  # Maks. liczba masek w osi czestotliwosci
    AUGMENT_SPECMASK_MAX_TIME_WIDTH: int = 4  # Maks. szerokosc jednej maski czasowej
    AUGMENT_SPECMASK_MAX_FREQ_WIDTH: int = 6  # Maks. szerokosc jednej maski czestotliwosciowej

    # Transpose (transponowanie spektrogramu - zmiana tonacji)
    AUGMENT_TRANSPOSE_ENABLED: bool = False  # Czy uzywac tej augmentacji (online)
    AUGMENT_TRANSPOSE_PROB: float = 0.4  # Szansa na zastosowanie dla probki (0-1)
    AUGMENT_TRANSPOSE_MIN: int = -6  # Min liczba poltonow do transponowania
    AUGMENT_TRANSPOSE_MAX: int = 6  # Max liczba poltonow do transponowania

@dataclass
class BuilderConfig:
    CQT_METHOD: str = 'cqt' # 'cqt' lub 'cqt_fast'
    MAX_WORKERS: int = 17  # Maksymalna liczba procesów do budowania datasetu (nie więcej niż liczba rdzeni CPU - 2)
    SUPPORT_SEVENTHS: bool = False  # If True, include seventh chords (e.g., C7, Cm7) in VOCAB


@dataclass
class AnalysisConfig:
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "out", "analysis")
    CHORD_SIMILARITY_ROOT_WEIGHT: float = 0.55
    CHORD_SIMILARITY_QUALITY_WEIGHT: float = 0.30
    CHORD_SIMILARITY_KEY_WEIGHT: float = 0.15
    MUSIC_METRICS_DATA_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "isophonics_dataset")
    MUSIC_METRICS_DEFAULT_KEY: str | None = "A"
    DATASET_SONGS: int | None = 120


@dataclass
class HpoConfig:
    STUDY_NAME: str = "chord_hpo"
    DIRECTION: str = "minimize"
    N_TRIALS: int = 20
    TIMEOUT_SECONDS: int | None = None
    STORAGE_PATH: str = os.path.join(BASE_DIR, "out", "hpo", "optuna_study.db")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "out", "hpo")


# Instancje konfiguracji do importowania w całym projekcie
cfg_paths = PathsConfig()
cfg_audio = AudioConfig()
cfg_model = ModelConfig()
cfg_train = TrainConfig()
cfg_builder = BuilderConfig()
cfg_analysis = AnalysisConfig()
cfg_logger = LoggerConfig()
cfg_hpo = HpoConfig()


def sync_model_config_with_builder() -> None:
    """Keep model output size aligned with chord vocabulary settings."""
    cfg_model.NUM_CLASSES = 169 if cfg_builder.SUPPORT_SEVENTHS else 25


sync_model_config_with_builder()


def get_config_snapshot() -> dict:
    return {
        "logger": asdict(cfg_logger),
        "paths": asdict(cfg_paths),
        "audio": asdict(cfg_audio),
        "model": asdict(cfg_model),
        "train": asdict(cfg_train),
        "builder": asdict(cfg_builder),
        "analysis": asdict(cfg_analysis),
        "hpo": asdict(cfg_hpo),
    }
