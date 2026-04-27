import os
from dataclasses import dataclass

# Dynamiczne ustalenie głównego folderu projektu (AI-CHORD-ANALYSIS),
# niezależnie od tego, z jakiego miejsca w terminalu odpalisz skrypt.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PathsConfig:
    RAW_DATA: str = os.path.join(BASE_DIR, "raw_dataset")
    
    PROCESSED_DATA: str = os.path.join(BASE_DIR, "out", "full_dataset")
    MODEL_SAVE_PATH: str = os.path.join(BASE_DIR, "out", "best_crnn_model.pth")
    
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
class TrainConfig:
    BATCH_SIZE: int = 64
    EPOCHS: int = 30
    LEARNING_RATE: float = 0.0001
    PATIENCE: int = 5

    # Augmentacja danych (MVP): działa tylko dla train dataset.
    AUGMENT_ENABLED: bool = False

    AUGMENT_GAIN_ENABLED: bool = True
    AUGMENT_GAIN_PROB: float = 0.5
    AUGMENT_GAIN_DB_MIN: float = -6.0
    AUGMENT_GAIN_DB_MAX: float = 6.0

    AUGMENT_NOISE_ENABLED: bool = True
    AUGMENT_NOISE_PROB: float = 0.4
    AUGMENT_NOISE_SNR_DB_MIN: float = 20.0
    AUGMENT_NOISE_SNR_DB_MAX: float = 35.0

    AUGMENT_SPECMASK_ENABLED: bool = True
    AUGMENT_SPECMASK_PROB: float = 0.2
    AUGMENT_SPECMASK_MAX_TIME_MASKS: int = 1
    AUGMENT_SPECMASK_MAX_FREQ_MASKS: int = 1
    AUGMENT_SPECMASK_MAX_TIME_WIDTH: int = 4
    AUGMENT_SPECMASK_MAX_FREQ_WIDTH: int = 8

@dataclass
class BuilderConfig:
    CQT_METHOD: str = 'cqt' # 'cqt' lub 'cqt_fast'


# Instancje konfiguracji do importowania w całym projekcie
cfg_paths = PathsConfig()
cfg_audio = AudioConfig()
cfg_train = TrainConfig()
cfg_builder = BuilderConfig()
