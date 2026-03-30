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


@dataclass
class TrainConfig:
    BATCH_SIZE: int = 64
    EPOCHS: int = 30
    LEARNING_RATE: float = 0.0001
    PATIENCE: int = 5


# Instancje konfiguracji do importowania w całym projekcie
cfg_paths = PathsConfig()
cfg_audio = AudioConfig()
cfg_train = TrainConfig()
