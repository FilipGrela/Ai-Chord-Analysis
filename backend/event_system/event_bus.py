from PyQt6.QtCore import QObject, pyqtSignal
from enum import Enum


class LogLevel(Enum):
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    SUCCESS = 4
    DEBUG = 5


class AppEventBus(QObject):
    log_message = pyqtSignal(LogLevel, str)  # (level, message)
    progress_updated = pyqtSignal(int, str)  # (procent, status)
    inference_finished = pyqtSignal(list)  # (wyniki)
    inference_error = pyqtSignal(str)  # (błąd)


event_bus = AppEventBus()
