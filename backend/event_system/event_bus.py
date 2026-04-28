from PyQt6.QtCore import QObject, pyqtSignal
from enum import Enum


class LogLevel(Enum):
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3
    SUCCESS = 4


class AppEventBus(QObject):
    log_message = pyqtSignal(LogLevel, str)  # (level, message)
    progress_updated = pyqtSignal(int, str)  # (procent, status)
    inference_finished = pyqtSignal(list)  # (wyniki)
    inference_error = pyqtSignal(str)  # (błąd)


event_bus = AppEventBus()
