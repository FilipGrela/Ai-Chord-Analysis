import time

from PyQt6.QtCore import QThread

from backend.api.inference import ChordInferenceEngine
from backend.event_system.event_bus import *


class InferenceWorker(QThread):
    def __init__(self, audio_path: str):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        """Ten kod wykonuje się w tle, nie blokując interfejsu."""
        try:
            event_bus.log_message.emit(LogLevel.INFO, f"Rozpoczęto analizę pliku: {self.audio_path}")

            try:
                engine = ChordInferenceEngine()
                results = engine.predict(self.audio_path)
            except FileNotFoundError as e:
                event_bus.log_message(LogLevel.ERROR, e)
                return

            event_bus.inference_finished.emit(results)

        except Exception as e:
            event_bus.inference_error.emit(str(e))
            event_bus.log_message.emit("ERROR", f"Wystąpił błąd: {str(e)}")

