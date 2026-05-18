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
            event_bus.progress_updated.emit(0, "Rozpoczynanie analizy")
            try:
                event_bus.log_message.emit(LogLevel.DEBUG, "Tworzenie silnika")
                engine = ChordInferenceEngine()
                event_bus.log_message.emit(LogLevel.DEBUG, "Rozpoczynanie analizy")
                results = engine.predict(self.audio_path)
            except FileNotFoundError as e:
                event_bus.log_message.emit(LogLevel.ERROR, str(e))
                event_bus.inference_error.emit(str(e))
                return
            event_bus.log_message.emit(LogLevel.SUCCESS, "Zakończono analizę ścieżki audio")
            event_bus.progress_updated.emit(100, "Analiza zakończona")

            event_bus.inference_finished.emit(results)

        except Exception as e:
            event_bus.log_message.emit(LogLevel.ERROR, f"Wystąpił błąd: {str(e)}")
            event_bus.inference_error.emit(str(e))

