import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit, QFileDialog, QGroupBox
)
from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from backend.event_system.event_bus import *
from backend.api.worker import InferenceWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("Chord Classifier - AI Audio Analysis")
        self.resize(700, 500)
        self.results = []
        self.current_audio_path = None

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.player.positionChanged.connect(self.sync_chords)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.btn_upload = QPushButton("🎵 Upload Piosenki (.mp3, .wav)")
        self.btn_upload.setMinimumHeight(40)
        self.btn_upload.clicked.connect(self.upload_file)
        layout.addWidget(self.btn_upload)

        chords_group = QGroupBox("Odtwarzanie i Akordy")
        chords_layout = QVBoxLayout(chords_group)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        chords_layout.addWidget(self.btn_play)

        labels_layout = QHBoxLayout()
        self.lbl_prev = QLabel("-")
        self.lbl_curr = QLabel("Czekam...")
        self.lbl_next = QLabel("-")

        for lbl in (self.lbl_prev, self.lbl_curr, self.lbl_next):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_prev.setStyleSheet("font-size: 20px; color: gray;")
        self.lbl_curr.setStyleSheet("font-size: 40px; font-weight: bold; color: #2ecc71;")
        self.lbl_next.setStyleSheet("font-size: 20px; color: gray;")

        labels_layout.addWidget(self.lbl_prev)
        labels_layout.addWidget(self.lbl_curr)
        labels_layout.addWidget(self.lbl_next)

        chords_layout.addLayout(labels_layout)
        layout.addWidget(chords_group)

        status_group = QGroupBox("Status Analizy")
        status_layout = QVBoxLayout(status_group)

        self.lbl_status = QLabel("Gotowy")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.log_console)
        layout.addWidget(status_group)

        event_bus.progress_updated.connect(self.on_progress_update)
        event_bus.log_message.connect(self.on_log_message)
        event_bus.inference_finished.connect(self.on_inference_done)
        event_bus.inference_error.connect(self.on_inference_error)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik audio", "", "Audio Files (*.mp3 *.wav)"
        )
        if file_path:
            self.current_audio_path = file_path

            if hasattr(self, 'player'):
                self.player.stop()
            self.btn_play.setText("▶ Play")

            self.btn_upload.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.progress_bar.setValue(0)
            self.log_console.clear()
            self.lbl_curr.setText("Analiza...")

            self.worker = InferenceWorker(file_path)
            self.worker.start()

    def on_progress_update(self, percent: int, status: str):
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(status)

    def on_log_message(self, level: LogLevel, message: str):
        color = "white"
        if level == LogLevel.ERROR:
            color = "red"
        elif level == LogLevel.WARNING:
            color = "yellow"
        elif level == LogLevel.INFO:
            color = "#569cd6"
        elif level == LogLevel.DEBUG:
            color = "gray"
        elif level == LogLevel.SUCCESS:
            color = "green"

        html_msg = f'<span style="color:{color}">[{level.name}] {message}</span>'
        self.log_console.append(html_msg)

    def on_inference_done(self, results: list):
        self.results = results
        self.btn_upload.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.lbl_status.setText("Zakończono. Możesz odtworzyć utwór.")

        if self.current_audio_path:
            self.player.setSource(QUrl.fromLocalFile(self.current_audio_path))

        if self.results:
            self.lbl_curr.setText(self.results[0]['chord'])
            if len(self.results) > 1:
                self.lbl_next.setText(self.results[1]['chord'])

    def on_inference_error(self, error_msg: str):
        self.btn_upload.setEnabled(True)
        self.lbl_status.setText("Błąd analizy.")
        self.lbl_curr.setText("BŁĄD")

    def toggle_playback(self):
        """Pauzuje lub wznawia odtwarzanie muzyki."""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶ Play")
        else:
            self.player.play()
            self.btn_play.setText("⏸ Pause")

    def sync_chords(self, position_ms: int):
        """Wywolywane dziesiątki razy na sekundę przez odtwarzacz. Aktualizuje akordy na ekranie."""
        if not self.results:
            return

        current_sec = position_ms / 1000.0

        for i, interval in enumerate(self.results):
            if interval['start'] <= current_sec <= interval['end']:

                curr_chord = interval['chord']
                prev_chord = self.results[i - 1]['chord'] if i > 0 else "-"
                next_chord = self.results[i + 1]['chord'] if i < len(self.results) - 1 else "-"

                if self.lbl_curr.text() != curr_chord:
                    self.lbl_prev.setText(prev_chord)
                    self.lbl_curr.setText(curr_chord)
                    self.lbl_next.setText(next_chord)

                break


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
