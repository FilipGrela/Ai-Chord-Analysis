import sys
import os
import html

# Ignoring mp3float warnings
try:
    stderr_fd = sys.stderr.fileno()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, stderr_fd)
    sys.stderr = sys.stdout
except Exception:
    pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit, QFileDialog, QGroupBox, QSlider
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from backend.event_system.event_bus import LogLevel, event_bus
from backend.api.worker import InferenceWorker


class MainWindow(QMainWindow):
    SHARP_TO_FLAT = {
        "C#": "D♭", "D#": "E♭", "F#": "G♭", "G#": "A♭", "A#": "B♭",
        "C#m": "D♭m", "D#m": "E♭m", "F#m": "G♭m", "G#m": "A♭m", "A#m": "B♭m"
    }

    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("Chord Classifier - AI Audio Analysis")
        self.resize(700, 500)
        self.results = []
        self.current_audio_path = None

        self.use_flats = False

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)

        # Connecting to audio events
        self.player.positionChanged.connect(self.sync_playback)
        self.player.durationChanged.connect(self.update_duration)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)  # Wykrywanie końca piosenki

        # --- UI LAYOUT ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.btn_upload = QPushButton("🎵 Upload Piosenki (.mp3, .wav)")
        self.btn_upload.setMinimumHeight(40)
        self.btn_upload.clicked.connect(self.upload_file)
        layout.addWidget(self.btn_upload)

        chords_group = QGroupBox("Odtwarzanie i Akordy")
        chords_layout = QVBoxLayout(chords_group)

        # Buttons
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.btn_play)

        self.btn_toggle_notation = QPushButton("Zmień na bemole (♭)")
        self.btn_toggle_notation.clicked.connect(self.toggle_notation)
        controls_layout.addWidget(self.btn_toggle_notation)
        chords_layout.addLayout(controls_layout)

        # Slider
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position) # User slides slider event
        self.lbl_time = QLabel("00:00 / 00:00")

        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_time)
        chords_layout.addLayout(slider_layout)

        # Accords
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

        # Status
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

    def format_time(self, ms: int) -> str:
        seconds = (ms // 1000) % 60
        minutes = (ms // 60000) % 60
        return f"{minutes:02}:{seconds:02}"

    # Audio logic
    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶ Play")
        else:
            # Clicking play after the end of the song
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                self.player.setPosition(0)
            self.player.play()
            self.btn_play.setText("⏸ Pause")

    def set_position(self, position: int):
        self.player.setPosition(position)

    def update_duration(self, duration: int):
        self.slider.setRange(0, duration)

    def sync_playback(self, position_ms: int):
        if not self.slider.isSliderDown():
            self.slider.setValue(position_ms)

        duration_ms = self.player.duration()
        self.lbl_time.setText(f"{self.format_time(position_ms)} / {self.format_time(duration_ms)}")

        if not self.results:
            return

        current_sec = position_ms / 1000.0

        target_idx = 0
        for i, interval in enumerate(self.results):
            if current_sec <= interval['end']:
                target_idx = i
                break
        else:
            target_idx = len(self.results) - 1

        curr_chord = self.format_chord(self.results[target_idx]['chord'])
        prev_chord = self.format_chord(self.results[target_idx - 1]['chord']) if target_idx > 0 else "-"
        next_chord = self.format_chord(self.results[target_idx + 1]['chord']) if target_idx < len(
            self.results) - 1 else "-"

        if (self.lbl_curr.text() != curr_chord or
                self.lbl_prev.text() != prev_chord or
                self.lbl_next.text() != next_chord):
            self.lbl_prev.setText(prev_chord)
            self.lbl_curr.setText(curr_chord)
            self.lbl_next.setText(next_chord)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.stop()
            self.btn_play.setText("🔁 Odtwórz ponownie")
            self.slider.setValue(0)
            self.lbl_time.setText(f"00:00 / {self.format_time(self.player.duration())}")

    def format_chord(self, chord: str) -> str:
        if self.use_flats and chord in self.SHARP_TO_FLAT:
            return self.SHARP_TO_FLAT[chord]
        return chord

    def toggle_notation(self):
        self.use_flats = not self.use_flats
        self.btn_toggle_notation.setText("Zmień na krzyżyki (♯)" if self.use_flats else "Zmień na bemole (♭)")
        if self.results:
            self.sync_playback(self.player.position())

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik audio", "", "Audio Files (*.mp3 *.wav)")
        if file_path:
            self.current_audio_path = file_path

            if hasattr(self, 'player'):
                self.player.stop()
            self.btn_play.setText("▶ Play")

            self.btn_upload.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.progress_bar.setValue(0)
            self.log_console.clear()

            self.results = []
            self.lbl_prev.setText("-")
            self.lbl_curr.setText("Analiza...")
            self.lbl_next.setText("-")
            self.slider.setValue(0)
            self.lbl_time.setText("00:00 / 00:00")

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

        safe_message = html.escape(message)

        html_msg = f'<span style="color:{color}">[{level.name}] {safe_message}</span>'
        self.log_console.append(html_msg)

    def on_inference_done(self, results: list):
        self.results = results
        self.btn_upload.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.lbl_status.setText("Zakończono. Możesz odtworzyć utwór.")

        if self.current_audio_path:
            self.player.setSource(QUrl.fromLocalFile(self.current_audio_path))

        if self.results:
            self.lbl_curr.setText(self.format_chord(self.results[0]['chord']))
            if len(self.results) > 1:
                self.lbl_next.setText(self.format_chord(self.results[1]['chord']))

    def on_inference_error(self, error_msg: str):
        self.btn_upload.setEnabled(True)
        self.lbl_status.setText("Błąd analizy.")
        self.lbl_curr.setText("BŁĄD")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()