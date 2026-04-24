import matplotlib.pyplot as plt
import numpy as np
import os
from backend.config import cfg_audio
from backend.logger.logger import Logger

logger = Logger(__name__)

class SpectrogramVisualizer:
    """Narzędzie do interaktywnej wizualizacji i zapisu cech audio (CQT, Chromagram)."""
    
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    @classmethod
    def plot_chromagram(cls, chroma_matrix: np.ndarray, hop_size_ms: int | None = None):
        if hop_size_ms is None:
            hop_size_ms = cfg_audio.HOP_SIZE_MS

        num_frames = chroma_matrix.shape[1]
        total_time_sec = num_frames * (hop_size_ms / 1000.0)
        
        plt.figure(figsize=(14, 4))
        plt.imshow(chroma_matrix, aspect='auto', origin='lower', cmap='magma',
                   extent=(0, total_time_sec, 0, 12), interpolation='nearest')
        
        plt.colorbar(label='Energy')
        plt.yticks(np.arange(12), cls.NOTES)
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Class')
        plt.title('Chromagram - 12 Tone Representation')
        plt.tight_layout()
        plt.show()

    @classmethod
    def _draw_cqt_base_logic(cls, ax, cqt_matrix: np.ndarray, total_time_sec: float, custom_text: str | None):
        """Prywatna metoda pomocnicza rysująca główny szkielet wykresu."""
        im = ax.imshow(cqt_matrix, aspect='auto', origin='lower', cmap='magma', 
                       extent=(0, total_time_sec, 0, 84), interpolation='nearest')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('MIDI Pitch')
        
        plt.colorbar(im, ax=ax, label='Amplitude (Normalized)')
        
        y_ticks = np.arange(0, 84, 12) 
        y_labels = [f"C{i+1}" for i in range(len(y_ticks))]
        ax.set_yticks(y_ticks, y_labels)
        
        # Linie siatki dla oktaw
        for y in y_ticks:
            ax.axhline(y=y, color='white', linestyle='--', alpha=0.3)
        
        # Renderowanie boksu z parametrami
        if custom_text:
            ax.text(0.01, 0.95, custom_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', color='black',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    @classmethod
    def plot_cqt(cls, cqt_matrix: np.ndarray, hop_size_ms: int | None = None, custom_text: str | None = None):
        if hop_size_ms is None:
            hop_size_ms = cfg_audio.HOP_SIZE_MS

        num_bins, num_frames = cqt_matrix.shape
        total_time_sec = num_frames * (hop_size_ms / 1000.0)
        
        plt.figure(figsize=(14, 6))
        ax = plt.gca()
        
        cls._draw_cqt_base_logic(ax, cqt_matrix, total_time_sec, custom_text)
        ax.set_title('CQT Spectrogram - Hover to read notes')

        # Interaktywne formatowanie współrzędnych (po najechaniu myszką)
        def custom_format_coord(x, y):
            if 0 <= y < 84:
                note_idx = int(y)
                note_name = cls.NOTES[note_idx % 12]
                octave = (note_idx // 12) + 1
                col_idx = int(x / (hop_size_ms / 1000.0))
                
                if 0 <= col_idx < num_frames:
                    amplitude = cqt_matrix[note_idx, col_idx]
                    return f"Time: {x:.2f}s | Note: {note_name}{octave} | Amp: {amplitude:.2f}"
            return f"Time: {x:.2f}s | Y: {y:.2f}"
            
        ax.format_coord = custom_format_coord
        plt.tight_layout()
        plt.show()

    @classmethod
    def save_cqt_image(cls, cqt_matrix: np.ndarray, output_filename: str, hop_size_ms: int | None = None, custom_text: str | None = None):
        if hop_size_ms is None:
            hop_size_ms = cfg_audio.HOP_SIZE_MS

        num_bins, num_frames = cqt_matrix.shape
        total_time_sec = num_frames * (hop_size_ms / 1000.0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        cls._draw_cqt_base_logic(ax, cqt_matrix, total_time_sec, custom_text)
        plt.tight_layout()
        
        # Zabezpieczenie katalogu docelowego
        directory = os.path.dirname(output_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Zapis i zwolnienie pamięci (kluczowe przy masowym przetwarzaniu)
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close(fig) 
        logger.info(f"Zapisano spektrogram: {output_filename}")