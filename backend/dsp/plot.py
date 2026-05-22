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

        # Handle both (bins, frames) and (frames, bins) orderings defensively
        shape = chroma_matrix.shape
        if shape[0] > shape[1]:
            # Likely (bins, frames) - standard
            num_bins, num_frames = shape
        else:
            # Might be (frames, bins) - transpose if needed
            if shape[1] <= 100 and shape[0] > shape[1]:
                # (bins, frames)
                num_bins, num_frames = shape
            else:
                # Transpose to (bins, frames)
                chroma_matrix = chroma_matrix.T
                num_bins, num_frames = chroma_matrix.shape
        
        total_time_sec = num_frames * (hop_size_ms / 1000.0)
        
        plt.figure(figsize=(14, 6 if num_bins > 12 else 4))
        im = plt.imshow(chroma_matrix, aspect='auto', origin='lower', cmap='magma',
                   extent=(0, total_time_sec, 0, num_bins), interpolation='nearest')
        
        plt.colorbar(im, label='Energy')
        
        # Set y-ticks based on chromagram type
        if num_bins == 12:
            # Traditional chromagram: one label per semitone
            plt.yticks(np.arange(12), cls.NOTES)
            y_label = 'Pitch Class'
            title = 'Chromagram - 12 Tone Representation'
        else:
            # High-resolution chromagram: labels for every semitone
            bins_per_semitone = num_bins // 12
            if bins_per_semitone > 0:
                y_tick_positions = np.arange(0, num_bins, bins_per_semitone)
                y_tick_labels = [cls.NOTES[i % 12] for i in range(len(y_tick_positions))]
                plt.yticks(y_tick_positions, y_tick_labels)
            else:
                # num_bins < 12: space labels evenly
                step = max(1, 12.0 // num_bins)
                y_tick_positions = np.arange(0, num_bins, step)
                y_tick_labels = [cls.NOTES[i % 12] for i in range(0, num_bins, step)]
                plt.yticks(y_tick_positions, y_tick_labels)
            
            y_label = f'Pitch (bins, {bins_per_semitone} per semitone)' if bins_per_semitone > 0 else 'Pitch'
            title = f'Chromagram - {num_bins} Bins'
        
        plt.xlabel('Time (s)')
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @classmethod
    def _draw_cqt_base_logic(cls, ax, cqt_matrix: np.ndarray, total_time_sec: float, custom_text: str | None):
        """Prywatna metoda pomocnicza rysująca główny szkielet wykresu."""
        num_bins = cqt_matrix.shape[0]
        
        # Dynamically determine octaves and note labels
        octaves = num_bins / cfg_audio.BINS_PER_OCTAVE
        y_max = num_bins
        
        im = ax.imshow(cqt_matrix, aspect='auto', origin='lower', cmap='magma', 
                       extent=(0, total_time_sec, 0, y_max), interpolation='nearest')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('MIDI Pitch')
        
        plt.colorbar(im, ax=ax, label='Amplitude (Normalized)')
        
        # Grid lines for octaves
        octave_spacing = cfg_audio.BINS_PER_OCTAVE
        y_ticks = np.arange(0, num_bins, octave_spacing)
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
        
        plt.figure(figsize=(14, 6 if num_bins > 100 else 5))
        ax = plt.gca()
        
        cls._draw_cqt_base_logic(ax, cqt_matrix, total_time_sec, custom_text)
        ax.set_title(f'CQT Spectrogram ({num_bins} bins/octave={cfg_audio.BINS_PER_OCTAVE}) - Hover to read notes')

        # Interaktywne formatowanie współrzędnych (po najechaniu myszką)
        def custom_format_coord(x, y):
            if 0 <= y < num_bins:
                note_idx = int(y)
                note_name = cls.NOTES[note_idx % 12]
                octave = (note_idx // cfg_audio.BINS_PER_OCTAVE) + 1
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
        
        fig, ax = plt.subplots(figsize=(14, 6 if num_bins > 100 else 5))
        
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