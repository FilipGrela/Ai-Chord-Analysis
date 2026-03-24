import subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _draw_cqt_base_logic(ax, cqt_matrix, total_time_sec, custom_text):
    # Core rendering logic (pixel-perfect interpolation)
    im = ax.imshow(cqt_matrix, aspect='auto', origin='lower', cmap='magma', 
                   extent=[0, total_time_sec, 0, 84], interpolation='nearest')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MIDI Pitch')
    
    plt.colorbar(im, ax=ax, label='Amplitude (Normalized)')
    
    y_ticks = np.arange(0, 84, 12) 
    y_labels = [f"C{i+1}" for i in range(len(y_ticks))]
    ax.set_yticks(y_ticks, y_labels)
    
    # Octave gridlines
    for y in y_ticks:
        ax.axhline(y=y, color='white', linestyle='--', alpha=0.3)
    
    # Render parameters bounding box
    if custom_text:
        ax.text(0.01, 0.95, custom_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))


def plot_cqt(cqt_matrix, hop_size_ms=50, custom_text=None):
    num_bins, num_frames = cqt_matrix.shape
    total_time_sec = num_frames * (hop_size_ms / 1000.0)
    
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    _draw_cqt_base_logic(ax, cqt_matrix, total_time_sec, custom_text)
    ax.set_title('CQT Spectrogram - Hover to read notes')

    # Interactive coordinates formatting
    def custom_format_coord(x, y):
        if 0 <= y < 84:
            note_idx = int(y)
            note_name = NOTES[note_idx % 12]
            octave = (note_idx // 12) + 1
            col_idx = int(x / (hop_size_ms / 1000.0))
            
            if 0 <= col_idx < num_frames:
                amplitude = cqt_matrix[note_idx, col_idx]
                return f"Time: {x:.2f}s | Note: {note_name}{octave} | Amp: {amplitude:.2f}"
                
        return f"Time: {x:.2f}s | Y: {y:.2f}"
        
    ax.format_coord = custom_format_coord
        
    plt.tight_layout()
    plt.show()


def save_cqt_image(cqt_matrix, output_filename, hop_size_ms=50, custom_text=None):
    num_bins, num_frames = cqt_matrix.shape
    total_time_sec = num_frames * (hop_size_ms / 1000.0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    _draw_cqt_base_logic(ax, cqt_matrix, total_time_sec, custom_text)
    plt.tight_layout()
    
    # Ensure target directory exists
    directory = os.path.dirname(output_filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save to file and release memory (prevent RAM leaks during batch processing)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig) 
    print(f"Saved spectrogram: {output_filename}")