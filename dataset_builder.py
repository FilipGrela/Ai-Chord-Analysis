import numpy as np
import glob
import os
import traceback

from spectograms.spectograms import read_audio_universal, generate_spectrogram
from labels_parser import parse_labels

# Define the vocabulary of chord labels (simplified)
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
VOCAB = NOTES + [n + 'm' for n in NOTES] + ['N']

# Lookup tables for encoding/decoding chord labels
CHORD_TO_INT = {chord: idx for idx, chord in enumerate(VOCAB)}
INT_TO_CHORD = {idx: chord for chord, idx in CHORD_TO_INT.items()}


def align_frames_with_labels(num_frames, parsed_labels, hop_size_ms=50):
    """
    Maps each CQT frame to a single integer label based on its physical time in seconds.
    """
    frame_labels_int = np.full(num_frames, CHORD_TO_INT['N'], dtype=np.int32)
    
    for frame_idx in range(num_frames):
        # Calculate the exact time center of the current frame
        time_sec = frame_idx * (hop_size_ms / 1000.0)
        
        # Find which chord was playing at this exact second
        for start, end, chord in parsed_labels:
            if start <= time_sec < end:
                # Fallback to 'N' if parser let something weird slip through
                safe_chord = chord if chord in CHORD_TO_INT else 'N'
                frame_labels_int[frame_idx] = CHORD_TO_INT[safe_chord]
                break # Found the chord, move to the next frame
                
    return frame_labels_int


def create_sequences(cqt_matrix, frame_labels_int, seq_len=40, hop_seq=10):
    """
    Slices the full song into small, overlapping windows (sequences) for the CRNN.
    - seq_len: How many frames the network sees at once (e.g., 40 frames = 2 seconds).
    - hop_seq: How many frames to move forward for the next sequence (overlap).
    """
    num_bins, num_frames = cqt_matrix.shape
    
    X_sequences = []
    y_labels = []
    
    # Slide a window across the song
    for start_idx in range(0, num_frames - seq_len + 1, hop_seq):
        end_idx = start_idx + seq_len
        
        # 1. Extract the 2D feature patch (Bins x Frames)
        patch = cqt_matrix[:, start_idx:end_idx]
        
        # 2. Transpose to (Time_Steps, Features) - standard format for PyTorch/Keras RNNs
        patch_t = patch.T 
        
        # 3. Get the label for the center of this sequence
        center_idx = start_idx + (seq_len // 2)
        label = frame_labels_int[center_idx]
        
        X_sequences.append(patch_t)
        y_labels.append(label)
        
    return np.array(X_sequences), np.array(y_labels)

def save_dataset(X, y, output_dir, prefix_name):
    """
    Saves the processed sequences and labels as binary .npy files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    x_path = os.path.join(output_dir, f"{prefix_name}_X.npy")
    y_path = os.path.join(output_dir, f"{prefix_name}_y.npy")
    
    np.save(x_path, X)
    np.save(y_path, y)
    
    print(f"Dataset saved: {X.shape[0]} sequences of shape {X.shape[1:]}")
    print(f"X saved to: {x_path}")
    print(f"y saved to: {y_path}")

def build_entire_dataset(dataset_root, output_dir, hop_size_ms=50, seq_len=40):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]

    print(f"Znaleziono {len(subfolders)} podfolderów w {dataset_root}.")

    success_count = 0
    error_count = 0

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        audio_files= glob.glob(os.path.join(folder, '*.mp3')) + glob.glob(os.path.join(folder, '*.wav'))
        label_files = glob.glob(os.path.join(folder, '*.jams'))

        if not audio_files or not label_files:
            print(f"\n[POMINIĘTO] {folder_name}: Brakuje pliku audio lub etykiet.")
            error_count += 1
            continue

        audio_path = audio_files[0]
        label_path = label_files[0]

        try:
            print(f"Przetwarzanie: {folder_name}...")
            
            # 1. Odczyt Audio
            audio_data, sample_rate = read_audio_universal(audio_path)
            if audio_data is None:
                raise ValueError("Błąd odczytu pliku audio.")

            # 2. Odczyt Etykiet
            parsed_labels = parse_labels(label_path)

            # 3. Generowanie CQT
            cqt_matrix = generate_spectrogram(
                audio_data, 
                sample_rate, 
                method='cqt', 
                hop_size_ms=hop_size_ms,
                apply_smoothing=True, 
                apply_whitening=False, 
                apply_denoise=True
            )

            # 4. Synchronizacja milisekund z etykietami
            num_bins, num_frames = cqt_matrix.shape
            frame_labels_int = align_frames_with_labels(num_frames, parsed_labels, hop_size_ms=hop_size_ms)

            # 5. Pocięcie na sekwencje
            X, y = create_sequences(cqt_matrix, frame_labels_int, seq_len=seq_len, hop_seq=10)

            # 6. Zapis
            save_dataset(X, y, output_dir, prefix_name=folder_name)
            
            success_count += 1
        except Exception as e:
            print(f"\n[BŁĄD] Wystąpił problem przy utworze {folder_name}: {e}")
            traceback.print_exc()
            error_count += 1
            continue


if __name__ == "__main__":
    dataset_root = 'isophonics_dataset'  # Folder z podfolderami utworów
    output_dir = os.path.join('out', 'full_dataset')
    
    build_entire_dataset(dataset_root, output_dir, hop_size_ms=50, seq_len=40)