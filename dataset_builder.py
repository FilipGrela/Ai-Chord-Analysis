import numpy as np
import os


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