import os
from tqdm import tqdm

from spectograms.spectograms import read_audio_universal, generate_spectrogram
from spectograms.plot import save_cqt_image
from labels_parser import parse_labels
from dataset_builder import align_frames_with_labels, create_sequences, save_dataset

def process_single_song(audio_path, label_path, output_dir, hop_size_ms=50, seq_len=40):
    """
    Process single song and prase labels, generate CQT, align frames with labels, create sequences and save dataset.
    """
    song_name = os.path.basename(audio_path)
    base_name = os.path.splitext(song_name)[0]
    
    print(f"\n--- Przetwarzanie: {song_name} ---")
    
    audio_data, sample_rate = read_audio_universal(audio_path)
    if audio_data is None:
        print(f"Błąd: Nie udało się wczytać audio dla {song_name}")
        return False

    # 2. Prasing of the labels
    try:
        parsed_labels = parse_labels(label_path)
        print(f"Pomyślnie wczytano etykiety z: {os.path.basename(label_path)}")
    except Exception as e:
        print(f"Błąd: Nie udało się wczytać etykiet: {e}")
        return False


    print("Generowanie macierzy CQT...")
    cqt_matrix = generate_spectrogram(
        audio_data, 
        sample_rate, 
        method='cqt', 
        hop_size_ms=hop_size_ms,
        apply_smoothing=True, 
        apply_whitening=False, 
        apply_denoise=True
    )
    num_bins, num_frames = cqt_matrix.shape

    # Save CQT image for visual inspection 
    img_path = os.path.join(output_dir, f"{base_name}_cqt_check.png")
    save_cqt_image(cqt_matrix, img_path, hop_size_ms=hop_size_ms, custom_text=f"CQT: {base_name}")

    # Alignment of frames with labels
    print("Synchronizacja ramek z etykietami...")
    frame_labels_int = align_frames_with_labels(num_frames, parsed_labels, hop_size_ms=hop_size_ms)

    # Cuting into training windows
    print(f"Cięcie na okna treningowe (długość: {seq_len} ramek)...")
    # hop_seq=10 oznacza, że okna nachodzą na siebie (data augmentation z automatu)
    X, y = create_sequences(cqt_matrix, frame_labels_int, seq_len=seq_len, hop_seq=10)

    save_dataset(X, y, output_dir, prefix_name=base_name)
    
    return True

def main():
    audio_file = os.path.join('single_test_data', 'isophonics_0', 'isophonics_0.mp3')
    label_file = os.path.join('single_test_data', 'isophonics_0', 'isophonics_0.jams')
    output_directory = os.path.join('out', 'dataset_output')

    success = process_single_song(
        audio_path=audio_file,
        label_path=label_file,
        output_dir=output_directory,
        hop_size_ms=50, # 50 ms per frame (20 fps)
        seq_len=40      # 40 frames per sequence (2 seconds)
    )
    
    if success:
        print("\nSukces! Dane zostały przetworzone i zapisane.")

if __name__ == "__main__":
    main()