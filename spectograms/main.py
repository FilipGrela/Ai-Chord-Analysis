from    spectograms import *

import itertools
import os

def main():
    file_path = 'spectograms\\samples\\Gravity.mp3'   
    audio_data, sample_rate = read_audio_universal(file_path)

    if audio_data is None:
        print("Audio loading failed.")
        return

    song_name = os.path.basename(file_path)
    base_name = os.path.splitext(song_name)[0]
    
    print(f"Song name: {song_name}")
    print(f"Sample Rate: {sample_rate} Hz, Audio Length: {len(audio_data) / sample_rate:.2f} seconds")

    # Generate all (True, False) combinations for the 3 parameters
    param_combinations = list(itertools.product([True, False], repeat=3))

    for apply_smoothing, apply_whitening, apply_denoise in param_combinations:
        
        print(f"\nProcessing -> Smoothing: {apply_smoothing} | Whitening: {apply_whitening} | Denoise: {apply_denoise}")

        spectrogram = generate_spectrogram(
            audio_data, 
            sample_rate, 
            method='cqt', 
            apply_smoothing=apply_smoothing, 
            apply_whitening=apply_whitening, 
            apply_denoise=apply_denoise
        )
        
        # Format custom text for the image
        custom_text = f"Smoothing: {apply_smoothing}\nWhitening: {apply_whitening}\nDenoise: {apply_denoise}"
        
        # Dynamic filename generation (e.g., Gravity_S1_W0_D1.png)
        suffix = f"S{int(apply_smoothing)}_W{int(apply_whitening)}_D{int(apply_denoise)}"
        output_file = f"spectograms\\output\\{base_name}_{suffix}.png"
        
        save_cqt_image(spectrogram, output_file, custom_text=custom_text)

    print("\nBatch processing complete. All combinations saved.")

if __name__ == "__main__":
    main()