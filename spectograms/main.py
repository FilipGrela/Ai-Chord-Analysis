from spectograms import generate_spectrogram, create_chromagram, read_audio_universal
from plot import plot_cqt, save_cqt_image, plot_chromagram

import itertools
import os


def generate_combinations(audio_data, sample_rate, base_name):
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
        
        suffix = f"S{int(apply_smoothing)}_W{int(apply_whitening)}_D{int(apply_denoise)}"
        output_file = f"spectograms\\output\\{base_name}_{suffix}.png"
        
        save_cqt_image(spectrogram, output_file, custom_text=custom_text)

    print("\nBatch processing complete. All combinations saved.")

def generate_single_spectrogram(audio_data, sample_rate):

    apply_smoothing = True
    apply_whitening = True
    apply_denoise = True
    apply_short_noises = True

    spectrogram = generate_spectrogram(audio_data, sample_rate, method='cqt', 
                                       apply_smoothing=apply_smoothing, apply_whitening=apply_whitening, apply_denoise=apply_denoise, apply_short_noises=apply_short_noises)
    
    # Everything quieted down unver the threshold
    chroma = create_chromagram(spectrogram, threshold_percent=10)

    plot_chromagram(chroma)
    plot_cqt(spectrogram, custom_text=f"Applied Smoothing: {apply_smoothing},Short noises removed: {apply_short_noises}, Whitening: {apply_whitening}, Denoise: {apply_denoise}")



def main():
    file_path = 'spectograms\\samples\\Beethoven - Moonlight Sonata (FULL)(1).mp3'   
    audio_data, sample_rate = read_audio_universal(file_path)

    if audio_data is None:
        print("Audio loading failed.")
        return

    song_name = os.path.basename(file_path)
    base_name = os.path.splitext(song_name)[0]

    # generate_combinations(audio_data, sample_rate, base_name)
    generate_single_spectrogram(audio_data, sample_rate)
    
    print(f"Song name: {song_name}")
    print(f"Sample Rate: {sample_rate} Hz, Audio Length: {len(audio_data) / sample_rate:.2f} seconds")

   

if __name__ == "__main__":
    main()