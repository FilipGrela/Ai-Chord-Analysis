import os
import sys
import itertools
from glob import glob

# Pozwala uruchamiac skrypt jako plik: `python backend/scripts/run_single_tests.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import cfg_paths
from backend.dsp.spectrograms import AudioProcessor
from backend.dsp.plot import SpectrogramVisualizer

def run_combinations(processor: AudioProcessor, audio_data, base_name: str):
    """
    Testuje wszystkie kombinacje filtrów (wygładzanie, wybielanie, odszumianie)
    i zapisuje wyniki jako obrazy .png w folderze wyjściowym.
    """
    print("\n--- Generowanie kombinacji filtrów (Zapis do plików) ---")
    param_combinations = list(itertools.product([True, False], repeat=3))
    
    # Upewniamy się, że folder wyjściowy istnieje
    if not os.path.exists(cfg_paths.TEST_OUTPUT):
        os.makedirs(cfg_paths.TEST_OUTPUT)
    
    for apply_smoothing, apply_whitening, apply_denoise in param_combinations:
        print(f"Przetwarzanie -> Smoothing: {apply_smoothing} | Whitening: {apply_whitening} | Denoise: {apply_denoise}")
        
        spectrogram = processor.generate_spectrogram(
            audio_data,
            apply_smoothing=apply_smoothing,
            apply_whitening=apply_whitening,
            apply_denoise=apply_denoise,
            apply_short_noises=True  # Standardowo włączone dla czystości
        )
        
        custom_text = f"Smoothing: {apply_smoothing}\nWhitening: {apply_whitening}\nDenoise: {apply_denoise}"
        suffix = f"S{int(apply_smoothing)}_W{int(apply_whitening)}_D{int(apply_denoise)}"
        
        output_file = os.path.join(cfg_paths.TEST_OUTPUT, f"{base_name}_{suffix}.png")
        SpectrogramVisualizer.save_cqt_image(spectrogram, output_file, custom_text=custom_text)
        
    print(f"Zakończono. Pliki zapisano w: {cfg_paths.TEST_OUTPUT}\n")

def run_single_interactive(processor: AudioProcessor, audio_data):
    """
    Generuje pojedynczy spektrogram i chromagram z pełnym filtrowaniem,
    a następnie wyświetla je w interaktywnym oknie Matplotlib.
    """
    print("\n--- Generowanie interaktywnego spektrogramu i chromagramu ---")
    apply_smoothing = False
    apply_whitening = False
    apply_denoise = False
    apply_short_noises = False
    
    spectrogram = processor.generate_spectrogram(
        audio_data,
        apply_smoothing=apply_smoothing,
        apply_whitening=apply_whitening,
        apply_denoise=apply_denoise,
        apply_short_noises=apply_short_noises
    )
    
    # Generowanie chromagramu (12 tonów) z wycięciem szumów tła poniżej 10%
    chroma = processor.create_chromagram(spectrogram, threshold_percent=10)
    
    custom_text = (f"Smoothing: {apply_smoothing} | Short noises: {apply_short_noises}\n"
                   f"Whitening: {apply_whitening} | Denoise: {apply_denoise}")
    
    print("Wyświetlam Chromagram (zamknij okno, aby przejść dalej)...")
    SpectrogramVisualizer.plot_chromagram(chroma)
    
    print("Wyświetlam Spektrogram CQT...")
    SpectrogramVisualizer.plot_cqt(spectrogram, custom_text=custom_text)

def main():
    print("--- Moduł Diagnostyki DSP (Digital Signal Processing) ---")
    
    # 1. Inicjalizacja procesora audio (pobiera ustawienia z config.py)
    processor = AudioProcessor()
    
    
    # 2. Ścieżka do testowego pliku (izolowany folder testowy)
    search_path = cfg_paths.SINGLE_TEST_DATA

    # Szukamy wszystkich plików mp3 i wav
    audio_files = glob(os.path.join(search_path, "*.mp3")) + \
                  glob(os.path.join(search_path, "*.wav"))
    
    if not audio_files:
        print("Błąd: Nie znaleziono plików testowych w folderze 'single_test_data/'.")
        print("Upewnij się, że masz pliki '.mp3' lub '.wav' w folderze 'single_test_data/'.")
        return
        
    file_path = audio_files[0]
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 3. Odczyt Audio
    print(f"\nWczytywanie pliku: {base_name}.mp3")
    audio_data, sample_rate = processor.read_audio_universal(file_path)
    
    if audio_data is None or sample_rate is None:
        print("Krytyczny błąd: Nie udało się zdekodować pliku audio.")
        return
        
    print(f"Sample Rate: {sample_rate} Hz, Długość: {len(audio_data) / sample_rate:.2f} s")
    
    # =================================================================
    # ODKOMENTUJ ZADANIE, KTÓRE CHCESZ WYKONAĆ:
    # =================================================================
    
    # Opcja A: Zapisz wszystkie 8 kombinacji filtrów do folderu 'out/dataset_output'
    # run_combinations(processor, audio_data, base_name)
    
    # Opcja B: Wyświetl na żywo Chromagram i CQT dla optymalnych ustawień
    run_single_interactive(processor, audio_data)

if __name__ == "__main__":
    main()