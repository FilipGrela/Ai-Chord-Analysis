import subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def read_audio_universal(file_path, target_sr=44100):
    command = [
        'ffmpeg',
        '-i', file_path,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', str(target_sr),
        '-loglevel', 'quiet',
        '-'
    ]
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0:
            print(f"Błąd FFmpeg: {err.decode('utf-8')}")
            return None, None

        signal = np.frombuffer(out, dtype=np.int16)
        
        return signal, target_sr
        
    except FileNotFoundError:
        print("Błąd: Nie znaleziono FFmpeg w systemie! Musisz go zainstalować i dodać do zmiennych środowiskowych (PATH).")
        return None, None


# ==========================================
#            ENHANCEMENT PIPELINE
# ==========================================

def smooth_harmonics(spectrogram, kernel_size=15):
    """
    Smooth the spectrogram by averaging each bin with its neighbors.
    This can help reduce percussion sounds, noise and make harmonic patterns more visible.
    """

    smoothed_cqt = np.zeros_like(spectrogram)

    kernel = np.ones(kernel_size) / kernel_size

    for i in tqdm(range(spectrogram.shape[0]), desc="Smoothing Spectrogram", leave=False):
        smoothed_cqt[i, :] = np.convolve(spectrogram[i, :], kernel, mode='same')

    return smoothed_cqt

def denoise_normalize_audio(cqt_db, dynamic_range = 35):

    max_db = np.max(cqt_db)

    threshold_db = max_db - dynamic_range
    
    cqt_db[cqt_db < threshold_db] = threshold_db
    
    cqt_normalized = (cqt_db - threshold_db) / dynamic_range
    
    return cqt_normalized

def generate_hann_window(window_size):
    n = np.arange(window_size)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (window_size - 1))
    return window

def spectral_whitening(spectrogram):
    """
    It balances energy between frequency ranges. It prevents low bass from dominating the highs.
    """
    whitened_cqt = np.zeros_like(spectrogram)
    
    for i in tqdm(range(spectrogram.shape[0]), desc="Spectral Whitening", leave=False):
        row = spectrogram[i, :]
        
        row_median = np.median(row)
        whitened_cqt[i, :] = row - row_median
        
    return whitened_cqt

def remove_short_noises(chroma_matrix, min_duration_frames=3):
    """
    Removes short-lived energy bursts (percussion/noise) from the chromagram.
    min_duration_frames: sounds shorter than number of frames will be zeroed.
    """
    cleaned_chroma = np.copy(chroma_matrix)

    for i in range(12):
        row = cleaned_chroma[i, :]
        

        for f in range(1, len(row) - 1):
            # If the current frame is active but neighbors are silent, kill it.
            if row[f] > 0 and row[f-1] == 0 and row[f+1] == 0:
                cleaned_chroma[i, f] = 0
                
    return cleaned_chroma

# ==========================================
#                CORE MATH
# ==========================================

def calculate_spectogram_cqt(audio_data, sample_rate, fmin=32.703, n_bins=84, bins_per_octave=12, hop_size_ms=50):

    Q = 1.0 / (2**(1.0 / bins_per_octave) - 1.0)
    
    # Frequencies for each bin
    freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
    
    # Window size in samples for each frequency bin
    window_lengths = np.ceil(Q * sample_rate / freqs).astype(int)
    max_window = window_lengths[0] 

    filters = []

    for k in range(n_bins):
        N = window_lengths[k]
        window = generate_hann_window(N)

        complex_wave = np.exp(-2j * np.pi * freqs[k] * np.arange(N) / sample_rate)
        
        filters.append((window * complex_wave) / N)
    
    # 5. Przygotowanie do skanowania piosenki
    hop_length = int(sample_rate * (hop_size_ms / 1000.0))
    # Całkowita liczba ramek (uwzględniamy max_window, by nie wyjść poza plik przy najniższym basie)
    total_frames = 1 + (len(audio_data) - max_window) // hop_length
    
    # Tworzymy pustą macierz docelową (84 klawisze x okna czasowe)
    cqt_result = np.zeros((n_bins, total_frames))
    
    # 6. Analiza sygnału (Przesuwamy się po nagraniu)
    for i in tqdm(range(total_frames), desc="Analyzing Audio Frames", leave=False):
        start_idx = i * hop_length
        
        # Sprawdzamy występowanie każdej z 84 nut w tym konkretnym ułamku sekundy
        for k in range(n_bins):
            N = window_lengths[k]
            
            # Pobieramy fragment sygnału idealnie docięty pod tę nutę
            frame = audio_data[start_idx : start_idx + N]
            
            # Iloczyn skalarny (dot product) naszego filtra z surowym dźwiękiem
            # Wyciągamy moduł z liczby zespolonej (amplitudę)
            cqt_result[k, i] = np.abs(np.dot(frame, filters[k]))
            
    # Konwersja do skali decybelowej (skala logarytmiczna)
    cqt_db = 20 * np.log10(cqt_result + 1e-10)

    
    # Wynik jest już w formacie (Częstotliwości, Okna), więc zwracamy bez transpozycji
    return cqt_db

def calculate_spectogram_rfft(audio_data, sample_rate, window_size=100, hop_size=50):
    window_size_samples = int(sample_rate * (window_size / 1000.0))
    hop_lenght = int(sample_rate * (hop_size/1000.0))

    total_sample_num = 1 + int(len(audio_data) - window_size_samples) // hop_lenght


    # Hann Window - Allows audio to overlap during sampling
    window = generate_hann_window(window_size_samples)

    spectrogram = []

    for i in tqdm(range(total_sample_num), desc="Calculating RFFT Frames", leave=False):
        start = i * hop_lenght
        end = start + window_size_samples
        frame = audio_data[start:end]

        windowed_frame = frame * window

        fft_result = np.fft.rfft(windowed_frame)
        
        magnitude = np.abs(fft_result)
        db_magnitude = 20 * np.log10(magnitude + 1e-10)
        
        spectrogram.append(db_magnitude)

    return np.array(spectrogram).T


def create_chromagram(cqt_matrix, threshold_percent=25):
    """
    Enhanced Chromagram with non-linear scaling to reduce noise.
    """
    num_bins, num_frames = cqt_matrix.shape
    chroma = np.zeros((12, num_frames))
    
    # Non-linear scaling  to enhance strong harmonics and suppress noise
    cqt_enhanced = np.power(cqt_matrix, 3) 

    for i in range(num_bins):
        pitch_class = i % 12
        chroma[pitch_class, :] += cqt_enhanced[i, :]
        
    # Dynamic Range Clipping & Normalization per frame
    for f in range(num_frames):
        col = chroma[:, f]
        col_max = np.max(col)
        
        if col_max > 0:
            # Re-normalize to 0.0 - 1.0 range
            col /= col_max
            
            # 4. Final Thresholding (Optional: zeros out everything below 20% of max)
            col[col < threshold_percent / 100.0] = 0
            
            # Re-normalize again after thresholding
            final_max = np.max(col)
            if final_max > 0:
                col /= final_max
        
        chroma[:, f] = col
            
    return chroma

# ==========================================
#                 PIPELINE
# ==========================================

def generate_spectrogram(audio_data, sample_rate, method='cqt', 
                         apply_denoise=True, apply_short_noises=True, apply_whitening=True, apply_smoothing=True, **kwargs):

    if method == 'cqt':
        spectrogram = calculate_spectogram_cqt(audio_data, sample_rate,n_bins=84, **kwargs)
    elif method == 'rfft':
        spectrogram = calculate_spectogram_rfft(audio_data, sample_rate, **kwargs)
    else:
        raise ValueError(f"Nieznana metoda spektrogramu: {method}. Dostępne opcje: 'cqt', 'rfft'.")
    

    
    if apply_denoise:
        spectrogram = denoise_normalize_audio(spectrogram)
    if apply_short_noises:
        spectrogram = remove_short_noises(spectrogram)
    if apply_whitening:
        spectrogram = spectral_whitening(spectrogram)
    if apply_smoothing:
        spectrogram = smooth_harmonics(spectrogram)


        
    return spectrogram