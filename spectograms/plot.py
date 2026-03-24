import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_cqt(cqt_matrix, hop_size_ms=50):
    """
    Ulepszone rysowanie macierzy CQT z prawdziwą osią czasu i siatką.
    """
    # 1. Obliczamy całkowity czas trwania fragmentu w sekundach
    num_bins, num_frames = cqt_matrix.shape
    total_time_sec = num_frames * (hop_size_ms / 1000.0)
    
    plt.figure(figsize=(14, 6))
    
    # 2. Używamy 'extent', aby narzucić prawdziwe wartości na osie X i Y.
    # extent = [lewa_krawędź_X, prawa_krawędź_X, dolna_krawędź_Y, górna_krawędź_Y]
    # Używamy interpolation='nearest', aby obraz był ostry jak brzytwa (tzw. pixel-perfect)
    plt.imshow(cqt_matrix, aspect='auto', origin='lower', cmap='magma', 
               extent=[0, total_time_sec, 0, 84], interpolation='nearest')
    
    plt.colorbar(label='Amplituda (Znormalizowana)')
    
    # Zmieniona oś X - teraz to prawdziwe sekundy!
    plt.xlabel('Czas (Sekundy)')
    plt.ylabel('Klawisze fortepianu')
    plt.title('Spektrogram CQT - Ręczna implementacja 84 klawiszy')
    
    # 3. Podmiana surowych numerów na nazwy nut (tak jak miałeś)
    y_ticks = np.arange(0, 84, 12) 
    y_labels = [f"C{i+1}" for i in range(len(y_ticks))]
    plt.yticks(y_ticks, y_labels)
    
    # 4. NOWOŚĆ: Dodajemy poziome, półprzezroczyste linie na każdym 'C', 
    # żeby łatwiej było czytać wykres po prawej stronie
    for y in y_ticks:
        plt.axhline(y=y, color='white', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.show()

# Wywołanie pozostaje proste:
# plot_cqt_better(spectrogram)


def plot_spectrogram(spectrogram, sample_rate, window_size=100, hop_size=50):
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.title('Spectrogram')
    plt.show()