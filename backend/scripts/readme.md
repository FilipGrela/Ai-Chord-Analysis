# SKRYPTY URUCHOMIENIOWE (ENTRY POINTS)

Ten folder zawiera punkty wejścia do aplikacji. Skrypty te pełnią rolę "łączników" – importują klasy z pozostałych modułów backendu, konfigurację z `config.py` i uruchamiają konkretne procesy.

##  ZASADY URUCHAMIANIA
Wszystkie skrypty należy wywoływać z GŁÓWNEGO FOLDERU projektu (root), aby zachować poprawność ścieżek importów:
`python -m backend.scripts.<nazwa_skryptu>`

---

## 🛠 OPIS SKRYPTÓW

### 1. run_build_dataset.py
**Cel:** Przetworzenie surowych danych audio na format zrozumiały dla modelu (ETL).
- **Działanie**: Skanuje folder `dataset/`, wywołuje `AudioProcessor` do ekstrakcji cech CQT oraz `ChordLabelParser` do normalizacji akordów.
- **Wyjście**: Tworzy pliki binarne `.npy` (X - cechy, y - etykiety) w folderze `out/full_dataset/`.
- **Technologia**: Wykorzystuje wielowątkowość (ProcessPoolExecutor), aby maksymalnie obciążyć procesor podczas obliczeń DSP.

### 2. run_training.py
**Cel:** Trenowanie sieci neuronowej na przygotowanych danych.
- **Działanie**: Inicjalizuje architekturę `ChordCRNN`, ładuje dane przez `DataLoaderFactory` i uruchamia pętlę treningową `Trainer`.
- **Funkcje**: Automatycznie dobiera wagi klas (Loss Smoothing), obsługuje Early Stopping oraz Scheduler uczenia.
- **Wyjście**: Zapisuje najlepszy stan wag do pliku `out/best_crnn_model.pth`.
- **Sprzęt**: Zoptymalizowany pod GPU RTX 5070 (wymaga CUDA 12.8).

### 3. run_predict.py
**Cel:** Testowe wnioskowanie (Inference) na pojedynczym pliku.
- **Działanie**: Ładuje wytrenowany model `best_crnn_model.pth`, przetworzy wskazany plik MP3 i wyświetli wynik w konsoli.
- **Wyjście**: Lista sformatowanych bloków czasowych, np. `[00.00s - 02.50s] : C#m`.
- **Zastosowanie**: Szybka weryfikacja poprawności "rozumowania" modelu bez uruchamiania serwera.

### 4. run_single_tests.py
**Cel:** Diagnostyka sygnału i wizualizacja (DSP Debugging).
- **Działanie**: Pozwala wygenerować spektrogram CQT oraz Chromagram dla jednego utworu.
- **Funkcje**: Możliwość przetestowania 8 kombinacji filtrów (Smoothing, Whitening, Denoise) i zapisu ich do plików `.png` w celu porównania jakości cech.
- **Interakcja**: Otwiera okno wykresu, które pozwala sprawdzić parametry dźwięku po najechaniu myszką.

---