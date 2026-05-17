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

### 5. run_music_metrics.py
**Cel:** Analiza muzyczna datasetu i metryki jakości akordu.
- **Działanie**: Czyta ustawienia bezpośrednio z `backend/config.py` (`cfg_analysis`) i przetwarza dataset `isophonics_dataset`.
- **Zakres analizy**: Liczy rozkład rootów, jakości akordów (rozszerzone kategorie: `7`, `maj7`, `m7`, `sus2`, `sus4`, `aug`, `dim`, `other`), odległości interwałowe, procent segmentów zgodnych z tonacją oraz najczęstsze przejścia między akordami.
- **Dodatkowe funkcje:**
	- raport **per-pliki** (jeśli liczba plików ≤ 50) z top-root i top-quality,
	- zliczanie i wypis top-10 przejść akordowych,
	- eksport wyników do prostego `HTML`
- **Zastosowanie**: Służy jako szybki punkt wejścia do sprawdzenia, czy dane są muzycznie spójne i jak można je później wykorzystać do transpozycji oraz balansowania klas.

---
