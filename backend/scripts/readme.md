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

### 6. inspect_checkpoint.py
**Cel:** Podgląd zapisanych metadanych i pełnego configu z checkpointu modelu.
- **Działanie**: Ładuje plik `.pth`, wykrywa nowy format checkpointu z `metadata` i wypisuje pełny snapshot konfiguracji.
- **Wyjście**: Czytelny podgląd sekcji `logger`, `paths`, `audio`, `model`, `train`, `builder` i `analysis` zapisanych przy treningu.
- **Użycie**:
```bash
python -m backend.scripts.inspect_checkpoint --checkpoint out/model.pth
```
- **Uwaga**: Jeśli nie podasz `--checkpoint`, skrypt użyje domyślnego pliku z `backend/config.py` (`cfg_paths.MODEL_SAVE_PATH`).
- **Menu wyboru**: Jeśli w podanej lokalizacji jest kilka plików `.pth`, skrypt pokaże prostą listę numerowaną i poprosi o wybór konkretnego modelu.

### 7. run_test_model.py
**Cel:** Pełniejsza ewaluacja modelu na zbiorze walidacyjnym.
- **Działanie**: Ładuje checkpoint, wykonuje predykcje na walidacji i generuje metryki jakości oraz wykresy.
- **Wyjście**: CSV z predykcjami, podsumowanie metryk i wykresy confusion matrix / per-class metrics.
- **Zastosowanie**: Przydatny do porównywania checkpointów po treningu bez uruchamiania pełnego pipeline inference.

### 8. run_transpose_offline.py
**Cel:** Offline augmentacja danych przez transpozycję.
- **Działanie**: Generuje transponowane warianty materiału wejściowego, aby zwiększyć różnorodność danych treningowych.
- **Wyjście**: Nowe pliki/wersje danych w zadanym zakresie półtonów.
- **Zastosowanie**: Przygotowanie bardziej zbalansowanego zbioru do treningu i eksperymentów.

---
