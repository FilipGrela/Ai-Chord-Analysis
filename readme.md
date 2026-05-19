# AI Chord Analysis — Core System

Zaawansowany pipeline do automatycznego rozpoznawania akordów z plików audio (CQT + CRNN). Zoptymalizowany pod GPU z architekrurą Blackwell.

## Wymagania Systemowe
* **Python**: 3.10 lub nowszy.
* **FFmpeg**: Wymagany do dekodowania audio (musi być w systemowym PATH) Pobierz [tutaj](https://ffmpeg.org/download.html).

## Instalacja

1. **Przygotowanie środowiska:**
```bash
   python -m venv .venv
   .\.venv\Scripts\activate
```

> [!IMPORTANT]
> **Kluczowy krok dla RTX 5070:**
> Przed instalacją pozostałych bibliotek, zainstaluj dedykowaną wersję PyTorch komendą:
> ```pip install torch==2.11.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128```
> Dodatkowo wymagana jest [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).

1. **Instalacja pozostałych zależności:**
   ```bash 
   pip install -r requirements.txt
   ```

## Workflow Wykonawczy

Wszystkie komendy uruchamiaj z głównego folderu projektu:

1. **Budowa Datasetu (ETL):** Zamiana audio na macierze obliczeniowe.
   ```bash
   python -m backend.scripts.run_build_dataset
   ```
   
   **Augmentacja danych:** Możesz dodatkowo wygenerować transpozycje offline dla zwiększenia różnorodności danych. (--semitones opcjonalne)
   ```bash
   python -m backend.scripts.run_transpose_offline --semitones -6 -5 -4 -3 -2 -1 1 2 3 4 5 6
   ```

2. **Trening Modelu (Train):** Nauka sieci neuronowej na GPU.
   
   ```bash
   python -m backend.scripts.run_training
   ```

3. **Podgląd checkpointu:** Sprawdzenie zapisanych metadanych i pełnego configu treningu.
   ```bash
   python -m backend.scripts.inspect_checkpoint --checkpoint out/model.pth
   ```

   flaga `--checkpoint` aktywuje odczyt configu z metadanych, brak określenia konkretnego modelu .pth otwiera menu wyboru plikow we wskazanym folderze.

4. **Predykcja (Inference):** Analiza akordów w nowym pliku.
   ```bash
   python -m backend.scripts.run_predict
   ```

5. **Diagnostyka DSP:** Interaktywne wykresy i testy filtrów.
   ```bash
   python -m backend.scripts.run_single_tests
   ```
   
5. **Aplikacja Okienkowa (UI):** Narzędzie graficzne z odtwarzaczem do wizualizacji akordów "na żywo".
   ```bash
   python -m frontend.MainWindow
   ```

## Skrypty Uruchomieniowe

Wszystkie poniższe komendy uruchamiaj z głównego folderu projektu.

### Przygotowanie i trening
```bash
python -m backend.scripts.run_build_dataset
python -m backend.scripts.run_transpose_offline --semitones -6 -5 -4 -3 -2 -1 1 2 3 4 5 6
python -m backend.scripts.run_training
```

### Podgląd checkpointów i testy modelu
```bash
python -m backend.scripts.inspect_checkpoint --checkpoint out/model.pth
python -m backend.scripts.run_test_model --checkpoint out/model.pth
```

### Inferencja i diagnostyka
```bash
python -m backend.scripts.run_predict
python -m backend.scripts.run_single_tests
python -m backend.scripts.run_music_metrics
```

## 📁 Struktura Projektu
* **backend/config.py**: Centralna konfiguracja projektu, w tym parametry audio, modelu, treningu i analizy.
* **backend/logger/**: Wspólny logger używany przez cały backend.
* **backend/data/**: ETL, parser etykiet i DataLoadery dla zbioru treningowego.
* **backend/dsp/**: Przetwarzanie sygnału audio i wizualizacja cech.
* **backend/models/**: Architektura sieci CRNN.
* **backend/training/**: Funkcje straty, pętla treningowa i checkpointing.
* **backend/metrics/**: Ewaluacja modelu i generowanie wykresów metryk.
* **backend/analysis/**: Analiza muzyczna i raporty jakości danych/modeli.
* **backend/api/**: Inferencja i ładowanie checkpointów.
* **backend/scripts/**: Skrypty uruchomieniowe do treningu, testów, analizy i podglądu checkpointów.
* **dataset/**: Folder na Twoje piosenki do nauki (.mp3 + .jams).
* **out/**: Pliki wynikowe (wagi .pth i dane .npy).
* **frontend/**: Aplikacja okienkowa do wizualizacji działania modelu
