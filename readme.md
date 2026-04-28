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
> **Kluczowy krok dla posiadaczy RTX 5070:**
> Przed instalacją pozostałych bibliotek, zainstaluj dedykowaną wersję PyTorch komendą:
> ```pip install torch==2.11.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128```
> Dodatkowo wymagana jest [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).

2. **Instalacja pozostałych zależności:**
   ```bash 
   pip install -r requirements.txt
   ```

## Workflow Wykonawczy

Wszystkie komendy uruchamiaj z głównego folderu projektu:

1. **Budowa Datasetu (ETL):** Zamiana audio na macierze obliczeniowe.
   ```bash
   python -m backend.scripts.run_build_dataset
   ```
   **Augumentacja danych:** Możesz dodatkowo wygenerować transpozycje offline dla zwiększenia różnorodności danych. (--semitones opcjonalne)
   ```bash
   python -m backend.scripts.run_transpose_offline --semitones -6 -5 -4 -3 -2 -1 1 2 3 4 5 6
   ```

2. **Trening Modelu (Train):** Nauka sieci neuronowej na GPU.
   
   ```bash
   python -m backend.scripts.run_training
   ```

3. **Predykcja (Inference):** Analiza akordów w nowym pliku.
   ```bash
   python -m backend.scripts.run_predict
   ```

4. **Diagnostyka DSP:** Interaktywne wykresy i testy filtrów.
   ```bash
   python -m backend.scripts.run_single_tests
   ```

## 📁 Struktura Projektu
* **backend/config.py**: Centralna konfiguracja (LR, Batch Size, SR).
* **backend/dsp/**: Przetwarzanie sygnału i wizualizacja.
* **backend/models/**: Architektura sieci CRNN.
* **dataset/**: Folder na Twoje piosenki do nauki (.mp3 + .jams).
* **out/**: Pliki wynikowe (wagi .pth i dane .npy).