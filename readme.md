# AI Chord Analysis

Projekt do rozpoznawania akordów z audio na podstawie spektrogramu CQT i modelu CRNN.

Repozytorium obejmuje cały pipeline:

1. przygotowanie danych (audio + etykiety),
2. budowę datasetu treningowego (`.npy`),
3. trening modelu,
4. predykcję akordów dla nowej piosenki.

---

## 1. Wymagania

### System

- Python 3.10+ (zalecane 3.10 lub 3.11)
- FFmpeg dostępny w systemowym `PATH`
- Windows / Linux / macOS

### Biblioteki Python (z `requirements.txt`)

- `numpy` - obliczenia numeryczne i macierze
- `matplotlib` - wizualizacje CQT/chroma
- `tqdm` - paski postępu
- `torch` - model CRNN i trening
- `torchvision`, `torchaudio` - zależności ekosystemu PyTorch
- `scikit-learn` - podział danych train/val/test
- `librosa` - wsparcie audio (przydatne narzędzia DSP)

---

## 2. Instalacja krok po kroku

### 2.1. Wejście do katalogu projektu

```powershell
cd ../Ai-Chord-Analysis
```

### 2.2. Utworzenie środowiska virtualenv

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3. Instalacja pakietów

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.4. Sprawdzenie FFmpeg

```bash
ffmpeg -version
```

Jeśli komenda nie działa, doinstaluj FFmpeg i dodaj jego folder `bin` do `PATH`.

---

## 3. Jak przygotować dane Isophonics

### 3.1. Struktura folderów

Każdy utwór powinien być w osobnym folderze:

```text
isophonics_dataset/
	isophonics_0/
		isophonics_0.mp3   (lub .wav)
		isophonics_0.jams  (lub .csv/.txt)
	isophonics_1/
		isophonics_1.mp3
		isophonics_1.jams
	...
```

Wymagania minimalne na podfolder:

- dokładnie co najmniej 1 plik audio (`.mp3` lub `.wav`),
- co najmniej 1 plik etykiet (`.jams`; parser obsługuje też `.csv` i `.txt`).

Skrypt `dataset_builder.py` skanuje wszystkie podfoldery i pomija te, w których brakuje audio lub etykiet.

### 3.2. Obsługiwane formaty etykiet

- `.jams` - preferowany (namespace `chord`)
- `.csv` / `.txt` - format wiersza: `start_time, end_time, chord_label`

Przykład CSV/TXT:

```text
0.00,1.25,C:maj
1.25,2.80,G:maj
2.80,4.10,A:min
4.10,5.00,N
```

---

## 4. Co robi parser etykiet (`labels_parser.py`)

Parser:

1. rozpoznaje format (`.jams`, `.csv`, `.txt`),
2. czyta przedziały czasu akordów,
3. upraszcza nazwy akordów do wspólnego słownika 25 klas:
	 - 12 durowych,
	 - 12 molowych,
	 - `N` (brak akordu).

Reguły upraszczania:

- enharmonie są normalizowane (np. `Db -> C#`, `Bb -> A#`),
- inwersje są usuwane (`C:maj/5 -> C:maj`),
- jakości typu `maj7`, `sus`, `aug` traktowane są jako major,
- `min`/`m`/`dim` mapowane są do minor,
- `N`, `X`, `Z`, puste wartości -> `N`.

---

## 5. Przygotowanie datasetu

### 5.1. Przetworzenie jednego utworu (test pipeline)

Uruchom:

```bash
python main.py
```

Domyślne wejście:

- `single_test_data/isophonics_0/isophonics_0.mp3`
- `single_test_data/isophonics_0/isophonics_0.jams`

Wyjście:

- `out/dataset_output/isophonics_0_X.npy`
- `out/dataset_output/isophonics_0_y.npy`
- `out/dataset_output/isophonics_0_cqt_check.png`

### 5.2. Budowa pełnego datasetu

Uruchom:

```bash
python dataset_builder.py
```

Domyślne ścieżki:

- wejście: `isophonics_dataset/`
- wyjście: `out/full_dataset/`

Dla każdego utworu powstaną:

- `<song_folder>_X.npy`
- `<song_folder>_y.npy`

### 5.3. Parametry ważne dla datasetu

- `hop_size_ms=50` - krok czasowy ramek CQT (~20 fps)
- `seq_len=40` - długość okna modelu (~2.0 s)
- `hop_seq=10` - przesuw okna przy tworzeniu sekwencji

---

## 6. Pojedynczy test spektrogramu (`single_tests.py`)

Uruchom:

```bash
python single_tests.py
```

Do czego służy:

- szybki podgląd CQT/chromagramu,
- porównanie wpływu filtrów (`smoothing`, `whitening`, `denoise`),
- ręczna weryfikacja jakości cech przed treningiem.

Uwaga: w pliku jest hardcodowana ścieżka audio (`file_path`).
Przed uruchomieniem ustaw ją na lokalizację Twojego pliku.

---

## 7. Trening modelu (`model/train.py`)

### 7.1. Co robi trening

Skrypt:

1. wczytuje wszystkie pary `*_X.npy` i `*_y.npy`,
2. dzieli dane na zbiory train/val (na poziomie plików-utworów),
3. buduje model `ChordCRNN`,
4. trenuje z `CrossEntropyLoss` i wagami klas,
5. zapisuje najlepszy model do `best_crnn_model.pth`,
6. stosuje `Early Stopping`.

### 7.2. Uruchomienie

```bash
python model/train.py
```

Przed uruchomieniem sprawdź w `model/train.py` zmienną:

- `DATASET_FOLDER` - ustaw na folder z wygenerowanymi plikami `*_X.npy` i `*_y.npy`.

### 7.3. Wyjście treningu

- plik wag: `best_crnn_model.pth`
- logi epok: accuracy/loss walidacji i treningu

---

## 8. Analiza piosenki (predykcja) `model/predict.py`

### 8.1. Co robi analiza

1. ładuje wytrenowany model (`best_crnn_model.pth`),
2. generuje CQT dla podanego pliku audio,
3. tnie audio na przesuwne okna,
4. przewiduje akord dla każdego okna,
5. grupuje wyniki w czytelne bloki czasu:
	 - `[start - end] : chord`.

### 8.2. Uruchomienie

```bash
python model/predict.py
```

Przed uruchomieniem ustaw w pliku:

- `MODEL_FILE` - ścieżka do wag modelu,
- `TEST_SONG` - ścieżka do analizowanego audio (`.mp3`/`.wav`).

---

## 9. Opis formatu danych wyjściowych

### `*_X.npy`

- typ: `float32`
- kształt: `(N, seq_len, 84)`
- zawartość: sekwencje cech CQT

### `*_y.npy`

- typ: `int`
- kształt: `(N,)`
- zawartość: indeks klasy akordu

Mapowanie klas:

- `0..11` - akordy durowe,
- `12..23` - akordy molowe,
- `24` - `N`.

---

## 10. Najczęstsze problemy

### 1) `Nie znaleziono FFmpeg w systemie`

Rozwiązanie: zainstaluj FFmpeg i dodaj do `PATH`.

### 2) Puste lub małe zbiory `.npy`

Możliwe przyczyny:

- utwór jest krótszy niż wymagane okno,
- błędne etykiety czasowe,
- za agresywne filtrowanie sygnału.

### 3) Brak plików `*_X.npy` / `*_y.npy` przy treningu

Rozwiązanie:

- najpierw uruchom `python dataset_builder.py`,
- upewnij się, że `DATASET_FOLDER` wskazuje poprawny katalog.

### 4) Brak modelu przy predykcji

Rozwiązanie:

- najpierw uruchom `python model/train.py`,
- upewnij się, że `MODEL_FILE` wskazuje na istniejący `best_crnn_model.pth`.

---

## 11. Szybki workflow (skrót)

1. Skopiuj dane Isophonics do `isophonics_dataset/` według struktury z sekcji 3.
2. Zainstaluj zależności i FFmpeg.
3. Sprawdź pipeline na jednym utworze: `python main.py`.
4. Zbuduj pełny dataset: `python dataset_builder.py`.
5. Wytrenuj model: `python model/train.py`.
6. Zrób analizę piosenki: `python model/predict.py`.

---

## 12. Dodatkowe pliki pomocnicze

- `model/test_gpu.py` - szybki test wykrycia i obliczeń CUDA/GPU.
- `spectograms/plot.py` - zapis i podgląd spektrogramów/chromagramu.


