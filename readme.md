# AI Chord Analysis

Projekt do analizy akordów z plików audio: generowanie spektrogramów CQT, parsowanie etykiet akordów, synchronizacja ramek z etykietami i budowa datasetu sekwencji do trenowania modeli (np. CRNN).

## 1. Co robi projekt

Pipeline projektu:

1. Wczytanie audio (`.mp3`/`.wav`) przez `ffmpeg`.
2. Obliczenie spektrogramu (domyślnie CQT).
3. Opcjonalne przetwarzanie: odszumianie, usuwanie krótkich szumów, whitening, smoothing.
4. Parsowanie etykiet akordów (`.jams`, `.csv`, `.txt`) i uproszczenie do słownika:
	 - 12 akordów durowych,
	 - 12 akordów molowych,
	 - `N` (brak akordu).
5. Wyrównanie ramek czasowych do etykiet.
6. Cięcie na sekwencje treningowe i zapis do plików `.npy`.

## 2. Wymagania systemowe

- Python 3.10+ (zalecane 3.10 lub 3.11)
- `ffmpeg` dostępny w systemowym `PATH`
- System: Windows/Linux/macOS

## 3. Instalacja krok po kroku

### 3.1. Wejście do katalogu projektu

```powershell
cd D:\SI_Studia\Ai-Chord-Analysis
```

### 3.2. Utworzenie i aktywacja środowiska wirtualnego

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3. Instalacja bibliotek Pythona

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4. Instalacja `ffmpeg`

`ffmpeg` jest wymagany do działania funkcji `read_audio_universal`.

Sprawdzenie instalacji:

```bash
ffmpeg -version
```

Jeśli komenda nie działa, doinstaluj `ffmpeg` i dodaj go do `PATH`.

## 4. Opis pakietów (requirements.txt)

### `numpy`
- Operacje numeryczne na macierzach i wektorach.
- Używany m.in. przy obliczeniach CQT, normalizacji i budowie sekwencji.

### `matplotlib`
- Wizualizacja spektrogramów/chromagramów.
- Używany przez moduł `spectograms/plot.py` do podglądu i zapisu obrazów.

### `tqdm`
- Paski postępu dla dłuższych pętli (analiza ramek, whitening, smoothing).

### `torch`
- Framework deep learning (PyTorch).
- W aktualnym kodzie dataset builder nie trenuje modelu, ale pakiet jest przygotowany pod kolejne etapy trenowania.

### `scikit-learn`
- Narzędzia ML i preprocessing.
- W aktualnym kodzie nie jest centralny dla pipeline’u ekstrakcji, ale przydatny do podziału danych i ewaluacji.

## 5. Struktura kluczowych plików

- `main.py`:
	- Przetwarzanie pojedynczego utworu.
	- Tworzy CQT, etykiety ramek, sekwencje i zapisuje `.npy`.

- `dataset_builder.py`:
	- Przetwarzanie całego katalogu datasetu (wiele utworów).
	- Dla każdego podfolderu zapisuje osobne pliki `{nazwa}_X.npy` i `{nazwa}_y.npy`.

- `labels_parser.py`:
	- Parser etykiet (`.jams`, `.csv`, `.txt`).
	- Upraszcza akordy do słownika klasyfikacji.

- `spectograms/spectograms.py`:
	- Odczyt audio przez `ffmpeg`.
	- Obliczanie CQT/RFFT i kroki przetwarzania sygnału.

- `spectograms/plot.py`:
	- Rysowanie i zapis obrazów spektrogramu/chromagramu.

- `single_tests.py`:
	- Testy manualne i szybki podgląd wpływu parametrów.

## 6. Format wejścia danych

Przykładowy układ dla jednego utworu:

```text
isophonics_X/
	isophonics_X.mp3   (lub .wav)
	isophonics_X.jams  (lub labels.csv/.txt)
```

Dla pełnego datasetu:

```text
isophonics_dataset/
	isophonics_0/
		isophonics_0.mp3
		isophonics_0.jams
	isophonics_1/
		...
```

## 7. Przykłady użycia

### 7.1. Przetworzenie jednego utworu

Uruchomienie:

```bash
python main.py
```

Domyślnie skrypt używa:

- audio: `single_test_data/isophonics_0/isophonics_0.mp3`
- etykiety: `single_test_data/isophonics_0/isophonics_0.jams`
- output: `out/dataset_output`

Wynik:

- `out/dataset_output/isophonics_0_X.npy`
- `out/dataset_output/isophonics_0_y.npy`
- `out/dataset_output/isophonics_0_cqt_check.png`

### 7.2. Budowa całego datasetu

Uruchomienie:

```bash
python dataset_builder.py
```

Domyślnie:

- wejście: `isophonics_dataset`
- wyjście: `out/full_dataset`
- parametry: `hop_size_ms=50`, `seq_len=40`

Dla każdego utworu powstaną pliki:

- `out/full_dataset/<folder>_X.npy`
- `out/full_dataset/<folder>_y.npy`

### 7.3. Szybki test wizualny spektrogramu

Uruchomienie:

```bash
python single_tests.py
```

Uwaga: w `single_tests.py` jest aktualnie hardcodowana ścieżka `D:\isophonics_dataset\...`.
Jeśli dane są gdzie indziej, zmień `file_path` przed uruchomieniem.

## 8. Opis najważniejszych parametrów

- `hop_size_ms` (np. `50`):
	- Krok czasowy między ramkami CQT.
	- `50 ms` oznacza około `20` ramek/s.

- `seq_len` (np. `40`):
	- Długość sekwencji wejściowej dla modelu.
	- `40` ramek przy `50 ms` to około `2 s` kontekstu.

- `hop_seq` (w `create_sequences`, domyślnie `10`):
	- O ile ramek przesuwane jest okno sekwencji.
	- Mniejsza wartość = większe nakładanie okien = więcej przykładów.

- `apply_denoise`, `apply_short_noises`, `apply_whitening`, `apply_smoothing`:
	- Przełączniki kolejnych etapów obróbki sygnału.

## 9. Format danych wyjściowych

### `*_X.npy`
- Typ: `numpy.ndarray`
- Kształt: `(liczba_sekwencji, seq_len, liczba_cech)`
- Dla CQT zwykle `liczba_cech = 84` (84 biny częstotliwości).

### `*_y.npy`
- Typ: `numpy.ndarray`
- Kształt: `(liczba_sekwencji,)`
- Każda wartość to klasa akordu jako liczba całkowita.

## 10. Mapowanie klas akordów

Słownik etykiet zawiera:

- durowe: `C, C#, D, ..., B`
- molowe: `Cm, C#m, Dm, ..., Bm`
- `N` (no chord)

Mapowanie realizowane jest przez `CHORD_TO_INT` i `INT_TO_CHORD` w `dataset_builder.py`.

## 11. Typowy workflow end-to-end

1. Przygotuj dane audio + etykiety (`.jams`/`.csv`/`.txt`).
2. Upewnij się, że działa `ffmpeg -version`.
3. Zainstaluj zależności (`pip install -r requirements.txt`).
4. Dla szybkiej walidacji uruchom `python main.py` na jednym utworze.
5. Sprawdź wygenerowany obraz CQT (`*_cqt_check.png`).
6. Uruchom `python dataset_builder.py` dla całego datasetu.
7. Załaduj pliki `.npy` w kodzie treningowym modelu.

## 12. Najczęstsze problemy i rozwiązania

### Problem: `Nie znaleziono FFmpeg w systemie`
- Przyczyna: brak `ffmpeg` w `PATH`.
- Rozwiązanie: doinstaluj `ffmpeg` i dodaj katalog binarny do zmiennej środowiskowej `PATH`.

### Problem: `Brakuje pliku audio lub etykiet`
- Przyczyna: w podfolderze datasetu brakuje `.mp3/.wav` albo `.jams`.
- Rozwiązanie: sprawdź strukturę folderu i nazwy plików.

### Problem: puste/małe wyjście datasetu
- Przyczyna: zbyt krótki utwór względem `seq_len`, błędy etykiet, lub agresywne filtrowanie.
- Rozwiązanie:
	- zmniejsz `seq_len`,
	- sprawdź poprawność etykiet,
	- przetestuj ustawienia `apply_*`.

### Problem: parser nie rozpoznaje akordów
- Przyczyna: nietypowy format etykiet.
- Rozwiązanie: sprawdź funkcję `simplify_chord` w `labels_parser.py` i rozszerz reguły.

## 13. Szybki start (skrót)

```bash
pip install -r requirements.txt
python main.py
python dataset_builder.py
```

Po tych krokach otrzymasz gotowe pliki `.npy` do treningu i obrazy kontrolne CQT.
