# Augmentacja przez Transpozycję - Dokumentacja

## Przegląd

Implementacja dwóch rodzajów transpozycji:
1. **Online Transposition** - stosowana podczas treningu, na gorąco
2. **Offline Transposition** - generuje augmentowane wielokrotnie dane przed treningiem

## Komponenty

### 1. `backend/data/augument/label_ops.py` - Transpozycja Etykiet

Klasa `ChordTranspose` transpozycjonuje etykiety akordów:

```python
from backend.data.augument.label_ops import ChordTranspose

# Transpozycja pojedynczego akordu
ChordTranspose.transpose_chord_label('C', 1)      # -> 'C#'
ChordTranspose.transpose_chord_label('Am', -2)    # -> 'G#m'
ChordTranspose.transpose_chord_label('N', 5)      # -> 'N' (cisza nie zmienia się)
```

### 2. `backend/data/augument/transforms.py` - RandomTranspose

Klasa `RandomTranspose` do online augmentacji spektrogramów CQT:

```python
from backend.data.augument.transforms import RandomTranspose
import numpy as np

# Inicjalizacja
transpose_aug = RandomTranspose(
    prob=0.3,              # Prawdopodobieństwo aplikacji
    min_semitones=-6,      # Min półtony
    max_semitones=6        # Max półtony
)

# Aplikacja na spektrogram
spec = np.random.randn(50, 84)  # (time_frames, frequency_bins)
spec_transposed = transpose_aug.apply(spec)
```

### 3. `backend/config.py` - Parametry Konfiguracji

Nowe parametry w `TrainConfig`:

```python
AUGMENT_TRANSPOSE_ENABLED: bool = False       # Włącz/wyłącz
AUGMENT_TRANSPOSE_PROB: float = 0.0           # Szansa na aplikację (0-1)
AUGMENT_TRANSPOSE_MIN: int = -6               # Min półtonów
AUGMENT_TRANSPOSE_MAX: int = 6                # Max półtonów
```

### 4. `backend/data/builder.py` - Offline Transpozycja

Metody do offline augmentacji już wygenerowanego datasetu:

```python
from backend.data.builder import DatasetBuilder

builder = DatasetBuilder()

# Aplikuj offline transpozycję
builder.apply_offline_transposition(
    semitones_list=[-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
)
```

## Użycie

### Online Augmentacja (podczas treningu)

1. Włącz w `backend/config.py`:
```python
AUGMENT_TRANSPOSE_ENABLED: bool = True
AUGMENT_TRANSPOSE_PROB: float = 0.3        # 30% szans
AUGMENT_TRANSPOSE_MIN: int = -6
AUGMENT_TRANSPOSE_MAX: int = 6
```

2. Pipeline automatycznie włączy `RandomTranspose` w augmentacji

### Offline Augmentacja (przed treningiem)

Uruchom skrypt po wygenerowaniu datasetu:

```bash
python backend/scripts/run_transpose_offline.py --semitones -2 -1 1 2 3 4 5 6
```

Możliwe flagi:
- `--semitones` - lista półtonów do wygenerowania (domyślnie: -6 do +6)

**Wynik:**
- Originalne pliki: `track_X.npy`, `track_y.npy`
- Augmentowane: `track_T-2_X.npy`, `track_T-2_y.npy`, etc.

## Mechanika

### Transpozycja Spektrogramu CQT

```
Spektrogram CQT: (num_bins=84, num_frames)
                    ↓
Przesunięcie o N półtonów = przesunięcie binów o N pozycji
(każdy bin = 1 półton)
                    ↓
Wypełnienie zerami margines (unikana wraparound na krawędziach)
```

### Transpozycja Etykiet Akordów

```
C (indeks 0) + 1 półton → C# (indeks 1)
Am (minor)  + 2 półtony → Bm (minor zachowany)
N (cisza)   + N półtonów → N (niezmienione)
```

## Przykłady

### Test transpozycji

```python
# Zapatrz się u konsoli
python -c "from backend.data.augument.label_ops import ChordTranspose; \
print(ChordTranspose.transpose_chord_label('C', 1))"
# Wynik: C#
```

### Generowanie augmentacji offline

```bash
# Generuj transpozycje dla -6 do +6 półtonów
python backend/scripts/run_transpose_offline.py --semitones -6 -5 -4 -3 -2 -1 1 2 3 4 5 6
```

## Ograniczenia & Uwagi

1. **Zawijanie modulo 12**: Transpozycja większa niż ±12 półtonów wraca do zakresu -11...+11 (jedna oktawa)
2. **Dane marginalne**: Przesunięte biny są zerowane, aby uniknąć wraparound
3. **Offline vs Online**: 
   - Online=szybka, mniejsza spotrzeba RAM
   - Offline=więcej danych, lepsza augmentacja, większe pliki na dysku

## Integracja z Treningiem

Offline augmentacja zwiększa rozmiar datasetu (np. z 53 tworów x12 wariantów transpozycji = 636 par plików).

Podczas treningu DataLoader może losowo wybierać zarówno oryginalne, jak i augmentowane wersje.