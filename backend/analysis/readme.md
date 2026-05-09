# ANALIZA: METRYKI MUZYCZNE I DSP

Ten folder zbiera narzędzia analityczne dla projektu.

## Zawartość

### `music_metrics.py`
Moduł z podstawowymi funkcjami muzycznymi i metrykami jakości akordu:
- `parse_chord()` - parsuje root i **szerszą paletę jakości** (np. `maj`, `m`, `7`, `maj7`, `m7`, `sus2`, `sus4`, `aug`, `dim`, `other`)
- `parse_key()` - parsowanie tonacji (major/minor)
- `in_key()` - sprawdzenie, czy root akordu należy do skali danej tonacji
- `interval_distance()` - minimalna odległość między rootami w półtonach
- `chord_similarity()` - diagnostyczny score uwzględniający dopasowanie rootu, jakość i zgodność z tonacją (wagi konfigurowalne)

Uwaga: parser obsługuje bardziej złożone zapisy (np. `C:maj7`, `Am7`, `Csus4`, `D#dim`) i klasyfikuje niewspierane lub bardzo złożone typy jako `other`.

## Konfiguracja

Domyślne ustawienia dla analizy znajdują się w [backend/config.py](../config.py):
- `cfg_analysis.MUSIC_METRICS_DATA_DIR`
- `cfg_analysis.MUSIC_METRICS_DEFAULT_KEY`
- `cfg_analysis.DATASET_SONGS`
- `cfg_analysis.CHORD_SIMILARITY_ROOT_WEIGHT`
- `cfg_analysis.CHORD_SIMILARITY_QUALITY_WEIGHT`
- `cfg_analysis.CHORD_SIMILARITY_KEY_WEIGHT`

## Jak używać

Najprostszy test uruchomieniowy:
```powershell
python backend/scripts/run_music_metrics.py
```

Skrypt odczytuje dane z `cfg_analysis` i przetwarza folder datasetu w prostym raporcie tekstowym.
