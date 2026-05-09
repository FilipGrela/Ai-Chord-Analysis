import csv
import json
import os
import sys
from pathlib import Path

# Pozwala uruchamiać skrypt jako plik: `python backend/scripts/run_music_metrics_demo.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.analysis.music_metrics import (
    chord_quality,
    chord_root,
    chord_similarity,
    in_key,
    interval_distance,
    parse_chord,
    parse_key,
)
from backend.config import cfg_analysis

# Bierze CSV i wyciaga timestampy oraz akord (wzorowalem sie .csv z isophonics_dataset)
def _load_labels_csv(csv_path: Path) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start = float(row.get("start_sec", ""))
                end = float(row.get("end_sec", ""))
                chord = str(row.get("chord_label", "")).strip()
            except ValueError:
                continue
            if chord:
                segments.append((start, end, chord))
    return segments



# Funkcja wczytujaca pliki .jams - tez moze byc przydatne

def _load_key_segments(jams_path: Path) -> list[tuple[float, float, str]]:
    try:
        with jams_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    key_segments: list[tuple[float, float, str]] = []
    for annotation in data.get("annotations", []):
        namespace = str(annotation.get("namespace", "")).lower()
        if "key" not in namespace:
            continue
        for obs in annotation.get("data", []):
            value = obs.get("value") or obs.get("label") or obs.get("text")
            if value is None:
                continue
            time = float(obs.get("time", 0.0))
            duration = float(obs.get("duration", 0.0))
            key_segments.append((time, time + duration, str(value)))
    key_segments.sort(key=lambda item: item[0])
    return key_segments

# Jaka tonacja wystepuje w danym momencie (na podstw. pliku .jams)
def _match_key_for_time(time_sec: float, key_segments: list[tuple[float, float, str]], default_key: str | None) -> str | None:
    if not key_segments:
        return default_key
    for start, end, key in key_segments:
        if start <= time_sec < end:
            return key
    return default_key


def _process_dataset(data_dir: Path, default_key: str | None, limit: int | None) -> None:
    # Cale dane do zbierania
    root_counts: dict[str, int] = {} # Ile danych C, C#, ... wystapilo
    quality_counts: dict[str, int] = {} # ile mor/dur
    interval_distances: list[int] = [] # jakie przeskoki akordowe wystepowaly
    in_key_total = 0  # Ile akordów miało w ogóle przypisaną (i znaną) tonację
    in_key_matches = 0  # Ile akordów faktycznie pasowało muzycznie do tej tonacji
    total_segments = 0 # Całkowita, bezwzględna liczba wierszy (akordów) przeczytanych w ogóle
    processed_tracks = 0 # Liczba piosenek, przez które z sukcesem przeszedł skrypt

    # patrzy na folder ze wszystkimi piosenkami i zbiera .csv oraz .jams
    folders = [path for path in data_dir.iterdir() if path.is_dir()]
    for folder in sorted(folders):
        if limit is not None and processed_tracks >= limit:
            break
        
        # Szukamy .csv oraz .jams
        labels_path = folder / "labels.csv"
        jams_path = next(iter(folder.glob("*.jams")), None)
        if not labels_path.exists():
            continue

        # jak znajdzie to laduje je do pamieci
        segments = _load_labels_csv(labels_path)
        if not segments:
            continue

        key_segments = _load_key_segments(jams_path) if jams_path else []

        # Dla kazdego akordu zliczamy odpowiednie statystyki
        prev_chord = None
        for start, end, chord in segments:
            total_segments += 1
            root = chord_root(chord) or "N"
            quality = chord_quality(chord)
            root_counts[root] = root_counts.get(root, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

            # sprawdzanie tonacji
            key = _match_key_for_time(start, key_segments, default_key)
            if key is not None:
                in_key_total += 1
                if in_key(chord, key):
                    in_key_matches += 1

            # mierzenie interwalu poprzedniego akordu
            if prev_chord is not None:
                distance = interval_distance(prev_chord, chord)
                if distance is not None:
                    interval_distances.append(distance)
            prev_chord = chord

        processed_tracks += 1


    # podsumowanie wynikow

    # sredni skok interwalowy akordow
    mean_distance = (
        sum(interval_distances) / len(interval_distances)
        if interval_distances
        else 0.0
    )

    # Ile akordow bylo w tonacji
    in_key_ratio = (
        in_key_matches / in_key_total
        if in_key_total > 0
        else 0.0
    )

    print("=== Dataset music metrics ===")
    print(f"Tracks processed: {processed_tracks}")
    print(f"Total segments: {total_segments}")
    if in_key_total > 0:
        print(f"In-key ratio: {in_key_ratio:.3f} ({in_key_matches}/{in_key_total})")
    else:
        print("In-key ratio: n/a (no key annotations detected)")
    print(f"Mean interval distance: {mean_distance:.3f}")

    print("\nTop roots:")
    for root, count in sorted(root_counts.items(), key=lambda item: item[1], reverse=True)[:12]:
        print(f"  {root}: {count}")

    print("\nChord quality distribution:")
    for quality, count in sorted(quality_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {quality}: {count}")


def main():
    data_dir = Path(cfg_analysis.MUSIC_METRICS_DATA_DIR)
    default_key = cfg_analysis.MUSIC_METRICS_DEFAULT_KEY
    limit = cfg_analysis.MUSIC_METRICS_LIMIT

    print("=== Music metrics config ===")
    print(f"Data dir: {data_dir}")
    print(f"Default key: {default_key}")
    print(f"Limit: {limit}")
    print()

    _process_dataset(data_dir, default_key, limit)


if __name__ == "__main__":
    main()
