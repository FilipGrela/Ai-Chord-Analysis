import csv
import json
import os
import sys
import statistics
from pathlib import Path
from collections import defaultdict

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
    # Agregowane dane dla całego datasetu
    root_counts: dict[str, int] = {} # Ile danych C, C#, ... wystapilo
    quality_counts: dict[str, int] = {} # ile każdej jakości
    interval_distances: list[int] = [] # jakie przeskoki akordowe wystepowaly
    chord_transitions: dict[str, int] = {} # najczęstsze przejścia między akordami
    in_key_total = 0  # Ile akordów miało przypisaną (i znaną) tonację
    in_key_matches = 0  # Ile akordów faktycznie pasowało do tej tonacji
    total_segments = 0 # Całkowita liczba wierszy (akordów)
    segment_durations: list[float] = [] # Długości segmentów
    
    # Dane per-plik
    file_stats: dict[str, dict] = {} # Statystyki dla każdego pliku
    processed_tracks = 0 # Liczba piosenek
    
    # Patrzy na folder ze wszystkimi piosenkami i zbiera .csv oraz .jams
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
        
        # Inicjalizuj statystyki dla tego pliku
        file_name = folder.name
        file_stats[file_name] = {
            "segment_count": 0,
            "in_key_count": 0,
            "in_key_total": 0,
            "root_counts": {},
            "quality_counts": {},
            "interval_distances": [],
            "segment_durations": [],
            "chords": [],  # lista wszystkich akordów w pliku
            "keys": [],    # lista wszystkich tonacji w pliku
        }

        # Dla kazdego akordu zliczamy odpowiednie statystyki
        prev_chord = None
        for start, end, chord in segments:
            total_segments += 1
            duration = end - start
            segment_durations.append(duration)
            file_stats[file_name]["segment_durations"].append(duration)
            file_stats[file_name]["segment_count"] += 1
            file_stats[file_name]["chords"].append(chord)
            
            root = chord_root(chord) or "N"
            quality = chord_quality(chord)
            root_counts[root] = root_counts.get(root, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            file_stats[file_name]["root_counts"][root] = file_stats[file_name]["root_counts"].get(root, 0) + 1
            file_stats[file_name]["quality_counts"][quality] = file_stats[file_name]["quality_counts"].get(quality, 0) + 1

            # sprawdzanie tonacji
            key = _match_key_for_time(start, key_segments, default_key)
            if key is not None:
                in_key_total += 1
                file_stats[file_name]["in_key_total"] += 1
                file_stats[file_name]["keys"].append(key)
                if in_key(chord, key):
                    in_key_matches += 1
                    file_stats[file_name]["in_key_count"] += 1

            # mierzenie interwalu poprzedniego akordu
            if prev_chord is not None:
                distance = interval_distance(prev_chord, chord)
                if distance is not None:
                    interval_distances.append(distance)
                    file_stats[file_name]["interval_distances"].append(distance)
                
                # Zliczaj przejścia między akordami
                transition = f"{prev_chord} -> {chord}"
                chord_transitions[transition] = chord_transitions.get(transition, 0) + 1
            
            prev_chord = chord

        processed_tracks += 1


    # Obliczanie statystyk
    mean_distance = (
        sum(interval_distances) / len(interval_distances)
        if interval_distances
        else 0.0
    )
    
    std_distance = (
        statistics.stdev(interval_distances)
        if len(interval_distances) > 1
        else 0.0
    )

    # Ile akordow bylo w tonacji
    in_key_ratio = (
        in_key_matches / in_key_total
        if in_key_total > 0
        else 0.0
    )
    
    # Średnia długość segmentu
    mean_segment_duration = (
        sum(segment_durations) / len(segment_durations)
        if segment_durations
        else 0.0
    )

    # Drukowanie rezultatów
    print("=" * 60)
    print("╔ ANALIZA MUZYCZNA DATASETU ╗")
    print("=" * 60)
    
    print(f"\nOGÓLNE STATYSTYKI:")
    print(f"  Plików przetworzonych: {processed_tracks}")
    print(f"  Razem segmentów: {total_segments}")
    print(f"  Średnia długość segmentu: {mean_segment_duration:.2f}s")
    print(f"  Razem czasu muzyki: {sum(segment_durations):.1f}s")
    
    print(f"\nTONACJA:")
    if in_key_total > 0:
        print(f"  Akordy w tonacji: {in_key_ratio:.1%} ({in_key_matches}/{in_key_total})")
    else:
        print(f"  Akordy w tonacji: brak danych (no key annotations)")
    
    print(f"\nPRZESKOKI AKORDOWE (INTERVAL DISTANCES):")
    print(f"  Średnia odległość: {mean_distance:.3f} półtonów")
    print(f"  Odch. std: {std_distance:.3f}")
    if interval_distances:
        print(f"  Min: {min(interval_distances)}, Max: {max(interval_distances)}")
    
    print(f"\nROZKŁAD ROOTÓW (TOP 15):")
    for root, count in sorted(root_counts.items(), key=lambda item: item[1], reverse=True)[:15]:
        pct = 100 * count / total_segments if total_segments > 0 else 0
        print(f"  {root:3s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nROZKŁAD JAKOŚCI AKORDÓW (CHORD QUALITY DISTRIBUTION):")
    for quality, count in sorted(quality_counts.items(), key=lambda item: item[1], reverse=True):
        pct = 100 * count / total_segments if total_segments > 0 else 0
        print(f"  {quality:6s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nTOP 10 PRZEJŚĆ MIĘDZY AKORDAMI:")
    sorted_transitions = sorted(chord_transitions.items(), key=lambda item: item[1], reverse=True)[:10]
    for idx, (transition, count) in enumerate(sorted_transitions, 1):
        pct = 100 * count / (total_segments - processed_tracks) if total_segments > processed_tracks else 0
        print(f"  {idx:2d}. {transition:30s} x{count:3d} ({pct:5.1f}%)")
    
    # Wypisz statystyki per-plik jeśli mamy nie za dużo plików
    if processed_tracks <= 20:
        print(f"\nSTATYSTYKI PER-PLIK:")
        print("-" * 60)
        for file_name in sorted(file_stats.keys()):
            stats = file_stats[file_name]
            file_in_key_ratio = (
                stats["in_key_count"] / stats["in_key_total"]
                if stats["in_key_total"] > 0
                else 0.0
            )
            file_mean_distance = (
                sum(stats["interval_distances"]) / len(stats["interval_distances"])
                if stats["interval_distances"]
                else 0.0
            )
            top_quality = max(stats["quality_counts"].items(), key=lambda x: x[1])[0] if stats["quality_counts"] else "N/A"
            top_root = max(stats["root_counts"].items(), key=lambda x: x[1])[0] if stats["root_counts"] else "N/A"
            
            print(f"  {file_name}:")
            print(f"    - Segmenty: {stats['segment_count']}")
            print(f"    - Top root: {top_root}, Top quality: {top_quality}")
            if stats["in_key_total"] > 0:
                print(f"    - In-key: {file_in_key_ratio:.1%} ({stats['in_key_count']}/{stats['in_key_total']})")
            print(f"    - Mean interval: {file_mean_distance:.2f}")
    
    print("\n" + "=" * 60)


def main():
    data_dir = Path(cfg_analysis.MUSIC_METRICS_DATA_DIR)
    default_key = cfg_analysis.MUSIC_METRICS_DEFAULT_KEY
    limit = cfg_analysis.DATASET_SONGS

    _process_dataset(data_dir, default_key, limit)


if __name__ == "__main__":
    main()
