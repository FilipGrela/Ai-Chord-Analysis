import csv
import json
import os
import sys
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np

# Pozwala uruchamiać skrypt jako plik: `python backend/scripts/run_music_metrics_demo.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.analysis.music_metrics import (
    chord_quality,
    canonical_chord_report_label,
    canonical_transition_label,
    canonical_note_name,
    in_key,
    interval_distance,
    parse_chord,
    parse_key,
)
from backend.config import cfg_analysis
from backend.analysis.visualization.segment_analysis import (
    build_transition_heatmap_matrices,
    generate_segment_duration_graph,
    generate_top_class_duration_barchart,
    generate_transition_heatmap,
)

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


def _load_chord_segments_jams(jams_path: Path) -> list[tuple[float, float, str]]:
    try:
        with jams_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    chord_segments: list[tuple[float, float, str]] = []
    for annotation in data.get("annotations", []):
        namespace = str(annotation.get("namespace", "")).lower()
        if "chord" not in namespace:
            continue
        for obs in annotation.get("data", []):
            value = obs.get("value") or obs.get("label") or obs.get("text")
            if value is None:
                continue
            try:
                time = float(obs.get("time", 0.0))
                duration = float(obs.get("duration", 0.0))
            except (TypeError, ValueError):
                continue
            chord_segments.append((time, time + duration, str(value)))

    chord_segments.sort(key=lambda item: item[0])
    return chord_segments

# Jaka tonacja wystepuje w danym momencie (na podstw. pliku .jams)
# UWAGA - jesli brak podanej tonacji w pliku .jams bierze jak z configa
def _match_key_for_time(time_sec: float, key_segments: list[tuple[float, float, str]], default_key: str | None) -> str | None:
    if not key_segments:
        return default_key
    for start, end, key in key_segments:
        if start <= time_sec < end:
            return key
    return default_key


def _is_song_folder(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    if (folder / "labels.csv").exists() or (folder / "annotation.jams").exists():
        return True
    return any(folder.glob("*.jams"))


def _discover_song_folders(data_dir: Path) -> list[Path]:
    folders: set[Path] = set()

    # Wspieramy zarówno stare płaskie drzewa, jak i nowe korpusy zagnieżdżone.
    for candidate in data_dir.rglob("labels.csv"):
        folders.add(candidate.parent)
    for candidate in data_dir.rglob("annotation.jams"):
        folders.add(candidate.parent)
    for candidate in data_dir.rglob("*.jams"):
        folders.add(candidate.parent)

    if _is_song_folder(data_dir):
        folders.add(data_dir)

    return sorted(folders)


def _export_html_report(
    processed_tracks: int,
    total_segments: int,
    mean_segment_duration: float,
    total_time_s: float,
    in_key_ratio: float,
    mean_interval: float,
    std_interval: float,
    root_durations: dict[str, float],
    quality_durations: dict[str, float],
    key_root_durations: dict[str, dict[str, float]],
    key_root_quality_durations: dict[str, dict[str, dict[str, float]]],
    transition_counts: dict[str, int],
    transition_durations: dict[str, float],
    no_chord_count: int,
    no_chord_duration: float,
    segment_durations: list[float],
    chord_class_durations: dict[str, float],
) -> None:
    """Zapisuje raport analizy tylko do HTML w cfg_analysis.OUTPUT_DIR."""
    out_dir = Path(cfg_analysis.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"raportDatasetu_{ts}.html"

    with html_path.open("w", encoding="utf-8") as fh:
        fh.write('<!doctype html><html><head><meta charset="utf-8"><title>Music Metrics</title>')
        fh.write('<style>body{font-family:Arial,Helvetica,sans-serif}h2{margin-top:1em}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}</style>')
        fh.write('</head><body>')
        fh.write(f'<h1>Music Metrics report ({ts})</h1>')
        fh.write(
            '<p>Raport podsumowuje cały dataset: najpierw metryki ogólne, potem globalny rozkład rootów i typów akordów, '
            'następnie szczegóły per-tonacja (root + quality) oraz najczęstsze przejścia między akordami.</p>'
        )
        fh.write('<h2>Summary</h2>')
        fh.write('<p>Podstawowe metryki całego zbioru: liczba utworów, segmentów, in-key ratio i statystyki interwałowe.</p>')
        fh.write('<ul>')
        fh.write(f'<li><strong>processed_tracks:</strong> {processed_tracks}</li>')
        fh.write(f'<li><strong>total_segments:</strong> {total_segments}</li>')
        fh.write(f'<li><strong>mean_segment_duration:</strong> {mean_segment_duration}</li>')
        fh.write(f'<li><strong>total_time_s:</strong> {total_time_s}</li>')
        fh.write(f'<li><strong>in_key_ratio:</strong> {in_key_ratio}</li>')
        fh.write(f'<li><strong>No-Chord (N) segments:</strong> {no_chord_count} segments, {no_chord_duration:.3f}s</li>')
        fh.write(f'<li><strong>mean_interval:</strong> {mean_interval}</li>')
        fh.write(f'<li><strong>std_interval:</strong> {std_interval}</li>')
        fh.write('</ul>')

        fh.write('<h2>Segment Duration Distribution</h2>')
        fh.write('<p>Graf rozkładu długości segmentów. Pokazuje czy dataset zawiera wiele ultrakrótkich akordów (długi ogon) czy rozkład jest normalny.</p>')
        script_tag, div_tag = generate_segment_duration_graph(segment_durations)
        fh.write(script_tag)
        fh.write(div_tag)

        fh.write('<h2>Top Classes by Duration</h2>')
        fh.write('<p>Posortowany malejąco wykres słupkowy dla najczęstszych klas akordów (np. G:maj, E:min). Oś Y to łączny czas w sekundach.</p>')
        class_script, class_div = generate_top_class_duration_barchart(chord_class_durations)
        fh.write(class_script)
        fh.write(class_div)

        fh.write('<h2>Rozkład rootów (globalnie)</h2>')
        fh.write('<p>Udział rootów akordów w całym zbiorze. Procent liczony względem total_time_s.</p>')
        fh.write('<table><tr><th>Root</th><th>Duration (s)</th><th>Percent</th></tr>')
        for root, dur in sorted(root_durations.items(), key=lambda x: x[1], reverse=True):
            pct = (100.0 * dur / total_time_s) if total_time_s > 0 else 0.0
            fh.write(f'<tr><td>{root}</td><td>{dur:.3f}</td><td>{pct:.2f}%</td></tr>')
        fh.write('</table>')

        fh.write('<h2>Rozkład typów akordów (globalnie)</h2>')
        fh.write('<p>Udział jakości akordów (np. maj, m, 7, sus4) w całym zbiorze. Procent liczony względem total_time_s.</p>')
        fh.write('<table><tr><th>Quality</th><th>Duration (s)</th><th>Percent</th></tr>')
        for quality, dur in sorted(quality_durations.items(), key=lambda x: x[1], reverse=True):
            pct = (100.0 * dur / total_time_s) if total_time_s > 0 else 0.0
            fh.write(f'<tr><td>{quality}</td><td>{dur:.3f}</td><td>{pct:.2f}%</td></tr>')
        fh.write('</table>')

        fh.write('<h2>Per-key breakdowns</h2>')
        fh.write('<p>Dla każdej tonacji pokazane są rooty i ich liczności. Dodatkowo przy każdym rootcie jest rozbicie na quality.</p>')
        for key_name in sorted(key_root_durations.keys()):
            fh.write(f'<h3>Tonacja {key_name}</h3>')
            fh.write('<table><tr><th>Root</th><th>Duration (s)</th><th>Percent in key</th><th>Quality breakdown</th></tr>')
            roots = key_root_durations[key_name]
            total_in_key = sum(roots.values())
            for root, dur in sorted(roots.items(), key=lambda x: x[1], reverse=True):
                quals = key_root_quality_durations.get(key_name, {}).get(root, {})
                qhtml = '<br>'.join([
                    f'{q}: {qd:.3f}s ({(100.0 * qd / dur) if dur > 0 else 0.0:.1f}%)'
                    for q, qd in sorted(quals.items(), key=lambda x: x[1], reverse=True)
                ])
                pct_in_key = (100.0 * dur / total_in_key) if total_in_key > 0 else 0.0
                fh.write(f'<tr><td>{root}</td><td>{dur:.3f}</td><td>{pct_in_key:.2f}%</td><td>{qhtml}</td></tr>')
            fh.write('</table>')

        fh.write('<h2>Top transitions</h2>')
        fh.write('<p>Najczęstsze przejścia pomiędzy kolejnymi akordami. Procent liczony względem liczby wszystkich przejść (count).</p>')
        fh.write('<h3>Transition Matrix Heatmap</h3>')
        fh.write('<p>Heatmapa pokazuje uproszczone klasy 25x25 (12 rootów * maj/min + N). Jaśniejsze pola oznaczają większą łączną długość przejść; w hoverze widać też count i prawdopodobieństwo.</p>')
        heatmap_labels, heatmap_counts, heatmap_durations = build_transition_heatmap_matrices(
            transition_counts,
            transition_durations,
        )
        heatmap_script, heatmap_div = generate_transition_heatmap(
            heatmap_labels,
            heatmap_durations,
            heatmap_counts,
        )
        fh.write(heatmap_script)
        fh.write(heatmap_div)
        fh.write('<table><tr><th>Transition</th><th>Count</th><th>Percent</th></tr>')
        total_trans = sum(transition_counts.values()) if transition_counts else 0
        for t, cnt in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:50]:
            pct_t = (100.0 * cnt / total_trans) if total_trans > 0 else 0.0
            fh.write(f'<tr><td>{t}</td><td>{cnt}</td><td>{pct_t:.2f}%</td></tr>')
        fh.write('</table>')

        fh.write('</body></html>')


def _process_dataset(data_dir: Path, default_key: str | None, limit: int | None) -> None:
    # Agregowane dane dla całego datasetu
    root_counts: dict[str, int] = {} # Ile danych C, C#, ... wystapilo
    quality_counts: dict[str, int] = {} # ile każdej jakości
    # Duration-based aggregations (sum of segment durations in seconds)
    root_durations: dict[str, float] = {}
    quality_durations: dict[str, float] = {}
    chord_class_durations: dict[str, float] = {}
    interval_distances: list[int] = [] # jakie przeskoki akordowe wystepowaly
    chord_transitions: dict[str, int] = {} # najczęstsze przejścia między akordami
    transition_durations: dict[str, float] = {}
    no_chord_count: int = 0
    no_chord_duration: float = 0.0
    in_key_total = 0  # Ile akordów miało przypisaną (i znaną) tonację
    in_key_matches = 0  # Ile akordów faktycznie pasowało do tej tonacji
    total_segments = 0 # Całkowita liczba wierszy (akordów)
    segment_durations: list[float] = [] # Długości segmentów
    key_root_counts: dict[str, dict[str, int]] = {} # zlicza ile razy wystapil dany root (np. C, G#) wystapil w obrebie danej tonacji
    key_quality_counts: dict[str, dict[str, int]] = {} # ile razy dany typ (np. min, sus4) wystapil w obrebie danej tonacji
    key_root_durations: dict[str, dict[str, float]] = {}
    key_root_quality_durations: dict[str, dict[str, dict[str, float]]] = {}

    # Dodatkowy zagnieżdżony słownik: dla każdej tonacji -> root -> typ -> count
    # Pozwala wypisać, dla przykładowo rootu 'C', ile razy wystąpiło 'C:maj', 'C:m', etc.
    key_root_quality_counts: dict[str, dict[str, dict[str, int]]] = {}
    
    # Dane per-plik
    file_stats: dict[str, dict] = {} # Statystyki dla każdego pliku
    processed_tracks = 0 # Liczba piosenek
    
    # Szukamy folderów utworów rekurencyjnie, żeby obsłużyć całe /data.
    folders = _discover_song_folders(data_dir)
    for folder in sorted(folders):
        if limit is not None and processed_tracks >= limit:
            break
        
        # Szukamy lokalnego formatu CSV albo JAMS.
        labels_path = folder / "labels.csv"
        jams_path = folder / "annotation.jams"
        if not jams_path.exists():
            jams_path = next(iter(folder.glob("*.jams")), None)

        # Wczytujemy segmenty z CSV albo z JAMS.
        if labels_path.exists():
            segments = _load_labels_csv(labels_path)
        elif jams_path is not None and jams_path.exists():
            segments = _load_chord_segments_jams(jams_path)
        else:
            continue

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
            
            parsed = parse_chord(chord)
            canonical_root = canonical_note_name(parsed.root_pc) or "N"
            root = canonical_root
            quality = chord_quality(chord)
            if parsed.quality == "N" or canonical_root == "N":
                class_label = "N"
            else:
                class_label = canonical_chord_report_label(chord)
            if quality == "N":
                no_chord_count += 1
                no_chord_duration += duration
            root_counts[root] = root_counts.get(root, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            root_durations[root] = root_durations.get(root, 0.0) + duration
            quality_durations[quality] = quality_durations.get(quality, 0.0) + duration
            chord_class_durations[class_label] = chord_class_durations.get(class_label, 0.0) + duration
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

            # Zliczamy wystapienia rootow akordow i ich typow dla danej tonacji
            if key is not None:
                _, _, key_norm = parse_key(key)
                if not key_norm:
                    key_norm = str(key)
                kr = key_root_counts.setdefault(key_norm, {})
                kq = key_quality_counts.setdefault(key_norm, {})
                kr[root] = kr.get(root, 0) + 1
                kq[quality] = kq.get(quality, 0) + 1
                # Duration-based per-key aggregates
                krd = key_root_durations.setdefault(key_norm, {})
                krd[root] = krd.get(root, 0.0) + duration
                krqd = key_root_quality_durations.setdefault(key_norm, {})
                root_map_d = krqd.setdefault(root, {})
                root_map_d[quality] = root_map_d.get(quality, 0.0) + duration
                # Tutaj mapujemy jeszcze typ akordu z jego rootem i to tez zliczamy
                krq = key_root_quality_counts.setdefault(key_norm, {})
                root_map = krq.setdefault(root, {})
                root_map[quality] = root_map.get(quality, 0) + 1

            # mierzenie interwalu poprzedniego akordu
            if prev_chord is not None:
                distance = interval_distance(prev_chord, chord)
                if distance is not None:
                    interval_distances.append(distance)
                    file_stats[file_name]["interval_distances"].append(distance)
                
                # Zliczaj przejścia między akordami
                transition = f"{canonical_transition_label(prev_chord)} -> {canonical_transition_label(chord)}"
                chord_transitions[transition] = chord_transitions.get(transition, 0) + 1
                transition_durations[transition] = transition_durations.get(transition, 0.0) + duration
            
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
    # Report No-Chord / silence summary
    total_time_s = sum(segment_durations)
    if total_segments > 0:
        pct_segments = 100.0 * no_chord_count / total_segments
    else:
        pct_segments = 0.0
    pct_time = 100.0 * no_chord_duration / total_time_s if total_time_s > 0 else 0.0
    print(f"\nNo-Chord (N): {no_chord_count} segments, {no_chord_duration:.1f}s ({pct_time:.1f}% of audio, {pct_segments:.1f}% of segments)")
    
    print(f"\nPRZESKOKI AKORDOWE (INTERVAL DISTANCES):")
    print(f"  Średnia odległość: {mean_distance:.3f} półtonów")
    print(f"  Odch. std: {std_distance:.3f}")
    if interval_distances:
        print(f"  Min: {min(interval_distances)}, Max: {max(interval_distances)}")
    
    print(f"\nROZKŁAD ROOTÓW (TOP 15) - wg czasu:")
    total_time_s = sum(segment_durations)
    for root, dur in sorted(root_durations.items(), key=lambda item: item[1], reverse=True)[:15]:
        pct = 100 * dur / total_time_s if total_time_s > 0 else 0
        print(f"  {root:3s}: {dur:7.2f}s ({pct:5.1f}%)")
    
    print(f"\nROZKŁAD TYPÓW AKORDÓW (CHORD QUALITY DISTRIBUTION) - wg czasu:")
    for quality, dur in sorted(quality_durations.items(), key=lambda item: item[1], reverse=True):
        pct = 100 * dur / total_time_s if total_time_s > 0 else 0
        print(f"  {quality:6s}: {dur:7.2f}s ({pct:5.1f}%)")

    # Rozkłady per-tonacja
    if key_root_durations:
        print(f"\nROZKŁADY PER-TONACJA - wg czasu:")
        for key_name in sorted(key_root_durations.keys()):
            roots = key_root_durations[key_name]
            total = sum(roots.values())
            print(f"  Tonacja {key_name}: {total:7.2f}s")
            for root, dur in sorted(roots.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = 100 * dur / total if total > 0 else 0
                print(f"    {root:3s}: {dur:7.2f}s ({pct:5.1f}%)")
                # Jeśli mamy też rozkład jakości dla tego rootu, pokażemy go pod spodem
                root_quals = key_root_quality_durations.get(key_name, {}).get(root, {})
                if root_quals:
                    for q, qd in sorted(root_quals.items(), key=lambda x: x[1], reverse=True):
                        pct_root = 100 * qd / dur if dur > 0 else 0
                        print(f"      - {q:6s}: {qd:7.2f}s ({pct_root:5.1f}% of {root})")
    
    print(f"\nTOP 10 PRZEJŚĆ MIĘDZY AKORDAMI - wg liczby (count):")
    total_transitions = sum(chord_transitions.values()) if chord_transitions else 0
    sorted_transitions = sorted(chord_transitions.items(), key=lambda item: item[1], reverse=True)[:10]
    for idx, (transition, cnt) in enumerate(sorted_transitions, 1):
        pct = 100 * cnt / total_transitions if total_transitions > 0 else 0
        print(f"  {idx:2d}. {transition:30s} {cnt:6d} ({pct:5.1f}%)  ")
    
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
    
    # Eksport wyników: HTML
    try:
        _export_html_report(
            processed_tracks=processed_tracks,
            total_segments=total_segments,
            mean_segment_duration=mean_segment_duration,
            total_time_s=sum(segment_durations),
            in_key_ratio=in_key_ratio,
            mean_interval=mean_distance,
            std_interval=std_distance,
            root_durations=root_durations,
            quality_durations=quality_durations,
            key_root_durations=key_root_durations,
            key_root_quality_durations=key_root_quality_durations,
            transition_counts=chord_transitions,
            transition_durations=transition_durations,
            no_chord_count=no_chord_count,
            no_chord_duration=no_chord_duration,
            segment_durations=segment_durations,
            chord_class_durations=chord_class_durations,
        )
    except Exception:
        pass

    print("\n" + "=" * 60)


def main():
    data_dir = Path(cfg_analysis.MUSIC_METRICS_DATA_DIR)
    default_key = cfg_analysis.MUSIC_METRICS_DEFAULT_KEY
    limit = cfg_analysis.DATASET_SONGS

    _process_dataset(data_dir, default_key, limit)


if __name__ == "__main__":
    main()
