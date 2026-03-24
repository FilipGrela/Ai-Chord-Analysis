import csv
import json
import re
import os

ENHARMONICS = {
    'Cb': 'B',  'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 
    'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 
    'E#': 'F',  'B#': 'C'
}

def simplify_chord(chord_string):
    """
    Simpliies complex chords to simplest possible.
    """
    if chord_string in ['N', 'X', 'Z', '']:
        return 'N'
        
    # Remove inversions. C:maj/5 -> C:maj
    chord = chord_string.split('/')[0]
    

    # Extraxt root and quality using regex. C:maj7 -> root: C, quality: maj7
    match = re.match(r'^([A-G][#b]?)(.*)', chord)
    if not match:
        return 'N'
        
    root, quality = match.groups()
    
    # Standardize enharmonics. Db -> C#, etc.
    root = ENHARMONICS.get(root, root)
    
    # Min vs Maj
    if 'min' in quality or 'm' in quality and 'maj' not in quality:
        return f"{root}m"
    elif 'dim' in quality:
        return f"{root}m"
    else:
        # sus, maj7, aug, => treat as major
        return root
    

def parse_csv(file_path):
    """
    Czyta prosty plik CSV w formacie: start_time, end_time, chord_label
    Zakłada separację przecinkiem lub tabulacją.
    """
    parsed_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Próba odgadnięcia separatora (często pliki tekstowe mają tabulacje)
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        reader = csv.reader(f, dialect=dialect)
        
        for row in reader:
            if len(row) >= 3:
                try:
                    start = float(row[0])
                    end = float(row[1])
                    chord = simplify_chord(row[2].strip())
                    parsed_labels.append((start, end, chord))
                except ValueError:
                    # Pomijamy nagłówki (jeśli wiersz ma tekst w miejscu liczb)
                    continue
                    
    return parsed_labels

def parse_jams(file_path):
    """
    Wyciąga etykiety akordów (chord annotations) z plików .jams (JSON).
    """
    parsed_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Szukamy sekcji z akordami (namespace 'chord')
    for annotation in data.get('annotations', []):
        if annotation.get('namespace') == 'chord':
            for obs in annotation.get('data', []):
                start = float(obs['time'])
                end = start + float(obs['duration'])
                chord = simplify_chord(obs['value'])
                parsed_labels.append((start, end, chord))
            # Po znalezieniu pierwszej poprawnej warstwy akordów wychodzimy
            break
            
    return parsed_labels

def parse_labels(file_path):
    """
    Główny ruter - rozpoznaje rozszerzenie i odpala odpowiedni parser.
    Zwraca listę krotek: [(0.0, 1.5, 'C'), (1.5, 3.2, 'Am'), ...]
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.csv' or ext == '.txt':
        return parse_csv(file_path)
    elif ext == '.jams':
        return parse_jams(file_path)
    else:
        raise ValueError(f"Nieobsługiwany format pliku etykiet: {ext}")
