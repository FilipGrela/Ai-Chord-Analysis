import csv
import json
import re
import os

class ChordLabelParser:
    """Klasa odpowiedzialna za odczyt i normalizację etykiet akordów."""
    
    ENHARMONICS = {
        'Cb': 'B',  'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 
        'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 
        'E#': 'F',  'B#': 'C'
    }

    @classmethod
    def simplify_chord(cls, chord_string: str) -> str:
        if chord_string in ['N', 'X', 'Z', '']:
            return 'N'
            
        chord = chord_string.split('/')[0]
        match = re.match(r'^([A-G][#b]?)(.*)', chord)
        if not match:
            return 'N'
            
        root, quality = match.groups()
        root = cls.ENHARMONICS.get(root, root)
        
        if 'min' in quality or 'm' in quality and 'maj' not in quality:
            return f"{root}m"
        elif 'dim' in quality:
            return f"{root}m"
        else:
            return root

    @classmethod
    def parse_csv(cls, file_path: str) -> list[tuple[float, float, str]]:
        parsed_labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            reader = csv.reader(f, dialect=dialect)
            
            for row in reader:
                if len(row) >= 3:
                    try:
                        start, end = float(row[0]), float(row[1])
                        chord = cls.simplify_chord(row[2].strip())
                        parsed_labels.append((start, end, chord))
                    except ValueError:
                        continue
        return parsed_labels

    @classmethod
    def parse_jams(cls, file_path: str) -> list[tuple[float, float, str]]:
        parsed_labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for annotation in data.get('annotations', []):
            if annotation.get('namespace') == 'chord':
                for obs in annotation.get('data', []):
                    start = float(obs['time'])
                    end = start + float(obs['duration'])
                    chord = cls.simplify_chord(obs['value'])
                    parsed_labels.append((start, end, chord))
                break
        return parsed_labels

    @classmethod
    def parse(cls, file_path: str) -> list[tuple[float, float, str]]:
        """Główny ruter rozpoznający format pliku."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.csv', '.txt']:
            return cls.parse_csv(file_path)
        elif ext == '.jams':
            return cls.parse_jams(file_path)
        else:
            raise ValueError(f"Nieobsługiwany format pliku etykiet: {ext}")