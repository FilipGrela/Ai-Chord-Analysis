import csv
import json
import re
import os

from backend.config import cfg_builder
from backend.logger.logger import Logger

logger = Logger(__name__)

class ChordLabelParser:
    """Klasa odpowiedzialna za odczyt i normalizację etykiet akordów."""
    
    ENHARMONICS = {
        'Cb': 'B',  'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 
        'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 
        'E#': 'F',  'B#': 'C'
    }

    @classmethod
    def simplify_chord(cls, chord_string: str) -> str:
        """
        Funkcja upraszcza akordy, usuwa akordy maj, 7, 9, 13, / itp. 
        Upraszcza je do akordów moll oraz dur.
        Normalizuje też ciszę do symbolu 'N'
        """
        if chord_string in ['N', 'X', 'Z', '']:
            return 'N'
            
        chord = chord_string.split('/')[0]
        match = re.match(r'^([A-G][#b]?)(.*)', chord)
        if not match:
            return 'N'
            
        root, quality = match.groups()
        root = cls.ENHARMONICS.get(root, root)

        support_sevenths = getattr(cfg_builder, "SUPPORT_SEVENTHS", False)

        if support_sevenths:
            # Preserve seventh chords in a compact form compatible with VOCAB.
            # - minor seventh -> Cm7
            # - any other seventh-ish quality -> C7
            if '7' in quality:
                if 'min' in quality or ('m' in quality and 'maj' not in quality):
                    return f"{root}m7"
                return f"{root}7"

        if 'min' in quality or ('m' in quality and 'maj' not in quality):
            return f"{root}m"
        elif 'dim' in quality:
            return f"{root}m"
        else:
            return root

    @classmethod
    def parse_csv(cls, file_path: str) -> list[tuple[float, float, str]]:
        """
        Funkcja wczytuje akordy z pliku i zwraca je w formie listy:
        [(start, koniec, akord), ..., (...)]

        Format pliku:
        start, end, chord

        Przykład:
        0.000,1.245,C:maj
        1.245,2.510,G:min
        2.510,3.000,F:maj
        3.000,4.500,N

        Uwagi:
        - Separator jest autodetektowany (przecinek, średnik, tabulator).
        - Wiersze za krótkie lub z błędem są pomijane.
        """
        # logger.info(f"Parsowanie CSV: {file_path}")
        parsed_labels = []
        with open(file_path, 'r', encoding='utf-8') as f:

            # Detekcja formatowania pliku, wykrywa automatycznie seperatory kolumn itp.
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            # Użycie wykrytego formatowania do odczytu pliku CSV
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
        """
        Funkcja wczytuje akordy z pliku i zwraca je w formie listy:
        [(start, koniec, akord), ..., (...)]

        Format pliku (fragment JSON):
        {
          "annotations": [
            {
              "namespace": "chord",
              "data": [
                {"time": 0.0, "duration": 1.245, "value": "C:maj"},
                {"time": 1.245, "duration": 1.265, "value": "G:min"},
                {"time": 2.510, "duration": 0.490, "value": "F:maj"},
                {"time": 3.0, "duration": 1.5, "value": "N"}
              ]
            }
          ]
        }

        Uwagi:
        - Szukane są annotacje z namespace="chord".
        - Z pola "time" i "duration" wyliczane jest: end_sec = time + duration.
        - Tylko pierwsze znalezione annotacje z chord są przetwarzane.
        """

        # TODO: przeanalizować czy mozna użyć innych annotacji z pliku JAMS
        # logger.info(f"Parsowanie JAMS: {file_path}")
        parsed_labels = []

        # Ładuje plij json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Szuka annotanions i znajuje pojedyńcze akordy.
        for annotation in data.get('annotations', []):
            if annotation.get('namespace') == 'chord':
                # Odczytuje pojedyńczy akord i czas w którym wybrzmiewa
                for obs in annotation.get('data', []):
                    start = float(obs['time'])
                    end = start + float(obs['duration'])
                    chord = cls.simplify_chord(obs['value']) # Normalizacja akordu
                    parsed_labels.append((start, end, chord))
                break
        return parsed_labels

    @classmethod
    def parse(cls, file_path: str) -> list[tuple[float, float, str]]:
        """Główny ruter rozpoznający format pliku."""
        _, ext = os.path.splitext(file_path) # Ekstrakcja rozszerzenia.
        ext = ext.lower()

        if ext in ['.csv', '.txt']:
            return cls.parse_csv(file_path)
        elif ext == '.jams':
            return cls.parse_jams(file_path)
        else:
            raise ValueError(f"Nieobsługiwany format pliku etykiet: {ext}")