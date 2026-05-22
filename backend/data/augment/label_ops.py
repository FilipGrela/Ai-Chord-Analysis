"""
Operacje na etykietach akordów dla augmentacji danych.
"""

from backend.config import cfg_builder
from backend.logger.logger import Logger

logger = Logger(__name__)

class ChordTranspose:
    """Klasa do transponowania etykiet akordów."""

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    VOCAB = NOTES + [n + 'm' for n in NOTES] + ['N']

    @staticmethod
    def transpose_chord_label(chord: str, semitones: int) -> str:
        """
        Transpozycja etykiety akordu o podaną liczbę półtonów.

        Args:
            chord: str - etykieta akordu np. 'C', 'C#', 'Am', 'N' (cisza)
            semitones: int - liczba półtonów do transponowania (dodatnia lub ujemna)

        Returns:
            str - transponowana etykieta akordu

        Example:
            transpose_chord_label('C', 1)  -> 'C#'
            transpose_chord_label('C', 12) -> 'C'  (oktawa w górę)
            transpose_chord_label('Am', -2) -> 'G#m'
            transpose_chord_label('N', 5)  -> 'N'  (cisza pozostaje ciszą)
        """
        # Cisza nie zmienia się przy transpozycji
        if chord == 'N':
            return 'N'

        # Podziel etykietę na nutę bazową i typ akordu (major/minor/seventh)
        suffix = ""
        if getattr(cfg_builder, "SUPPORT_SEVENTHS", False):
            if chord.endswith('m7'):
                base_note = chord[:-2]
                suffix = 'm7'
            elif chord.endswith('7'):
                base_note = chord[:-1]
                suffix = '7'
            elif chord.endswith('m'):
                base_note = chord[:-1]
                suffix = 'm'
            else:
                base_note = chord
        else:
            if chord.endswith('m'):
                base_note = chord[:-1]
                suffix = 'm'
            else:
                base_note = chord

        # Znajdź obecny indeks nuty w słowniku
        if base_note not in ChordTranspose.NOTES:
            # Jeśli nota nie jest znana, zwróć oryginał
            return chord

        current_idx = ChordTranspose.NOTES.index(base_note)

        # Oblicz nowy indeks z zawinięciem modulo 12
        new_idx = (current_idx + semitones) % 12
        new_note = ChordTranspose.NOTES[new_idx]

        # Odbuduj etykietę akordu
        return new_note + suffix

    @staticmethod
    def transpose_label_array(label_array: list, semitones: int) -> list:
        """
        Transpozycja całej listy etykiet akordów.

        Args:
            label_array: list[str] - lista etykiet akordów
            semitones: int - liczba półtonów do transponowania

        Returns:
            list[str] - lista transponowanych etykiet
        """
        logger.info(f"Transponowanie etykiet: count={len(label_array)}, semitony={semitones}")
        return [ChordTranspose.transpose_chord_label(chord, semitones) for chord in label_array]

