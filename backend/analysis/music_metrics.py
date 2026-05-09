"""Muzyczne metryki jakości akordu.

Ten moduł jest celowo mały i skupia się na trzech rzeczach:
- czy akord jest w tonacji,
- jaka jest odległość interwałowa między akordami,
- jak podobne są dwa akordy w prostym, diagnostycznym sensie.

Kod nie zakłada pełnego modelu harmonicznego. Wspiera tylko uproszczone
akordy major/minor oraz oznaczenie ciszy jako ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

# Przypisujemy kazdemu dzwieku jego index np. C - 0, C# - 1 ...
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PC = {note: index for index, note in enumerate(NOTE_NAMES)}

# Normalizacja bemoli z krzyzykami
ENHARMONIC_TO_PC = {
    "Cb": NOTE_TO_PC["B"],
    "Db": NOTE_TO_PC["C#"],
    "Eb": NOTE_TO_PC["D#"],
    "Fb": NOTE_TO_PC["E"],
    "Gb": NOTE_TO_PC["F#"],
    "Ab": NOTE_TO_PC["G#"],
    "Bb": NOTE_TO_PC["A#"],
    "E#": NOTE_TO_PC["F"],
    "B#": NOTE_TO_PC["C"],
}


# Struct akordu - potem mozna rozbudoac o 7 czy inne interwaly
@dataclass(frozen=True)
class ParsedChord:
    """Uproszczona reprezentacja akordu."""

    raw: str
    root: str | None
    root_pc: int | None
    is_minor: bool | None
    quality: str


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_chord(chord: str | None) -> ParsedChord:
    """Parsuje akord do uproszczonej reprezentacji root + major/minor.

    Obsługiwane przykłady:
    - C
    - Am
    - C:maj
    - D:min7
    - N
    """

    # cisza albo cos gorszego
    chord_text = _normalize_text(chord)
    if chord_text in {"", "N", "X", "Z"}:
        return ParsedChord(raw=chord_text, root=None, root_pc=None, is_minor=None, quality="N")

    #Odrzuca bas z akordow np. C/E -> zostanie samo E
    chord_text = chord_text.split("/")[0]
    # dzieli akord na Dźwięk i reszte tekstu np. D# i maj
    match = re.match(r"^([A-G][#b]?)(.*)$", chord_text)
    if not match:
        return ParsedChord(raw=chord_text, root=None, root_pc=None, is_minor=None, quality="N")


    # Ustawianie wszystkiego pod zdefiniowanie akordu
    root, quality_part = match.groups()
    root_pc = ENHARMONIC_TO_PC.get(root, NOTE_TO_PC.get(root))
    quality_lower = quality_part.lower()

    if "dim" in quality_lower or "aug" in quality_lower:
        is_minor = True if "dim" in quality_lower else False
        quality = "m" if "dim" in quality_lower else "maj"
    elif "min" in quality_lower or ("m" in quality_lower and "maj" not in quality_lower):
        is_minor = True
        quality = "m"
    else:
        is_minor = False
        quality = "maj"

    return ParsedChord(raw=chord_text, root=root, root_pc=root_pc, is_minor=is_minor, quality=quality)


def parse_key(key: str | None) -> tuple[int | None, bool | None, str]:
    """Parsuje tonację w prostym formacie major/minor.

    Zwraca:
    - tonic_pc: pitch class toniki,
    - is_minor: czy tonacja jest molowa,
    - normalized: uproszczony tekst tonacji.
    """

    key_text = _normalize_text(key)
    if key_text in {"", "N", "X", "Z"}:
        return None, None, "N"

    match = re.match(r"^([A-G][#b]?)(.*)$", key_text)
    if not match:
        return None, None, key_text

    root, suffix = match.groups()
    tonic_pc = ENHARMONIC_TO_PC.get(root, NOTE_TO_PC.get(root))
    suffix_lower = suffix.lower()

    if suffix_lower.startswith("m") or "min" in suffix_lower:
        return tonic_pc, True, f"{root}m"
    if "maj" in suffix_lower:
        return tonic_pc, False, root
    if key_text.endswith("m"):
        return tonic_pc, True, f"{root}m"
    return tonic_pc, False, root


# Jakie akordy (te ich indexy) wystepuja by okreslic czy dur czy mol
def _scale_for_key(tonic_pc: int, is_minor: bool) -> set[int]:
    """Zbiór pitch classes należących do skali major/minor."""

    if is_minor:
        intervals = {0, 2, 3, 5, 7, 8, 10}
    else:
        intervals = {0, 2, 4, 5, 7, 9, 11}
    return {(tonic_pc + interval) % 12 for interval in intervals}


def in_key(chord: str | None, key: str | None) -> bool:
    """Sprawdza, czy root akordu należy do skali danej tonacji."""

    chord_info = parse_chord(chord)
    tonic_pc, is_minor, _ = parse_key(key)

    if chord_info.root_pc is None or tonic_pc is None or is_minor is None:
        return False

    return chord_info.root_pc in _scale_for_key(tonic_pc, is_minor)


def interval_distance(note_a: str | None, note_b: str | None) -> int | None:
    """Minimalna odległość między dwiema nutami w półtonach (0..6).

    Funkcja liczy odległość po kole chromatycznym, więc C do G daje 5,
    a C do F# daje 6.
    """

    parsed_a = parse_chord(note_a)
    parsed_b = parse_chord(note_b)

    if parsed_a.root_pc is None or parsed_b.root_pc is None:
        return None

    raw_distance = abs(parsed_a.root_pc - parsed_b.root_pc) % 12
    return min(raw_distance, 12 - raw_distance)


# po prostu porownywanie 2 akordow i wyrzucanie ich roznic
# wagi:
# baza akordu - 55%
# Jakość akordu 30 % (Dur/Mol)
# Zgodność z tonacją - 15 %
# ! USTAWIANE w Config.py
def chord_similarity(
    predicted: str | None,
    target: str | None,
    key: str | None = None,
) -> dict[str, float | bool | None]:
    """Prosty score diagnostyczny między dwoma akordami.

    Zwraca słownik z polami:
    - score: 0..1
    - root_match: czy zgadza się root
    - quality_match: czy zgadza się major/minor
    - in_key: czy przewidziany akord jest w tonacji
    - interval_distance: odległość między rootami w półtonach
    """

    pred = parse_chord(predicted)
    true = parse_chord(target)

    if pred.quality == "N" and true.quality == "N":
        return {
            "score": 1.0,
            "root_match": True,
            "quality_match": True,
            "in_key": True if key is None else in_key(predicted, key),
            "interval_distance": 0.0,
        }

    if pred.root_pc is None or true.root_pc is None:
        return {
            "score": 0.0,
            "root_match": False,
            "quality_match": False,
            "in_key": False if key is None else in_key(predicted, key),
            "interval_distance": None,
        }

    distance = interval_distance(predicted, target)
    root_score = 0.0 if distance is None else max(0.0, 1.0 - (distance / 6.0))
    root_match = pred.root_pc == true.root_pc
    quality_match = (
        pred.is_minor == true.is_minor
        if pred.is_minor is not None and true.is_minor is not None
        else False
    )
    key_match = in_key(predicted, key) if key is not None else True

    score = (0.55 * root_score) + (0.30 * (1.0 if quality_match else 0.0)) + (0.15 * (1.0 if key_match else 0.0))
    return {
        "score": float(max(0.0, min(1.0, score))),
        "root_match": root_match,
        "quality_match": quality_match,
        "in_key": key_match,
        "interval_distance": float(distance) if distance is not None else None,
    }


def chord_root(chord: str | None) -> str | None:
    """Zwraca root akordu bez jakości."""

    return parse_chord(chord).root


def chord_quality(chord: str | None) -> str:
    """Zwraca uproszczoną jakość akordu: maj, m albo N."""

    return parse_chord(chord).quality


def count_values(values: Iterable[str | None]) -> dict[str, int]:
    """Prosty licznik tekstowych wartości z normalizacją pustych wpisów."""

    counts: dict[str, int] = {}
    for value in values:
        key = _normalize_text(value) or "N"
        counts[key] = counts.get(key, 0) + 1
    return counts
