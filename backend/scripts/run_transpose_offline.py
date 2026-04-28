"""
Skrypt do aplikowania offline transpozycji na już wygenerowany dataset.
Generuje transponowane wersje spektrogramów i etykiet.

Użycie:
    python backend/scripts/run_transpose_offline.py --semitones -2 -1 1 2
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.data.builder import DatasetBuilder
from backend.logger.logger import Logger

logger = Logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Aplikuj offline transpozycję na dataset"
    )
    parser.add_argument(
        "--semitones",
        type=int,
        nargs="+",
        default=[-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
        help="Lista półtonów do transpozycji (domyślnie: -6 do +6 oprócz 0)",
    )

    args = parser.parse_args()

    logger.info("--- Offline Transponowanie Datasetu ---")
    logger.info(f"Półtony: {args.semitones}")

    builder = DatasetBuilder()
    builder.apply_offline_transposition(semitones_list=args.semitones)

    logger.info("Ukończono!")


if __name__ == "__main__":
    main()

