import argparse
from pathlib import Path
from pprint import pformat
from typing import Any

import torch

from backend.config import cfg_paths


def _extract_checkpoint_payload(checkpoint: Any) -> tuple[dict, dict]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        metadata = checkpoint.get("metadata", {})
        return checkpoint["state_dict"], metadata if isinstance(metadata, dict) else {}
    return checkpoint, {}


def _print_section(title: str, value: Any) -> None:
    print(f"\n{title}:")
    print(pformat(value, sort_dicts=False, width=100))


def _discover_checkpoints(path_hint: str) -> list[Path]:
    candidate_path = Path(path_hint)

    if candidate_path.is_file():
        return [candidate_path]

    if candidate_path.is_dir():
        return sorted(candidate_path.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)

    if candidate_path.suffix.lower() == ".pth":
        parent_dir = candidate_path.parent
        if parent_dir.exists():
            return sorted(parent_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)

    return []


def _choose_checkpoint(checkpoints: list[Path]) -> Path:
    if not checkpoints:
        raise FileNotFoundError("Nie znaleziono żadnych plików checkpoint (.pth) w podanej lokalizacji.")

    if len(checkpoints) == 1:
        return checkpoints[0]

    print("Znaleziono kilka checkpointów:")
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"  {index}. {checkpoint.name}")

    while True:
        choice = input("Wybierz numer modelu: ").strip()
        if not choice.isdigit():
            print("Podaj numer z listy.")
            continue

        selected_index = int(choice)
        if 1 <= selected_index <= len(checkpoints):
            return checkpoints[selected_index - 1]

        print("Numer poza zakresem listy.")


def inspect_checkpoint(checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, metadata = _extract_checkpoint_payload(checkpoint)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Type: {'full checkpoint' if metadata else 'legacy state_dict'}")
    print(f"State dict keys: {len(state_dict)}")

    if metadata:
        _print_section("Metadata", metadata)
        config_snapshot = metadata.get("config")
        if config_snapshot:
            _print_section("Config snapshot", config_snapshot)
    else:
        print("\nMetadata: brak (stary format checkpointu)")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved model checkpoint and print stored config metadata")
    parser.add_argument("--checkpoint", "-c", default=cfg_paths.MODEL_SAVE_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    checkpoints = _discover_checkpoints(args.checkpoint)
    selected_checkpoint = _choose_checkpoint(checkpoints)
    inspect_checkpoint(str(selected_checkpoint))


if __name__ == "__main__":
    cli()
