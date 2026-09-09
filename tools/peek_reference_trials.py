"""Print a few HDF5 trial/event rows for adapter verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def clean(value: object) -> object:
    return value.decode("utf-8") if isinstance(value, bytes) else value.item() if hasattr(value, "item") else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    with h5py.File(args.file, "r") as handle:
        for name in ("trials", "events", "cue_events"):
            print(name)
            rows = handle[name][:12]
            for row in rows:
                print({field: clean(row[field]) for field in row.dtype.names})


if __name__ == "__main__":
    main()
