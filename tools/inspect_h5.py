"""Read-only HDF5 inventory helper used before writing a source adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def inspect_file(path: Path, root: Path) -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, value: object) -> None:
            if isinstance(value, h5py.Dataset):
                datasets.append({
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "attrs": {str(key): str(item) for key, item in value.attrs.items()},
                })

        handle.visititems(visitor)
        return {
            "file": str(path.relative_to(root)),
            "attrs": {str(key): str(item) for key, item in handle.attrs.items()},
            "datasets": datasets,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted(args.root.rglob("session.h5"))
    print(json.dumps([inspect_file(path, args.root) for path in files], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
