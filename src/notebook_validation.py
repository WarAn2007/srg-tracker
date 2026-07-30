"""Execute project notebooks in memory for reproducibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from src.config import ROOT_DIR


def validate_notebook(path: Path, timeout: int = 600) -> None:
    """Execute one notebook from the project root without overwriting it."""
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT_DIR)}},
    )
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", help="Notebook paths relative to the project root.")
    args = parser.parse_args()
    for notebook_name in args.notebooks:
        path = ROOT_DIR / notebook_name
        validate_notebook(path)
        print(f"PASS {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
