#!/usr/bin/env python3
"""
Walk a directory tree, find every .csv/.tsv and .xlsx/.xls/.xlsm file,
and print the header row (column names) of each.

For Excel files, every sheet is opened and its header is printed
separately, since different sheets can have different columns.

Usage:
    python print_headers.py                # scans current directory
    python print_headers.py /path/to/data  # scans a given root folder
"""

import sys
from pathlib import Path

import pandas as pd

CSV_EXTS = {".csv", ".tsv"}
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}

# how many header rows to try reading (some files have 1 header row, this
# just reads row 0 as the header — change nrows/header below if your files
# have multi-row headers or need skiprows)
CSV_SEP_FOR_EXT = {".csv": ",", ".tsv": "\t"}


def print_header(label: str, columns) -> None:
    print(f"\n--- {label} ---")
    print(list(columns))


def handle_csv(path: Path) -> None:
    sep = CSV_SEP_FOR_EXT.get(path.suffix.lower(), ",")
    try:
        df = pd.read_csv(path, sep=sep, nrows=0)
        print_header(str(path), df.columns)
    except Exception as e:
        print(f"\n--- {path} ---")
        print(f"  [ERROR reading file: {e}]")


def handle_excel(path: Path) -> None:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"\n--- {path} ---")
        print(f"  [ERROR opening file: {e}]")
        return

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)
            print_header(f"{path} :: sheet '{sheet_name}'", df.columns)
        except Exception as e:
            print(f"\n--- {path} :: sheet '{sheet_name}' ---")
            print(f"  [ERROR reading sheet: {e}]")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    files = sorted(p for p in root.rglob("*") if p.is_file())

    if not files:
        print(f"No files found under {root}")
        return

    for path in files:
        ext = path.suffix.lower()
        if ext in CSV_EXTS:
            handle_csv(path)
        elif ext in EXCEL_EXTS:
            handle_excel(path)

    print("\nDone.")


if __name__ == "__main__":
    main()