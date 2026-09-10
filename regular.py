"""
build_regular.py

Run this from the ROOT of your dataset folder (sibling to indicator_*.csv files
and to the `dataset/` folder).

What it does:
1. Reads indicator_56.csv, indicator_57.csv, indicator_58.csv, indicator_59.csv
   from the root folder.
2. Reads every .csv file under dataset/energy consumption/polluted/.
3. Each source file has a junk title row, then a real header row
   (Date, Latitude, Longitude, Place, Parameter, Value, Error, Unit of measure),
   then data rows. We skip the title row and use the real header.
4. Drops Latitude, Longitude, Place, Error -- keeps only:
   Date, Parameter, Value, Unit of measure
5. Excludes any row whose Date falls inside one of the Entry-Exit ranges
   below (inclusive of both endpoints).
6. Combines everything, then removes exact duplicate rows where
   (Date, Parameter, Value) match -- if the Value differs for the same
   Date+Parameter, BOTH rows are kept (that's not a duplicate).
7. Writes the result to regular.csv in the root folder.

Empty / unreadable / missing files are skipped with a warning instead of
crashing the whole run.
"""

import glob
import os
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))

INDICATOR_FILES = [
    "indicator_56.csv",
    "indicator_57.csv",
    "indicator_58.csv",
    "indicator_59.csv",
]

MONTHLY_DIR = os.path.join(ROOT, "dataset", "energy consumption", "polluted")

OUTPUT_PATH = os.path.join(ROOT, "regular.csv")

# Entry/Exit pairs (inclusive range to EXCLUDE), as given.
ENTRY_EXIT_PAIRS = [
    ("Jan-93", "Feb-93"),
    ("Jan-94", "Apr-94"),
    ("Apr-95", "Apr-95"),
    ("Apr-96", "Apr-96"),
    ("Apr-97", "Apr-97"),
    ("Mar-98", "Mar-98"),
    ("Jan-99", "Feb-99"),
    ("Mar-03", "Apr-04"),
    ("Dec-12", "Feb-13"),
    ("Sep-14", "Apr-15"),
    ("Dec-14", "May-15"),
]

FINAL_COLUMNS = ["Date", "Parameter", "Value", "Unit of measure"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_month_year(s):
    """Parse strings like 'Jan-93' -> datetime(1993, 1, 1).
    Returns None if it can't be parsed (row will then be kept, not silently
    dropped, since we can't prove it's inside an excluded range)."""
    if s is None:
        return None
    s = str(s).strip()
    try:
        return datetime.strptime(s, "%b-%y")
    except ValueError:
        return None


EXCLUDED_RANGES = [
    (parse_month_year(start), parse_month_year(end))
    for start, end in ENTRY_EXIT_PAIRS
]


def is_excluded(date_val):
    dt = parse_month_year(date_val)
    if dt is None:
        return False
    for start, end in EXCLUDED_RANGES:
        if start is not None and end is not None and start <= dt <= end:
            return True
    return False


def load_one_file(path):
    """Read a single source file, skip its junk title row, return a
    DataFrame with just FINAL_COLUMNS (or None if unusable)."""
    if not os.path.isfile(path):
        print(f"  [skip] not found: {path}")
        return None
    if os.path.getsize(path) == 0:
        print(f"  [skip] empty file: {path}")
        return None

    try:
        # Row 0 is the junk "Indicator NN, title..." row -> skip it.
        # Row 1 becomes the real header.
        df = pd.read_csv(path, skiprows=1)
    except Exception as e:
        print(f"  [skip] could not read {path}: {e}")
        return None

    if df.empty:
        print(f"  [skip] no data rows: {path}")
        return None

    required = {"Date", "Parameter", "Value", "Unit of measure"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [skip] missing columns {missing} in {path}")
        return None

    df = df[list(FINAL_COLUMNS)].copy()
    df["Date"] = df["Date"].astype(str).str.strip()

    before = len(df)
    df = df[~df["Date"].apply(is_excluded)]
    after = len(df)
    print(f"  loaded {path}: {before} rows -> {after} after entry/exit exclusion")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    frames = []

    print("Reading indicator_*.csv files from root...")
    for fname in INDICATOR_FILES:
        path = os.path.join(ROOT, fname)
        df = load_one_file(path)
        if df is not None:
            frames.append(df)

    print("\nReading Monthly *.csv files from dataset/energy consumption/polluted/...")
    monthly_paths = sorted(glob.glob(os.path.join(MONTHLY_DIR, "*.csv")))
    if not monthly_paths:
        print(f"  [warning] no .csv files found in {MONTHLY_DIR}")
    for path in monthly_paths:
        df = load_one_file(path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise SystemExit("\nNo usable data found in any source file. Nothing to write.")

    combined = pd.concat(frames, ignore_index=True)
    total_before_dedup = len(combined)

    # Duplicate = same Date + Parameter + Value.
    # If Value differs for the same Date+Parameter, both rows are kept.
    combined = combined.drop_duplicates(subset=["Date", "Parameter", "Value"], keep="first")

    total_after_dedup = len(combined)

    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCombined rows before de-duplication: {total_before_dedup}")
    print(f"Rows after removing exact duplicates: {total_after_dedup}")
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()