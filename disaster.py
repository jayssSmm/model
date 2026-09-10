import pandas as pd
import glob
import os

# ---------------------------------------------------------------
# 1. Disaster entry/exit periods (as provided, kept as Mon-YY text)
# ---------------------------------------------------------------
DISASTER_PERIODS = [
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

# Convert Mon-YY -> real datetime for range comparison only
# (POSIX rule: 69-99 -> 1900s, 00-68 -> 2000s; matches this dataset's 1993-2015 range)
def parse_monyy(s):
    return pd.to_datetime(s.strip(), format="%b-%y")

DISASTER_RANGES = [(parse_monyy(e), parse_monyy(x)) for e, x in DISASTER_PERIODS]

# ---------------------------------------------------------------
# 2. Locate source files
# ---------------------------------------------------------------
ROOT = "."
MONTHLY_DIR = os.path.join(ROOT, "dataset", "energy consumption", "polluted")

monthly_files = glob.glob(os.path.join(MONTHLY_DIR, "Monthly*.csv"))
indicator_files = glob.glob(os.path.join(ROOT, "indicator*.csv"))

all_files = monthly_files + indicator_files

if not all_files:
    print("No Monthly*/indicator* CSV files found. Check that this script is")
    print("run from the folder containing 'dataset/' and the indicator_*.csv files.")

# ---------------------------------------------------------------
# 3. Read + filter each file
# ---------------------------------------------------------------
REQUIRED_COLS = ["Date", "Parameter", "Value", "Unit of measure"]
collected = []

for path in all_files:
    try:
        # Row 0 of the file is a title row ("Indicator 56 ..."), the real
        # column headers are on the next line -> header=1
        df = pd.read_csv(path, header=1)
    except Exception as e:
        print(f"Skipping {path}: could not read ({e})")
        continue

    # Drop any blank/unnamed trailing columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"Skipping {path}: missing expected columns {missing}")
        continue

    df = df.dropna(subset=["Date"])

    # Parse date for range comparison, keep original text for output
    try:
        parsed_dates = pd.to_datetime(df["Date"], format="%b-%y", errors="coerce")
    except Exception as e:
        print(f"Skipping {path}: could not parse dates ({e})")
        continue

    df = df.assign(_parsed_date=parsed_dates)
    df = df.dropna(subset=["_parsed_date"])

    # Keep rows that fall inside ANY disaster period (inclusive)
    mask = pd.Series(False, index=df.index)
    for start, end in DISASTER_RANGES:
        mask |= (df["_parsed_date"] >= start) & (df["_parsed_date"] <= end)

    matched = df.loc[mask, REQUIRED_COLS]
    if not matched.empty:
        collected.append(matched)
        print(f"{os.path.basename(path)}: {len(matched)} matching row(s)")
    else:
        print(f"{os.path.basename(path)}: 0 matching rows")

# ---------------------------------------------------------------
# 4. Combine, dedupe, write out
# ---------------------------------------------------------------
if collected:
    result = pd.concat(collected, ignore_index=True)

    # Dedupe: same Date + Parameter + Value + Unit of measure = same reading,
    # only keep one. Different Value for same Date is kept as a separate row.
    before = len(result)
    result = result.drop_duplicates(subset=REQUIRED_COLS, keep="first")
    after = len(result)

    result.to_csv("disaster.csv", index=False)
    print(f"\nWrote disaster.csv: {after} row(s) ({before - after} duplicate(s) removed)")
else:
    pd.DataFrame(columns=REQUIRED_COLS).to_csv("disaster.csv", index=False)
    print("\nNo matching rows found in any file. Wrote empty disaster.csv with headers only.")