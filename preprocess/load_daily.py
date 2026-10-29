"""
Preprocess 2-year inverter + irradiation data from data_2_years/.

Key logic
---------
- Inverter energy   : Daily energy(kWh) is a *cumulative* counter that resets
                      at midnight.  Interval energy = diff(A[i] - A[i-1])
                      within each (ManageObject, date) group.
- Irradiation 2025  : Irradiance(W/㎡) is instantaneous power.
                      kWh/m² per 15-min interval = W/m² × (15/60) / 1000
- Irradiation 2026  : Daily irradiation(Energy)(kWh/㎡) is also cumulative
                      within the day → same diff approach as inverter energy.

Outputs
-------
  input_data/inverter_daily.csv     – daily yield per inverter
  input_data/irradiation_daily.csv  – daily irradiation
  input_data/transformed.csv        – merged + PR, ready for modelling
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_2_years"
OUTPUT_DIR = PROJECT_ROOT / "input_data"

SHUNDAO_DIRS = {
    1: DATA_DIR / "Shundao 1",
    2: DATA_DIR / "Shundao 2",
}
IRR_DIR = DATA_DIR / "Irradiation"

# Column aliases used in irradiation files (differ between 2025 and 2026)
IRR_COL_CUMULATIVE = "Daily irradiation(Energy)(kWh/\u33a1)"   # 2026 format
IRR_COL_INSTANT    = "Irradiance(W/\u33a1)"                    # 2025 format


# ---------------------------------------------------------------------------
# Installed capacity
# ---------------------------------------------------------------------------

def load_installed_capacity(path: Path) -> pd.DataFrame:
    """
    Parse installed_capacity.xlsx and return a DataFrame with columns:
        device_name (str)  – e.g. 'HF1 Inverter 1'
        capacity_kw (float)
    """
    raw = pd.read_excel(path, sheet_name="INV Infor", header=None)

    records = []
    for _, row in raw.iterrows():
        device_name = row.iloc[2]   # alias column (e.g. 'HF1 Inverter 1')
        capacity    = row.iloc[6]   # CAPACITY (KW) column
        if pd.notna(device_name) and pd.notna(capacity):
            try:
                capacity = float(capacity)
                records.append({"device_name": str(device_name).strip(), "capacity_kw": capacity})
            except (ValueError, TypeError):
                pass

    df = pd.DataFrame(records).drop_duplicates("device_name")
    print(f"  Loaded {len(df)} inverter capacity records")
    return df


# ---------------------------------------------------------------------------
# Inverter data
# ---------------------------------------------------------------------------

def _parse_inverter_file(path: Path, zone: int) -> pd.DataFrame:
    """
    Read one monthly inverter file and return daily yield per (device, date).

    Steps
    -----
    1. Sort by ManageObject + Start Time.
    2. For each (ManageObject, date) group compute interval energy as the
       forward diff of the cumulative Daily energy column.
    3. Fill the first interval of each day with its raw value (no previous row).
    4. Clip negative diffs to 0 (data gaps / reset artefacts).
    5. Sum intervals → daily yield_kwh per device.
    """
    df = pd.read_excel(path, sheet_name=0, header=3)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Start Time":        "timestamp",
        "Daily energy(kWh)": "cumulative_energy",
        "ManageObject":      "manage_object",
        "Site Name":         "site_name",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["cumulative_energy"] = pd.to_numeric(df["cumulative_energy"], errors="coerce")
    df = df.dropna(subset=["timestamp", "manage_object", "cumulative_energy"])

    df["device_name"] = df["manage_object"].str.split("/").str[-1].str.strip()
    df["date"]        = df["timestamp"].dt.date
    df["zone"]        = zone

    # Sort within each device so diff is in time order
    df = df.sort_values(["manage_object", "timestamp"])

    # Interval energy = diff within (manage_object, date)
    df["interval_energy"] = df.groupby(["manage_object", "date"])["cumulative_energy"].diff()

    # First interval of the day has no previous row → use its cumulative value directly
    first_mask = df["interval_energy"].isna()
    df.loc[first_mask, "interval_energy"] = df.loc[first_mask, "cumulative_energy"]

    # Clip negatives (artefacts from resets or missing rows)
    df["interval_energy"] = df["interval_energy"].clip(lower=0)

    # Aggregate to daily
    daily = (
        df.groupby(["zone", "device_name", "date"], as_index=False)
        .agg(yield_kwh=("interval_energy", "sum"))
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def load_inverter_data() -> pd.DataFrame:
    """Load and combine all Shundao 1 & 2 monthly inverter files."""
    frames: list[pd.DataFrame] = []
    for zone, folder in SHUNDAO_DIRS.items():
        files = sorted(folder.glob("Inverter_*.xlsx"))
        print(f"  Zone {zone}: {len(files)} monthly files")
        for f in files:
            frames.append(_parse_inverter_file(f, zone))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["zone", "device_name", "date"]).reset_index(drop=True)
    print(f"  Inverter daily rows: {len(df):,}  |  "
          f"devices: {df['device_name'].nunique()}  |  "
          f"date range: {df['date'].min().date()} – {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Irradiation data
# ---------------------------------------------------------------------------

def _parse_irradiation_file(path: Path) -> pd.DataFrame:
    """
    Read one monthly irradiation file and return daily irradiation (kWh/m²).

    Two formats are handled automatically:
    - 2025: Irradiance(W/㎡) instantaneous  → kWh/m² = W/m² × 0.25h / 1000
    - 2026: Daily irradiation(Energy)(kWh/㎡) cumulative → diff approach
    """
    df = pd.read_excel(path, sheet_name=0, header=3)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Start Time": "timestamp", "ManageObject": "manage_object"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df = df.sort_values(["manage_object", "timestamp"])

    if IRR_COL_CUMULATIVE in df.columns:
        # 2026 cumulative format
        df["irr_raw"] = pd.to_numeric(df[IRR_COL_CUMULATIVE], errors="coerce").fillna(0)
        df["interval_irr"] = df.groupby(["manage_object", "date"])["irr_raw"].diff()
        first_mask = df["interval_irr"].isna()
        df.loc[first_mask, "interval_irr"] = df.loc[first_mask, "irr_raw"]
        df["interval_irr"] = df["interval_irr"].clip(lower=0)

    elif IRR_COL_INSTANT in df.columns:
        # 2025 instantaneous format:  W/m² → kWh/m² per 15-min slot
        df["irr_raw"] = pd.to_numeric(df[IRR_COL_INSTANT], errors="coerce").fillna(0)
        df["interval_irr"] = (df["irr_raw"] * 0.25 / 1000).clip(lower=0)

    else:
        available = list(df.columns)
        raise ValueError(
            f"Unrecognised irradiation column in {path.name}. "
            f"Available columns: {available}"
        )

    daily = (
        df.groupby("date", as_index=False)
        .agg(irradiation_kwh_m2=("interval_irr", "sum"))
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def load_irradiation_data() -> pd.DataFrame:
    """Load all irradiation files from 2025/ and 2026/ sub-folders."""
    frames: list[pd.DataFrame] = []
    for year_dir in sorted(IRR_DIR.iterdir()):
        files = sorted(year_dir.glob("*.xlsx"))
        print(f"  Irradiation {year_dir.name}: {len(files)} files")
        for f in files:
            frames.append(_parse_irradiation_file(f))

    df = pd.concat(frames, ignore_index=True)

    # If multiple sensors exist for the same day, average them
    df = (
        df.groupby("date", as_index=False)
        .agg(irradiation_kwh_m2=("irradiation_kwh_m2", "mean"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"  Irradiation daily rows: {len(df):,}  |  "
          f"date range: {df['date'].min().date()} – {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Transform: merge + PR
# ---------------------------------------------------------------------------

def build_transformed(
    inv_df: pd.DataFrame,
    irr_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join inverter daily yield with irradiation and installed capacity,
    then compute Performance Ratio:

        PR = yield_kwh / (irradiation_kwh_m2 × capacity_kw)
    """
    df = inv_df.merge(capacity_df[["device_name", "capacity_kw"]], on="device_name", how="left")
    df = df.merge(irr_df.rename(columns={"date": "date"}), on="date", how="left")

    # PR: only defined when irradiation and capacity are available and non-zero
    denom = df["irradiation_kwh_m2"] * df["capacity_kw"]
    df["performance_ratio"] = (
        df["yield_kwh"] / denom.replace(0, float("nan"))
    ).round(4)

    df["date"] = pd.to_datetime(df["date"])

    col_order = [
        "zone", "date", "device_name", "capacity_kw",
        "yield_kwh", "irradiation_kwh_m2", "performance_ratio",
    ]
    return df[col_order].sort_values(["zone", "date", "device_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Installed capacity ──")
    capacity_df = load_installed_capacity(DATA_DIR / "installed_capacity.xlsx")

    print("\n── Inverter data ──")
    inv_df = load_inverter_data()

    print("\n── Irradiation data ──")
    irr_df = load_irradiation_data()

    print("\n── Building transformed dataset ──")
    transformed_df = build_transformed(inv_df, irr_df, capacity_df)

    inv_path         = output_dir / "inverter_daily.csv"
    irr_path         = output_dir / "irradiation_daily.csv"
    transformed_path = output_dir / "transformed.csv"

    inv_df.to_csv(inv_path, index=False, encoding="utf-8-sig")
    irr_df.to_csv(irr_path, index=False, encoding="utf-8-sig")
    transformed_df.to_csv(transformed_path, index=False, encoding="utf-8-sig")

    print(f"\n── Output ──")
    print(f"  {inv_path}")
    print(f"  {irr_path}")
    print(f"  {transformed_path}")
    print(f"\n  Transformed rows : {len(transformed_df):,}")
    print(f"  Devices          : {transformed_df['device_name'].nunique()}")
    print(f"  Date range       : {transformed_df['date'].min().date()} – {transformed_df['date'].max().date()}")
    missing_cap = transformed_df["capacity_kw"].isna().sum()
    missing_irr = transformed_df["irradiation_kwh_m2"].isna().sum()
    missing_pr  = transformed_df["performance_ratio"].isna().sum()
    print(f"  Missing capacity : {missing_cap:,}")
    print(f"  Missing irradiation: {missing_irr:,}")
    print(f"  Missing PR       : {missing_pr:,}")
    print()
    print(transformed_df.head(5).to_string(index=False))

    return inv_path, irr_path, transformed_path


if __name__ == "__main__":
    run()
