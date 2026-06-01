"""
Transform 2-year inverter + irradiation data → input_data/2_years/

Logic
-----
- Daily energy(kWh) in inverter files is a cumulative counter that resets at
  midnight per inverter.  To get the actual energy in each 15-min slot:
      interval_energy = diff(cumulative[i] - cumulative[i-1])
  grouped by (ManageObject, date).  The first row of each group keeps its
  raw value (no previous row to subtract).  Negative diffs are clipped to 0.

- Irradiation 2025  : Irradiance(W/㎡) is instantaneous power.
      kWh/m² per slot = W/m² × (15 min / 60) / 1000
- Irradiation 2026  : Daily irradiation(Energy)(kWh/㎡) is also a cumulative
  counter → same diff approach as inverter energy.

Outputs (input_data/2_years/)
-------------------------------
  inverter_daily.csv     – daily yield per device
  irradiation_daily.csv  – daily irradiation (kWh/m²)
  transformed.csv        – merged: yield + irradiation + capacity + PR
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data_2_years"
OUTPUT_DIR   = PROJECT_ROOT / "input_data" / "2_years"

SHUNDAO = {1: DATA_DIR / "Shundao 1", 2: DATA_DIR / "Shundao 2"}
IRR_DIR  = DATA_DIR / "Irradiation"

# Irradiation column names differ by year
COL_CUMULATIVE = "Daily irradiation(Energy)(kWh/\u33a1)"   # 2026
COL_INSTANT    = "Irradiance(W/\u33a1)"                    # 2025


# ── helpers ──────────────────────────────────────────────────────────────────

def _daily_from_cumulative(df: pd.DataFrame, group_cols: list[str], cum_col: str) -> pd.Series:
    """
    Given a DataFrame sorted by (group_cols + timestamp), return per-row
    interval values derived from a cumulative column.
    """
    diff = df.groupby(group_cols)[cum_col].diff()
    # First row of each group: no previous row → use the cumulative value itself
    diff = diff.fillna(df[cum_col])
    return diff.clip(lower=0)


# ── installed capacity ────────────────────────────────────────────────────────

def load_capacity() -> pd.DataFrame:
    """
    Return DataFrame with device_name and installed_capacity_kwp.

    Source: sheet 'INV Infor (2)'
      col[2] = device alias  (e.g. 'HF1 Inverter 1')
      col[5] = Công suất lắp đặt (kWp DC string capacity)
    """
    raw = pd.read_excel(DATA_DIR / "installed_capacity.xlsx",
                        sheet_name="INV Infor (2)", header=None)
    records = []
    for _, row in raw.iterrows():
        name     = row.iloc[2]   # device alias
        capacity = row.iloc[5]   # Công suất lắp đặt (kWp)
        if pd.notna(name) and pd.notna(capacity):
            try:
                records.append({
                    "device_name":            str(name).strip(),
                    "installed_capacity_kwp": float(capacity),
                })
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(records).drop_duplicates("device_name")
    print(f"  Capacity records loaded: {len(df)}")
    return df


# ── inverter data ─────────────────────────────────────────────────────────────

def _read_inverter_file(path: Path, zone: int) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=3,
                       usecols=["ManageObject", "Start Time", "Daily energy(kWh)"])
    df.columns = ["manage_object", "timestamp", "cumulative_energy"]
    df["timestamp"]         = pd.to_datetime(df["timestamp"], errors="coerce")
    df["cumulative_energy"] = pd.to_numeric(df["cumulative_energy"], errors="coerce")
    df = df.dropna(subset=["timestamp", "manage_object", "cumulative_energy"])

    df["device_name"] = df["manage_object"].str.split("/").str[-1].str.strip()
    df["date"]        = df["timestamp"].dt.date
    df["zone"]        = zone

    # Sort within inverter so diff is chronological
    df = df.sort_values(["manage_object", "timestamp"])
    df["interval_energy"] = _daily_from_cumulative(df, ["manage_object", "date"], "cumulative_energy")

    daily = (
        df.groupby(["zone", "device_name", "date"], as_index=False)
          .agg(yield_kwh=("interval_energy", "sum"))
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def load_inverter_daily() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for zone, folder in SHUNDAO.items():
        files = sorted(folder.glob("Inverter_*.xlsx"))
        print(f"  Zone {zone}: reading {len(files)} files", end="", flush=True)
        for f in files:
            frames.append(_read_inverter_file(f, zone))
            print(".", end="", flush=True)
        print()
    df = pd.concat(frames, ignore_index=True).sort_values(["zone", "device_name", "date"])
    print(f"  → {len(df):,} device-days  |  "
          f"{df['device_name'].nunique()} devices  |  "
          f"{df['date'].min().date()} – {df['date'].max().date()}")
    return df.reset_index(drop=True)


# ── irradiation data ──────────────────────────────────────────────────────────

def _read_irr_file(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=3)
    raw.columns = raw.columns.str.strip()
    raw["timestamp"] = pd.to_datetime(raw.get("Start Time", raw.iloc[:, 3]), errors="coerce")
    raw["manage_object"] = raw.get("ManageObject", raw.iloc[:, 2]).astype(str)
    raw = raw.dropna(subset=["timestamp"])
    raw["date"] = raw["timestamp"].dt.date
    raw = raw.sort_values(["manage_object", "timestamp"])

    if COL_CUMULATIVE in raw.columns:
        raw["irr_raw"]      = pd.to_numeric(raw[COL_CUMULATIVE], errors="coerce").fillna(0)
        raw["interval_irr"] = _daily_from_cumulative(raw, ["manage_object", "date"], "irr_raw")
    elif COL_INSTANT in raw.columns:
        raw["irr_raw"]      = pd.to_numeric(raw[COL_INSTANT], errors="coerce").fillna(0)
        raw["interval_irr"] = (raw["irr_raw"] * 0.25 / 1000).clip(lower=0)   # W/m² → kWh/m²
    else:
        raise ValueError(f"No recognised irradiation column in {path.name}. "
                         f"Columns: {list(raw.columns)}")

    return (
        raw.groupby("date", as_index=False)
           .agg(irradiation_kwh_m2=("interval_irr", "sum"))
    )


def load_irradiation_daily() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year_dir in sorted(IRR_DIR.iterdir()):
        files = sorted(year_dir.glob("*.xlsx"))
        print(f"  {year_dir.name}: reading {len(files)} files", end="", flush=True)
        for f in files:
            frames.append(_read_irr_file(f))
            print(".", end="", flush=True)
        print()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # Average across sensors if multiple exist for the same day
    df = (df.groupby("date", as_index=False)
            .agg(irradiation_kwh_m2=("irradiation_kwh_m2", "mean"))
            .sort_values("date")
            .reset_index(drop=True))
    print(f"  → {len(df):,} daily rows  |  "
          f"{df['date'].min().date()} – {df['date'].max().date()}")
    return df


# ── transform ─────────────────────────────────────────────────────────────────

def build_transformed(inv: pd.DataFrame,
                       irr: pd.DataFrame,
                       cap: pd.DataFrame) -> pd.DataFrame:
    df = inv.merge(cap[["device_name", "installed_capacity_kwp"]], on="device_name", how="left")
    df = df.merge(irr, on="date", how="left")

    # PR = Yield(kWh) / (Irradiation(kWh/m²) × Installed DC capacity(kWp))
    denom = df["irradiation_kwh_m2"] * df["installed_capacity_kwp"]
    df["performance_ratio"] = (
        df["yield_kwh"] / denom.replace(0, float("nan"))
    ).round(4)

    return df[[
        "zone", "date", "device_name", "installed_capacity_kwp",
        "yield_kwh", "irradiation_kwh_m2", "performance_ratio",
    ]].sort_values(["zone", "date", "device_name"]).reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Installed capacity")
    cap = load_capacity()

    print("\n[2/4] Inverter daily yield")
    inv = load_inverter_daily()

    print("\n[3/4] Irradiation daily")
    irr = load_irradiation_daily()

    print("\n[4/4] Building transformed dataset")
    tfm = build_transformed(inv, irr, cap)

    inv_path = OUTPUT_DIR / "inverter_daily.csv"
    irr_path = OUTPUT_DIR / "irradiation_daily.csv"
    tfm_path = OUTPUT_DIR / "transformed.csv"

    inv.to_csv(inv_path, index=False, encoding="utf-8-sig")
    irr.to_csv(irr_path, index=False, encoding="utf-8-sig")
    tfm.to_csv(tfm_path, index=False, encoding="utf-8-sig")

    print(f"\n── Done ──")
    print(f"  {inv_path}")
    print(f"  {irr_path}")
    print(f"  {tfm_path}")
    print(f"\n  Rows             : {len(tfm):,}")
    print(f"  Devices          : {tfm['device_name'].nunique()}")
    print(f"  Date range       : {tfm['date'].min().date()} – {tfm['date'].max().date()}")
    print(f"  Missing capacity : {tfm['installed_capacity_kwp'].isna().sum():,}")
    print(f"  Missing irr      : {tfm['irradiation_kwh_m2'].isna().sum():,}")
    print(f"  Missing PR       : {tfm['performance_ratio'].isna().sum():,}")
    print()
    print(tfm.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
