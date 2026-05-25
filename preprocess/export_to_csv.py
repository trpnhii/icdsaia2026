"""Export Irradiation and Inverter Report Excel files to CSV."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "Data_raw"
INVERTER_DIR = DATA_RAW_DIR / "Inverter Report"
IRRADIATION_FILE = DATA_RAW_DIR / "Irradiation.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "input_data"

INVERTER_FILENAME_PATTERN = re.compile(
    r"Zone(?P<zone>\d+)_Inverter Report_(?P<report_date>\d{2}-\d{2}-\d{4})\.xlsx$"
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_inverter_reports(inverter_dir: Path) -> pd.DataFrame:
    if not inverter_dir.is_dir():
        raise FileNotFoundError(f"Inverter report folder not found: {inverter_dir}")

    frames: list[pd.DataFrame] = []
    for file_path in sorted(inverter_dir.glob("Zone*_Inverter Report_*.xlsx")):
        match = INVERTER_FILENAME_PATTERN.match(file_path.name)
        if match is None:
            continue

        df = pd.read_excel(file_path, sheet_name=0, header=1)
        df = _clean_columns(df)
        df["zone"] = int(match.group("zone"))
        df["report_date"] = match.group("report_date")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No inverter report files found in: {inverter_dir}")

    return pd.concat(frames, ignore_index=True)


def load_irradiation(irradiation_file: Path) -> pd.DataFrame:
    if not irradiation_file.is_file():
        raise FileNotFoundError(f"Irradiation file not found: {irradiation_file}")

    excel_file = pd.ExcelFile(irradiation_file)
    frames: list[pd.DataFrame] = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(irradiation_file, sheet_name=sheet_name, header=1)
        if df.empty:
            continue

        df = _clean_columns(df)
        df["source_sheet"] = sheet_name
        frames.append(df)

    if not frames:
        raise ValueError(f"No data sheets found in: {irradiation_file}")

    return pd.concat(frames, ignore_index=True)


def export_to_csv(
    data_raw_dir: Path = DATA_RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inverter_df = load_inverter_reports(data_raw_dir / "Inverter Report")
    irradiation_df = load_irradiation(data_raw_dir / "Irradiation.xlsx")

    inverter_csv = output_dir / "inverter_report.csv"
    irradiation_csv = output_dir / "irradiation.csv"

    inverter_df.to_csv(inverter_csv, index=False, encoding="utf-8-sig")
    irradiation_df.to_csv(irradiation_csv, index=False, encoding="utf-8-sig")

    return inverter_csv, irradiation_csv


def main() -> None:
    inverter_csv, irradiation_csv = export_to_csv()
    print(f"Exported inverter report -> {inverter_csv}")
    print(f"Exported irradiation     -> {irradiation_csv}")


if __name__ == "__main__":
    main()
