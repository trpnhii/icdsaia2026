"""Transform exported CSV data by selecting and merging required columns."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input_data"
OUTPUT_DIR = PROJECT_ROOT / "input_data"

INVERTER_COLUMNS = [
    "Plant Name",
    "Device Name",
    "Total String Capacity (kWp)",
    "Yield (kWh)",
]
IRRADIATION_COLUMNS = [
    "Global Irradiation (kWh/㎡)",
]
OUTPUT_COLUMNS = INVERTER_COLUMNS + IRRADIATION_COLUMNS


def to_snake_case(name: str) -> str:
    name = name.strip()
    name = name.replace("㎡", "m2").replace("²", "2")
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip().lower()
    name = name.replace(" ", "_")
    return re.sub(r"_+", "_", name)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(column) for column in df.columns]
    return df


def _parse_inverter_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d-%m-%Y")


def _parse_irradiation_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series)


def transform_data(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    inverter_df = pd.read_csv(input_dir / "inverter_report.csv")
    irradiation_df = pd.read_csv(input_dir / "irradiation.csv")

    inverter_df = inverter_df[INVERTER_COLUMNS + ["zone", "report_date"]].copy()
    irradiation_df = irradiation_df[["Statistical Period", *IRRADIATION_COLUMNS]].copy()

    inverter_df["date"] = _parse_inverter_date(inverter_df["report_date"])
    irradiation_df["date"] = _parse_irradiation_date(irradiation_df["Statistical Period"])

    transformed_df = inverter_df.merge(
        irradiation_df[["date", *IRRADIATION_COLUMNS]],
        on="date",
        how="left",
    )
    transformed_df["date"] = transformed_df["date"].dt.strftime("%Y-%m-%d")
    transformed_df = transformed_df[["zone", "date", *OUTPUT_COLUMNS]]
    transformed_df = _normalize_column_names(transformed_df)

    # Performance Ratio = Yield / (Irradiation × Capacity)
    # Only defined when irradiation is available and capacity > 0.
    irr_col = "global_irradiation_kwh_m2"
    cap_col = "total_string_capacity_kwp"
    yield_col = "yield_kwh"
    denominator = transformed_df[irr_col] * transformed_df[cap_col]
    transformed_df["performance_ratio"] = (
        transformed_df[yield_col] / denominator.replace(0, float("nan"))
    ).round(4)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transformed_data.csv"
    transformed_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def main() -> None:
    output_path = transform_data()
    print(f"Exported transformed data -> {output_path}")


if __name__ == "__main__":
    main()
