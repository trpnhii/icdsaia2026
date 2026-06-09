import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
candidates = pd.read_csv(ROOT / 'input_data' / 'risk' / 'replacement_candidates.csv', parse_dates=['risk_date'])
transformed = pd.read_csv(ROOT / 'input_data' / 'base' / 'transformed.csv', parse_dates=['date'])
cap = transformed[['zone','device_name','installed_capacity_kwp']].drop_duplicates()
# Merge on device_name only to avoid zone type mismatches; show both zones for inspection
merged = candidates.merge(cap, on=['device_name'], how='left', suffixes=('_cand','_trans'))
missing = merged[merged['installed_capacity_kwp'].isna()]
print(f"Total candidate rows: {len(candidates)}")
print(f"Missing capacity rows: {len(missing)}")
if not missing.empty:
    print(missing[['risk_date','zone_cand','zone_trans','device_name']].drop_duplicates().to_string(index=False))
else:
    print('No missing devices')
