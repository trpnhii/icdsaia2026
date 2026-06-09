import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
trans = pd.read_csv(ROOT / 'input_data' / 'base' / 'transformed.csv', dtype=str)
names = trans['device_name'].dropna().unique().tolist()
res = [n for n in names if isinstance(n,str) and ('100' in n or 'ktl' in n.lower() or 'com1' in n.lower())]
print(len(res))
for r in res[:50]:
    print(r)
