import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
transformed = pd.read_csv(ROOT / 'input_data' / 'base' / 'transformed.csv', parse_dates=['date'])
trans_names = transformed['device_name'].dropna().unique()

missing = [
"HF12 Inverter 07",
"HF 11 Inverter 02",
"HF18 Inverter 09",
"HF02 Inverter 09",
"HF01 Inverter 01",
"HF02 Inverter 06",
"HF02 Inverter 07",
"HF02 Inverter 08",
"HF04 Inverter 05",
"HF12 Inverter 01",
"HF12 Inverter 07",
"HF15 Inverter 02",
"HF15 Inverter 05",
"HF21 Inverter 03",
"HF16 Inverter 02",
"100KTL-M1 (COM1-1)",
"HF16 Inverter 09",
"HF16 Inverter 09",
"HF22 Inverter 08",
"HF02 Inverter 01",
"HF10 Inverter 06",
"HF09 Inverter 09",
"HF05 Inverter 04",
"HF02 Inverter 01",
]

for m in sorted(set(missing)):
    m_norm = ''.join(ch.lower() for ch in m if ch.isalnum())
    matches = [n for n in trans_names if m_norm in ''.join(ch.lower() for ch in (n if isinstance(n,str) else '') if ch.isalnum())]
    print(f"\nMissing: {m}")
    if matches:
        for s in matches[:10]:
            print("  ->", s)
    else:
        # also try substring search
        subs = [n for n in trans_names if (isinstance(n,str) and m.split()[0] in n)]
        if subs:
            for s in subs[:10]:
                print("  ~>", s)
        else:
            print("  (no match found)")
