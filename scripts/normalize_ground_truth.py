import pandas as pd
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / 'input_data' / 'risk' / 'ground_truth_replacements.csv'
TRANS = ROOT / 'input_data' / 'base' / 'transformed.csv'
OUT = ROOT / 'input_data' / 'risk' / 'ground_truth_replacements.mapped.csv'

def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    s = re.sub(r"0+(\d)", r"\1", s)  # remove leading zeros before digits
    return s


gt = pd.read_csv(GT, dtype=str)
trans = pd.read_csv(TRANS, dtype=str)
trans_names = trans['device_name'].dropna().unique().tolist()

mapping = {}
for idx, row in gt.iterrows():
    dev = str(row['device_name']).strip()
    norm = normalize(dev)
    # find exact normalized matches
    candidates = [t for t in trans_names if normalize(t) == norm]
    if len(candidates) == 1:
        mapping[dev] = candidates[0]
    elif len(candidates) > 1:
        mapping[dev] = candidates[0]
    else:
        # try contains heuristic
        parts = re.findall(r"[a-z]+|\d+", dev.lower())
        nums = [p for p in parts if p.isdigit()]
        found = []
        for t in trans_names:
            tl = t.lower()
            ok = True
            for n in nums:
                if n not in tl:
                    ok = False
                    break
            if ok and parts[0] in tl:
                found.append(t)
        if len(found) == 1:
            mapping[dev] = found[0]
        elif len(found) > 1:
            mapping[dev] = found[0]
        else:
            mapping[dev] = None

# create mapped dataframe
mapped_rows = []
for idx, row in gt.iterrows():
    dev = str(row['device_name']).strip()
    mapped = mapping.get(dev)
    new_dev = mapped if mapped else dev
    mapped_rows.append({**row, 'device_name': new_dev})

mapped_df = pd.DataFrame(mapped_rows)
mapped_df.to_csv(OUT, index=False)

print('Mappings:')
for k,v in mapping.items():
    print(f"{k} -> {v}")
print(f"Wrote mapped ground-truth to: {OUT}")
