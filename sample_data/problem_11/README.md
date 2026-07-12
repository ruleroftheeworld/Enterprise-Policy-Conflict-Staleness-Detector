# Problem 11 Sample Data (Policy Conflict & Staleness Detector)

This folder contains the regenerated full sample dataset for Problem 11.

## Files
- `policies/` (30 markdown policy documents)
- `policy_metadata.csv` and `policy_metadata.json`
- `obligation_extracts_labels.csv` and `obligation_extracts_labels.json`
- `findings_labels.csv` and `findings_labels.json`
- `generate_ps11_data.py`

## Counts
| File | Records |
|---|---|
| `policies/` | 30 documents |
| `policy_metadata.csv/.json` | 30 policies |
| `obligation_extracts_labels.csv/.json` | 350 obligations |
| `findings_labels.csv/.json` | 80 findings |

## Findings distribution (`finding_subtype`)
- `DIRECT_CONFLICT`: 15
- `PARTIAL_CONFLICT`: 10
- `REDUNDANCY`: 20
- `STALE_POLICY`: 12
- `STALE_REFERENCE`: 8
- `FALSE_POSITIVE_PRONE`: 15

## Notes
- The main label category remains in `finding_type` (`CONFLICT`, `REDUNDANCY`, `STALE`).
- The required detailed split is provided in `finding_subtype`.
- Findings use two schemas: `policy` (single-policy findings) and `policy_a`/`policy_b` (cross-policy findings).
- Running `generate_ps11_data.py` refreshes all CSV/JSON files and rewrites exactly 30 policy markdown files.

## Quick Python conversion example
```python
import json
import pandas as pd

for name in ["policy_metadata", "obligation_extracts_labels", "findings_labels"]:
    with open(f"{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    pd.DataFrame(data).to_csv(f"{name}.csv", index=False)
```
