import csv
import sys
import codecs
from pathlib import Path

# Resolve repo root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "obligation_extracts_labels.csv"

def main():
    topics = set()
    strengths = set()
    scopes = set()
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            topics.add(row.get("topic", ""))
            strengths.add(row.get("strength", ""))
            scopes.add(row.get("scope", ""))
            
    print(f"Unique Topics in CSV: {sorted(list(topics))}")
    print(f"Unique Strengths in CSV: {sorted(list(strengths))}")
    print(f"Unique Scopes in CSV: {sorted(list(scopes))}")

if __name__ == "__main__":
    main()
