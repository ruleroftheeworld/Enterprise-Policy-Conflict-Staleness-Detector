import csv
import sys
import codecs
from pathlib import Path

# Resolve repo root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "obligation_extracts_labels.csv"

def main():
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    
    # Load all raw sentences from all policies
    all_raw_sentences = set()
    for path in policy_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("- "):
                    # strip leading "- " and trailing punctuation/spaces
                    all_raw_sentences.add(line[2:].strip().lower())

    # Load label CSV
    labels = []
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(row)

    print(f"Total raw policy sentences found: {len(all_raw_sentences)}")
    print(f"Total labeled obligations: {len(labels)}")

    exact_matches = 0
    contains_matches = 0
    for lbl in labels:
        txt = lbl["obligation_text"].strip().lower()
        if txt in all_raw_sentences:
            exact_matches += 1
        else:
            # check if it is a substring of any raw sentence
            found = False
            for rs in all_raw_sentences:
                if txt in rs or rs in txt:
                    found = True
                    break
            if found:
                contains_matches += 1

    print(f"Exact string matches: {exact_matches}")
    print(f"Partial/contains matches: {contains_matches}")

    # Let's print a sample of the raw policy sentences
    print("\nSample of raw policy sentences:")
    for rs in list(all_raw_sentences)[:10]:
        print(f"  - {rs}")

if __name__ == "__main__":
    main()
