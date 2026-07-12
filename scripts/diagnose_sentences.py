import csv
import sys
import codecs
import re
from pathlib import Path

# Resolve repo root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "findings_labels.csv"

def read_policy_sentences(policy_name: str) -> list[str]:
    path = _POLICY_DIR / policy_name
    if not path.exists():
        return []
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                sentences.append(line[2:])
    return sentences

def main():
    redundancies = []
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ft = row.get("finding_type", "").strip().upper()
            pol1 = row.get("policy_a", "").strip()
            pol2 = row.get("policy_b", "").strip()
            desc = row.get("description", "").strip()
            expl = row.get("explanation", "").strip()
            if ft == "REDUNDANCY" and pol1 and pol2:
                redundancies.append((pol1, pol2, desc, expl))

    print(f"Loaded {len(redundancies)} REDUNDANCY labels from CSV.\n")

    for idx, (pol_a, pol_b, desc, expl) in enumerate(redundancies):
        print(f"Row {idx+1}: {pol_a} <-> {pol_b}")
        print(f"  Description: {desc}")
        print(f"  Explanation: {expl}")

        # Parse key keyword from explanation, e.g. "Both require the same action on encryption"
        # Extract last word (or match against known topics)
        match = re.search(r'action on ([a-zA-Z0-9_-]+)', expl)
        keyword = match.group(1) if match else ""
        print(f"  Target Keyword: {keyword!r}")

        sents_a = read_policy_sentences(pol_a)
        sents_b = read_policy_sentences(pol_b)

        # Print all sentences matching the keyword in policy A
        matching_a = [s for s in sents_a if keyword in s.lower()]
        matching_b = [s for s in sents_b if keyword in s.lower()]

        # If keyword is "third-party", also check for "vendor" and vice versa
        if keyword == "third-party":
            matching_a.extend([s for s in sents_a if "vendor" in s.lower() and s not in matching_a])
            matching_b.extend([s for s in sents_b if "vendor" in s.lower() and s not in matching_b])
        elif keyword == "vendor":
            matching_a.extend([s for s in sents_a if "third-party" in s.lower() and s not in matching_a])
            matching_b.extend([s for s in sents_b if "third-party" in s.lower() and s not in matching_b])

        print(f"  Matching Sentences in {pol_a}:")
        for s in matching_a:
            print(f"    - {s}")
        print(f"  Matching Sentences in {pol_b}:")
        for s in matching_b:
            print(f"    - {s}")
        print("-" * 80)

if __name__ == "__main__":
    main()
