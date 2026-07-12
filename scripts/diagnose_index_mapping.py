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

def read_policy_sentences(policy_name: str) -> list[str]:
    path = _POLICY_DIR / policy_name
    if not path.exists():
        return []
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                sentences.append(line[2:].strip())
    return sentences

def main():
    # Load label CSV and group by policy_file
    labels_by_policy = {}
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pfile = row["policy_file"]
            labels_by_policy.setdefault(pfile, []).append(row)

    print("Index mapping comparison:")
    print(f"{'Policy File':<15} | {'Sentences Count':<15} | {'Labels Count':<12}")
    print("-" * 50)
    for pfile in sorted(labels_by_policy.keys()):
        sents = read_policy_sentences(pfile)
        lbls = labels_by_policy[pfile]
        print(f"{pfile:<15} | {len(sents):<15} | {len(lbls):<12}")

    # Let's inspect policy_01.md index by index
    print("\npolicy_01.md Detail:")
    sents = read_policy_sentences("policy_01.md")
    lbls = labels_by_policy["policy_01.md"]
    for i in range(max(len(sents), len(lbls))):
        sent_str = sents[i] if i < len(sents) else "N/A"
        lbl_row = lbls[i] if i < len(lbls) else {}
        lbl_str = f"Text: \"{lbl_row.get('obligation_text', 'N/A')}\" | Topic: {lbl_row.get('topic')} | Strength: {lbl_row.get('strength')} | Scope: {lbl_row.get('scope')}"
        print(f"  Index {i+1}:")
        print(f"    Raw sentence: {sent_str}")
        print(f"    Label obligation: {lbl_str}")

if __name__ == "__main__":
    main()
