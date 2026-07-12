import csv
import sys
import codecs
from pathlib import Path

# Resolve repo root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.pipeline import analyze_policy_documents
from shared.contracts.policy_analysis import NormalizedObligation

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "obligation_extracts_labels.csv"

def main():
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    result = analyze_policy_documents(policy_paths, llm_provider=None)
    extracted_obls = result.obligations
    policies = result.policies

    # Build policy_id -> filename basename mapping
    policy_id_to_file = {p.policy_id: Path(p.source_file).name for p in policies}

    # Group extracted obligations by policy file name
    extracted_by_policy: dict[str, list[NormalizedObligation]] = {}
    for o in extracted_obls:
        policy_name = policy_id_to_file.get(o.policy_id, o.policy_id)
        extracted_by_policy.setdefault(policy_name, []).append(o)

    # Load label CSV
    labels = []
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(row)

    missed_by_strength = {}
    total_by_strength = {}

    for idx, lbl in enumerate(labels):
        policy_file = lbl["policy_file"]
        topic = lbl["topic"]
        strength = lbl["strength"]

        total_by_strength[strength] = total_by_strength.get(strength, 0) + 1

        candidates = extracted_by_policy.get(policy_file, [])
        match_found = False
        for ext in candidates:
            if ext.topic == topic:
                match_found = True
                break

        if not match_found:
            missed_by_strength[strength] = missed_by_strength.get(strength, 0) + 1

    print("Missed Obligations Breakdown by Labeled Strength:")
    print(f"{'Strength':<15} | {'Missed':<6} | {'Total':<6} | {'Recall':<6}")
    print("-" * 45)
    for strength in sorted(total_by_strength.keys()):
        total = total_by_strength[strength]
        missed = missed_by_strength.get(strength, 0)
        matched = total - missed
        recall = matched / total if total > 0 else 0.0
        print(f"{strength:<15} | {missed:<6} | {total:<6} | {recall*100:.2f}%")

if __name__ == "__main__":
    main()
