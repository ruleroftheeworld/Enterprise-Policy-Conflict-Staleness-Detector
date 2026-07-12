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

    print("Sample Matches:")
    match_count = 0
    for idx, lbl in enumerate(labels):
        policy_file = lbl["policy_file"]
        topic = lbl["topic"]
        lbl_text = lbl["obligation_text"]

        candidates = extracted_by_policy.get(policy_file, [])
        for ext in candidates:
            if ext.topic == topic:
                match_count += 1
                print(f"\nMatch #{match_count}: Row {idx+2} in CSV")
                print(f"  Label file: {policy_file} | Topic: {topic}")
                print(f"  Label text: \"{lbl_text}\" | Strength: {lbl['strength']} | Scope: {lbl['scope']}")
                print(f"  Extracted text: \"{ext.sentence_text}\"")
                print(f"  Extracted details: Action: {ext.action!r} | Object: {ext.object!r} | Subject: {ext.subject!r} | Modality: {ext.modality!r} | Negated: {ext.negated!r} | Scope: {ext.scope!r}")
                if match_count >= 10:
                    return

if __name__ == "__main__":
    main()
