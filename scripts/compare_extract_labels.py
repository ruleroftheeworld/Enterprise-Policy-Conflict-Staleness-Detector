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

    print(f"Total labeled obligations: {len(labels)}")
    print(f"Total extracted obligations: {len(extracted_obls)}\n")

    # Match each label to an extracted obligation
    # Let's see if we can match them by policy_file and topic
    matched_count = 0
    missed_labels = []

    for idx, lbl in enumerate(labels):
        policy_file = lbl["policy_file"]
        topic = lbl["topic"]
        strength = lbl["strength"]
        scope = lbl["scope"]
        lbl_text = lbl["obligation_text"]

        # Find matching extracted obligations for this policy and topic
        candidates = extracted_by_policy.get(policy_file, [])
        match_found = False
        for ext in candidates:
            # If the extracted obligation's topic matches the label's topic, is it a match?
            if ext.topic == topic:
                match_found = True
                break

        if match_found:
            matched_count += 1
        else:
            missed_labels.append((idx + 2, lbl))

    print(f"Matched by (policy_file, topic): {matched_count} / {len(labels)} ({matched_count / len(labels) * 100:.2f}%)")
    print(f"Missed: {len(missed_labels)}")

    print("\nSample missed labels:")
    for row_num, lbl in missed_labels[:15]:
        print(f"  Line {row_num}: Policy: {lbl['policy_file']} | Text: {lbl['obligation_text']} | Topic: {lbl['topic']} | Strength: {lbl['strength']} | Scope: {lbl['scope']}")

if __name__ == "__main__":
    main()
