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
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    policy_sentences = {path.name: read_policy_sentences(path.name) for path in policy_paths}

    # Group labels by policy_file
    labels_by_policy = {}
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pfile = row["policy_file"]
            labels_by_policy.setdefault(pfile, []).append(row)

    print("Checking if labels are shuffled across policies:")
    # For each label group, find the policy file that has the highest overlap of topics/keywords
    for lbl_pfile, lbls in sorted(labels_by_policy.items()):
        lbl_topics = {lbl["topic"] for lbl in lbls}
        
        best_match = None
        best_overlap = -1
        best_overlap_topics = []

        for pfile, sents in policy_sentences.items():
            # Extract raw words/topics from sentences
            # simple topic mapping based on keywords
            from engine.extraction.obligation_extractor import _classify_topic
            p_topics = set()
            for s in sents:
                t = _classify_topic(s)
                if t:
                    p_topics.add(t)

            overlap = len(lbl_topics & p_topics)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = pfile
                best_overlap_topics = list(lbl_topics & p_topics)

        print(f"Labels for {lbl_pfile:<15} -> Best raw policy match: {best_match:<15} (Overlap: {best_overlap}/12, Topics: {best_overlap_topics})")

if __name__ == "__main__":
    main()
