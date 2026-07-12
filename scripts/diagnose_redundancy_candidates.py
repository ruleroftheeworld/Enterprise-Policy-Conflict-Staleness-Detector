#!/usr/bin/env python
"""
diagnose_redundancy_candidates.py
==================================
Diagnose REDUNDANCY candidates ahead of the full metrics report.
Pulls all REDUNDANCY candidates with cosine similarity >= 0.90.
"""
from __future__ import annotations

import sys
import csv
import codecs
import warnings as py_warnings
from pathlib import Path

py_warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.pipeline import analyze_policy_documents
from engine.candidates import generate_candidate_pairs
from engine.embeddings import embed_obligations
from engine.extraction import extract_policy_obligations
from engine.normalization import normalize_obligations

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "findings_labels.csv"

def load_redundancy_labels() -> set[frozenset[str]]:
    redundancies = set()
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ft   = row.get("finding_type", "").strip().upper()
            pol1 = row.get("policy_a", "").strip()
            pol2 = row.get("policy_b", "").strip()
            if ft == "REDUNDANCY" and pol1 and pol2:
                redundancies.add(frozenset({pol1, pol2}))
    return redundancies

def main() -> None:
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    if not policy_paths:
        sys.exit(f"ERROR: no .md files found in {_POLICY_DIR}")

    # Run pipeline steps manually up to candidate generation
    from engine.ingestion import load_policy_document
    
    policies = []
    obligations = []
    for path in policy_paths:
        try:
            policy = load_policy_document(path)
            policies.append(policy)
            extracted = extract_policy_obligations(policy)
            normalized = normalize_obligations(extracted)
            obligations.extend(normalized)
        except Exception as e:
            print(f"Failed to process {path.name}: {e}")

    print("Embedding obligations...")
    embedded_obligations, _ = embed_obligations(obligations)

    print("Generating candidate pairs...")
    candidates = generate_candidate_pairs(
        embedded_obligations,
        similarity_threshold=0.90,
        top_k_per_obligation=None,
    )

    policy_id_to_file = {}
    for policy in policies:
        policy_id_to_file[policy.policy_id] = Path(policy.source_file).name

    redundancy_labels = load_redundancy_labels()

    # Filter candidate pairs that meet REDUNDANCY conditions:
    # 1. same target (same action, same object)
    # 2. same subject (subject compatible)
    # 3. same scope (scope compatible)
    # 4. same technology (technology compatible)
    # 5. same modality, same negation, same frequency
    # 6. cosine_similarity >= 0.90
    # Wait, the engine's REDUNDANCY check in deterministic_detector.py is:
    # return (
    #     first.modality == second.modality
    #     and first.negated == second.negated
    #     and first.frequency == second.frequency
    #     and candidate.cosine_similarity >= 0.90
    # )
    # Let's filter on these criteria or just use candidates >= 0.90 that match the target checks.
    # Actually, the user prompt says:
    # "pull all REDUNDANCY candidates with cosine similarity >= 0.90.
    # For each one, print side by side..."
    # Let's check how many candidate pairs satisfy the REDUNDANCY criteria.
    from engine.detection.deterministic_detector import _is_redundancy

    redundancy_candidates = []
    for cand in candidates:
        first = embedded_obligations[cand.source_index]
        second = embedded_obligations[cand.target_index]
        if _is_redundancy(first, second, cand):
            redundancy_candidates.append((cand, first, second))

    # Sort by cosine_similarity descending
    redundancy_candidates.sort(key=lambda x: x[0].cosine_similarity, reverse=True)

    print()
    print("=" * 100)
    print(f"REDUNDANCY CANDIDATES (cosine_similarity >= 0.90) - Total: {len(redundancy_candidates)}")
    print("=" * 100)

    tp_count = 0
    fp_count = 0

    for i, (cand, first, second) in enumerate(redundancy_candidates):
        policy_a = policy_id_to_file.get(first.policy_id, first.policy_id)
        policy_b = policy_id_to_file.get(second.policy_id, second.policy_id)
        pair = frozenset({policy_a, policy_b})
        is_labeled = pair in redundancy_labels

        if is_labeled:
            tp_count += 1
            label_status = "TRUE POSITIVE (in ground-truth)"
        else:
            fp_count += 1
            label_status = "FALSE POSITIVE (NOT in ground-truth)"

        print(f"\nCandidate #{i+1} | Cosine: {cand.cosine_similarity:.4f} | Status: {label_status}")
        print(f"  Policies: {policy_a} <-> {policy_b}")
        print(f"    Obligation A: {first.sentence_text}")
        print(f"    Obligation B: {second.sentence_text}")

    print("\n" + "=" * 100)
    print("SUMMARY STATS")
    print("=" * 100)
    print(f"Total REDUNDANCY predictions (cosine >= 0.90): {len(redundancy_candidates)}")
    print(f"  True Positives (match label pair)       : {tp_count}")
    print(f"  False Positives (no match label pair)   : {fp_count}")
    print("=" * 100)

if __name__ == "__main__":
    main()
