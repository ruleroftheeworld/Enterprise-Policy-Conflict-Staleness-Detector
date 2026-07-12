# -*- coding: utf-8 -*-
"""
diagnose_redundancy_misses.py
=============================
Diagnoses why genuine REDUNDANCY labels are not detected.
"""
from __future__ import annotations

import csv
import sys
import codecs
import re
from pathlib import Path
import numpy as np

# Resolve repo root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.pipeline import analyze_policy_documents
from engine.candidates.candidate_generator import cosine_similarity
from shared.contracts.policy_analysis import NormalizedObligation
from engine.detection.deterministic_detector import (
    _subjects_compatible,
    _scope_compatible,
    _technology_compatible,
)

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "findings_labels.csv"

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def main():
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    if not policy_paths:
        sys.exit(f"ERROR: no .md files found in {_POLICY_DIR}")

    print("Loading pipeline to embed obligations and calculate frequencies...")
    result = analyze_policy_documents(policy_paths, llm_provider=None)
    obligations = result.obligations
    policies = result.policies

    # Group obligations by policy filename
    policy_id_to_file = {p.policy_id: Path(p.source_file).name for p in policies}
    file_to_policy_id = {Path(p.source_file).name: p.policy_id for p in policies}

    policy_to_obls: dict[str, list[NormalizedObligation]] = {}
    for obl in obligations:
        policy_to_obls.setdefault(obl.policy_id, []).append(obl)

    # Compute corpus frequencies
    normalized_texts = {obl.obligation_id: normalize_text(obl.sentence_text) for obl in obligations}
    obl_to_freq: dict[str, int] = {}
    
    for obl_a in obligations:
        matching_policies = {obl_a.policy_id}
        norm_a = normalized_texts[obl_a.obligation_id]
        emb_a = obl_a.embedding

        for obl_b in obligations:
            if obl_b.policy_id == obl_a.policy_id:
                continue
            if obl_b.policy_id in matching_policies:
                continue

            norm_b = normalized_texts[obl_b.obligation_id]
            if norm_a == norm_b:
                matching_policies.add(obl_b.policy_id)
                continue

            if emb_a and obl_b.embedding:
                sim = cosine_similarity(emb_a, obl_b.embedding)
                if sim >= 0.95:
                    matching_policies.add(obl_b.policy_id)

        obl_to_freq[obl_a.obligation_id] = len(matching_policies)

    # Load REDUNDANCY labels
    redundancies = []
    with codecs.open(str(_LABELS_CSV), "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ft = row.get("finding_type", "").strip().upper()
            pol1 = row.get("policy_a", "").strip()
            pol2 = row.get("policy_b", "").strip()
            desc = row.get("description", "").strip()
            if ft == "REDUNDANCY" and pol1 and pol2:
                redundancies.append((pol1, pol2, desc))

    print(f"\nAnalyzing {len(redundancies)} REDUNDANCY policy pairs...")
    print("=" * 120)

    for idx, (pol_a, pol_b, desc) in enumerate(redundancies):
        id_a = file_to_policy_id.get(pol_a)
        id_b = file_to_policy_id.get(pol_b)
        
        if not id_a or not id_b:
            print(f"\nRow {idx+1}: policy ids not found for {pol_a} or {pol_b}")
            continue

        # Find non-boilerplate obligations (freq < 5)
        obls_a = [o for o in policy_to_obls.get(id_a, []) if obl_to_freq[o.obligation_id] < 5]
        obls_b = [o for o in policy_to_obls.get(id_b, []) if obl_to_freq[o.obligation_id] < 5]

        print(f"\nRow {idx+1}: {pol_a} <-> {pol_b} | Description: {desc}")
        print(f"  Available non-boilerplate obligations in {pol_a}: {len(obls_a)}")
        print(f"  Available non-boilerplate obligations in {pol_b}: {len(obls_b)}")

        if not obls_a or not obls_b:
            print("  [ERROR] Cannot check similarity: one or both policies have 0 non-boilerplate obligations.")
            continue

        best_sim = -1.0
        best_pair = (None, None)

        for oa in obls_a:
            for ob in obls_b:
                if oa.embedding and ob.embedding:
                    sim = cosine_similarity(oa.embedding, ob.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (oa, ob)

        oa, ob = best_pair
        if oa and ob:
            print(f"  Highest similarity pair (Cos = {best_sim:.4f}):")
            print(f"    Obl A: \"{oa.sentence_text}\"")
            print(f"           Action: {oa.action!r} | Object: {oa.object!r} | Topic: {oa.topic!r} | Freq: {obl_to_freq[oa.obligation_id]}")
            print(f"    Obl B: \"{ob.sentence_text}\"")
            print(f"           Action: {ob.action!r} | Object: {ob.object!r} | Topic: {ob.topic!r} | Freq: {obl_to_freq[ob.obligation_id]}")

            # Check _is_redundancy checks
            same_action = oa.action == ob.action
            same_object = oa.object == ob.object
            subj_comp = _subjects_compatible(oa, ob)
            scope_comp = _scope_compatible(oa, ob)
            tech_comp = _technology_compatible(oa, ob)
            mod_match = oa.modality == ob.modality
            neg_match = oa.negated == ob.negated
            freq_match = oa.frequency == ob.frequency
            cosine_pass = best_sim >= 0.95

            passes_all = (
                same_action and same_object and subj_comp and scope_comp and 
                tech_comp and mod_match and neg_match and freq_match and cosine_pass
            )

            print(f"  Gates:")
            print(f"    1. same_action (exact)         : {'PASS' if same_action else 'FAIL'}")
            print(f"    2. same_object (exact)         : {'PASS' if same_object else 'FAIL'}")
            print(f"    3. _subjects_compatible        : {'PASS' if subj_comp else 'FAIL'}")
            print(f"    4. _scope_compatible           : {'PASS' if scope_comp else 'FAIL'}")
            print(f"    5. _technology_compatible      : {'PASS' if tech_comp else 'FAIL'}")
            print(f"    6. modality matches            : {'PASS' if mod_match else 'FAIL'} (A: {oa.modality!r}, B: {ob.modality!r})")
            print(f"    7. negation matches            : {'PASS' if neg_match else 'FAIL'} (A: {oa.negated!r}, B: {ob.negated!r})")
            print(f"    8. frequency matches           : {'PASS' if freq_match else 'FAIL'} (A: {oa.frequency!r}, B: {ob.frequency!r})")
            print(f"    9. cosine_similarity >= 0.95   : {'PASS' if cosine_pass else 'FAIL'}")
            print(f"  RESULT: {'PASS' if passes_all else 'FAIL'}")
        else:
            print("  [ERROR] No valid similarity pair found.")

if __name__ == "__main__":
    main()
