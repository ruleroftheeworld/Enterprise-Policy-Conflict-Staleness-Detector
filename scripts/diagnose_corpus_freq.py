# -*- coding: utf-8 -*-
"""
diagnose_corpus_freq.py
======================
Computes corpus frequency for all obligations and analyzes ground-truth
redundancy pairs to help determine the boilerplate cutoff.
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

    print("Running pipeline to load and embed obligations...")
    result = analyze_policy_documents(policy_paths, llm_provider=None)
    obligations = result.obligations
    policies = result.policies

    print(f"Loaded {len(obligations)} obligations across {len(policies)} policies.")

    # Map policy_id to filename basename
    policy_id_to_file = {p.policy_id: Path(p.source_file).name for p in policies}

    # Group obligations by policy_id for easy lookups
    policy_to_obls: dict[str, list[NormalizedObligation]] = {}
    for obl in obligations:
        policy_to_obls.setdefault(obl.policy_id, []).append(obl)

    # 1. Compute corpus_frequency for each obligation
    # For obligation obl in policy A, count how many distinct policies contain a near-duplicate.
    # A near-duplicate is defined as:
    #   - normalized text match OR
    #   - cosine similarity >= 0.95
    # obligation.corpus_frequency is the count of distinct policies.
    print("Computing corpus frequencies...")
    
    # Precompute normalized texts to save time
    normalized_texts = {obl.obligation_id: normalize_text(obl.sentence_text) for obl in obligations}

    obl_to_freq: dict[str, int] = {}
    
    for i, obl_a in enumerate(obligations):
        matching_policies = {obl_a.policy_id} # starts with its own policy (count 1)
        norm_a = normalized_texts[obl_a.obligation_id]
        emb_a = obl_a.embedding

        for obl_b in obligations:
            if obl_b.policy_id == obl_a.policy_id:
                continue
            if obl_b.policy_id in matching_policies:
                # already matched this policy
                continue

            # check normalized text first (cheap)
            norm_b = normalized_texts[obl_b.obligation_id]
            if norm_a == norm_b:
                matching_policies.add(obl_b.policy_id)
                continue

            # check cosine similarity (if embeddings are present)
            if emb_a and obl_b.embedding:
                sim = cosine_similarity(emb_a, obl_b.embedding)
                if sim >= 0.95:
                    matching_policies.add(obl_b.policy_id)

        obl_to_freq[obl_a.obligation_id] = len(matching_policies)

    # 2. Report distribution of corpus frequencies (histogram)
    # Categories: 1, 2, 3, 4-6, 7-10, 11+
    histogram = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4-6": 0,
        "7-10": 0,
        "11+": 0
    }
    for obl_id, freq in obl_to_freq.items():
        if freq == 1:
            histogram["1"] += 1
        elif freq == 2:
            histogram["2"] += 1
        elif freq == 3:
            histogram["3"] += 1
        elif 4 <= freq <= 6:
            histogram["4-6"] += 1
        elif 7 <= freq <= 10:
            histogram["7-10"] += 1
        else:
            histogram["11+"] += 1

    print("\n" + "=" * 60)
    print("OBLIGATION CORPUS FREQUENCY HISTOGRAM")
    print("=" * 60)
    for cat, count in histogram.items():
        print(f"  Corpus Frequency {cat:<6}: {count} obligations")

    # 3. Cross-reference with known 20 ground-truth REDUNDANCY policy-pairs
    # Load ground truth redundancy labels
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

    # Map filename basenames back to policy_id
    file_to_policy_id = {Path(p.source_file).name: p.policy_id for p in policies}

    print("\n" + "=" * 60)
    print("GROUND TRUTH REDUNDANCY ANALYSIS")
    print("=" * 60)
    print(f"Found {len(redundancies)} redundancy rows in CSV.")

    print(f"{'Policy Pair':<45} | {'Max Cos':<7} | {'Freq A':<6} | {'Freq B':<6} | {'Subj/ObjMatch?'}")
    print("-" * 90)

    for pol_a, pol_b, desc in redundancies:
        id_a = file_to_policy_id.get(pol_a)
        id_b = file_to_policy_id.get(pol_b)
        
        if not id_a or not id_b:
            print(f"Warning: policy ids not found for {pol_a} or {pol_b}")
            continue

        obls_a = policy_to_obls.get(id_a, [])
        obls_b = policy_to_obls.get(id_b, [])

        best_sim = -1.0
        best_pair = (None, None)

        for oa in obls_a:
            for ob in obls_b:
                # highest cosine similarity same-topic pair as proxy
                if oa.topic == ob.topic:
                    sim = cosine_similarity(oa.embedding, ob.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (oa, ob)

        oa, ob = best_pair
        if oa and ob:
            freq_a = obl_to_freq[oa.obligation_id]
            freq_b = obl_to_freq[ob.obligation_id]
            subj_obj_match = (oa.subject == ob.subject and oa.action == ob.action and oa.object == ob.object)
            print(f"{pol_a + ' <-> ' + pol_b:<45} | {best_sim:.4f} | {freq_a:<6} | {freq_b:<6} | {str(subj_obj_match)}")
            # Let's print the texts too if they have high or low frequency
            if freq_a >= 4 or freq_b >= 4:
                print(f"    [High Freq text A]: {oa.sentence_text}")
                print(f"    [High Freq text B]: {ob.sentence_text}")
        else:
            print(f"{pol_a + ' <-> ' + pol_b:<45} | No same-topic pair found")

if __name__ == "__main__":
    main()
