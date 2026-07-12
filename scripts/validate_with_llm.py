#!/usr/bin/env python
"""
validate_with_llm.py
====================
Runs validate_against_labels.py metrics calculation but with the
qwen2.5:3b LLM verifier enabled (cached for robustness), so the gate fix
precision impact is visible.
"""
from __future__ import annotations

import sys
import warnings as py_warnings
import time
import json
import csv
import codecs
import threading
from pathlib import Path

py_warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.pipeline import analyze_policy_documents
from engine.llm.ollama_provider import OllamaProvider

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "findings_labels.csv"
_CACHE_PATH = Path("C:/Users/JEEVIKA/.gemini/antigravity-ide/brain/ec674117-9203-4c6c-93ae-2be8c562ff70/scratch/llm_cache.json")

# ── same mapping as validate_against_labels.py ────────────────────────────────
ENGINE_TO_LABEL = {
    "CONTRADICTION":          ("CONFLICT",   "DIRECT_CONFLICT"),
    "FREQUENCY_MISMATCH":     ("CONFLICT",   "PARTIAL_CONFLICT"),
    "MODALITY_INCONSISTENCY": ("CONFLICT",   "PARTIAL_CONFLICT"),
    "REDUNDANCY":             ("REDUNDANCY", "REDUNDANCY"),
    "STALE_POLICY":           ("STALE",      "STALE_POLICY"),
    "STALE_REFERENCE":        ("STALE",      "STALE_REFERENCE"),
}

class CachedOllamaProvider:
    def __init__(self, provider: OllamaProvider, cache_path: Path):
        self.provider = provider
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} LLM responses from cache: {self.cache_path}")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def generate(self, prompt: str) -> str:
        with self.lock:
            if prompt in self.cache:
                return self.cache[prompt]
        
        res = self.provider.generate(prompt)
        
        with self.lock:
            self.cache[prompt] = res
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error writing cache: {e}")
        return res

def main() -> None:
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    if not policy_paths:
        sys.exit(f"ERROR: no .md files found in {_POLICY_DIR}")

    raw_provider = OllamaProvider(model="qwen2.5:3b")
    provider = CachedOllamaProvider(raw_provider, _CACHE_PATH)

    print("=" * 60)
    print("POLICY LABEL VALIDATION SCRIPT (with Cached LLM: qwen2.5:3b)")
    print("=" * 60)
    print(f"  Policy dir : {_POLICY_DIR}")
    print(f"  Labels CSV : {_LABELS_CSV}")
    print(f"Running pipeline on {len(policy_paths)} policy files …")

    start_time = time.time()
    result = analyze_policy_documents(policy_paths, llm_provider=provider)
    elapsed_time = time.time() - start_time

    policy_id_to_file: dict[str, str] = {}
    for policy in result.policies:
        filename = Path(policy.source_file).name
        policy_id_to_file[policy.policy_id] = filename

    stats = result.statistics
    print(f"  Policies parsed:   {len(result.policies)}")
    print(f"  Files failed:      {stats['documents_failed']}")
    print(f"  Obligations:       {len(result.obligations)}")
    print(f"  Raw findings:      {len(result.findings)}")
    print()
    print(f"  LLM gate stats:")
    print(f"    eligible  (0.50-0.95): {stats['llm_eligible_findings']}")
    print(f"    verified  (LLM=true) : {stats['llm_verified_findings']}")
    print(f"    rejected  (LLM=false): {stats['llm_rejected_findings']}")
    print(f"    bypassed  (>=0.95)   : {stats['llm_bypassed_findings']}")
    print(f"    dropped   (<0.50)    : {stats['llm_dropped_findings']}")

    # ── Load ground-truth labels ──────────────────────────────────────────────
    from typing import NamedTuple

    class FindingKey(NamedTuple):
        label_type: str
        label_subtype: str
        policy_key: frozenset | tuple

    def _finding_to_key(finding, policy_id_to_file) -> FindingKey | None:
        engine_type = finding.finding_type
        if engine_type not in ENGINE_TO_LABEL:
            return None
        label_type, label_subtype = ENGINE_TO_LABEL[engine_type]
        ev = finding.evidence or {}
        if label_type == "STALE":
            raw_policy_id = ev.get("policy_id")
            if raw_policy_id is None:
                return None
            filename = policy_id_to_file.get(raw_policy_id)
            if filename is None:
                return None
            return FindingKey(label_type, label_subtype, (filename,))
        else:
            src_id = ev.get("source_policy_id")
            tgt_id = ev.get("target_policy_id")
            if src_id is None or tgt_id is None:
                return None
            src_file = policy_id_to_file.get(src_id)
            tgt_file = policy_id_to_file.get(tgt_id)
            if src_file is None or tgt_file is None:
                return None
            if src_file == tgt_file:
                return None
            return FindingKey(label_type, label_subtype, frozenset({src_file, tgt_file}))

    def _build_predicted_keys(findings, policy_id_to_file) -> list[FindingKey]:
        keys = []
        for f in findings:
            key = _finding_to_key(f, policy_id_to_file)
            if key is not None:
                keys.append(key)
        return keys

    def _load_label_keys() -> list[tuple[FindingKey, dict]]:
        rows = []
        with open(_LABELS_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ftype    = (row.get("finding_type") or "").strip()
                fsubtype = (row.get("finding_subtype") or "").strip()
                policy_a = (row.get("policy_a") or "").strip()
                policy_b = (row.get("policy_b") or "").strip()
                policy   = (row.get("policy") or "").strip()
                if not ftype or not fsubtype:
                    continue
                if ftype == "STALE":
                    if not policy:
                        continue
                    key = FindingKey(ftype, fsubtype, (policy,))
                else:
                    if not policy_a or not policy_b:
                        continue
                    key = FindingKey(ftype, fsubtype, frozenset({policy_a, policy_b}))
                rows.append((key, dict(row)))
        return rows

    def _match(predicted_keys, label_rows):
        label_key_set = {}
        for key, _ in label_rows:
            label_key_set[key] = label_key_set.get(key, 0) + 1
        pred_key_pool = {}
        for key in predicted_keys:
            pred_key_pool[key] = pred_key_pool.get(key, 0) + 1

        matched_label_indices = set()
        matched_pred_keys = []
        for i, (lkey, _) in enumerate(label_rows):
            if pred_key_pool.get(lkey, 0) > 0:
                pred_key_pool[lkey] -= 1
                matched_label_indices.add(i)
                matched_pred_keys.append(lkey)

        fn_rows = [row for i, row in enumerate(label_rows) if i not in matched_label_indices]
        fp_keys = []
        for key, cnt in pred_key_pool.items():
            fp_keys.extend([key] * cnt)

        y_true = [1] * len(label_rows) + [0] * len(fp_keys)
        y_pred = [1 if i in matched_label_indices else 0 for i in range(len(label_rows))] + [1] * len(fp_keys)
        return y_true, y_pred, fp_keys, fn_rows

    label_rows = _load_label_keys()
    predicted_keys = _build_predicted_keys(result.findings, policy_id_to_file)

    y_true, y_pred, fp_keys, fn_rows = _match(predicted_keys, label_rows)

    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        sys.exit("ERROR: sklearn is not installed")

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    print()
    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f:.4f}")
    print()
    print(f"  Labels (ground truth) : {len(label_rows)}")
    print(f"  Predictions           : {len(predicted_keys)}")
    print(f"  True Positives (TP)   : {sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)}")
    print(f"  False Positives (FP)  : {len(fp_keys)}")
    print(f"  False Negatives (FN)  : {len(fn_rows)}")

    print()
    print("=" * 60)
    print("PER-TYPE BREAKDOWN (finding_type level)")
    print("=" * 60)
    for ftype in ("CONFLICT", "REDUNDANCY", "STALE"):
        lbl_keys  = [k for k, _ in label_rows if k.label_type == ftype]
        pred_k    = [k for k in predicted_keys if k.label_type == ftype]
        fn_type   = [row for row in fn_rows if row[0].label_type == ftype]
        fp_type   = [k for k in fp_keys if k.label_type == ftype]

        lbl_set  = {}
        for k in lbl_keys:
            lbl_set[k] = lbl_set.get(k, 0) + 1
        pred_set = {}
        for k in pred_k:
            pred_set[k] = pred_set.get(k, 0) + 1

        tp_t = 0
        tmp_pred = dict(pred_set)
        for k in lbl_keys:
            if tmp_pred.get(k, 0) > 0:
                tp_t += 1
                tmp_pred[k] -= 1

        fp_t = sum(pred_set.get(k, 0) - min(pred_set.get(k, 0), lbl_set.get(k, 0))
                   for k in set(list(lbl_keys) + list(pred_k)))
        fn_t = len(fn_type)

        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        rec_t  = tp_t / (tp_t + fn_t)  if (tp_t + fn_t) > 0  else 0.0
        f1_t   = (2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0)

        print(f"\n  {ftype}")
        print(f"    Labels  : {len(lbl_keys)}  Predictions: {len(pred_k)}")
        print(f"    TP: {tp_t}  FP: {fp_t}  FN: {fn_t}")
        print(f"    Precision: {prec_t:.4f}  Recall: {rec_t:.4f}  F1: {f1_t:.4f}")

    print()
    print("=" * 60)
    print("EXECUTION METRICS")
    print("=" * 60)
    print(f"  Total wall-clock time : {elapsed_time:.2f} seconds")
    print(f"  Total LLM calls made  : {stats['llm_eligible_findings']}")

    # ── Pull 5 surviving false-positive REDUNDANCY predictions ───────────────
    redundancy_fp_findings = []
    for f in result.findings:
        if f.finding_type == "REDUNDANCY":
            src_file = policy_id_to_file.get((f.evidence or {}).get("source_policy_id", ""), "")
            tgt_file = policy_id_to_file.get((f.evidence or {}).get("target_policy_id", ""), "")
            if src_file and tgt_file:
                key = FindingKey("REDUNDANCY", "REDUNDANCY", frozenset({src_file, tgt_file}))
                if key in fp_keys:
                    redundancy_fp_findings.append(f)

    if redundancy_fp_findings:
        print()
        print("=" * 60)
        print("SURVIVING FALSE-POSITIVE REDUNDANCY PREDICTIONS (UP TO 5)")
        print("=" * 60)
        for i, f in enumerate(redundancy_fp_findings[:5]):
            print(f"\nFP REDUNDANCY #{i+1}:")
            print(f"  Obligation A: {(f.evidence or {}).get('source_sentence', '')}")
            print(f"  Obligation B: {(f.evidence or {}).get('target_sentence', '')}")
            print(f"  Cosine Similarity: {(f.evidence or {}).get('cosine_similarity', 0.0)}")
            print(f"  Deterministic Score: {f.deterministic_score}")
            print(f"  LLM Explanation: {(f.evidence or {}).get('llm_verification', {}).get('explanation', '')}")

    print("\nDone.")

if __name__ == "__main__":
    main()
