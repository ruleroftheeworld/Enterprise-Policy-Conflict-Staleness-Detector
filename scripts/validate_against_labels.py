# -*- coding: utf-8 -*-
"""
validate_against_labels.py
==========================
Runs the deterministic pipeline against all 30 policies in
sample_data/problem_11/policies/ and compares the predicted findings to the
ground-truth labels in findings_labels.csv.

Outputs
-------
* Overall precision / recall / F1 (micro-averaged, sklearn)
* Per-type breakdown (CONFLICT, REDUNDANCY, STALE)
* Confusion tables: misses (FN) and false positives (FP)
* A summary of which policy files failed to ingest (if any)

Usage
-----
    python scripts/validate_against_labels.py

No LLM provider is used — deterministic detection only.
"""
from __future__ import annotations

import codecs
import csv
import io
import sys
import traceback
import warnings as py_warnings
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# ① MAPPING TABLE — edit this dict to adjust the schema bridge.
#    engine finding_type  →  (label finding_type, label finding_subtype)
# ---------------------------------------------------------------------------
ENGINE_TO_LABEL: dict[str, tuple[str, str]] = {
    "CONTRADICTION":         ("CONFLICT",   "DIRECT_CONFLICT"),
    "FREQUENCY_MISMATCH":    ("CONFLICT",   "PARTIAL_CONFLICT"),
    "MODALITY_INCONSISTENCY":("CONFLICT",   "PARTIAL_CONFLICT"),
    "REDUNDANCY":            ("REDUNDANCY", "REDUNDANCY"),
    "STALE_POLICY":          ("STALE",      "STALE_POLICY"),
    "STALE_REFERENCE":       ("STALE",      "STALE_REFERENCE"),
}

# ---------------------------------------------------------------------------
# Paths (resolved relative to this script's location so it works regardless
# of where the caller's cwd is).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"
_LABELS_CSV  = _REPO_ROOT / "sample_data" / "problem_11" / "findings_labels.csv"


# ---------------------------------------------------------------------------
# Internal key type — the canonical unit of comparison.
# ---------------------------------------------------------------------------
class FindingKey(NamedTuple):
    """Normalised finding identity used for matching predicted ↔ labelled."""
    label_type: str     # CONFLICT | REDUNDANCY | STALE
    label_subtype: str  # DIRECT_CONFLICT | PARTIAL_CONFLICT | REDUNDANCY | …
    # For STALE_* types: (policy,)   — single-policy key
    # For pair types:    frozenset({policy_a, policy_b})
    policy_key: frozenset | tuple


# ---------------------------------------------------------------------------
# Step 1: Load the 30 policies and run the deterministic pipeline.
# ---------------------------------------------------------------------------

def _load_and_run() -> tuple[list, list[str], dict[str, str]]:
    """
    Returns:
        findings      – list of Finding objects from the engine
        failed_files  – list of filenames that could not be processed
        policy_id_to_file – mapping of internal policy_id → filename basename
    """
    # Collect all .md files, sorted for reproducibility.
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    if not policy_paths:
        sys.exit(f"ERROR: no .md files found in {_POLICY_DIR}")

    # Import engine here so import errors are caught cleanly.
    try:
        from engine.pipeline import analyze_policy_documents
    except ImportError as exc:
        sys.exit(f"ERROR: cannot import engine.pipeline — {exc}")

    # Suppress chatty log output from the engine during the run.
    py_warnings.filterwarnings("ignore")

    print(f"Running pipeline on {len(policy_paths)} policy files …")
    result = analyze_policy_documents(
        policy_paths,
        llm_provider=None,   # deterministic only
    )

    # Build policy_id → filename mapping from successfully parsed policies.
    policy_id_to_file: dict[str, str] = {}
    for policy in result.policies:
        filename = Path(policy.source_file).name
        policy_id_to_file[policy.policy_id] = filename

    # Identify which input files failed (appear in pipeline warnings).
    failed_files: list[str] = []
    succeeded_files = {Path(p.source_file).name for p in result.policies}
    for path in policy_paths:
        if path.name not in succeeded_files:
            failed_files.append(path.name)

    print(f"  Policies parsed:   {len(result.policies)}")
    print(f"  Files failed:      {len(failed_files)}")
    print(f"  Obligations:       {len(result.obligations)}")
    print(f"  Raw findings:      {len(result.findings)}")
    if result.warnings:
        print(f"  Pipeline warnings: {len(result.warnings)}")

    return result.findings, failed_files, policy_id_to_file


# ---------------------------------------------------------------------------
# Step 2: Convert engine findings → normalised FindingKey set.
# ---------------------------------------------------------------------------

def _finding_to_key(
    finding,
    policy_id_to_file: dict[str, str],
) -> FindingKey | None:
    """Map one engine Finding to a FindingKey.  Returns None if unrecognised."""
    engine_type = finding.finding_type
    if engine_type not in ENGINE_TO_LABEL:
        return None

    label_type, label_subtype = ENGINE_TO_LABEL[engine_type]
    ev = finding.evidence

    if label_type == "STALE":
        # Staleness findings are anchored to a single policy.
        raw_policy_id = ev.get("policy_id")
        if raw_policy_id is None:
            return None
        filename = policy_id_to_file.get(raw_policy_id)
        if filename is None:
            return None
        return FindingKey(label_type, label_subtype, (filename,))

    else:
        # Pair findings.
        src_id = ev.get("source_policy_id")
        tgt_id = ev.get("target_policy_id")
        if src_id is None or tgt_id is None:
            return None
        src_file = policy_id_to_file.get(src_id)
        tgt_file = policy_id_to_file.get(tgt_id)
        if src_file is None or tgt_file is None:
            return None
        # Skip intra-policy findings (same source & target).
        if src_file == tgt_file:
            return None
        return FindingKey(
            label_type, label_subtype,
            frozenset({src_file, tgt_file}),
        )


def _build_predicted_keys(
    findings: list,
    policy_id_to_file: dict[str, str],
) -> list[FindingKey]:
    """Convert all engine findings to FindingKeys, dropping unknowns."""
    keys = []
    for f in findings:
        key = _finding_to_key(f, policy_id_to_file)
        if key is not None:
            keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Step 3: Load ground-truth labels → FindingKey set.
# ---------------------------------------------------------------------------

def _load_label_keys() -> list[tuple[FindingKey, dict]]:
    """Load findings_labels.csv and return list of (key, raw_row) pairs."""
    rows: list[tuple[FindingKey, dict]] = []
    with open(_LABELS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ftype    = (row.get("finding_type") or "").strip()
            fsubtype = (row.get("finding_subtype") or "").strip()
            policy_a = (row.get("policy_a") or "").strip()
            policy_b = (row.get("policy_b") or "").strip()
            policy   = (row.get("policy") or "").strip()

            if not ftype or not fsubtype:
                continue   # skip blank/header rows

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


# ---------------------------------------------------------------------------
# Step 4: Match predicted ↔ labelled.
# ---------------------------------------------------------------------------

def _match(
    predicted_keys: list[FindingKey],
    label_rows: list[tuple[FindingKey, dict]],
) -> tuple[list[bool], list[bool], list[tuple[FindingKey, dict]]]:
    """
    Returns:
        y_true_list  – binary label for each item in combined (labels+preds) space
        y_pred_list  – binary prediction for each item
        fp_keys      – predicted keys with no matching label
        fn_rows      – label rows with no matching prediction
    """
    label_key_set:     dict[FindingKey, int] = {}  # key → count remaining
    for key, _ in label_rows:
        label_key_set[key] = label_key_set.get(key, 0) + 1

    pred_key_pool: dict[FindingKey, int] = {}
    for key in predicted_keys:
        pred_key_pool[key] = pred_key_pool.get(key, 0) + 1

    # Greedy 1-to-1 matching (order-independent, by key equality).
    tp_keys:  set[FindingKey] = set()
    matched_label_indices: set[int] = set()
    matched_pred_keys: list[FindingKey] = []

    for i, (lkey, _) in enumerate(label_rows):
        if pred_key_pool.get(lkey, 0) > 0:
            pred_key_pool[lkey] -= 1
            matched_label_indices.add(i)
            matched_pred_keys.append(lkey)

    # TP = labels that were matched.
    tp_count = len(matched_label_indices)
    # FN = labels that were not matched.
    fn_rows  = [
        row for i, row in enumerate(label_rows)
        if i not in matched_label_indices
    ]
    # FP = predicted keys with count > 0 after matching.
    fp_keys: list[FindingKey] = []
    for key, cnt in pred_key_pool.items():
        fp_keys.extend([key] * cnt)

    # Build sklearn-compatible binary vectors (one entry per label + per FP).
    y_true: list[int] = [1] * len(label_rows) + [0] * len(fp_keys)
    y_pred: list[int] = (
        [1 if i in matched_label_indices else 0
         for i in range(len(label_rows))]
        + [1] * len(fp_keys)
    )

    return y_true, y_pred, fp_keys, fn_rows


# ---------------------------------------------------------------------------
# Step 5: Metrics (overall + per-type).
# ---------------------------------------------------------------------------

def _metrics(
    y_true: list[int],
    y_pred: list[int],
    label_rows: list[tuple[FindingKey, dict]],
    predicted_keys: list[FindingKey],
    fp_keys: list[FindingKey],
    fn_rows: list[tuple[FindingKey, dict]],
) -> None:
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        sys.exit("ERROR: sklearn is not installed — run: pip install scikit-learn")

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f:.4f}")
    total_labels = len(label_rows)
    total_preds  = len(predicted_keys)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    print(f"\n  Labels (ground truth) : {total_labels}")
    print(f"  Predictions           : {total_preds}")
    print(f"  True Positives (TP)   : {tp}")
    print(f"  False Positives (FP)  : {len(fp_keys)}")
    print(f"  False Negatives (FN)  : {len(fn_rows)}")

    # --- Per-type breakdown ---
    print("\n" + "=" * 60)
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
        f1_t   = (2 * prec_t * rec_t / (prec_t + rec_t)
                  if (prec_t + rec_t) > 0 else 0.0)

        print(f"\n  {ftype}")
        print(f"    Labels  : {len(lbl_keys)}  Predictions: {len(pred_k)}")
        print(f"    TP: {tp_t}  FP: {fp_t}  FN: {fn_t}")
        print(f"    Precision: {prec_t:.4f}  Recall: {rec_t:.4f}  F1: {f1_t:.4f}")


# ---------------------------------------------------------------------------
# Step 6: Print confusion tables.
# ---------------------------------------------------------------------------

def _fmt_policy_key(key: FindingKey) -> str:
    if key.label_type == "STALE":
        return key.policy_key[0]
    parts = sorted(key.policy_key)
    return f"{parts[0]}  <->  {parts[1]}"


def _print_confusion_tables(
    fn_rows: list[tuple[FindingKey, dict]],
    fp_keys: list[FindingKey],
    findings: list,
    policy_id_to_file: dict[str, str],
) -> None:
    print("\n" + "=" * 60)
    print("FALSE NEGATIVES — labeled findings with NO matching prediction")
    print("=" * 60)
    if not fn_rows:
        print("  (none — all labels matched)")
    else:
        col_w = [14, 25, 40, 35]
        hdr = (f"{'Subtype':<{col_w[0]}}  {'Policy key':<{col_w[1]}}  "
               f"{'Description':<{col_w[2]}}  {'Explanation':<{col_w[3]}}")
        print(hdr)
        print("-" * (sum(col_w) + 6))
        for key, row in fn_rows:
            desc    = (row.get("description") or "")[:col_w[2]]
            expl    = (row.get("explanation") or "")[:col_w[3]]
            pk      = _fmt_policy_key(key)[:col_w[1]]
            subtype = key.label_subtype[:col_w[0]]
            print(f"{subtype:<{col_w[0]}}  {pk:<{col_w[1]}}  "
                  f"{desc:<{col_w[2]}}  {expl:<{col_w[3]}}")

    print("\n" + "=" * 60)
    print("FALSE POSITIVES — predicted findings with NO matching label")
    print("=" * 60)
    if not fp_keys:
        print("  (none — no extra predictions beyond labels)")
    else:
        # Enrich FPs with the engine's own explanation text.
        # Build a lookup: FindingKey → explanation.
        fp_details: dict[FindingKey, str] = {}
        for f in findings:
            key = _finding_to_key(f, policy_id_to_file)
            if key is not None and key in [k for k in fp_keys]:
                fp_details[key] = fp_details.get(key, f.explanation)

        col_w = [14, 25, 60]
        hdr = (f"{'Subtype':<{col_w[0]}}  {'Policy key':<{col_w[1]}}  "
               f"{'Engine explanation':<{col_w[2]}}")
        print(hdr)
        print("-" * (sum(col_w) + 4))
        for key in fp_keys:
            expl    = fp_details.get(key, "")[:col_w[2]]
            pk      = _fmt_policy_key(key)[:col_w[1]]
            subtype = key.label_subtype[:col_w[0]]
            print(f"{subtype:<{col_w[0]}}  {pk:<{col_w[1]}}  {expl:<{col_w[2]}}")


# ---------------------------------------------------------------------------
# Step 7: Report failed files.
# ---------------------------------------------------------------------------

def _report_failures(failed_files: list[str], pipeline_warnings: list[str]) -> None:
    if not failed_files:
        return
    print("\n" + "=" * 60)
    print("FILES THAT FAILED INGESTION (skipped)")
    print("=" * 60)
    for fn in failed_files:
        # Try to find a matching warning message.
        matching = [w for w in pipeline_warnings if fn in w]
        reason = matching[0] if matching else "unknown reason"
        print(f"  {fn}  →  {reason}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_stdout() -> None:
    """Re-wrap stdout/stderr to UTF-8 so Unicode chars print cleanly on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def main() -> None:
    _setup_stdout()  # must be first, before any print
    print("\n" + "=" * 60)
    print("POLICY LABEL VALIDATION SCRIPT")
    print("=" * 60)
    print(f"  Policy dir : {_POLICY_DIR}")
    print(f"  Labels CSV : {_LABELS_CSV}")

    # ① Run pipeline.
    findings, failed_files, policy_id_to_file = _load_and_run()

    # ② Build predicted keys.
    predicted_keys = _build_predicted_keys(findings, policy_id_to_file)
    print(f"  Mapped prediction keys: {len(predicted_keys)}")

    # ③ Load labels.
    label_rows = _load_label_keys()
    print(f"  Ground-truth label keys: {len(label_rows)}")

    # ④ Match.
    y_true, y_pred, fp_keys, fn_rows = _match(predicted_keys, label_rows)

    # ⑤ Metrics.
    _metrics(y_true, y_pred, label_rows, predicted_keys, fp_keys, fn_rows)

    # ⑥ Confusion tables.
    _print_confusion_tables(fn_rows, fp_keys, findings, policy_id_to_file)

    # ⑦ Failure report.
    if failed_files:
        print("\n" + "=" * 60)
        print("FILES THAT FAILED INGESTION (skipped from scoring)")
        print("=" * 60)
        for fn in failed_files:
            print(f"  ✗  {fn}")

    print("\nDone.\n")


if __name__ == "__main__":
    # Make sure the repo root is on sys.path so `engine` and `shared` are importable.
    sys.path.insert(0, str(_REPO_ROOT))
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
