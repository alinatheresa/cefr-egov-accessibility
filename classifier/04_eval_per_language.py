# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 04_eval_per_language.py
# Purpose: Generate per-language confusion matrices from the saved model without retraining.
#          Loads the tokenized val/test splits and the final model, runs a forward pass,
#          then saves confusion_matrix_{split}_{lang}.png for both window-level and
#          document-level predictions.
# Pipeline step: Classifier evaluation per language (forward pass only, no training).
# Thesis: Chapter 4 per-language results; called by b6b_eval_and_rater_check.py
#         (tab:clf-prf-doc, tab:clf-errors-doc and the saved predictions behind the
#         Section 4.3.1 confusion matrices).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Paths 
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RUN = "cefr_20260817_231925"

PREP_DIR   = os.path.join(SCRIPT_DIR, "00_prepare_output")
PACKED_DIR = os.path.join(PREP_DIR, "tokenized_packed")
RUNS_DIR   = os.path.join(SCRIPT_DIR, "02_train_output", "runs")

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match 02_train_final.py exactly
# ──────────────────────────────────────────────────────────────────────────────
LABELS  = ["A", "B1", "B2", "C"]
ID2LANG = {0: "de", 1: "en", 2: "nl"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mirror the functions in 02_train_final.py)
# ──────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, out_png: str, title: str) -> None:
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"  Saved: {os.path.basename(out_png)}")


def per_lang_confusion_matrices(y_true, y_pred, lang_ids, out_dir: str,
                                split_name: str, level: str) -> None:
    for lid, lname in ID2LANG.items():
        mask = lang_ids == lid
        if mask.sum() < 2:
            print(f"  Skipping {lname} ({split_name}/{level}): only {mask.sum()} samples")
            continue
        fname = f"confusion_matrix_{split_name}_{level}_{lname}.png"
        save_confusion_matrix(
            y_true[mask], y_pred[mask],
            os.path.join(out_dir, fname),
            f"CEFR Confusion Matrix ({split_name} / {level} / {lname.upper()})",
        )
        mf1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        print(f"    {lname} macro-F1: {mf1:.4f}  (n={mask.sum()})")


def aggregate_doc_level(logits, y_true, sample_ids, lang_ids):
    """Mean-pool window logits per source document — mirrors 02_train_final.py."""
    doc_logits: dict = defaultdict(list)
    doc_label: dict  = {}
    doc_lang: dict   = {}
    for i, sid in enumerate(sample_ids):
        doc_logits[sid].append(logits[i])
        doc_label[sid] = int(y_true[i])
        doc_lang[sid]  = int(lang_ids[i])
    doc_ids      = sorted(doc_logits.keys())
    y_true_doc   = np.array([doc_label[d] for d in doc_ids])
    y_pred_doc   = np.array([np.argmax(np.mean(doc_logits[d], axis=0)) for d in doc_ids])
    lang_ids_doc = np.array([doc_lang[d]  for d in doc_ids])
    return y_true_doc, y_pred_doc, lang_ids_doc


# ──────────────────────────────────────────────────────────────────────────────
# Thesis statistics verification  (run with --verify-stats, no model needed)
# ──────────────────────────────────────────────────────────────────────────────

def verify_stats(packed_dir: str, stride: int = 128) -> None:
    """
    Compute and print thesis table verification statistics.
    Covers: text length distribution, raw/final row counts, rater agreement,
    plus-level claim, and the collapsed CEFR class distribution.
    """
    import json
    from datasets import load_from_disk

    SEP = "=" * 64

    # ─────────────────────────────────────────────────────────────────────────
    # 1. TEXT LENGTH DISTRIBUTION (derived from tokenized windows)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("TEXT LENGTH — mDeBERTa-v3-base tokens (all 5,078 texts)")
    print(SEP)

    sample_wins: dict = defaultdict(list)
    for spl in ("train", "validation", "test"):
        ds = load_from_disk(os.path.join(packed_dir, spl))
        for row in ds:
            # Key includes split name to prevent sample_id collisions across splits
            # (train/val/test each independently index from 0)
            sample_wins[(spl, int(row["sample_id"]))].append(
                (list(row["attention_mask"]), int(row["lang_id"]))
            )

    lang_toks: dict = defaultdict(list)
    all_toks = []
    for wins in sample_wins.values():
        tok = (len(wins) - 1) * stride + int(sum(wins[-1][0]))
        lang_toks[wins[0][1]].append(tok)
        all_toks.append(tok)

    def _report_lengths(label, arr):
        a = np.array(arr)
        print(f"  {label} (n={len(a):,}):  median={int(np.median(a))}  "
              f"min={int(a.min())}  max={int(a.max())}  "
              f">512 tokens: {int((a > 512).sum())} ({100*(a > 512).mean():.1f}%)")

    _report_lengths("Overall", all_toks)
    for lid, lname in sorted(ID2LANG.items()):
        if lang_toks[lid]:
            _report_lengths(lname.upper(), lang_toks[lid])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DEDUP STATS — read from dedup_stats.json (written by 00_prepare_data.py)
    #    Uses the pipeline's own text normalisation; no re-download needed.
    #    This also resolves the elg_cefr_nl 1,196/1,195 discrepancy: the JSON
    #    was produced using the pipeline's HTML-stripping + whitespace collapse,
    #    so text grouping is identical to the final 5,078-row dataset.
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("TABLE 1 — RAW vs FINAL ROW COUNTS  (from dedup_stats.json)")
    print(SEP)

    dedup_path = os.path.join(os.path.dirname(packed_dir), "dedup_stats.json")
    with open(dedup_path) as _f:
        ds_json = json.load(_f)

    per_ds     = ds_json["per_dataset"]
    en_within  = ds_json["en_within_dups"]
    cross_dup  = ds_json["en_cross_dataset_dups"]
    rater_json = ds_json["rater_agreement"]

    DATASET_ORDER = [
        ("merlin_de",          "de"),
        ("elg_cefr_de",        "de"),
        ("elg_cefr_nl",        "nl"),
        ("elg_cefr_en",        "en"),
        ("icle500_en",         "en"),
        ("cambridge_exams_en", "en"),
        ("cople2_pt",          "pt"),
        ("peapl2_pt",          "pt"),
    ]
    MULTI_RATER  = {"elg_cefr_nl", "cople2_pt"}
    THESIS_RAW   = {"merlin_de": 1033, "elg_cefr_de": 509, "elg_cefr_nl": 3596,
                    "elg_cefr_en": 712, "icle500_en": 495, "cambridge_exams_en": 331,
                    "cople2_pt": 942, "peapl2_pt": 481}
    THESIS_FINAL = {"merlin_de": 1033, "elg_cefr_de": 509, "elg_cefr_nl": 1195,
                    "elg_cefr_en": 706, "icle500_en": 488, "cambridge_exams_en": 325,
                    "cople2_pt": 471, "peapl2_pt": 481}

    multi_rater_stats: dict = {}
    for name, _ in DATASET_ORDER:
        info     = per_ds.get(name, {})
        n_raw    = info.get("n_raw", "?")
        n_na     = info.get("n_removed_na", "?")
        n_plus   = info.get("n_plus", "?")
        n_within = en_within.get(name, 0)
        n_final  = info.get("n_final", "?")
        tr       = THESIS_RAW.get(name, "?")
        tf       = THESIS_FINAL.get(name, "?")
        raw_ok   = "MATCH" if n_raw == tr else f"MISMATCH (computed {n_raw})"
        fin_ok   = "MATCH" if n_final == tf else f"MISMATCH (computed {n_final})"
        print(f"  {name}")
        print(f"    raw={n_raw} [{raw_ok}]  na={n_na}  within_dup={n_within}  "
              f"final={n_final} [{fin_ok}]  plus={n_plus}")
        if name in MULTI_RATER and name in rater_json:
            r = rater_json[name]
            multi_rater_stats[name] = {
                "n_texts":      r["n_texts"],
                "rater_counts": r["rater_dist"],
                "apart":        {int(k): v for k, v in r["apart"].items()},
            }

    total_raw = sum(per_ds[n]["n_raw"]         for n, _ in DATASET_ORDER if n in per_ds)
    total_na  = sum(per_ds[n]["n_removed_na"]  for n, _ in DATASET_ORDER if n in per_ds)
    en_wd     = sum(en_within.get(n, 0)        for n, lang in DATASET_ORDER if lang == "en")
    mr_rows   = sum(per_ds[n]["n_raw"] - per_ds[n]["n_final"]
                    for n in MULTI_RATER if n in per_ds)
    final_check = total_raw - total_na - en_wd - mr_rows - cross_dup
    print(f"\n  TOTAL raw:           {total_raw}  "
          f"[{'MATCH' if total_raw == 8099 else f'MISMATCH (thesis 8099)'}]")
    print(f"  NA/invalid removed:  {total_na}  (thesis 7)  "
          f"[{'MATCH' if total_na == 7 else 'MISMATCH'}]")
    print(f"  EN within-ds dups:   {en_wd}  (thesis 12)  "
          f"[{'MATCH' if en_wd == 12 else 'MISMATCH'}]")
    print(f"  Multi-rater rows:    {mr_rows}  (thesis 2872)  "
          f"[{'MATCH' if mr_rows == 2872 else 'MISMATCH'}]")
    print(f"  EN cross-ds dups:    {cross_dup}  (thesis 130)  "
          f"[{'MATCH' if cross_dup == 130 else 'MISMATCH'}]")
    print(f"  Reconciliation: {total_raw} - {total_na} - {en_wd} - {mr_rows} - {cross_dup} = {final_check}  "
          f"[{'MATCH' if final_check == 5078 else f'MISMATCH (expected 5078)'}]")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PLUS-LEVEL CLAIM
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PLUS-LEVEL CLAIM")
    print(SEP)
    total_plus = sum(per_ds[n].get("n_plus", 0) for n in per_ds)
    nl_plus    = per_ds.get("elg_cefr_nl", {}).get("n_plus", 0)
    nl_raw     = per_ds.get("elg_cefr_nl", {}).get("n_raw", 0)
    print(f"  Total plus-level labels: {total_plus}  "
          f"[{'MATCH' if total_plus == 1823 else f'MISMATCH (thesis 1,823)'}]")
    print(f"  elg_cefr_nl plus: {nl_plus} of {nl_raw}  "
          f"[{'MATCH' if nl_plus == 1505 else f'MISMATCH (thesis 1,505)'}]")
    for name, info in sorted(per_ds.items()):
        if info.get("n_plus", 0) > 0:
            print(f"    {name}: {info['n_plus']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TABLE 2 — RATER AGREEMENT
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("TABLE 2 — RATER AGREEMENT")
    print(SEP)
    THESIS_T2 = {
        "elg_cefr_nl": {"n_texts": 1195, "apart": {0: 290, 1: 684, 2: 212, 3: 9}},
        "cople2_pt":   {"n_texts":  471, "apart": {0: 308, 1: 163, 2:   0, 3: 0}},
    }
    for name, data in multi_rater_stats.items():
        th      = THESIS_T2.get(name, {})
        total   = data["n_texts"]
        th_tot  = th.get("n_texts", "?")
        rc      = data["rater_counts"]
        print(f"\n  {name}: {total} texts  "
              f"[{'MATCH' if total == th_tot else f'MISMATCH (thesis {th_tot})'}]  "
              f"rater distribution: {dict(sorted(rc.items()))}")
        for k in range(4):
            comp = data["apart"].get(k, 0)
            thes = th.get("apart", {}).get(k, "?")
            pct  = 100 * comp / total if total else 0
            ok   = "MATCH" if comp == thes else f"MISMATCH (thesis {thes})"
            print(f"    {k} apart: {comp} ({pct:.1f}%)  [{ok}]")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TABLE 3 — COLLAPSED CEFR CLASS DISTRIBUTION (from saved json)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("TABLE 3 — CEFR CLASS DISTRIBUTION (from label_counts_mergedC.json)")
    print(SEP)
    counts_path = os.path.join(os.path.dirname(packed_dir), "label_counts_mergedC.json")
    with open(counts_path) as f:
        lc = json.load(f)

    # Sum counts across all three splits
    full: dict = {lang: {lbl: 0 for lbl in LABELS}
                  for lang in ("de", "nl", "en", "pt")}
    for spl in ("train", "validation", "test"):
        for lang, cnts in lc["by_split"][spl]["by_lang_label"].items():
            for lbl, n in cnts.items():
                full[lang][lbl] += n

    THESIS_T3 = {
        "de": {"A": 484, "B1": 421, "B2": 408, "C": 229, "total": 1542},
        "nl": {"A": 186, "B1": 542, "B2": 380, "C":  87, "total": 1195},
        "en": {"A": 186, "B1": 344, "B2": 414, "C": 445, "total": 1389},
        "pt": {"A": 403, "B1": 367, "B2":  70, "C": 112, "total":  952},
    }
    LANG_ORDER = ["de", "nl", "en", "pt"]
    for lang in LANG_ORDER:
        total = sum(full[lang].values())
        th    = THESIS_T3[lang]
        th_total = th["total"]
        print(f"\n  {lang.upper()} total={total}  "
              f"[{'MATCH' if total == th_total else f'MISMATCH (thesis {th_total})'}]")
        for lbl in LABELS:
            comp = full[lang][lbl]
            thes = th[lbl]
            pct  = round(100 * comp / total) if total else 0
            ok   = "MATCH" if comp == thes else f"MISMATCH (thesis {thes})"
            print(f"    {lbl}: {comp} ({pct}%)  [{ok}]")

    # Overall totals
    overall = lc["overall"]
    print(f"\n  Overall totals: {overall['by_label']}  total={overall['total']}")
    THESIS_OVERALL = {"A": 1259, "B1": 1674, "B2": 1272, "C": 873}
    for lbl in LABELS:
        comp = overall["by_label"][lbl]
        thes = THESIS_OVERALL[lbl]
        pct  = round(100 * comp / overall["total"])
        print(f"    {lbl}: {comp} ({pct}%)  "
              f"[{'MATCH' if comp == thes else f'MISMATCH (thesis {thes})'}]")

    print(f"\n{SEP}")
    print("VERIFICATION COMPLETE")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-language confusion matrices from a saved classifier run."
    )
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help=f"Run directory name under 02_train_output/runs/ (default: {DEFAULT_RUN}).",
    )
    args = parser.parse_args()

    run_root  = os.path.join(RUNS_DIR, args.run)
    model_dir = os.path.join(run_root, "mdeberta_cefr_multi")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model not found: {model_dir}\n"
            f"Pass --run <run-name> with the correct timestamped folder."
        )

    print(f"Run:    {run_root}")
    print(f"Model:  {model_dir}")
    print(f"Data:   {PACKED_DIR}")

    # Load tokenized datasets
    print("\nLoading datasets...")
    ds_val  = load_from_disk(os.path.join(PACKED_DIR, "validation"))
    ds_test = load_from_disk(os.path.join(PACKED_DIR, "test"))
    print(f"  validation: {len(ds_val)} windows")
    print(f"  test:       {len(ds_test)} windows")

    # Device selection (MPS > CUDA > CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model (eval-only, no gradient tracking needed)
    print("\nLoading model...")
    AutoTokenizer.from_pretrained(model_dir)  # verify tokenizer loads
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    def run_forward(ds, desc: str, batch_size: int = 16):
        """Manual batched forward pass — avoids Trainer/TF/Keras dependency."""
        all_logits, all_labels = [], []
        ds_torch = ds.with_format("torch")
        n = len(ds_torch)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = ds_torch[start:end]
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            if (start // batch_size) % 10 == 0:
                print(f"  {desc}: {end}/{n}", flush=True)
        return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

    # Forward pass on val and test
    print("\nRunning forward pass on validation split...")
    val_logits, yv_true = run_forward(ds_val, "val")
    yv_pred_w = val_logits.argmax(axis=1)

    print("\nRunning forward pass on test split...")
    test_logits, yt_true = run_forward(ds_test, "test")
    yt_pred_w = test_logits.argmax(axis=1)

    val_lang_ids    = np.array(ds_val["lang_id"])
    test_lang_ids   = np.array(ds_test["lang_id"])
    val_sample_ids  = np.array(ds_val["sample_id"])
    test_sample_ids = np.array(ds_test["sample_id"])

    # Window-level per-language confusion matrices
    print("\n=== Window-level — Validation ===")
    per_lang_confusion_matrices(yv_true, yv_pred_w, val_lang_ids, run_root, "val", "window")
    print("\n=== Window-level — Test ===")
    per_lang_confusion_matrices(yt_true, yt_pred_w, test_lang_ids, run_root, "test", "window")

    # Document-level aggregation
    yv_true_d, yv_pred_d, val_lang_ids_d = aggregate_doc_level(
        val_logits, yv_true, val_sample_ids, val_lang_ids)
    yt_true_d, yt_pred_d, test_lang_ids_d = aggregate_doc_level(
        test_logits, yt_true, test_sample_ids, test_lang_ids)

    print("\n=== Document-level — Validation ===")
    per_lang_confusion_matrices(yv_true_d, yv_pred_d, val_lang_ids_d, run_root, "val", "doc")
    print("\n=== Document-level — Test ===")
    per_lang_confusion_matrices(yt_true_d, yt_pred_d, test_lang_ids_d, run_root, "test", "doc")

    print(f"\nAll per-language confusion matrices saved to:\n  {run_root}")


if __name__ == "__main__":
    import sys as _sys
    if "--verify-stats" in _sys.argv:
        verify_stats(PACKED_DIR)
    else:
        main()
