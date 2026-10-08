# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: check_rater_agreement.py
# Purpose: for test-set errors from multi-rater datasets (elg_cefr_nl, cople2_pt),
#          check whether any individual rater's original label matches the model's
#          prediction — not just the resolved gold label used for training/evaluation.
# Pipeline step: Rater check for classification errors (raw rater labels from UniversalCEFR).
# Thesis: Chapter 4 rater-supported error analysis (tab:clf-rater-nl, via
#         b6b_eval_and_rater_check.py, which reuses its matching functions).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
from collections import defaultdict, Counter

from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

ALL_TEST_CSV = os.path.join(SCRIPT_DIR, "output_errors", "all_test.csv")
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "rater_agreement_check.csv")
OUTPUT_TXT   = os.path.join(SCRIPT_DIR, "rater_agreement_report.txt")

LABELS    = ["A", "B1", "B2", "C"]
LABEL_SET = set(LABELS)


# ─────────────────────────────────────────────────────────────────────
# Label and text normalisation (mirrors 00_prepare_data.py exactly)
# ─────────────────────────────────────────────────────────────────────

def normalize_label(lbl):
    """Map a raw CEFR label string to the merged 4-class scale; None if invalid."""
    if not isinstance(lbl, str):
        return None
    y = lbl.strip().upper()
    base = y[:-1] if y.endswith("+") else y
    if base in {"A1", "A2"}:  return "A"
    if base == "B1":           return "B1"
    if base == "B2":           return "B2"
    if base in {"C1", "C2"}:  return "C"
    return base if base in LABEL_SET else None


def normalize_text(text):
    """Apply the same cleaning as pick_text_field() in 00_prepare_data.py."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+",      " ", text).strip()
    return text


def pick_raw_text(row):
    """Extract and normalise the text field from a raw HuggingFace row."""
    for key in ("text", "sentence", "content", "value"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return normalize_text(v)
        if isinstance(v, list):
            joined = " ".join(str(i) for i in v if isinstance(i, str))
            if joined.strip():
                return normalize_text(joined)
    return ""


# ─────────────────────────────────────────────────────────────────────
# Build per-text rater-label lookup from a raw multi-rater dataset
# ─────────────────────────────────────────────────────────────────────

def build_rater_lookup(hf_path):
    """
    Download the raw HuggingFace dataset and group raw per-rater labels by
    (normalised) text. Returns dict: normalised_text -> [raw_lbl, raw_lbl, ...]
    """
    ds = load_dataset(hf_path)
    skey = "train" if "train" in ds else list(ds.keys())[0]
    buckets = defaultdict(list)
    for row in ds[skey]:
        txt = pick_raw_text(row)
        lbl = row.get("cefr_level", row.get("label"))
        if txt and lbl:
            buckets[txt].append(str(lbl))
    return dict(buckets)


# ─────────────────────────────────────────────────────────────────────
# Aggregate window-level predictions to document level (majority vote)
# Exact for NL/PT since almost all documents fit in a single window.
# ─────────────────────────────────────────────────────────────────────

def load_doc_predictions():
    """
    Read all_test.csv (window-level) and return per-document majority-vote
    predictions. Returned dict: sample_id -> {lang, true_label, pred_label,
    is_correct, text_decoded (first window), n_windows}
    """
    rows = list(csv.DictReader(open(ALL_TEST_CSV, encoding="utf-8")))
    by_sid = defaultdict(list)
    for r in rows:
        by_sid[r["sample_id"]].append(r)

    docs = {}
    for sid, windows in by_sid.items():
        windows.sort(key=lambda r: int(r["idx"]))
        first     = windows[0]
        pred_doc  = Counter(w["pred_label"] for w in windows).most_common(1)[0][0]
        true_doc  = first["true_label"]
        docs[sid] = {
            "lang":         first["lang"],
            "true_label":   true_doc,
            "pred_label":   pred_doc,
            "is_correct":   true_doc == pred_doc,
            "text_decoded": first["text_decoded"],
            "n_windows":    len(windows),
        }
    return docs


# ─────────────────────────────────────────────────────────────────────
# Text matching: align tokeniser-decoded text against raw dataset texts
# ─────────────────────────────────────────────────────────────────────

def match_text(decoded_text, rater_lookup):
    """
    Attempt to find the entry in rater_lookup that corresponds to decoded_text.
    Tries exact match, then prefix matches of decreasing length.
    Returns (matched_key, rater_labels) or (None, []).
    """
    norm = normalize_text(decoded_text)

    if norm in rater_lookup:
        return norm, rater_lookup[norm]

    # Prefix match — the raw text may be longer than one decoded window
    for prefix_len in (150, 100, 60):
        prefix = norm[:prefix_len]
        if not prefix:
            continue
        hits = [k for k in rater_lookup if k.startswith(prefix)]
        if len(hits) == 1:
            return hits[0], rater_lookup[hits[0]]
        if len(hits) > 1:
            best = min(hits, key=lambda k: abs(len(k) - len(norm)))
            return best, rater_lookup[best]

    return None, []


# ─────────────────────────────────────────────────────────────────────
# Core analysis for one language's error set
# ─────────────────────────────────────────────────────────────────────

def analyse_errors(error_docs, rater_lookup, dataset_name):
    """
    For each document in error_docs, look up its per-rater labels and check
    whether any rater agrees with the model's prediction.
    Returns (list_of_detail_dicts, n_rater_agrees, n_not_found).
    """
    details      = []
    n_agree      = 0
    n_not_found  = 0

    for sid in sorted(error_docs, key=lambda s: int(s)):
        doc = error_docs[sid]
        matched_key, raw_labels = match_text(doc["text_decoded"], rater_lookup)

        if matched_key is None:
            n_not_found += 1
            norm_labels = []
            any_agree   = None   # unknown — text not found
        else:
            norm_labels = [normalize_label(l) for l in raw_labels]
            norm_labels = [l for l in norm_labels if l]
            any_agree   = doc["pred_label"] in norm_labels
            if any_agree:
                n_agree += 1

        details.append({
            "sample_id":         sid,
            "dataset":           dataset_name,
            "true_label":        doc["true_label"],
            "pred_label":        doc["pred_label"],
            "n_windows":         doc["n_windows"],
            "raw_rater_labels":  "|".join(raw_labels) if raw_labels else "NOT_FOUND",
            "norm_rater_labels": "|".join(norm_labels) if norm_labels else "NOT_FOUND",
            "any_rater_agrees":  ("YES" if any_agree else
                                  "NO"  if any_agree is False else "UNKNOWN"),
            "text_excerpt":      doc["text_decoded"][:140].replace("\n", " "),
        })

    return details, n_agree, n_not_found


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw multi-rater datasets from HuggingFace (uses local cache)...")
    rater_nl = build_rater_lookup("UniversalCEFR/elg_cefr_nl")
    rater_pt = build_rater_lookup("UniversalCEFR/cople2_pt")
    print(f"  elg_cefr_nl: {len(rater_nl)} unique texts")
    print(f"  cople2_pt:   {len(rater_pt)} unique texts")

    print("\nLoading document-level test predictions from all_test.csv...")
    docs = load_doc_predictions()
    n_docs = len(docs)
    print(f"  {n_docs} test documents")

    # Partition errors by language
    nl_errors = {s: d for s, d in docs.items()
                 if d["lang"] == "nl" and not d["is_correct"]}
    pt_all_errors = {s: d for s, d in docs.items()
                     if d["lang"] == "pt" and not d["is_correct"]}

    # PT: restrict to cople2_pt texts (multi-rater) by attempting lookup
    pt_mr_errors = {}
    pt_sr_errors = {}
    for sid, doc in pt_all_errors.items():
        key, _ = match_text(doc["text_decoded"], rater_pt)
        if key is not None:
            pt_mr_errors[sid] = doc
        else:
            pt_sr_errors[sid] = doc

    print(f"\n  NL errors:                 {len(nl_errors)}")
    print(f"  PT errors (total):         {len(pt_all_errors)}")
    print(f"  PT errors from cople2_pt:  {len(pt_mr_errors)}")
    print(f"  PT errors from peapl2_pt:  {len(pt_sr_errors)}  (single-rater; excluded)")

    # Run analysis
    nl_details, nl_agree, nl_nf = analyse_errors(nl_errors,    rater_nl, "elg_cefr_nl")
    pt_details, pt_agree, pt_nf = analyse_errors(pt_mr_errors, rater_pt, "cople2_pt")

    # ── Write CSV ──────────────────────────────────────────────────────
    all_rows = nl_details + pt_details
    fields   = ["sample_id","dataset","true_label","pred_label","n_windows",
                "raw_rater_labels","norm_rater_labels","any_rater_agrees","text_excerpt"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # ── Build text report ──────────────────────────────────────────────
    SEP   = "=" * 72
    sep2  = "-" * 72
    lines = []

    def h(title):
        lines.extend(["", SEP, title, SEP])

    h("RATER AGREEMENT CHECK — MULTI-RATER TEST-SET ERRORS")
    lines.append(
        "For each error from elg_cefr_nl (Dutch) and cople2_pt (Portuguese),\n"
        "this report checks whether ANY individual rater's original label\n"
        "(before median/lower-of-two resolution) matches the model's prediction.\n"
        "Raw labels are normalised to the 4-class scale before comparison."
    )

    # ── Dutch ────────────────────────────────────────────────────────
    h("DUTCH  —  elg_cefr_nl")
    n_found_nl = len(nl_errors) - nl_nf
    lines.append(f"Total NL test errors:                {len(nl_errors)}")
    lines.append(f"Texts matched in raw dataset:        {n_found_nl}")
    lines.append(f"Texts not matched (lookup failure):  {nl_nf}")
    lines.append(f"Errors with ≥1 rater agreeing:       "
                 f"{nl_agree} / {n_found_nl}  "
                 f"({100*nl_agree/max(1,n_found_nl):.1f}%)")
    lines.append(f"Errors with NO rater agreeing:       "
                 f"{n_found_nl - nl_agree} / {n_found_nl}  "
                 f"({100*(n_found_nl-nl_agree)/max(1,n_found_nl):.1f}%)")
    lines.append("")
    lines.append("Per-error breakdown:")
    lines.append(sep2)
    for d in nl_details:
        agree_tag = f"[{d['any_rater_agrees']}]"
        lines.append(
            f"  {agree_tag:<10}  true={d['true_label']}  pred={d['pred_label']}  "
            f"raw_labels={d['raw_rater_labels']}  norm={d['norm_rater_labels']}"
        )
        lines.append(f"             \"{d['text_excerpt']}\"")

    # ── Portuguese ───────────────────────────────────────────────────
    h("PORTUGUESE  —  cople2_pt  (peapl2_pt excluded: single-rater)")
    n_found_pt = len(pt_mr_errors) - pt_nf
    lines.append(f"Total PT test errors:                {len(pt_all_errors)}")
    lines.append(f"Of which from cople2_pt:             {len(pt_mr_errors)}")
    lines.append(f"Of which from peapl2_pt (excluded):  {len(pt_sr_errors)}")
    lines.append(f"Texts matched in raw dataset:        {n_found_pt}")
    lines.append(f"Texts not matched (lookup failure):  {pt_nf}")
    lines.append(f"Errors with ≥1 rater agreeing:       "
                 f"{pt_agree} / {n_found_pt}  "
                 f"({100*pt_agree/max(1,n_found_pt):.1f}%)")
    lines.append(f"Errors with NO rater agreeing:       "
                 f"{n_found_pt - pt_agree} / {n_found_pt}  "
                 f"({100*(n_found_pt-pt_agree)/max(1,n_found_pt):.1f}%)")
    lines.append("")
    lines.append("Per-error breakdown:")
    lines.append(sep2)
    for d in pt_details:
        agree_tag = f"[{d['any_rater_agrees']}]"
        lines.append(
            f"  {agree_tag:<10}  true={d['true_label']}  pred={d['pred_label']}  "
            f"raw_labels={d['raw_rater_labels']}  norm={d['norm_rater_labels']}"
        )
        lines.append(f"             \"{d['text_excerpt']}\"")

    # ── Specific examples from thesis ────────────────────────────────
    h("SPECIFIC THESIS EXAMPLES — EXPLICIT RATER LABELS")
    lines.append("NL true=B2 pred=A (the only NL B2→A distance-2 error):")
    b2a = [d for d in nl_details
           if d["true_label"] == "B2" and d["pred_label"] == "A"]
    if b2a:
        d = b2a[0]
        lines.append(f"  raw_labels={d['raw_rater_labels']}  "
                     f"norm={d['norm_rater_labels']}  "
                     f"any_rater_agrees={d['any_rater_agrees']}")
        lines.append(f"  \"{d['text_excerpt']}\"")
    else:
        lines.append("  (not found in Dutch error list)")

    lines.append("")
    lines.append("PT true=C pred=A (the two distance-3 errors):")
    ca = [d for d in pt_details
          if d["true_label"] == "C" and d["pred_label"] == "A"]
    if ca:
        for d in ca:
            lines.append(f"  raw_labels={d['raw_rater_labels']}  "
                         f"norm={d['norm_rater_labels']}  "
                         f"any_rater_agrees={d['any_rater_agrees']}")
            lines.append(f"  \"{d['text_excerpt']}\"")
    else:
        lines.append("  (none found — likely from peapl2_pt, excluded)")

    lines.append("")
    lines.append("PT true=C pred=B1 (distance-2 errors):")
    cb1 = [d for d in pt_details
           if d["true_label"] == "C" and d["pred_label"] == "B1"]
    if cb1:
        for d in cb1:
            lines.append(f"  raw_labels={d['raw_rater_labels']}  "
                         f"norm={d['norm_rater_labels']}  "
                         f"any_rater_agrees={d['any_rater_agrees']}")
            lines.append(f"  \"{d['text_excerpt']}\"")
    else:
        lines.append("  (none found in cople2_pt)")

    # Print and save
    report = "\n".join(lines)
    print(report)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nCSV  → {OUTPUT_CSV}")
    print(f"Text → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
