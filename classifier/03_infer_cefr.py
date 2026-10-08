# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 03_infer_cefr.py
# Purpose: run URL-level CEFR inference for one or all countries using the final classifier.
# Pipeline step: CEFR inference on the portal pages (512-token windows, stride 128).
# Thesis: Page-level CEFR labels and probabilities (<country>_cefr.csv) used in Chapters 5 and 6.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# ──────────────────────────────────────────────────────────────────────────────
# Paths anchored to this script's location (country_data/)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Current final model (DE/EN/NL retrain, Portuguese dropped). Pass --model-dir to override.
DEFAULT_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "classifier", "02_train_output", "runs",
    "cefr_20260817_231925", "mdeberta_cefr_multi",
)

# Tokenizer parameters must match the classifier training setup exactly
WINDOW_TOKENS = 512
STRIDE_TOKENS = 128

# ──────────────────────────────────────────────────────────────────────────────
# Per-country configuration
# Every path is derived from the country key; no hard-coded absolute paths.
# ──────────────────────────────────────────────────────────────────────────────
COUNTRY_CONFIG: Dict[str, Dict] = {
    "germany": {
        "categories": [
            "hilfs-problemloesungsdienste",
            "hilfs-problemloesungsdienste_ls",
            "lagen",
            "leistung",
            "rechte_pflichten",
        ],
    },
    "austria": {
        "categories": [
            "gesetzliche-neuerungen",
            "landingpages",
            "lebenslagen",
            "themen",
        ],
    },
    "ireland": {
        "categories": [
            "campaigns",
            "policies",
            "services",
        ],
    },
    "netherlands": {
        "categories": [
            "nl_onderwerpen",
        ],
    },
}

# CEFR level -> numeric index for computing averages
LABEL_TO_IDX = {"A": 1, "B1": 2, "B2": 3, "C": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# Output column order (country/raw_category/url match the SDG file for joining)
PAGE_FIELDS = [
    "country", "raw_category", "url",
    "pred_label", "pred_confidence",
    "prob_A", "prob_B1", "prob_B2", "prob_C",
]


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def combined_csv(country: str) -> str:
    return os.path.join(
        PROJECT_ROOT, "country_pipelines", country,
        f"02_scrape_output_{country}", f"combined_{country}.csv",
    )


def output_dir(country: str) -> str:
    return os.path.join(PROJECT_ROOT, "country_pipelines", country, f"03_cefr_output_{country}")


def per_country_csv(country: str) -> str:
    return os.path.join(output_dir(country), f"{country}_cefr.csv")


def summary_csv(country: str) -> str:
    return os.path.join(output_dir(country), f"{country}_cefr_summary.csv")


MASTER_CSV = os.path.join(SCRIPT_DIR, "all_countries_cefr.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Device selection (MPS > CUDA > CPU for Mac support)
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: str):
    """Load tokenizer + model. Raises a clear error if the directory is missing."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Pass the correct path with --model-dir, or check that the run "
            f"cefr_20260625_041029 finished and saved mdeberta_cefr_multi."
        )
    config    = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Recover label list from config (keys may be ints or strings)
    id2label = config.id2label
    try:
        label_ids = sorted(id2label.keys())
    except TypeError:
        label_ids = sorted(id2label.keys(), key=lambda x: int(x))
    labels = [id2label[k] for k in label_ids]

    device = get_device()
    model.to(device)
    return tokenizer, model, labels, device


# ──────────────────────────────────────────────────────────────────────────────
# Inference: token-based sliding window + logit mean-pooling
# Matches aggregate_doc_level() in 02_train_final.py: mean-pool raw logits
# across all windows, then softmax + argmax for label and confidence.
# ──────────────────────────────────────────────────────────────────────────────

def predict_page(
    text: str,
    tokenizer,
    model,
    labels: List[str],
    device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    """Return (pred_label, confidence, {label: prob}) for a single page."""
    if not text or not text.strip():
        return "", 0.0, {lbl: 0.0 for lbl in labels}

    # Tokenize with overlapping windows (no_batch, multiple overflow windows)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=WINDOW_TOKENS,
        stride=STRIDE_TOKENS,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )

    all_logits: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(enc["input_ids"].shape[0]):
            batch = {
                k: enc[k][i : i + 1].to(device)
                for k in ("input_ids", "attention_mask")
                if k in enc
            }
            out = model(**batch)
            all_logits.append(out.logits[0].cpu())  # (n_classes,)

    if not all_logits:
        return "", 0.0, {lbl: 0.0 for lbl in labels}

    # Mean of logits over all windows, then softmax for calibrated probabilities
    mean_logits = torch.stack(all_logits).mean(dim=0)
    probs       = torch.softmax(mean_logits, dim=0).numpy()

    best_idx  = int(probs.argmax())
    best_label = labels[best_idx]
    best_conf  = float(probs[best_idx])
    prob_dict  = {lbl: float(probs[i]) for i, lbl in enumerate(labels)}
    return best_label, best_conf, prob_dict


# ──────────────────────────────────────────────────────────────────────────────
# CEFR index utilities
# ──────────────────────────────────────────────────────────────────────────────

def avg_cefr(pred_labels: List[str]) -> Tuple[float, str]:
    """Average numeric CEFR index and the nearest discrete label."""
    vals = [LABEL_TO_IDX[l] for l in pred_labels if l in LABEL_TO_IDX]
    if not vals:
        return 0.0, ""
    avg = sum(vals) / len(vals)
    nearest = min(IDX_TO_LABEL, key=lambda x: abs(x - avg))
    return avg, IDX_TO_LABEL[nearest]


# ──────────────────────────────────────────────────────────────────────────────
# Per-country run
# ──────────────────────────────────────────────────────────────────────────────

def run_country(
    country: str,
    tokenizer,
    model,
    labels: List[str],
    device: torch.device,
) -> List[Dict]:
    """
    Classify every page in combined_<country>.csv. Writes the flat per-country
    CSV and the summary CSV. Returns the list of result dicts for the master file.
    """
    in_path = combined_csv(country)
    if not os.path.exists(in_path):
        print(f"  [WARN] Input not found, skipping {country}: {in_path}")
        return []

    os.makedirs(output_dir(country), exist_ok=True)

    # Read all pages
    rows = []
    with open(in_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {country}: {len(rows)} pages")

    results: List[Dict] = []
    labels_by_cat: Dict[str, List[str]] = {}

    for idx, row in enumerate(rows, 1):
        url      = row.get("url", "").strip()
        text     = row.get("text_joined", "").strip()
        raw_cat  = row.get("category", "").strip()

        pred_label, conf, prob_dict = predict_page(text, tokenizer, model, labels, device)

        rec = {
            "country":          country,
            "raw_category":     raw_cat,
            "url":              url,
            "pred_label":       pred_label,
            "pred_confidence":  f"{conf:.4f}",
            "prob_A":           f"{prob_dict.get('A',  0.0):.4f}",
            "prob_B1":          f"{prob_dict.get('B1', 0.0):.4f}",
            "prob_B2":          f"{prob_dict.get('B2', 0.0):.4f}",
            "prob_C":           f"{prob_dict.get('C',  0.0):.4f}",
        }
        results.append(rec)

        if raw_cat not in labels_by_cat:
            labels_by_cat[raw_cat] = []
        if pred_label:
            labels_by_cat[raw_cat].append(pred_label)

        if idx % 200 == 0:
            print(f"    {idx}/{len(rows)} done", flush=True)

    # Write flat per-country CSV
    with open(per_country_csv(country), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAGE_FIELDS)
        w.writeheader()
        w.writerows(results)
    print(f"  Wrote {len(results)} rows -> {per_country_csv(country)}")

    # Write summary CSV (URL-level granularity only)
    summary_rows = []
    all_labels: List[str] = []
    for cat in COUNTRY_CONFIG[country]["categories"]:
        cat_labels = labels_by_cat.get(cat, [])
        all_labels.extend(cat_labels)
        avg_idx, avg_lbl = avg_cefr(cat_labels)
        summary_rows.append({
            "granularity": "url",
            "category":    cat,
            "n_items":     len(cat_labels),
            "avg_cefr_index": f"{avg_idx:.4f}" if avg_idx else "",
            "avg_cefr_label": avg_lbl,
        })

    avg_idx_all, avg_lbl_all = avg_cefr(all_labels)
    summary_rows.append({
        "granularity": "url",
        "category":    "ALL",
        "n_items":     len(all_labels),
        "avg_cefr_index": f"{avg_idx_all:.4f}" if avg_idx_all else "",
        "avg_cefr_label": avg_lbl_all,
    })

    sum_fields = ["granularity", "category", "n_items", "avg_cefr_index", "avg_cefr_label"]
    with open(summary_csv(country), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Summary -> {summary_csv(country)}")

    # Print label distribution
    dist = {}
    for r in results:
        lbl = r["pred_label"]
        dist[lbl] = dist.get(lbl, 0) + 1
    dist_str = "  ".join(f"{k}={v}" for k, v in sorted(dist.items()))
    print(f"  Distribution: {dist_str}")
    print(f"  Overall avg CEFR index: {avg_idx_all:.3f} ({avg_lbl_all})")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CEFR inference for government pages (URL level, all countries)."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--country",
        choices=list(COUNTRY_CONFIG.keys()),
        help="Run inference for a single country.",
    )
    grp.add_argument(
        "--all",
        action="store_true",
        help="Run inference for all four countries in sequence.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=(
            f"Path to the trained mDeBERTa model directory "
            f"(default: {DEFAULT_MODEL_DIR})."
        ),
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    countries = list(COUNTRY_CONFIG.keys()) if args.all else [args.country]

    print(f"Model:   {model_dir}")
    print(f"Device:  {get_device()}")
    print(f"Window:  {WINDOW_TOKENS} tokens, stride {STRIDE_TOKENS}")
    print(f"Countries: {countries}")
    print()

    print("Loading model...")
    tokenizer, model, labels, device = load_model(model_dir)
    print(f"Labels: {labels}")
    print()

    all_results: List[Dict] = []

    for country in countries:
        print(f"=== {country.upper()} ===")
        results = run_country(country, tokenizer, model, labels, device)
        all_results.extend(results)
        print()

    # Write master file
    if all_results:
        with open(MASTER_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PAGE_FIELDS)
            w.writeheader()
            w.writerows(all_results)
        print(f"Master file: {MASTER_CSV}  ({len(all_results)} rows)")
    else:
        print("No results written.")


if __name__ == "__main__":
    main()
