# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: inspect_classifier_errors.py
# Purpose: inspect misclassified and correctly classified examples from the trained CEFR
#          classifier, writing per-split and per-language CSV files and a classification report.
# Pipeline step: Classifier error inspection (needs the model and the tokenised data).
# Thesis: Chapter 4 error analysis. Its output folder belongs to an earlier run and is
#         not included.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import argparse
import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# All paths anchor to this script's location so the repo works on any machine after cloning.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# analysis/error_analysis/ -> analysis/ -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

LABELS   = ["A", "B1", "B2", "C"]
ID2LANG  = {0: "de", 1: "en", 2: "nl", 3: "pt"}

# Default locations derived from the project root -- no machine-specific paths.
DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT, "classifier", "00_prepare_output", "tokenized_packed"
)
DEFAULT_RUNS_DIR = os.path.join(
    PROJECT_ROOT, "classifier", "02_train_output", "runs"
)
DEFAULT_OUT_DIR  = os.path.join(SCRIPT_DIR, "output_errors")


def find_latest_model_dir(runs_dir: str) -> str:
    """
    Walk timestamped run directories newest-first and return the first
    mdeberta_cefr_multi model directory found. Falls back to checking
    older xlmr_cefr_multi names for backward compatibility.
    """
    candidates = sorted(glob.glob(os.path.join(runs_dir, "cefr_*")), reverse=True)
    for run in candidates:
        for subpath in (
            "mdeberta_cefr_multi",                                       # current structure
            os.path.join("trials", "winner_refit", "mdeberta_cefr_multi"),  # old trial layout
            "xlmr_cefr_multi",                                           # very old runs
            os.path.join("trials", "winner_refit", "xlmr_cefr_multi"),
        ):
            model_dir = os.path.join(run, subpath)
            if os.path.isdir(model_dir):
                return model_dir
    raise FileNotFoundError(
        f"No model directory found under {runs_dir}.\n"
        f"Expected 'mdeberta_cefr_multi' inside a cefr_* run folder.\n"
        f"Pass the exact path with --model_dir."
    )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_predictions(model, dataset, device, batch_size=16):
    """Run batched inference manually — avoids Trainer/TF/Keras dependency."""
    ds_torch = dataset.with_format("torch")
    all_logits, all_labels = [], []
    n = len(ds_torch)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = ds_torch[start:end]
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(out.logits.cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return labels, logits.argmax(axis=1)


def decode_ids(input_ids, tokenizer) -> str:
    """Reverse tokenisation to recover close-to-original text for inspection."""
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def write_rows(split_name, y_true, y_pred, dataset, tokenizer,
               out_csv, correct=False, lang_filter=None):
    """Write a CSV of either correctly classified or misclassified rows."""
    mask = np.where(y_true == y_pred)[0] if correct else np.where(y_true != y_pred)[0]

    rows = []
    for idx in mask:
        row  = dataset[int(idx)]
        lang = ID2LANG.get(int(row["lang_id"]), "?")
        if lang_filter and lang != lang_filter:
            continue
        text = decode_ids(row["input_ids"], tokenizer)
        rows.append([
            int(idx),
            LABELS[int(y_true[idx])],
            LABELS[int(y_pred[idx])],
            lang,
            int(row["sample_id"]),
            text,
        ])

    kind  = "correct" if correct else "errors"
    label = f"{split_name} ({lang_filter})" if lang_filter else split_name
    print(f"  {label}: {len(rows)} {kind}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "true_label", "pred_label", "lang", "sample_id", "text_decoded"])
        writer.writerows(rows)
    print(f"  Saved -> {out_csv}")


def print_lang_summary(split_name, y_true, y_pred, dataset):
    lang_ids = np.array(dataset["lang_id"])
    print(f"  Per-language summary ({split_name}):")
    for lid, lname in ID2LANG.items():
        mask = lang_ids == lid
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        acc    = float((yt == yp).mean())
        f1     = float(f1_score(yt, yp, average="macro", zero_division=0))
        n_err  = int((yt != yp).sum())
        print(f"    {lname}: n={mask.sum():4d}  errors={n_err:3d}  "
              f"macro_f1={f1:.4f}  accuracy={acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect classifier errors on the packed validation and test splits."
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help="Path to the tokenized_packed DatasetDict from Step 00.",
    )
    parser.add_argument(
        "--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
        help="Directory containing timestamped training runs from Step 02.",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Explicit model directory (overrides auto-detection from --runs_dir).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=DEFAULT_OUT_DIR,
        help="Output directory for error/correct CSV files and reports.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir else find_latest_model_dir(args.runs_dir)
    print(f"Model:    {model_dir}")
    print(f"Data:     {args.data_dir}")
    print(f"Output:   {args.out_dir}")

    device    = get_device()
    print(f"Device:   {device}")

    tok_ds    = DatasetDict.load_from_disk(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("validation", "test"):
        print(f"\nProcessing {split}...")
        y_true, y_pred = get_predictions(model, tok_ds[split], device)

        # Combined file: ALL examples (correct + incorrect, all languages) in one CSV.
        # This is the primary output passed to compare_cefr_sdg.py for zero-shot comparison.
        combined_path = os.path.join(args.out_dir, f"all_{split}.csv")
        combined_rows = []
        for idx in range(len(y_true)):
            row  = tok_ds[split][int(idx)]
            lang = ID2LANG.get(int(row["lang_id"]), "?")
            text = decode_ids(row["input_ids"], tokenizer)
            combined_rows.append({
                "idx":         int(idx),
                "true_label":  LABELS[int(y_true[idx])],
                "pred_label":  LABELS[int(y_pred[idx])],
                "is_correct":  str(y_true[idx] == y_pred[idx]),
                "lang":        lang,
                "sample_id":   int(row["sample_id"]),
                "text_decoded": text,
            })
        combined_fields = ["idx", "true_label", "pred_label", "is_correct",
                           "lang", "sample_id", "text_decoded"]
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=combined_fields)
            w.writeheader()
            w.writerows(combined_rows)
        n_err = sum(1 for r in combined_rows if r["is_correct"] == "False")
        print(f"  all_{split}.csv: {len(combined_rows)} rows "
              f"({n_err} errors, {len(combined_rows)-n_err} correct) -> {combined_path}")

        # Detailed per-language / per-type breakdowns (kept for reference)
        write_rows(split, y_true, y_pred, tok_ds[split], tokenizer,
                   os.path.join(args.out_dir, f"errors_{split}.csv"), correct=False)
        for lang in ID2LANG.values():
            write_rows(split, y_true, y_pred, tok_ds[split], tokenizer,
                       os.path.join(args.out_dir, f"errors_{split}_{lang}.csv"),
                       correct=False, lang_filter=lang)

        print_lang_summary(split, y_true, y_pred, tok_ds[split])

        # Classification report
        report_str = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
        print(f"\n  Classification report ({split}):\n{report_str}")
        report_path = os.path.join(args.out_dir, f"classification_report_{split}.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"  Saved -> {report_path}")

        # Confusion matrix CSV
        cm      = confusion_matrix(y_true, y_pred)
        cm_path = os.path.join(args.out_dir, f"confusion_matrix_{split}.csv")
        with open(cm_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred"] + LABELS)
            for i, row in enumerate(cm):
                writer.writerow([LABELS[i]] + row.tolist())
        print(f"  Saved confusion matrix -> {cm_path}")

        # Adjacent-error rate (off-by-one on the ordinal scale)
        errors_mask = y_true != y_pred
        n_errors    = int(errors_mask.sum())
        if n_errors:
            n_adjacent = int(sum(
                abs(int(t) - int(p)) == 1
                for t, p in zip(y_true[errors_mask], y_pred[errors_mask])
            ))
            print(f"  Adjacent errors (off by 1 level): "
                  f"{n_adjacent}/{n_errors} = {100*n_adjacent/n_errors:.1f}%")

    print(f"\nPrimary outputs for zero-shot comparison:")
    print(f"  {os.path.join(args.out_dir, 'all_validation.csv')}")
    print(f"  {os.path.join(args.out_dir, 'all_test.csv')}")
    print(f"\nAll files saved to {args.out_dir}")


if __name__ == "__main__":
    main()
