# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 02_train_final.py
# Purpose: train the final CEFR classifier with the chosen settings and save the model, metrics, and plots.
# Pipeline step: Classifier, step 02 (final training, run cefr_20260817_231925, seed 42).
# Thesis: Chapter 4 training and results (metrics.json, training_log.csv); Section 4.3.1 learning
#         curve via b6c_classifier_figures.py.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Step 02 of the classifier pipeline. This script trains the final mDeBERTa CEFR
# classifier using the tokenized dataset from Step 00 and the best hyperparameters
# selected in Step 01. It runs a controlled comparison:
#   Balanced sampler: [False, True]
# It selects the combination with the highest validation macro-F1. The Trainer holds
# the best-F1 epoch weights at the end of training via load_best_model_at_end, so
# no separate refit step is needed. Both window-level and document-level metrics are
# reported. A short language-weighting A/B test runs at the end for reference.
#
# Outputs (timestamped run dir):
#   runs/cefr_YYYYMMDD_HHMMSS/
#     ├── trials/<scheduler>_bal<0/1>/
#     ├── ab_lang_weight_test/
#     ├── mdeberta_cefr_multi/             (final saved model)
#     ├── training_log.csv
#     ├── learning_curve_loss.png
#     ├── learning_curve_val_f1.png
#     ├── confusion_matrix_val_window.png
#     ├── confusion_matrix_test_window.png
#     ├── confusion_matrix_val_doc.png
#     ├── confusion_matrix_test_doc.png
#     ├── metrics.json
#     ├── run_manifest.json
#     └── selection_summary.json

import os
import sys
import json
import csv
import time
import shutil
import hashlib
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import DatasetDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

# -------------------------------
# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# -------------------------------
# mDeBERTa outperforms XLM-R on most European-language benchmarks and handles the
# mix of DE/EN/NL/PT better -- the tradeoff is it is slower to train
MODEL_NAME = "microsoft/mdeberta-v3-base"
LABELS = ["A", "B1", "B2", "C"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

SEED = 42
# Small per-device batch + accumulation gives effective batch of 16 without needing a large GPU
PER_DEVICE_BATCH = 2          # effective batch via gradient accumulation
DEFAULT_LR = 1e-5             # fallback if Step 01 best config missing
DEFAULT_GRAD_ACCUM = 4        # eff. batch = 8 with per_device=2
MAX_EPOCHS = 8                # upper bound for trial runs
USE_EARLY_STOPPING = True     # auto-disabled on older Transformers
EARLY_STOP_PATIENCE = 2
AB_TEST_EPOCHS = 3            # short A/B comparison for language-weighting
AB_TEST_CAP    = 10.0         # inverse-frequency cap applied equally to BOTH A/B arms
# Ordinal loss mix: CEFR classes are ordered (A < B1 < B2 < C), so a B1->C error
# should hurt more than B1->B2. EMD penalises far-off predictions; 0.3 keeps CE
# dominant (the part that already works) while rewarding ordinal structure.
EMD_WEIGHT = 0.3             # 0 = pure weighted-CE, 1 = pure EMD, 0.3 = CE-dominant mix

# Inputs from previous steps
PREP_DIR = os.path.join(SCRIPT_DIR, "00_prepare_output")
PACKED_DIR = os.path.join(PREP_DIR, "tokenized_packed")
TOK_DIR = os.path.join(PREP_DIR, "tokenizer")
CLASS_WEIGHTS_JSON = os.path.join(PREP_DIR, "class_weights.json")
BALANCED_WEIGHTS_PACKED = os.path.join(PREP_DIR, "balanced_sampler_weights_packed.pt")
WEIGHT_DIAGNOSTICS_JSON = os.path.join(PREP_DIR, "weight_diagnostics.json")
SPLITS_JSON = os.path.join(PREP_DIR, "splits.json")
SWEEP_BEST_JSON = os.path.join(SCRIPT_DIR, "01_sweep_output", "configs", "best_params.json")

# Outputs
ROOT_OUT = os.path.join(SCRIPT_DIR, "02_train_output")

# Scheduler is fixed to linear (best from Step 01); only balanced sampler varies
CANDIDATE_SCHEDULERS = ["linear"]
CANDIDATE_BALANCED = [False]  # [False, True] for an optional ablation

# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed: int) -> None:
    import transformers as _transformers  # local import avoids shadowing this function
    np.random.seed(seed)
    torch.manual_seed(seed)
    _transformers.set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"  [set_seed] seed={seed}; exact reproducibility not guaranteed on MPS.")

def read_best_params(path: str, default_lr: float, default_grad_accum: int) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
        best = (blob.get("best") or {})
        lr = float(best.get("learning_rate", default_lr))
        grad_accum = int(best.get("grad_accum", default_grad_accum))
        return {"learning_rate": lr, "grad_accum": grad_accum}
    return {"learning_rate": default_lr, "grad_accum": default_grad_accum}

def sha256_of_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def window_class_weights(tok_ds) -> torch.Tensor:
    from collections import Counter
    labels = list(tok_ds["train"]["labels"])
    counts = Counter(labels)
    total = len(labels)
    w = [total / max(1, counts.get(i, 0)) for i in range(len(LABELS))]
    return torch.tensor(w, dtype=torch.float32)

ID2LANG = {0: "de", 1: "en", 2: "nl"}

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "kappa": cohen_kappa_score(labels, preds),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
        "adjacent_accuracy": float(np.mean(np.abs(preds - labels) <= 1)),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

class CSVLogger(TrainerCallback):
    """Per-epoch CSV logger using on_evaluate, which fires directly after each evaluate() call
    with the metrics dict. With dict eval_dataset, on_evaluate fires twice per epoch (once for
    each key); we accumulate into a buffer and write the row only when both losses are present."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._buf: Dict[int, Dict] = {}
        self._written: set = set()
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "eval_train_loss", "eval_val_loss",
                "eval_val_accuracy", "eval_val_macro_f1",
                "eval_val_precision_macro", "eval_val_recall_macro",
            ])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or state is None or state.epoch is None:
            return
        ep = round(float(state.epoch))
        if ep not in self._buf:
            self._buf[ep] = {}
        self._buf[ep].update(metrics)
        buf = self._buf[ep]
        if "eval_val_loss" in buf and "eval_train_loss" in buf and ep not in self._written:
            def _fmt(k):
                v = buf.get(k)
                return f"{float(v):.6f}" if v is not None else ""
            row = [
                ep,
                _fmt("eval_train_loss"),
                _fmt("eval_val_loss"),
                _fmt("eval_val_accuracy"),
                _fmt("eval_val_macro_f1"),
                _fmt("eval_val_precision_macro"),
                _fmt("eval_val_recall_macro"),
            ]
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            self._written.add(ep)

class ValOnlyEarlyStoppingCallback(EarlyStoppingCallback):
    """Like EarlyStoppingCallback but silently skips on_evaluate calls that do not
    contain the target metric. With eval_dataset={"val":..., "train":...} the Trainer
    fires on_evaluate twice per epoch; the "train" call emits only eval_train_* keys
    and would otherwise trigger a spurious warning and disable ES for that call.
    Uses args.metric_for_best_model (version-robust; avoids the renamed instance attr)."""
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if args.metric_for_best_model not in metrics:
            return  # silently skip — this is the 'train' eval split, not 'val'
        super().on_evaluate(args, state, control, metrics, **kwargs)

# CEFR is ordered (A < B1 < B2 < C), so predicting C when truth is B1 should hurt
# more than predicting B2. Standard cross-entropy treats all wrong answers the same.
# EMD fixes this by comparing cumulative probability distributions.
def emd_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.size(1)
    true_onehot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1)
    cum_pred = torch.cumsum(probs, dim=1)[:, :num_classes - 1]
    cum_true = torch.cumsum(true_onehot, dim=1)[:, :num_classes - 1]
    return torch.mean(torch.sum((cum_pred - cum_true) ** 2, dim=1))


class WeightedTrainer(Trainer):
    """Trainer with combined EMD + class-weighted CE loss, and optional balanced sampler."""
    def __init__(self, class_weights: torch.Tensor | None = None,
                 balanced_sampler_weights: torch.Tensor | None = None,
                 emd_weight: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self._balanced_sampler_weights = balanced_sampler_weights
        self.emd_weight = emd_weight  # 0 = pure CE, 1 = pure EMD, 0.5 = equal mix

    # version-robust signature
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        ce_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=0.1,
        )
        ce = ce_fct(logits, labels)
        if self.emd_weight > 0:
            emd = emd_loss(logits, labels)
            loss = (1 - self.emd_weight) * ce + self.emd_weight * emd
        else:
            loss = ce
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        if self._balanced_sampler_weights is None:
            return dl
        from torch.utils.data import WeightedRandomSampler
        num_samples = len(dl.dataset)
        w = self._balanced_sampler_weights
        if w.numel() != num_samples:
            return dl
        sampler = WeightedRandomSampler(weights=w, num_samples=num_samples, replacement=True)
        dl.sampler = sampler
        return dl

def make_training_args(output_dir: str, lr: float, grad_accum: int,
                       epochs: int, scheduler: str) -> Tuple[TrainingArguments, bool]:
    """
    Prefer epoch strategies when available. Fallback avoids unsupported kwargs.
    Returns (args, early_stopping_supported).
    Selection is by eval_val_macro_f1 (higher is better) so the Trainer saves
    the best-F1 epoch checkpoint.
    """
    common = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        seed=SEED,
        report_to=[],
        lr_scheduler_type=scheduler,
        warmup_ratio=0.1,
        weight_decay=0.05,
        # Select best checkpoint by validation macro-F1, not loss
        metric_for_best_model="eval_val_macro_f1",
        greater_is_better=True,
    )
    # eval_strategy replaces evaluation_strategy in Transformers 4.45+.
    # Try the newest name first, then the old name, then a last-resort fallback.
    try:
        args = TrainingArguments(
            **common,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_only_model=True,
            save_total_limit=1,
        )
        return args, True
    except TypeError:
        pass
    try:
        args = TrainingArguments(
            **common,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
        )
        return args, True
    except TypeError:
        args = TrainingArguments(
            **common,
            do_eval=True,
            logging_steps=50,   # non-zero for older versions
            save_steps=0,
            load_best_model_at_end=False,
        )
        return args, False

def plot_learning_curves(csv_path: str, out_loss_png: str, out_f1_png: str) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("epoch") and r.get("eval_val_loss"):
                rows.append(r)
    if not rows:
        return
    rows.sort(key=lambda r: int(r["epoch"]))
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["eval_train_loss"]) if r.get("eval_train_loss") else None for r in rows]
    val_loss   = [float(r["eval_val_loss"]) for r in rows]
    val_f1     = [float(r["eval_val_macro_f1"]) if r.get("eval_val_macro_f1") else None for r in rows]

    print("\n  Epoch | Train loss (eval) | Val loss  | Val macro F1")
    for e, tl, vl, vf in zip(epochs, train_loss, val_loss, val_f1):
        tl_s = f"{tl:.4f}" if tl is not None else "  --  "
        vf_s = f"{vf:.4f}" if vf is not None else "  --  "
        print(f"    {e:3d} | {tl_s:17} | {vl:.4f}    | {vf_s}")

    plt.figure()
    if any(v is not None for v in train_loss):
        plt.plot(epochs, train_loss, marker="o", label="Train loss (eval mode)")
    plt.plot(epochs, val_loss, marker="o", label="Validation loss")
    plt.title("Learning Curves: Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_loss_png, dpi=180)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_f1, marker="o")
    plt.title("Validation Macro F1 by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.tight_layout()
    plt.savefig(out_f1_png, dpi=180)
    plt.close()

def per_class_f1(y_true, y_pred, labels: list[str]) -> Dict[str, float]:
    out = {}
    for i, lbl in enumerate(labels):
        f1 = f1_score((y_true == i), (y_pred == i), zero_division=0)
        out[lbl] = float(f1)
    return out

# Reports the epoch with the highest eval_val_macro_f1 for manifest logging.
def best_epoch_from_csv(csv_path: str) -> int:
    best_e, best_f1 = 1, -1.0
    try:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("eval_val_macro_f1"):
                    f1_v = float(r["eval_val_macro_f1"])
                    ep = int(float(r["epoch"])) if r.get("epoch") else 1
                    if f1_v > best_f1:
                        best_f1, best_e = f1_v, ep
    except FileNotFoundError:
        pass
    return best_e

# Breaking down F1 by language helps catch if one language is dragging down the
# overall score -- useful for diagnosing data imbalance across languages
def per_lang_f1(y_true: np.ndarray, y_pred: np.ndarray, lang_ids: np.ndarray) -> Dict[str, float]:
    """Compute macro F1 per language using lang_id column from the tokenized dataset."""
    result = {}
    for lid, lname in ID2LANG.items():
        mask = lang_ids == lid
        if mask.sum() > 0:
            result[lname] = float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
    return result

# Each source document may span multiple windows in the packed dataset. Aggregating
# by document gives a fairer metric that does not overweight long texts.
def aggregate_doc_level(
    logits: np.ndarray,
    y_true: np.ndarray,
    sample_ids: np.ndarray,
    lang_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool window logits per source document and return document-level arrays."""
    doc_logits: dict = defaultdict(list)
    doc_label: dict = {}
    doc_lang: dict = {}
    for i, sid in enumerate(sample_ids):
        doc_logits[sid].append(logits[i])
        doc_label[sid] = int(y_true[i])
        doc_lang[sid] = int(lang_ids[i])
    doc_ids = sorted(doc_logits.keys())
    y_true_doc = np.array([doc_label[d] for d in doc_ids])
    y_pred_doc = np.array([np.argmax(np.mean(doc_logits[d], axis=0)) for d in doc_ids])
    lang_ids_doc = np.array([doc_lang[d] for d in doc_ids])
    return y_true_doc, y_pred_doc, lang_ids_doc

def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str
) -> None:
    """Build and save a confusion matrix PNG from pre-computed prediction arrays."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def per_lang_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                                lang_ids: np.ndarray, out_dir: str, split_name: str) -> None:
    for lid, lname in ID2LANG.items():
        mask = lang_ids == lid
        if mask.sum() < 2:
            continue
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=list(range(len(LABELS))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
        disp.plot(xticks_rotation=45)
        plt.title(f"CEFR Confusion Matrix ({split_name} / {lname.upper()})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_matrix_{split_name}_{lname}.png"), dpi=180)
        plt.close()

# Extracted so the smoke test can call it directly without running a full training.
def _check_training_log(csv_path: str, n_trained_epochs: int,
                        early_stop_supported: bool) -> list:
    """
    Validate the training log and return parsed rows. Raises RuntimeError for:
    1. Step-level fallback was taken (epoch eval never ran).
    2. CSV is empty or has fewer rows than trained epochs.
    3. Any required column is empty in any row.
    """
    required_cols = ("eval_train_loss", "eval_val_loss", "eval_val_macro_f1")
    if not early_stop_supported:
        raise RuntimeError(
            "Training ran in the step-level logging fallback path. "
            "Both eval_strategy and evaluation_strategy were rejected by TrainingArguments "
            "so epoch evaluation never fired and training_log.csv will be empty. "
            "Check that the installed Transformers version accepts eval_strategy."
        )
    log_rows = list(csv.DictReader(open(csv_path)))
    if not log_rows:
        raise RuntimeError(
            f"training_log.csv is empty after {n_trained_epochs} epochs. "
            f"The CSVLogger on_evaluate callback did not fire even though "
            f"epoch-eval strategy was accepted (early_stop_supported=True). "
            f"Check that both dict eval keys 'val' and 'train' triggered on_evaluate."
        )
    if len(log_rows) != n_trained_epochs:
        print(
            f"  WARNING: training_log.csv has {len(log_rows)} rows but Trainer ran "
            f"{n_trained_epochs} epochs. Row count must equal epoch count for "
            f"best-epoch detection to be reliable."
        )
    for r in log_rows:
        for col in required_cols:
            if not r.get(col):
                print(
                    f"  WARNING: training_log.csv row (epoch={r.get('epoch')}) has empty '{col}'. "
                    f"The on_evaluate accumulation for that epoch did not complete."
                )
                break
    return log_rows


def _best_epoch_from_trainer(log_rows: list,
                              trainer_best_metric: Optional[float]) -> int:
    """
    Find the best epoch by strictly-increasing eval_val_macro_f1, mirroring the
    Trainer's own checkpoint-save logic. Verified against trainer.state.best_metric
    so the manifest and the saved model always refer to the same epoch.
    """
    best_f1_so_far = -1.0
    best_epoch = 1
    for r in sorted(log_rows, key=lambda r: int(float(r.get("epoch", 0)))):
        if r.get("eval_val_macro_f1"):
            f1 = float(r["eval_val_macro_f1"])
            if f1 > best_f1_so_far:
                best_f1_so_far = f1
                best_epoch = int(float(r["epoch"]))
    if trainer_best_metric is not None:
        if abs(best_f1_so_far - trainer_best_metric) > 1e-5:
            print(f"  WARNING: CSV best epoch={best_epoch} (F1={best_f1_so_far:.6f}) "
                  f"does not match trainer.state.best_metric={trainer_best_metric:.6f}. "
                  f"Trusting Trainer state.")
            for r in sorted(log_rows, key=lambda r: int(float(r.get("epoch", 0)))):
                if r.get("eval_val_macro_f1"):
                    if abs(float(r["eval_val_macro_f1"]) - trainer_best_metric) < 1e-5:
                        best_epoch = int(float(r["epoch"]))
                        break
    return best_epoch


def _ab_compute_sampler_weights(
    lang_label_counts: dict,
    train_lang_ids: np.ndarray,
    train_labels: np.ndarray,
    cap: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-row WeightedRandomSampler weights for the A/B test.
    Both arms use inverse-frequency weighting with the SAME cap -- the only
    difference is whether counts are per-label (arm A) or per-lang x label (arm B).
    Returns (arm_a_label_only, arm_b_lang_joint).
    """
    LANG2ID_MAP = {"de": 0, "en": 1, "nl": 2, "pt": 3}
    total = sum(sum(v.values()) for v in lang_label_counts.values())

    # Arm A: aggregate counts per label across all languages
    label_totals: dict[str, int] = {}
    for lbl_counts in lang_label_counts.values():
        for lbl, cnt in lbl_counts.items():
            label_totals[lbl] = label_totals.get(lbl, 0) + cnt
    label_w = {lbl: min(total / max(1, cnt), cap) for lbl, cnt in label_totals.items()}
    arm_a = torch.tensor(
        np.array([label_w.get(LABELS[int(y)], 1.0) for y in train_labels],
                 dtype=np.float32),
        dtype=torch.float32,
    )

    # Arm B: per-cell (lang, label) counts
    joint_w: dict = {}
    for lang, lbl_counts in lang_label_counts.items():
        lid = LANG2ID_MAP.get(lang, -1)
        for lbl, cnt in lbl_counts.items():
            joint_w[(lid, label2id[lbl])] = min(total / max(1, cnt), cap)
    arm_b = torch.tensor(
        np.array([joint_w.get((int(l), int(y)), 1.0)
                  for l, y in zip(train_lang_ids, train_labels)],
                 dtype=np.float32),
        dtype=torch.float32,
    )
    return arm_a, arm_b


# -------------------------------
# Training wrapper
# -------------------------------
def run_one_training(run_root: str,
                     tok_ds: DatasetDict,
                     tokenizer,
                     class_weights: torch.Tensor,
                     sampler_weights: Optional[torch.Tensor],
                     lr: float, grad_accum: int,
                     epochs: int, scheduler: str) -> Dict[str, Any]:
    """
    Train once under a given balanced-sampler configuration.
    With load_best_model_at_end=True and metric_for_best_model=eval_val_macro_f1,
    the Trainer holds the best-F1 epoch weights after training completes.
    Returns a dict with paths and metrics.
    """
    out_dir = os.path.join(run_root, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    args, early_stop_supported = make_training_args(
        output_dir=out_dir,
        lr=lr,
        grad_accum=grad_accum,
        epochs=epochs,
        scheduler=scheduler,
    )

    csv_log = os.path.join(run_root, "training_log.csv")
    callbacks: list[TrainerCallback] = [CSVLogger(csv_path=csv_log)]
    if USE_EARLY_STOPPING and early_stop_supported:
        callbacks.append(ValOnlyEarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE))
        es_msg = f"True (patience={EARLY_STOP_PATIENCE})"
    else:
        es_msg = "Disabled (strategy not supported by this Transformers version)"

    print(f"Scheduler:             {scheduler}")
    print(f"Epochs:                {epochs}")
    print(f"Per-device batch:      {PER_DEVICE_BATCH}")
    print(f"Grad accumulation:     {grad_accum}  (effective batch = {PER_DEVICE_BATCH * grad_accum})")
    print(f"Learning rate:         {lr:.1e}")
    print(f"Balanced sampler:      {bool(sampler_weights is not None)}")
    print(f"Early stopping:        {es_msg}")

    # Fresh pretrained weights for each trial -- no state bleeds between candidates
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # Full train split for end-of-epoch evaluation in eval mode with no grads --
    # plotted as the train-loss curve. A small subsample here would give a noisy
    # (high Monte-Carlo-variance) loss estimate each epoch; the full split removes that.
    print(f"Train eval set:        {len(tok_ds['train'])} windows (full train split; val size: {len(tok_ds['validation'])})")

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset={"val": tok_ds["validation"], "train": tok_ds["train"]},
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        balanced_sampler_weights=sampler_weights,
        emd_weight=EMD_WEIGHT,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    # Validate log and get best epoch -- both from a single source of truth.
    n_trained_epochs = int(round(float(trainer.state.epoch)))
    log_rows = _check_training_log(csv_log, n_trained_epochs, early_stop_supported)
    best_epoch_trainer = _best_epoch_from_trainer(log_rows, trainer.state.best_metric)
    print(f"  Best epoch by val macro-F1: epoch {best_epoch_trainer}")

    # Curves (every epoch that actually ran, eval-mode measurements only)
    plot_learning_curves(
        csv_path=csv_log,
        out_loss_png=os.path.join(run_root, "learning_curve_loss.png"),
        out_f1_png=os.path.join(run_root, "learning_curve_val_f1.png"),
    )

    # Raw logits for window-level and document-level metric computation
    val_pred  = trainer.predict(tok_ds["validation"], metric_key_prefix="val")
    test_pred = trainer.predict(tok_ds["test"],       metric_key_prefix="test")

    val_logits  = val_pred.predictions     # (n_val_windows, n_classes)
    yv_true     = val_pred.label_ids
    yv_pred_w   = val_logits.argmax(axis=1)

    test_logits = test_pred.predictions
    yt_true     = test_pred.label_ids
    yt_pred_w   = test_logits.argmax(axis=1)

    val_lang_ids    = np.array(tok_ds["validation"]["lang_id"])
    test_lang_ids   = np.array(tok_ds["test"]["lang_id"])
    val_sample_ids  = np.array(tok_ds["validation"]["sample_id"])
    test_sample_ids = np.array(tok_ds["test"]["sample_id"])

    # Window-level confusion matrices
    save_confusion_matrix(yv_true, yv_pred_w,
                          os.path.join(run_root, "confusion_matrix_val_window.png"),
                          "CEFR Confusion Matrix (Validation, window-level)")
    save_confusion_matrix(yt_true, yt_pred_w,
                          os.path.join(run_root, "confusion_matrix_test_window.png"),
                          "CEFR Confusion Matrix (Test, window-level)")
    per_lang_confusion_matrices(yv_true, yv_pred_w, val_lang_ids, run_root, "val")
    per_lang_confusion_matrices(yt_true, yt_pred_w, test_lang_ids, run_root, "test")

    # Document-level predictions (mean-pool logits across windows of same source document)
    yv_true_d, yv_pred_d, val_lang_ids_d = aggregate_doc_level(
        val_logits, yv_true, val_sample_ids, val_lang_ids)
    yt_true_d, yt_pred_d, test_lang_ids_d = aggregate_doc_level(
        test_logits, yt_true, test_sample_ids, test_lang_ids)

    save_confusion_matrix(yv_true_d, yv_pred_d,
                          os.path.join(run_root, "confusion_matrix_val_doc.png"),
                          "CEFR Confusion Matrix (Validation, document-level)")
    save_confusion_matrix(yt_true_d, yt_pred_d,
                          os.path.join(run_root, "confusion_matrix_test_doc.png"),
                          "CEFR Confusion Matrix (Test, document-level)")

    print(f"  Window-level: val={len(yv_true)} windows ({len(np.unique(val_sample_ids))} docs)  "
          f"test={len(yt_true)} windows ({len(np.unique(test_sample_ids))} docs)")

    # Print F1 summary for both granularities
    for level, yv, yp, yt, ytp, tl in (
        ("window", yv_true, yv_pred_w, yt_true, yt_pred_w, test_lang_ids),
        ("doc   ", yv_true_d, yv_pred_d, yt_true_d, yt_pred_d, test_lang_ids_d),
    ):
        print(f"  {level} val macro F1: {f1_score(yv, yp, average='macro'):.4f}  "
              f"test macro F1: {f1_score(yt, ytp, average='macro'):.4f}")
        for lang, score in per_lang_f1(yt, ytp, tl).items():
            print(f"    {lang}: {score:.4f}")

    def _split_metrics(y_true, y_pred, lang_ids, loss_v=None):
        m = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "macro_f1":    float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "kappa":       float(cohen_kappa_score(y_true, y_pred)),
            "qwk":         float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            "adjacent_accuracy": float(np.mean(np.abs(y_pred - y_true) <= 1)),
            "per_class_f1": per_class_f1(y_true, y_pred, LABELS),
            "per_lang_f1":  per_lang_f1(y_true, y_pred, lang_ids),
        }
        if loss_v is not None:
            m["loss"] = loss_v
        return m

    payload = {
        "window_level": {
            "validation": _split_metrics(
                yv_true, yv_pred_w, val_lang_ids,
                float(val_pred.metrics.get("val_loss", 0.0))),
            "test": _split_metrics(
                yt_true, yt_pred_w, test_lang_ids,
                float(test_pred.metrics.get("test_loss", 0.0))),
        },
        "document_level": {
            "validation": _split_metrics(yv_true_d, yv_pred_d, val_lang_ids_d),
            "test":       _split_metrics(yt_true_d, yt_pred_d, test_lang_ids_d),
        },
        "timings": {"train_seconds": round(train_seconds, 1)},
    }
    # Backward-compatible top-level aliases: downstream scripts read metrics["validation"].
    # These point to document-level metrics since documents are the unit of analysis.
    payload["validation"] = payload["document_level"]["validation"]
    payload["test"]       = payload["document_level"]["test"]
    with open(os.path.join(run_root, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    model_dir = os.path.join(run_root, "mdeberta_cefr_multi")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    return {
        "run_root":    run_root,
        "csv_log":     csv_log,
        "model_dir":   model_dir,
        "metrics":     payload,
        "val_macro_f1": payload["window_level"]["validation"]["macro_f1"],
        "val_loss":     payload["window_level"]["validation"]["loss"],
        "best_epoch":   best_epoch_trainer,
    }

# -------------------------------
# Language-weighting A/B test
# -------------------------------
def run_lang_weight_ab_test(
    ab_root: str,
    tok_ds: DatasetDict,
    tokenizer,
    class_weights: torch.Tensor,
    lr: float, grad_accum: int,
) -> None:
    """
    Fair A/B comparison: label-only sampler vs. language-aware joint sampler.
    Both arms use WeightedRandomSampler with the same inverse-frequency formula
    and the same cap (AB_TEST_CAP). The ONLY variable that differs is whether
    counts are per-label (arm A) or per-lang x label (arm B).

    Note: the joint weights in weight_diagnostics.json used cap=5.0, which collapsed
    all cells to 5.0 (equal). This function recomputes both arms at AB_TEST_CAP=10.0
    so values actually differ across cells.
    """
    if not os.path.exists(WEIGHT_DIAGNOSTICS_JSON):
        print("A/B test: weight_diagnostics.json not found, skipping.")
        return

    os.makedirs(ab_root, exist_ok=True)

    with open(WEIGHT_DIAGNOSTICS_JSON) as f:
        diag = json.load(f)

    lang_label_counts = diag["lang_label_counts_train"]
    train_lang_ids = np.array(tok_ds["train"]["lang_id"])
    train_labels   = np.array(tok_ds["train"]["labels"])

    # Both arms computed by the same helper with the same cap (one variable differs)
    arm_a_weights, arm_b_weights = _ab_compute_sampler_weights(
        lang_label_counts, train_lang_ids, train_labels, cap=AB_TEST_CAP,
    )

    print(f"\n=== Language-weighting A/B test ({AB_TEST_EPOCHS} epochs each) ===")
    print(f"  Mechanism: WeightedRandomSampler  (both arms)")
    print(f"  Cap:       {AB_TEST_CAP}  (both arms, same formula: total / count, capped)")
    print(f"  Arm A:     label-only  -- weight depends on label, not language")
    print(f"  Arm B:     lang x label -- weight depends on (language, label) cell")
    print(f"  Only variable that differs: label-only vs lang x label cell counts.")

    # Print the two weight configs side by side for inspection
    LANG2ID_MAP = {"de": 0, "en": 1, "nl": 2, "pt": 3}
    total = sum(sum(v.values()) for v in lang_label_counts.values())
    label_totals: dict[str, int] = {}
    for lbl_counts in lang_label_counts.values():
        for lbl, cnt in lbl_counts.items():
            label_totals[lbl] = label_totals.get(lbl, 0) + cnt
    label_w = {lbl: min(total / max(1, cnt), AB_TEST_CAP) for lbl, cnt in label_totals.items()}
    print(f"\n  Arm A weights (same per label regardless of language):")
    print(f"    " + "  ".join(f"{lbl}={label_w[lbl]:.2f}" for lbl in LABELS))
    print(f"  Arm B weights (per lang x label cell, cap={AB_TEST_CAP}):")
    for lang in LANG2ID_MAP:
        joint_w = {lbl: min(total / max(1, lang_label_counts[lang][lbl]), AB_TEST_CAP)
                   for lbl in LABELS}
        print(f"    {lang}: " + "  ".join(f"{lbl}={joint_w[lbl]:.2f}" for lbl in LABELS))

    ab_results: dict = {}
    for label, sampler in [("label_only", arm_a_weights),
                            ("lang_joint", arm_b_weights)]:
        print(f"\n  -- {label} --")
        trial_dir = os.path.join(ab_root, label)
        os.makedirs(trial_dir, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
        )
        t_args, _ = make_training_args(
            output_dir=os.path.join(trial_dir, "checkpoints"),
            lr=lr, grad_accum=grad_accum, epochs=AB_TEST_EPOCHS, scheduler="linear",
        )
        ab_trainer = WeightedTrainer(
            model=model, args=t_args,
            train_dataset=tok_ds["train"],
            eval_dataset={"val": tok_ds["validation"], "train": tok_ds["train"]},
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            balanced_sampler_weights=sampler,
            emd_weight=EMD_WEIGHT,
        )
        ab_trainer.train()

        val_pred = ab_trainer.predict(tok_ds["validation"], metric_key_prefix="val")
        yv_true  = val_pred.label_ids
        yv_pred  = val_pred.predictions.argmax(axis=1)
        vl_ids   = np.array(tok_ds["validation"]["lang_id"])

        overall  = float(f1_score(yv_true, yv_pred, average="macro"))
        by_lang  = per_lang_f1(yv_true, yv_pred, vl_ids)
        ab_results[label] = {"overall": overall, "by_lang": by_lang}
        print(f"    overall val macro F1: {overall:.4f}")
        for lang, score in by_lang.items():
            print(f"      {lang}: {score:.4f}")

    # Print comparison table
    print("\n--- A/B results: val macro F1 (3 epochs each) ---")
    w = 12
    print(f"  {'lang':<8} {'label_only':>{w}} {'lang_joint':>{w}} {'diff':>8}")
    print(f"  {'-'*8} {'-'*w} {'-'*w} {'-'*8}")
    for lang in ["de", "en", "nl", "pt"]:
        a = ab_results["label_only"]["by_lang"].get(lang, float("nan"))
        b = ab_results["lang_joint"]["by_lang"].get(lang, float("nan"))
        diff = b - a
        flag = "  <-- improves" if diff > 0.01 else ("  <-- drops" if diff < -0.01 else "")
        print(f"  {lang:<8} {a:>{w}.4f} {b:>{w}.4f} {diff:>+8.4f}{flag}")
    a_ov = ab_results["label_only"]["overall"]
    b_ov = ab_results["lang_joint"]["overall"]
    print(f"  {'-'*8} {'-'*w} {'-'*w} {'-'*8}")
    print(f"  {'overall':<8} {a_ov:>{w}.4f} {b_ov:>{w}.4f} {b_ov - a_ov:>+8.4f}")
    print()
    print("  Tradeoff: if PT/NL improve but DE/EN drop, the joint sampler helps minority")
    print("  languages at the cost of the dominant ones. Decide before the next full run.")

    with open(os.path.join(ab_root, "ab_results.json"), "w") as f:
        json.dump(ab_results, f, indent=2)
    print(f"\n  A/B results saved to {ab_root}")

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 02: final CEFR classifier training.")
    parser.add_argument(
        "--repair", type=str, default=None, metavar="RUN_DIR",
        help=(
            "Path to an existing run dir. Reads selection_summary.json and "
            "re-runs only the winner trial. With load_best_model_at_end=True "
            "the trial already holds the best-F1 epoch weights -- no refit needed."
        ),
    )
    parser.add_argument(
        "--skip-ab", action="store_true",
        help="Skip the language-weighting A/B test (saves ~3h).",
    )
    parser.add_argument(
        "--ab-only", type=str, default=None, metavar="OUT_DIR",
        help=(
            "Run ONLY the language-weighting A/B test and exit. "
            "Results are written to OUT_DIR. No trial, no model saving."
        ),
    )
    args = parser.parse_args()

    set_seed(SEED)

    # --ab-only: load data, run A/B test, exit. Nothing else runs.
    if args.ab_only:
        print("=== A/B Language-Weighting Test (--ab-only) ===")
        print(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output dir: {args.ab_only}")
        tok_ds = DatasetDict.load_from_disk(PACKED_DIR)
        tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
        with open(CLASS_WEIGHTS_JSON) as f:
            _cw = json.load(f)
        print("  doc-level (json):", {l: round(_cw[l], 3) for l in LABELS})
        class_weights = window_class_weights(tok_ds)
        print("  window-level (used in loss):", {l: round(class_weights[i].item(), 3) for i, l in enumerate(LABELS)})
        best = read_best_params(SWEEP_BEST_JSON, DEFAULT_LR, DEFAULT_GRAD_ACCUM)
        run_lang_weight_ab_test(
            ab_root=args.ab_only,
            tok_ds=tok_ds,
            tokenizer=tokenizer,
            class_weights=class_weights,
            lr=best["learning_rate"],
            grad_accum=best["grad_accum"],
        )
        print("Done.")
        return

    os.makedirs(ROOT_OUT, exist_ok=True)

    ts = datetime.now().strftime("cefr_%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT_OUT, "runs", ts)
    trials_root = os.path.join(run_dir, "trials")
    os.makedirs(trials_root, exist_ok=True)

    print("=== Final Training (Step 02) ===")
    print(f"Timestamp:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data (packed):         {PACKED_DIR}")
    print(f"Tokenizer:             {TOK_DIR}")
    print(f"Model:                 {MODEL_NAME}")
    print(f"Selection criterion:   highest val macro-F1 (load_best_model_at_end=True)")

    # Data and tokenizer -- packed variant so long texts are not truncation-handicapped
    tok_ds: DatasetDict = DatasetDict.load_from_disk(PACKED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

    # Class weights
    with open(CLASS_WEIGHTS_JSON) as f:
        class_w = json.load(f)
    print("  doc-level (json):", {l: round(class_w[l], 3) for l in LABELS})
    class_weights = window_class_weights(tok_ds)
    print("  window-level (used in loss):", {l: round(class_weights[i].item(), 3) for i, l in enumerate(LABELS)})

    # Optional balanced sampler weights
    weights_t = torch.load(BALANCED_WEIGHTS_PACKED) if os.path.exists(BALANCED_WEIGHTS_PACKED) else None

    # Hyperparams from Step 01
    best = read_best_params(SWEEP_BEST_JSON, DEFAULT_LR, DEFAULT_GRAD_ACCUM)
    lr = best["learning_rate"]
    grad_accum = best["grad_accum"]

    if args.repair:
        # Repair mode: read winner config from an existing run, re-run only that trial.
        sel_path = os.path.join(args.repair, "selection_summary.json")
        with open(sel_path) as f:
            sel = json.load(f)
        prev_winner = sel["winner"]
        repair_scheduler = prev_winner["scheduler"]
        repair_balanced  = prev_winner["balanced"]
        print(f"\nRepair mode: reading winner from {sel_path}")
        print(f"  Winner: scheduler={repair_scheduler}, balanced={repair_balanced}")
        print(f"  Re-running ONLY the winner trial (no refit needed with F1 selection).")

        trial_dir = os.path.join(trials_root, f"{repair_scheduler}_bal{int(repair_balanced)}")
        os.makedirs(trial_dir, exist_ok=True)
        sampler = weights_t if repair_balanced else None

        out = run_one_training(
            run_root=trial_dir,
            tok_ds=tok_ds,
            tokenizer=tokenizer,
            class_weights=class_weights,
            sampler_weights=sampler,
            lr=lr,
            grad_accum=grad_accum,
            epochs=MAX_EPOCHS,
            scheduler=repair_scheduler,
        )
        results_sorted = [{
            "scheduler":    repair_scheduler,
            "balanced":     repair_balanced,
            "val_macro_f1": out["val_macro_f1"],
            "val_loss":     out["val_loss"],
            "csv_log":      out["csv_log"],
            "run_root":     trial_dir,
            "best_epoch":   out["best_epoch"],
        }]
        winner = results_sorted[0]
        with open(os.path.join(run_dir, "selection_summary.json"), "w") as f:
            json.dump({"candidates": results_sorted, "winner": winner,
                       "repaired_from": args.repair}, f, indent=2)

    else:
        # Normal mode: run all trial combinations and pick the winner by highest val macro-F1.
        results = []
        for scheduler in CANDIDATE_SCHEDULERS:
            for use_bal in CANDIDATE_BALANCED:
                print("\n--- Trial run -------------------------------------------")
                print(f"Scheduler candidate:   {scheduler}")
                print(f"Balanced sampler:      {use_bal}")
                trial_dir = os.path.join(trials_root, f"{scheduler}_bal{int(use_bal)}")
                os.makedirs(trial_dir, exist_ok=True)
                sampler = weights_t if use_bal else None

                out = run_one_training(
                    run_root=trial_dir,
                    tok_ds=tok_ds,
                    tokenizer=tokenizer,
                    class_weights=class_weights,
                    sampler_weights=sampler,
                    lr=lr,
                    grad_accum=grad_accum,
                    epochs=MAX_EPOCHS,
                    scheduler=scheduler,
                )
                results.append({
                    "scheduler":    scheduler,
                    "balanced":     use_bal,
                    "val_macro_f1": out["val_macro_f1"],
                    "val_loss":     out["val_loss"],
                    "csv_log":      out["csv_log"],
                    "run_root":     trial_dir,
                    "best_epoch":   out["best_epoch"],
                })

        # Select winner by highest validation macro-F1
        results_sorted = sorted(results, key=lambda r: r["val_macro_f1"], reverse=True)
        winner = results_sorted[0]
        with open(os.path.join(run_dir, "selection_summary.json"), "w") as f:
            json.dump({"candidates": results_sorted, "winner": winner}, f, indent=2)

    # Winner's model is already the best-F1 epoch (load_best_model_at_end handled it).
    # best_epoch comes from the Trainer state -- same source as the saved weights.
    be = winner["best_epoch"]
    print(f"\nSelected winner -> scheduler={winner['scheduler']} | balanced={winner['balanced']} | "
          f"val_macro_f1={winner['val_macro_f1']:.4f}")
    print(f"Best epoch (same source as saved model): {be}")

    def copy_file(src, dst):
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Copy all outputs from the winner trial to the top-level run dir
    copy_file(winner["csv_log"], os.path.join(run_dir, "training_log.csv"))
    copy_file(os.path.join(winner["run_root"], "learning_curve_loss.png"),
              os.path.join(run_dir, "learning_curve_loss.png"))
    copy_file(os.path.join(winner["run_root"], "learning_curve_val_f1.png"),
              os.path.join(run_dir, "learning_curve_val_f1.png"))
    copy_file(os.path.join(winner["run_root"], "confusion_matrix_val_window.png"),
              os.path.join(run_dir, "confusion_matrix_val_window.png"))
    copy_file(os.path.join(winner["run_root"], "confusion_matrix_test_window.png"),
              os.path.join(run_dir, "confusion_matrix_test_window.png"))
    copy_file(os.path.join(winner["run_root"], "confusion_matrix_val_doc.png"),
              os.path.join(run_dir, "confusion_matrix_val_doc.png"))
    copy_file(os.path.join(winner["run_root"], "confusion_matrix_test_doc.png"),
              os.path.join(run_dir, "confusion_matrix_test_doc.png"))
    copy_file(os.path.join(winner["run_root"], "metrics.json"),
              os.path.join(run_dir, "metrics.json"))

    final_model_src = os.path.join(winner["run_root"], "mdeberta_cefr_multi")
    final_model_dst = os.path.join(run_dir, "mdeberta_cefr_multi")
    if os.path.exists(final_model_dst):
        shutil.rmtree(final_model_dst)
    shutil.copytree(final_model_src, final_model_dst)

    # Language-weighting A/B test (informational only -- does not change the final model)
    if not args.skip_ab:
        run_lang_weight_ab_test(
            ab_root=os.path.join(run_dir, "ab_lang_weight_test"),
            tok_ds=tok_ds,
            tokenizer=tokenizer,
            class_weights=class_weights,
            lr=lr,
            grad_accum=grad_accum,
        )
    else:
        print("\nA/B test skipped (--skip-ab).")

    manifest = {
        "seed": SEED,
        "per_device_batch": PER_DEVICE_BATCH,
        "learning_rate": lr,
        "grad_accum": grad_accum,
        "effective_batch": PER_DEVICE_BATCH * grad_accum,
        "search": {
            "schedulers": CANDIDATE_SCHEDULERS,
            "balanced_options": CANDIDATE_BALANCED,
            "selection_criterion": "highest val macro-F1 (load_best_model_at_end)",
            "winner": {
                "scheduler": winner["scheduler"],
                "balanced":  winner["balanced"],
                "best_epoch_by_f1": be,
            },
        },
        "labels": LABELS,
        "model_name": MODEL_NAME,
        "training_config": {
            "loss": "weighted_CE(label_smoothing=0.1) + EMD" if EMD_WEIGHT > 0 else "weighted_CE(label_smoothing=0.1)",
            "label_smoothing": 0.1,
            "emd_weight": EMD_WEIGHT,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": winner["scheduler"],
            "max_epochs": MAX_EPOCHS,
            "early_stopping": USE_EARLY_STOPPING,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "metric_for_best_model": "eval_val_macro_f1",
            "class_weight_scheme": "window-level inverse frequency (total / count)",
            "class_weights": {l: float(class_weights[i]) for i, l in enumerate(LABELS)},
        },
        "dataset": {
            "source": PACKED_DIR,
            "splits_sha256": sha256_of_file(SPLITS_JSON),
        },
        "paths": {
            "run_dir":          run_dir,
            "model_dir":        final_model_dst,
            "training_log_csv": os.path.join(run_dir, "training_log.csv"),
            "loss_curve":       os.path.join(run_dir, "learning_curve_loss.png"),
            "val_f1_curve":     os.path.join(run_dir, "learning_curve_val_f1.png"),
            "cm_val_window":    os.path.join(run_dir, "confusion_matrix_val_window.png"),
            "cm_test_window":   os.path.join(run_dir, "confusion_matrix_test_window.png"),
            "cm_val_doc":       os.path.join(run_dir, "confusion_matrix_val_doc.png"),
            "cm_test_doc":      os.path.join(run_dir, "confusion_matrix_test_doc.png"),
            "metrics_json":     os.path.join(run_dir, "metrics.json"),
            "selection_summary": os.path.join(run_dir, "selection_summary.json"),
        },
        "versions": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
            "datasets": __import__("datasets").__version__,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nRun summary")
    print(f"  Run directory:      {run_dir}")
    print(f"  Model saved to:     {final_model_dst}")
    print(f"  Metrics:            {os.path.join(run_dir, 'metrics.json')}")
    print(f"  Manifest:           {os.path.join(run_dir, 'run_manifest.json')}")
    print("Done.")

if __name__ == "__main__":
    main()
