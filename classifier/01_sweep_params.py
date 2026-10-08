# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 01_sweep_params.py
# Purpose: short hyperparameter search to pick the learning rate and batch size for the final model.
# Pipeline step: Classifier, step 01 (learning-rate and batch-size sweep, seed 42). The shipped result
#                was computed on the four-language data including Portuguese and was not repeated.
# Thesis: Section 4.2.2 (hyperparameters; 01_sweep_output/sweep_results.csv and
#         configs/best_params.json, which 02_train_final.py reads).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------- Configuration (small, safe grid) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "microsoft/mdeberta-v3-base"
LABELS = ["A", "B1", "B2", "C"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

SEED = 42
EPOCHS = 3                       # short runs
PER_DEVICE_BATCH = 2             # memory-friendly; eff. batch via grad_accum
LEARNING_RATES = [7e-6, 1e-05, 2e-05]
EFFECTIVE_BATCHES = [8, 16]      # grad_accum = eff_batch / PER_DEVICE_BATCH

DATA_DIR = os.path.join(SCRIPT_DIR, "00_prepare_output")
TOK_DIR = os.path.join(DATA_DIR, "tokenizer")
PACKED_DIR = os.path.join(DATA_DIR, "tokenized_packed")
CLASS_WEIGHTS_JSON = os.path.join(DATA_DIR, "class_weights.json")

OUT_ROOT = os.path.join(SCRIPT_DIR, "01_sweep_output")
RUNS_ROOT = os.path.join(OUT_ROOT, "runs")
RUN_STAMP = datetime.now().strftime("cefr_%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(RUNS_ROOT, RUN_STAMP)              # timestamped
CONFIGS_DIR = os.path.join(OUT_DIR, "configs")

# ---------- Utilities ----------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropyLoss (version-robust)."""
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    # Accept **kwargs to be compatible with HF versions that pass extra args
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=0.1,
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_training_args(output_dir: str, lr: float, grad_accum: int) -> TrainingArguments:
    """
    Create TrainingArguments compatibly across transformers versions.
    Modern path uses *strategy="epoch". Fallback avoids unsupported kwargs and
    sets a non-zero logging_steps to satisfy older versions.
    """
    common = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        seed=SEED,
        report_to=[],
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        load_best_model_at_end=False,
    )
    try:
        return TrainingArguments(**common, eval_strategy="epoch",
                                 logging_strategy="epoch", save_strategy="no")
    except TypeError:
        pass
    try:
        return TrainingArguments(**common, evaluation_strategy="epoch",
                                 logging_strategy="epoch", save_strategy="no")
    except TypeError:
        return TrainingArguments(**common, do_eval=True, logging_steps=50, save_steps=0)

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    # Keep legacy locations to remain compatible with step 02
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "configs"), exist_ok=True)

# ---------- Main ----------

def main():
    set_seed(SEED)
    ensure_dirs()

    ts_print = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=== Hyperparameter Sweep (Step 01) ===")
    print(f"Timestamp:           {ts_print}")
    print(f"Data (packed):       {PACKED_DIR}")
    print(f"Tokenizer:           {TOK_DIR}")
    print(f"Model:               {MODEL_NAME}")
    print(f"Epochs per run:      {EPOCHS}")
    print(f"Grid (LR):           {LEARNING_RATES}")
    print(f"Grid (eff. batch):   {EFFECTIVE_BATCHES}  (per_device_batch={PER_DEVICE_BATCH})")

    # Packed (sliding-window) dataset -- English texts do not lose content to truncation,
    # so the sweep LR comparison reflects the model's actual capability, not a truncation handicap
    tok_ds: DatasetDict = DatasetDict.load_from_disk(PACKED_DIR)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

    # Load class weights (from Step 00) -- shown for reference only
    with open(CLASS_WEIGHTS_JSON) as f:
        class_w_dict = json.load(f)
    print("  doc-level (json):", {l: round(class_w_dict[l], 3) for l in LABELS})
    # Recompute on window-level training labels to match the final model
    from collections import Counter
    _labels = list(tok_ds["train"]["labels"])
    _counts = Counter(_labels); _total = len(_labels)
    class_w = torch.tensor(
        [_total / max(1, _counts.get(i, 0)) for i in range(len(LABELS))],
        dtype=torch.float32)
    print("  window-level (used in loss):", {l: round(class_w[i].item(), 3) for i, l in enumerate(LABELS)})

    # Results CSV in timestamped dir
    csv_path = os.path.join(OUT_DIR, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "learning_rate", "effective_batch", "grad_accum", "val_macro_f1", "val_loss", "seconds"])

    best: Dict[str, Any] | None = None
    run_id = 0

    for lr in LEARNING_RATES:
        for eff_b in EFFECTIVE_BATCHES:
            run_id += 1
            grad_accum = max(1, eff_b // PER_DEVICE_BATCH)

            args = make_training_args(
                output_dir=os.path.join(OUT_DIR, f"run_{run_id:02d}"),
                lr=lr,
                grad_accum=grad_accum,
            )

            # Fresh model per run
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=len(LABELS),
                id2label=id2label,
                label2id=label2id,
            )

            trainer = WeightedTrainer(
                model=model,
                args=args,
                train_dataset=tok_ds["train"],
                eval_dataset=tok_ds["validation"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                class_weights=class_w,
            )

            t0 = time.time()
            trainer.train()
            metrics = trainer.evaluate(eval_dataset=tok_ds["validation"])
            secs = time.time() - t0

            val_f1 = float(metrics.get("eval_macro_f1", 0.0))
            val_loss = float(metrics.get("eval_loss", 0.0))

            # Persist one line per run
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([run_id, lr, eff_b, grad_accum, f"{val_f1:.6f}", f"{val_loss:.6f}", f"{secs:.1f}"])

            print(
                f"Run {run_id:02d} | lr={lr:.1e} | eff_batch={eff_b} | grad_accum={grad_accum} "
                f"-> val_macro_f1={val_f1:.4f}  val_loss={val_loss:.4f}  time={secs:.0f}s"
            )

            if (best is None) or (val_f1 > best["val_macro_f1"]):
                best = {
                    "learning_rate": lr,
                    "effective_batch": eff_b,
                    "grad_accum": grad_accum,
                    "val_macro_f1": val_f1,
                    "val_loss": val_loss,
                    "seconds": secs,
                }

            del model, trainer
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # Sort results deterministically
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows_sorted = sorted(
        rows,
        key=lambda r: (float(r["val_macro_f1"]), -float(r["val_loss"])),
        reverse=True,
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["run", "learning_rate", "effective_batch", "grad_accum", "val_macro_f1", "val_loss", "seconds"]
        )
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    # Save best params to timestamped configs
    best_payload = {
        "timestamp": ts_print,
        "seed": SEED,
        "model": MODEL_NAME,
        "per_device_batch": PER_DEVICE_BATCH,
        "epochs": EPOCHS,
        "best": best,
        "grid": {"learning_rates": LEARNING_RATES, "effective_batches": EFFECTIVE_BATCHES},
        "data_source": PACKED_DIR,
    }
    ts_conf = os.path.join(CONFIGS_DIR, "best_params.json")
    with open(ts_conf, "w") as f:
        json.dump(best_payload, f, indent=2)

    # Write legacy-compatible copies so step 02 can still read the same paths
    legacy_csv = os.path.join(OUT_ROOT, "sweep_results.csv")
    legacy_cfg_dir = os.path.join(OUT_ROOT, "configs")
    os.makedirs(legacy_cfg_dir, exist_ok=True)
    with open(csv_path, "r") as src, open(legacy_csv, "w") as dst:
        dst.write(src.read())
    with open(ts_conf, "r") as src, open(os.path.join(legacy_cfg_dir, "best_params.json"), "w") as dst:
        dst.write(src.read())

    print("\nSweep summary")
    print(f"  Results CSV:        {csv_path}")
    print(f"  Best params (conf): {ts_conf}")
    print(f"  Legacy copy CSV:    {legacy_csv}")
    print(f"  Legacy copy config: {os.path.join(legacy_cfg_dir, 'best_params.json')}")
    if best:
        print(
            "  Selected -> "
            f"lr={best['learning_rate']:.1e}, eff_batch={best['effective_batch']} "
            f"(grad_accum={best['grad_accum']}), val_macro_f1={best['val_macro_f1']:.4f}"
        )
    print("Done.")

if __name__ == "__main__":
    main()
