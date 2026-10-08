# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b6c_classifier_figures.py
# Purpose: classifier figures without titles, redrawn from saved data of run
#          cefr_20260817_231925: training_log.csv (learning curves) and the validation/test
#          predictions saved by b6b_eval_and_rater_check.py (confusion matrices).
# Pipeline step: Analysis: classifier figures from the saved training log and predictions. Also draws
#                further curves and matrices that are not in the thesis.
# Thesis: Section 4.3.1 clf_learning_curve_loss, clf_confusion_matrix_test_doc,
#         clf_confusion_matrix_test_doc_nl; Annex clf_confusion_matrix_test_doc_de,
#         clf_confusion_matrix_test_doc_en.

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from common import (RUN_DIR, FIG_DIR, RES_DIR, CEFR, CEFR_COLOUR, CM_CMAP, GRID_COLOUR,
                    cm_text_colour, save_fig, save_json)

LANG = {"de": "German", "en": "English", "nl": "Dutch"}

# Remove the earlier PNG-only copies of the training-script figures (superseded)
for f in glob.glob(os.path.join(FIG_DIR, "clf_*.png")):
    if not os.path.exists(f[:-4] + ".pdf"):
        os.remove(f)

# ── Learning curves ─────────────────────────────────────────────────────────
log = pd.read_csv(os.path.join(RUN_DIR, "training_log.csv"))
best = int(log.loc[log.eval_val_macro_f1.idxmax(), "epoch"])

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(log.epoch, log.eval_train_loss, marker="o", color=CEFR_COLOUR["B2"], lw=2, label="Training split")
ax.plot(log.epoch, log.eval_val_loss, marker="s", color=CEFR_COLOUR["C"], lw=2, ls="--", label="Validation split")
ax.axvline(best, color="#888888", lw=0.8, ls=":")
ax.text(best + 0.05, ax.get_ylim()[1] * 0.98, f"selected epoch {best}", fontsize=8,
        color="#555555", va="top")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (weighted CE + EMD)")
ax.set_xticks(log.epoch)
ax.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left")
save_fig(fig, "clf_learning_curve_loss")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(log.epoch, log.eval_val_macro_f1, marker="o", color=CEFR_COLOUR["B2"], lw=2, label="Macro-F1")
ax.plot(log.epoch, log.eval_val_accuracy, marker="s", color=CEFR_COLOUR["C"], lw=2, ls="--", label="Accuracy")
ax.axvline(best, color="#888888", lw=0.8, ls=":")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation score (window level)")
ax.set_xticks(log.epoch)
ax.set_ylim(0.45, 0.8)
ax.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
save_fig(fig, "clf_learning_curve_val_f1")


# ── Confusion matrices ──────────────────────────────────────────────────────
def draw_cm(ax, cm, show_ylabel=True):
    vmax = cm.max()
    ax.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=vmax)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color=cm_text_colour(cm[i, j], cm))
    ax.set_xticks(range(4), CEFR)
    ax.set_yticks(range(4), CEFR)
    ax.set_xlabel("Predicted level")
    if show_ylabel:
        ax.set_ylabel("Gold level")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


preds = {}
for split in ["test", "validation"]:
    preds[(split, "doc")] = pd.read_csv(os.path.join(RES_DIR, f"b6b_doc_predictions_{split}.csv"))
    preds[(split, "window")] = pd.read_csv(os.path.join(RES_DIR, f"b6b_window_logits_{split}.csv"))

cms = {}
for (split, level), d in preds.items():
    tag = "test" if split == "test" else "val"
    cm = confusion_matrix(d.true_label, d.pred_label, labels=CEFR)
    cms[f"{tag}_{level}"] = cm.tolist()
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    draw_cm(ax, cm)
    save_fig(fig, f"clf_confusion_matrix_{tag}_{level}")
    for lg in ["de", "en", "nl"]:
        cm_l = confusion_matrix(d[d.lang == lg].true_label, d[d.lang == lg].pred_label, labels=CEFR)
        cms[f"{tag}_{level}_{lg}"] = cm_l.tolist()
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        draw_cm(ax, cm_l)
        save_fig(fig, f"clf_confusion_matrix_{tag}_{level}_{lg}")
    # three-panel version per language
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for k, (ax, lg) in enumerate(zip(axes, ["de", "en", "nl"])):
        sub = d[d.lang == lg]
        cm_l = confusion_matrix(sub.true_label, sub.pred_label, labels=CEFR)
        draw_cm(ax, cm_l, show_ylabel=(k == 0))
        ax.set_xlabel(f"Predicted level\n{LANG[lg]} (n = {len(sub)})")
    fig.tight_layout()
    save_fig(fig, f"clf_confusion_matrix_{tag}_{level}_by_language")

# Overall test-window matrix must equal the one transcribed for B6
assert cms["test_window"] == [[74, 9, 1, 0], [27, 102, 22, 1], [2, 31, 110, 19], [0, 2, 36, 118]]
assert cms["test_doc"] == [[74, 9, 1, 0], [26, 85, 20, 0], [2, 23, 81, 15], [0, 2, 24, 51]]
save_json({"best_epoch": best, "confusion_matrices": cms}, "b6c_classifier_figures.json")
print("best epoch", best)
