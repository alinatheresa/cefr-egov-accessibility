# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b5_readability_and_copies.py
# Purpose: B5.8 convergent validity box plots (mean sentence length and Flesch score by
#          predicted CEFR level, per language) using the tokenisation, syllable and Flesch
#          functions of analysis/readability/convergent_validity_flesch.py unchanged.
#          B5.9 checks that the final model equals the trial model of run cefr_20260817_231925.
# Pipeline step: Analysis: readability by predicted level and model-copy check (needs the page
#                texts and the model weights).
# Thesis: Section 6.4.3: fig8a_sentence_length_by_cefr, fig8b_flesch_by_cefr,
#         tab:readability-by-level.

import os
import hashlib
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from common import (PROJECT_ROOT, RUN_DIR, FIG_DIR, TAB_DIR, CEFR, CEFR_COLOUR, GRID_COLOUR, save_fig,
                    save_json, tex_table)

spec = importlib.util.spec_from_file_location(
    "cv", os.path.join(PROJECT_ROOT, "analysis", "readability", "convergent_validity_flesch.py"))
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)

df = cv.load_and_join()
recs = []
for row in df.itertuples(index=False):
    r = cv.process_page_text(getattr(row, "text_joined", "") or "", row.language)
    if r is None:
        continue
    ns, nw, nsyl, asl, asw, fre = r
    recs.append({"country": row.country, "language": row.language, "url": row.url,
                 "pred_label": row.pred_label, "mean_sentence_length_words": asl, "flesch": fre})
out = pd.DataFrame(recs)

# Check against the existing per-page file written by the original script
old = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis", "readability", "readability_per_page.csv"))
m = out.merge(old, on="url", suffixes=("", "_old"))
check = {"n_new": len(out), "n_old": len(old), "n_matched": len(m),
         "max_abs_diff_flesch": float((m.flesch - m.flesch_old).abs().max()),
         "max_abs_diff_sentlen": float((m.mean_sentence_length_words
                                        - m.mean_sentence_length_words_old).abs().max()),
         "label_mismatch": int((m.pred_label != m.pred_label_old).sum())}
print("Check vs analysis/readability/readability_per_page.csv:", check)

LANGS = [("german", "German (Germany, Austria)"), ("dutch", "Dutch (Netherlands)"),
         ("english", "English (Ireland)")]
summary = []
for lang, _ in LANGS:
    for lvl in CEFR:
        d = out[(out.language == lang) & (out.pred_label == lvl)]
        summary.append({"language": lang, "cefr": lvl, "n": len(d),
                        "sentlen_median": d.mean_sentence_length_words.median(),
                        "sentlen_q1": d.mean_sentence_length_words.quantile(.25),
                        "sentlen_q3": d.mean_sentence_length_words.quantile(.75),
                        "flesch_median": d.flesch.median(),
                        "flesch_q1": d.flesch.quantile(.25), "flesch_q3": d.flesch.quantile(.75)})
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(TAB_DIR, "b5_fig8_readability_by_level.csv"), index=False,
               float_format="%.2f")
rho = []
for lang, _ in LANGS:
    d = out[out.language == lang]
    o = d.pred_label.map({l: i + 1 for i, l in enumerate(CEFR)})
    for meas in ["mean_sentence_length_words", "flesch"]:
        r, p = spearmanr(o, d[meas])
        rho.append({"language": lang, "measure": meas, "n": len(d), "rho": r, "p": p})
rho = pd.DataFrame(rho)
rho.to_csv(os.path.join(TAB_DIR, "b5_fig8_spearman_by_language.csv"), index=False,
           float_format="%.4g")
print(summary.to_string())
print(rho.to_string())

for meas, ylabel, name, clip in [
        ("mean_sentence_length_words", "Mean sentence length (words)", "fig8a_sentence_length_by_cefr", (0, 60)),
        ("flesch", "Flesch reading ease (language-specific)", "fig8b_flesch_by_cefr", (-60, 120))]:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4), sharey=True)
    for ax, (lang, title) in zip(axes, LANGS):
        data = [out[(out.language == lang) & (out.pred_label == l)][meas].to_numpy() for l in CEFR]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for patch, l in zip(bp["boxes"], CEFR):
            patch.set_facecolor(CEFR_COLOUR[l])
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, 5), [f"{l}\n{len(d):,}" for l, d in zip(CEFR, data)], fontsize=8)
        ax.set_xlabel(title, fontsize=9)
        ax.set_ylim(*clip)
        ax.yaxis.grid(True, color=GRID_COLOUR, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    fig.supxlabel("Predicted CEFR level (n pages below each level)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, name)

# ── B5.9: model identity check (figures now drawn by b6c_classifier_figures.py) ──
TRIAL = os.path.join(RUN_DIR, "trials", "linear_bal0")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# The final model is byte-identical to the single trial's model, so the trial's
# per-language matrices belong to the same run.
same_model = sha(os.path.join(TRIAL, "mdeberta_cefr_multi", "model.safetensors")) == \
    sha(os.path.join(RUN_DIR, "mdeberta_cefr_multi", "model.safetensors"))
assert same_model
# The classifier figures were first copied here; they are now redrawn without titles from
# the saved training log and predictions by b6c_classifier_figures.py, so no copy is made.
copied = {}

save_json({"check_against_existing": check, "summary": summary.to_dict(orient="records"),
           "spearman": rho.to_dict(orient="records"), "copied_figures": copied,
           "final_model_equals_trial_model": same_model}, "b5_readability_and_copies.json")

rows = []
for lang, title in LANGS:
    for i, lvl in enumerate(CEFR):
        r = summary[(summary.language == lang) & (summary.cefr == lvl)].iloc[0]
        rows.append([title.split(" ")[0] if i == 0 else "", lvl, f"{int(r.n):,}",
                     f"{r.sentlen_median:.1f} [{r.sentlen_q1:.1f}, {r.sentlen_q3:.1f}]",
                     f"{r.flesch_median:.1f} [{r.flesch_q1:.1f}, {r.flesch_q3:.1f}]"])
notes = ["Medians with interquartile range in brackets. Flesch variants: Amstad (German), Douma "
         "(Dutch), Flesch (English); higher means easier. Syllables use a vowel-cluster heuristic.",
         "Spearman correlation of predicted level (1 to 4) with sentence length and Flesch score: " +
         "; ".join(f"{l.capitalize()} {rho[(rho.language == l) & (rho.measure == 'mean_sentence_length_words')].rho.item():.2f} and "
                   f"{rho[(rho.language == l) & (rho.measure == 'flesch')].rho.item():.2f}"
                   for l, _ in LANGS) + "."]
tex_table(os.path.join(TAB_DIR, "b5_fig8_readability_by_level.tex"),
          caption="Sentence length and Flesch reading ease by predicted CEFR level",
          label="tab:readability-by-level",
          header=["Language", "Level", "n", "Sentence length", "Flesch"],
          rows=rows, midrules=(4, 8), notes=notes, colspec="llrrr", size=r"\small")
