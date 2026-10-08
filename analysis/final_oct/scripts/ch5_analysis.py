# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/ch5_analysis.py
# Purpose: additions for Chapter 5. Reads existing outputs only; nothing is re-scraped, re-classified
#          or re-categorised.
#          1. Classifier bias at the binary cut (A/B1 vs B2/C) on the saved test-set predictions
#             of the evaluation pass (results/b6b_doc_predictions_test.csv), per language.
#          2. Figure: CEFR distribution with Germany split into leistung pages and other pages.
#          3. Figure: expected CEFR index vs page length in words, all four countries.
#          4. Robustness: M3a binary logit plus log(word count), main estimation sample.
# Pipeline step: Analysis: binary-cut bias, length figure, length control. The Germany-split figure
#                it also draws is not in the thesis.
# Thesis: tab:clf-binary-cut; robustness table and full output tab:length-control; Annex
#         ch5_fig_expected_cefr_vs_words_by_country.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.formula.api as smf
from scipy import stats

from common import (TAB_DIR, RES_DIR, ANALYSIS_DF, COUNTRIES, COUNTRY_LABEL, COUNTRY_COLOUR, CEFR,
                    CEFR_COLOUR, CEFR_TEXT_COLOUR, GRID_COLOUR, load_cefr, combined_csv, tex_table,
                    fmt_p, fmt_int, save_fig, save_json)

LANG = {"de": "German", "en": "English", "nl": "Dutch"}
EASY = ["A", "B1"]
results = {}

# ── 1. Binary-cut bias on the test set (document level) ─────────────────────
# Document level is the unit used for the corpus: 03_infer_cefr.py averages window logits per page.
pred = pd.read_csv(os.path.join(RES_DIR, "b6b_doc_predictions_test.csv"))
assert len(pred) == 413
pred["true_easy"] = pred["true_label"].isin(EASY)
pred["pred_easy"] = pred["pred_label"].isin(EASY)

bias_rows = []
for lang in list(LANG) + ["all"]:
    d = pred if lang == "all" else pred[pred["lang"] == lang]
    n = len(d)
    # off-diagonal cells of the 2x2 table
    hard_as_easy = int((~d.true_easy & d.pred_easy).sum())
    easy_as_hard = int((d.true_easy & ~d.pred_easy).sum())
    n_hard = int((~d.true_easy).sum())
    mcn = stats.binomtest(hard_as_easy, hard_as_easy + easy_as_hard, 0.5).pvalue \
        if hard_as_easy + easy_as_hard else np.nan
    bias_rows.append({
        "language": "All" if lang == "all" else LANG[lang], "n": n,
        "binary_accuracy": float((d.true_easy == d.pred_easy).mean()),
        "recall_B2C": float((~d.true_easy & ~d.pred_easy).sum() / n_hard),
        "recall_AB1": float((d.true_easy & d.pred_easy).sum() / d.true_easy.sum()),
        "true_share_AB1": float(d.true_easy.mean()),
        "pred_share_AB1": float(d.pred_easy.mean()),
        "diff_pp": float(100 * (d.pred_easy.mean() - d.true_easy.mean())),
        "B2C_predicted_as_AB1": hard_as_easy, "AB1_predicted_as_B2C": easy_as_hard,
        "mcnemar_exact_p": float(mcn),
    })
bias = pd.DataFrame(bias_rows)
bias.to_csv(os.path.join(TAB_DIR, "ch5_binary_bias_test.csv"), index=False, float_format="%.4f")
print(bias.to_string())
results["binary_bias_test_doc"] = bias.to_dict("records")

tex_table(
    os.path.join(TAB_DIR, "ch5_binary_bias_test.tex"),
    caption="Classifier performance at the binary cut (easier, A or B1, versus harder, B2 or C), test set",
    label="tab:clf-binary-cut",
    header=["Language", "n", "Binary accuracy", "Recall harder", "Recall easier",
            "True easier (\\%)", "Predicted easier (\\%)", "Harder as easier", "Easier as harder",
            "McNemar p"],
    rows=[[r.language, fmt_int(r.n), f"{r.binary_accuracy:.3f}", f"{r.recall_B2C:.3f}",
           f"{r.recall_AB1:.3f}", f"{100 * r.true_share_AB1:.1f}", f"{100 * r.pred_share_AB1:.1f}",
           str(r.B2C_predicted_as_AB1), str(r.AB1_predicted_as_B2C), fmt_p(r.mcnemar_exact_p)]
          for r in bias.itertuples()],
    midrules=(3,),
    notes=[
        "Document-level predictions of run cefr\\_20260817\\_231925 on the 413 test texts (window "
        "logits averaged per text, as for the corpus pages). The four-level labels are collapsed "
        "to the binary outcome of the regression: easier (A or B1) and harder (B2 or C).",
        "The two error columns are the off-diagonal cells of the binary confusion matrix; the "
        "predicted share of easier texts exceeds the true share when harder as easier exceeds "
        "easier as harder. "
        "McNemar p: exact two-sided binomial test of these two counts.",
    ],
    colspec="lrrrrrrrrr",
    size=r"\small",
)

# ── Data for figures and the length-controlled model ────────────────────────
cefr = load_cefr()
assert len(cefr) == 8997
words = pd.concat([pd.read_csv(combined_csv(c), usecols=["category", "url", "word_count"])
                   for c in COUNTRIES], ignore_index=True)
cefr = cefr.merge(words, on="url", how="left", validate="one_to_one")
assert cefr["word_count"].notna().all() and (cefr["word_count"] > 0).all()

# ── 2. Figure: CEFR distribution, Germany split by leistung ─────────────────
ger = cefr[cefr.country == "germany"]
groups = [("Germany, service pages (leistung)", ger[ger.category == "leistung"]),
          ("Germany, other pages", ger[ger.category != "leistung"])]
groups += [(COUNTRY_LABEL[c], cefr[cefr.country == c]) for c in COUNTRIES[1:]]
groups += [("All portals", cefr)]
assert len(groups[0][1]) == 1049 and len(groups[1][1]) == 214

rows = []
for name, d in groups:
    shares = d["pred_label"].value_counts(normalize=True).reindex(CEFR, fill_value=0) * 100
    rows.append({"group": name, "n": len(d), **{lvl: shares[lvl] for lvl in CEFR}})
dist = pd.DataFrame(rows)
dist.to_csv(os.path.join(TAB_DIR, "ch5_fig_cefr_distribution_germany_split.csv"), index=False,
            float_format="%.2f")
print(dist.to_string())
results["fig_cefr_distribution_germany_split"] = dist.to_dict("records")

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ypos = [0, 1, 2, 3, 4, 5.6]
for y, r in zip(ypos, rows):
    left = 0
    for lvl in CEFR:
        w = r[lvl]
        ax.barh(y, w, left=left, height=0.72, color=CEFR_COLOUR[lvl], edgecolor="white",
                linewidth=1.0)
        if w >= 5:
            ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                    color=CEFR_TEXT_COLOUR[lvl])
        left += w
ax.set_yticks(ypos, [f"{r['group']}\n(n = {r['n']:,})" for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of pages (%)")
ax.axhline(4.8, color="#999999", lw=0.6, ls=(0, (4, 3)))
ax.legend(handles=[Patch(facecolor=CEFR_COLOUR[l], label=l) for l in CEFR],
          title="Predicted CEFR level", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          frameon=False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save_fig(fig, "ch5_fig_cefr_distribution_germany_split")

# ── 3. Figure: expected CEFR index vs words, four countries ─────────────────


def running_median(x, y, n_bins=40, min_count=30):
    """As in b5_figures.py (figure 4): log-spaced bins, medians for bins with >= min_count pages.
    Also returns the bin index, so the line can be broken where bins are skipped."""
    bins = np.logspace(np.log10(x.min()), np.log10(x.max() + 1), n_bins + 1)
    cx, cy, ci = [], [], []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_count:
            cx.append(np.sqrt(lo * hi))
            cy.append(np.median(y[m]))
            ci.append(i)
    return np.array(cx), np.array(cy), np.array(ci)


res_len = {}
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
for ax, c in zip(axes.flat, COUNTRIES):
    d = cefr[cefr.country == c]
    x, y = d["word_count"].to_numpy(dtype=float), d["E"].to_numpy()
    ax.scatter(x, y, s=6, alpha=0.25, color=COUNTRY_COLOUR[c], linewidths=0, rasterized=True)
    cx, cy, ci = running_median(x, y)
    # one line per run of adjacent bins; an isolated bin is drawn as a dot
    for run in np.split(np.arange(len(ci)), np.where(np.diff(ci) > 1)[0] + 1):
        ax.plot(cx[run], cy[run], color=COUNTRY_COLOUR[c], lw=2.2, zorder=5,
                marker="o" if len(run) == 1 else None, markersize=4)
    rho, p = stats.spearmanr(x, y)
    res_len[c] = {"n": len(d), "spearman_rho": float(rho), "p": float(p),
                  "median_words": float(np.median(x)), "n_running_median_bins": int(len(cx))}
    ax.text(0.03, 0.97, f"{COUNTRY_LABEL[c]}\nn = {len(d):,}, rho = {rho:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yticks([1, 2, 3, 4], ["1 (A)", "2 (B1)", "3 (B2)", "4 (C)"])
    ax.set_ylim(0.9, 4.1)
    ax.grid(True, color=GRID_COLOUR, lw=0.6)
    ax.set_axisbelow(True)
for ax in axes[1]:
    ax.set_xlabel("Page length (words, log scale)")
for ax in axes[:, 0]:
    ax.set_ylabel("Expected CEFR index")
fig.tight_layout()
save_fig(fig, "ch5_fig_expected_cefr_vs_words_by_country")
print(pd.DataFrame(res_len).T.to_string())
results["fig_expected_cefr_vs_words"] = res_len

# ── 4. M3a plus log(word count), main estimation sample ─────────────────────
df = pd.read_csv(ANALYSIS_DF)
assert len(df) == 8997
df = df.merge(words[["url", "word_count"]], on="url", how="left", validate="one_to_one")
df["country"] = pd.Categorical(df["country"], categories=COUNTRIES)
df["y"] = df["cefr_binary"].astype(int)
main = df[df["scope"].notna()].copy()
main["scope"] = pd.Categorical(main["scope"], categories=["citizen", "business"])
main["log_words"] = np.log(main["word_count"])
assert len(main) == 7359 and np.isfinite(main["log_words"]).all()

CTRY = "C(country, Treatment('germany'))"
SCP = "C(scope, Treatment('citizen'))"
TERMS = {"Austria": f"{CTRY}[T.austria]", "Netherlands": f"{CTRY}[T.netherlands]",
         "Ireland": f"{CTRY}[T.ireland]", "Business": f"{SCP}[T.business]",
         "log(words)": "log_words"}

fits, coef_rows = {}, []
for spec, rhs in [("A", f"{CTRY} + {SCP}"), ("L", f"{CTRY} + {SCP} + log_words")]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = smf.logit(f"y ~ {rhs}", main).fit(disp=False, maxiter=500, method="newton")
    fits[spec] = {"N": int(res.nobs), "converged": bool(res.mle_retvals.get("converged", False)),
                  "warnings": sorted({str(x.message)[:160] for x in w}),
                  "aic": float(res.aic), "mcfadden_r2": float(1 - res.llf / res.llnull)}
    ci = res.conf_int()
    for lab, t in TERMS.items():
        if t in res.params.index:
            coef_rows.append({"spec": spec, "term": lab, "coef": res.params[t], "se": res.bse[t],
                              "p": res.pvalues[t], "OR": np.exp(res.params[t]),
                              "OR_lo": np.exp(ci.loc[t, 0]), "OR_hi": np.exp(ci.loc[t, 1]),
                              "N": int(res.nobs)})
coefs = pd.DataFrame(coef_rows)

# Specification A must reproduce b4 M3a exactly
b4 = pd.read_csv(os.path.join(TAB_DIR, "b4_coefficients_binary.csv"))
for lab in ["Austria", "Netherlands", "Ireland", "Business"]:
    a = coefs[(coefs.spec == "A") & (coefs.term == lab)].iloc[0]
    ref = b4[(b4.model == "M3a") & (b4.label == lab)].iloc[0]
    assert abs(a.coef - ref.coef) < 1e-5 and abs(a.se - ref.se) < 1e-5, lab
print("Specification A reproduces b4 M3a.")

lw = coefs[(coefs.spec == "L") & (coefs.term == "log(words)")].iloc[0]
per_doubling = {"OR": float(2 ** lw.coef), "OR_lo": float(lw.OR_lo ** np.log(2)),
                "OR_hi": float(lw.OR_hi ** np.log(2))}
coefs.to_csv(os.path.join(TAB_DIR, "ch5_length_control.csv"), index=False, float_format="%.6g")
print(coefs.to_string())
print(fits)
results["length_control"] = {"coefficients": coefs.to_dict("records"), "fits": fits,
                             "log_words_per_doubling": per_doubling}


def cell(spec, term):
    r = coefs[(coefs.spec == spec) & (coefs.term == term)]
    if r.empty:
        return "", ""
    r = r.iloc[0]
    return f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]", fmt_p(r.p)


rows = [[t, *cell("A", t), *cell("L", t)] for t in TERMS]
rows += [["N", fmt_int(fits["A"]["N"]), "", fmt_int(fits["L"]["N"]), ""],
         ["McFadden $R^2$", f"{fits['A']['mcfadden_r2']:.3f}", "",
          f"{fits['L']['mcfadden_r2']:.3f}", ""],
         ["AIC", f"{fits['A']['aic']:,.1f}", "", f"{fits['L']['aic']:,.1f}", ""]]
tex_table(
    os.path.join(TAB_DIR, "ch5_length_control.tex"),
    caption="Country and scope effects with and without a control for page length",
    label="tab:length-control",
    header=["", "A. Main OR [95\\% CI]", "p", "With log(words) OR [95\\% CI]", "p"],
    rows=rows,
    midrules=(4, 5),
    notes=[
        "Binary logit of harder (predicted B2 or C, 1) versus easier (predicted A or B1, 0), M3a (country + scope; "
        "reference Germany, citizen), estimation sample of the main analysis. Specification A "
        "equals Table~\\ref{tab:regression-main}; the second model adds the natural log of the "
        "page's word count. Wald 95\\% confidence intervals.",
        f"log(words): odds ratio per one-unit increase in ln(words), i.e. per factor e "
        f"(2.72) in length; per doubling of length the odds ratio is {per_doubling['OR']:.2f} "
        f"[{per_doubling['OR_lo']:.2f}, {per_doubling['OR_hi']:.2f}].",
        "Words: \\texttt{word\\_count} in \\texttt{combined\\_<country>.csv} (matches of "
        "\\texttt{\\textbackslash w+}).",
    ],
    colspec="lrrrr",
    size=r"\small",
)

save_json(results, "ch5_analysis.json")
