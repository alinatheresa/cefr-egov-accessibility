# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/ch5b_cefr_by_scope.py
# Purpose: predicted CEFR level by scope, citizen (areas A to I) versus business (J to Q)
#          (table, figure, per-country figure for the annex). Reads existing outputs only;
#          nothing is re-classified, re-categorised or refitted.
#          Sample: the regression sample (pages with a CEFR label and a citizen or business
#          area, N = 7,359); the same sample as fig6_share_demanding_by_area (b5_figures.py),
#          whose scope colours the figures use. 95% CIs: Wilson, per level within scope.
#          Binary outcome: easier = A or B1, harder = B2 or C.
# Pipeline step: Analysis: CEFR level by scope (citizen versus business). The pooled figure
#                ch5b_fig_cefr_by_scope it also draws is not in the thesis.
# Thesis: Section 5.6 ch5b_fig_cefr_by_scope_country; tab:cefr-by-scope.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import (TAB_DIR, COUNTRIES, COUNTRY_LABEL, CEFR, SCOPE_COLOUR, GRID_COLOUR, LINE_GREY,
                    ANALYSIS_DF, load_cefr, tex_table, fmt_int, save_fig, save_json, wilson)

SCOPES = [("citizen", "Citizen areas (A to I)"), ("business", "Business areas (J to Q)")]

# ── Data ────────────────────────────────────────────────────────────────────
adf = pd.read_csv(ANALYSIS_DF)
reg = adf[adf.scope.notna()].copy()
assert len(reg) == 7359 and reg["url"].is_unique
assert set(reg["scope"]) == {"citizen", "business"}
cefr = load_cefr().set_index("url")["pred_label"]
assert (reg["pred_label"] == reg["url"].map(cefr)).all()


def summarise(d):
    n = len(d)
    out = {"n": n}
    for l in CEFR:
        k = int((d["pred_label"] == l).sum())
        lo, hi = wilson(k, n)
        out.update({f"k_{l}": k, f"pct_{l}": 100 * k / n, f"lo_{l}": 100 * lo, f"hi_{l}": 100 * hi})
    out["pct_easier"] = out["pct_A"] + out["pct_B1"]
    out["pct_harder"] = out["pct_B2"] + out["pct_C"]
    return out


rows = []
for country, d_c in [("all", reg)] + [(c, reg[reg.country == c]) for c in COUNTRIES]:
    for s, lab in SCOPES + [("all", "All areas")]:
        d = d_c if s == "all" else d_c[d_c.scope == s]
        rows.append({"country": country, "scope": s, "label": lab, **summarise(d)})
tab = pd.DataFrame(rows)
pooled = tab[tab.country == "all"].set_index("scope")
# scope counts and binary shares must agree with the regression input
assert pooled.loc["citizen", "n"] + pooled.loc["business", "n"] == 7359
assert abs(pooled.loc["all", "pct_harder"] - 100 * reg["cefr_binary"].mean()) < 1e-9
assert tab.groupby("country")["n"].sum().drop("all").sum() == 2 * 7359
tab.to_csv(os.path.join(TAB_DIR, "ch5b_cefr_by_scope.csv"), index=False, float_format="%.4f")
print(tab.drop(columns=[f"{p}_{l}" for p in ("k", "lo", "hi") for l in CEFR]).to_string())
save_json({"cefr_by_scope": tab.to_dict("records")}, "ch5b_cefr_by_scope.json")

# ── Table ───────────────────────────────────────────────────────────────────
tex_table(
    os.path.join(TAB_DIR, "ch5b_cefr_by_scope.tex"),
    caption="Predicted CEFR level by scope",
    label="tab:cefr-by-scope",
    header=["", "n", "A", "B1", "B2", "C", "Easier", "Harder"],
    rows=[[r.label, fmt_int(r.n), *[f"{r[f'pct_{l}']:.1f}" for l in CEFR], f"{r.pct_easier:.1f}",
           f"{r.pct_harder:.1f}"] for _, r in pooled.loc[["citizen", "business", "all"]].iterrows()],
    midrules=(2,),
    notes=[
        "Regression sample (7,359 pages with a CEFR label and a citizen or business area), "
        "pooled over the four countries; the same pages as Figure "
        "\\texttt{fig6\\_share\\_demanding\\_by\\_area}. A to C: share of pages per predicted "
        "level within the scope (\\%). Easier: A or B1; harder: B2 or C (\\%).",
        "Scope: citizen = SDGR Annex I areas A to I, business = areas J to Q, assigned by the "
        "zero-shot categoriser.",
    ],
    colspec="lrrrrrrr",
)


# ── Figures ─────────────────────────────────────────────────────────────────
def grouped_bars(ax, sub, fontsize):
    x = np.arange(len(CEFR))
    w = 0.38
    for i, (s, _) in enumerate(SCOPES):
        r = sub.loc[s]
        xs = x + (i - 0.5) * w
        pct = np.array([r[f"pct_{l}"] for l in CEFR])
        err = np.array([[r[f"pct_{l}"] - r[f"lo_{l}"] for l in CEFR],
                        [r[f"hi_{l}"] - r[f"pct_{l}"] for l in CEFR]])
        ax.bar(xs, pct, width=w, color=SCOPE_COLOUR[s], edgecolor="white", linewidth=0.8,
               zorder=2)
        ax.errorbar(xs, pct, yerr=err, fmt="none", ecolor=LINE_GREY, elinewidth=0.8, capsize=2,
                    capthick=0.8, zorder=3)
        for xi, p, top in zip(xs, pct, pct + err[1]):
            ax.text(xi, top + 1.2, f"{p:.0f}%", ha="center", va="bottom", fontsize=fontsize)
    ax.set_xticks(x, CEFR)
    ax.yaxis.grid(True, color=GRID_COLOUR, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


fig, ax = plt.subplots(figsize=(7.2, 3.8))
grouped_bars(ax, pooled, fontsize=8.5)
ax.set_ylim(0, 50)
ax.set_xlabel("Predicted CEFR level")
ax.set_ylabel("Share of pages within scope (%)")
ax.legend(handles=[Patch(facecolor=SCOPE_COLOUR[s], label=f"{lab}, n = {pooled.loc[s, 'n']:,}")
                   for s, lab in SCOPES],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save_fig(fig, "ch5b_fig_cefr_by_scope")

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.12})
for ax, c in zip(axes.flat, COUNTRIES):
    sub = tab[tab.country == c].set_index("scope")
    grouped_bars(ax, sub, fontsize=7)
    ax.set_title(f"{COUNTRY_LABEL[c]}\ncitizen n = {sub.loc['citizen', 'n']:,}, "
                 f"business n = {sub.loc['business', 'n']:,}", fontsize=9, loc="left")
    ax.set_ylim(0, 100)
for ax in axes[:, 0]:
    ax.set_ylabel("Share of pages within scope (%)")
for ax in axes[1]:
    ax.set_xlabel("Predicted CEFR level")
fig.legend(handles=[Patch(facecolor=SCOPE_COLOUR[s], label=lab) for s, lab in SCOPES],
           loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
save_fig(fig, "ch5b_fig_cefr_by_scope_country")
