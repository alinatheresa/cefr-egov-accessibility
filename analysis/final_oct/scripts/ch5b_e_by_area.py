# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/ch5b_e_by_area.py
# Purpose: mean expected CEFR index E per SDGR area (table, figure). Reads existing outputs only;
#          nothing is re-classified, re-categorised or refitted.
#          Sample: the regression sample (pages with a CEFR label and a citizen or business
#          area, N = 7,359), pooled over the four countries; the same sample as
#          fig6_share_demanding_by_area (b5_figures.py), whose style the figure copies.
#          E = 1*pA + 2*pB1 + 3*pB2 + 4*pC; 95% CI of the mean: percentile bootstrap,
#          2,000 resamples, seed 42 (a fresh generator per row, so each row is reproducible).
#          Binary outcome: easier = A or B1, harder = B2 or C.
# Pipeline step: Analysis: mean expected CEFR index E by SDGR area and scope. The figure
#                ch5b_fig_mean_e_by_area it also draws is not in the thesis.
# Thesis: Chapter 5 and 6 numbers from tab:e-by-area.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (TAB_DIR, CEFR, SCOPE_COLOUR, GRID_COLOUR, CITIZEN_AREAS, BUSINESS_AREAS,
                    AREA_NAMES, ANALYSIS_DF, load_cefr, tex_table, fmt_int, save_fig, save_json)

N_BOOT, SEED = 2000, 42

# ── Data ────────────────────────────────────────────────────────────────────
adf = pd.read_csv(ANALYSIS_DF)
reg = adf[adf.scope.notna()].copy()
assert len(reg) == 7359 and reg["url"].is_unique
cefr = load_cefr()[["url", "pred_label", "E"]].rename(columns={"pred_label": "pred_label_cefr"})
reg = reg.merge(cefr, on="url", how="left", validate="one_to_one")
assert reg["E"].notna().all() and (reg["pred_label"] == reg["pred_label_cefr"]).all()


def boot_ci(v):
    rng = np.random.default_rng(SEED)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarise(d):
    v = d["E"].to_numpy()
    lo, hi = boot_ci(v)
    shares = d["pred_label"].value_counts(normalize=True).reindex(CEFR, fill_value=0) * 100
    return {"n": len(d), "mean_E": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "median_E": float(np.median(v)), **{f"pct_{l}": float(shares[l]) for l in CEFR},
            "pct_harder": float(shares["B2"] + shares["C"])}


rows = [{"area": a, "name": AREA_NAMES[a],
         "scope": "citizen" if a in CITIZEN_AREAS else "business", **summarise(reg[reg.area == a])}
        for a in CITIZEN_AREAS + BUSINESS_AREAS]
rows += [{"area": "", "name": "Citizen areas", "scope": "citizen",
          **summarise(reg[reg.scope == "citizen"])},
         {"area": "", "name": "Business areas", "scope": "business",
          **summarise(reg[reg.scope == "business"])},
         {"area": "", "name": "All areas", "scope": "", **summarise(reg)}]
tab = pd.DataFrame(rows)
# n and harder share must equal figure 6 (b5_fig6_share_demanding_by_area.csv)
f6 = pd.read_csv(os.path.join(TAB_DIR, "b5_fig6_share_demanding_by_area.csv")).set_index("area")
chk = tab[tab.area != ""].set_index("area")
assert (chk["n"] == f6["n"]).all()
assert ((chk["pct_harder"] - 100 * f6["share"]).abs() < 0.01).all()
tab.to_csv(os.path.join(TAB_DIR, "ch5b_e_by_area.csv"), index=False, float_format="%.4f")
print(tab.to_string())
save_json({"n_boot": N_BOOT, "seed": SEED, "e_by_area": tab.to_dict("records")},
          "ch5b_e_by_area.json")


# ── Table ───────────────────────────────────────────────────────────────────
def cells(r):
    return [fmt_int(r.n), f"{r.mean_E:.2f} [{r.lo:.2f}, {r.hi:.2f}]", f"{r.median_E:.2f}",
            *[f"{getattr(r, 'pct_' + l):.1f}" for l in CEFR], f"{r.pct_harder:.1f}"]


areas = tab[tab.area != ""]
tex_rows, mids = [], []
for scope, lab in [("citizen", "Citizen (A to I)"), ("business", "Business (J to Q)")]:
    if tex_rows:
        mids.append(len(tex_rows))
    tex_rows.append([f"\\textbf{{{lab}}}"] + [""] * 8)
    for r in areas[areas.scope == scope].itertuples():
        tex_rows.append([f"\\quad {r.area}~~{r.name}"] + cells(r))
mids.append(len(tex_rows))
for r in tab[tab.area == ""].itertuples():
    tex_rows.append([r.name] + cells(r))
tex_table(
    os.path.join(TAB_DIR, "ch5b_e_by_area.tex"),
    caption="Expected CEFR index E by SDGR area",
    label="tab:e-by-area",
    header=["", "n", "Mean E [95\\% CI]", "Median E", "A", "B1", "B2", "C", "Harder"],
    rows=tex_rows,
    midrules=tuple(mids),
    notes=[
        "Regression sample (7,359 pages with a CEFR label and a citizen or business area), "
        "pooled over the four countries; the same pages as Figure "
        "\\texttt{fig6\\_share\\_demanding\\_by\\_area}.",
        "E: expected CEFR index, 1 $\\times$ pA + 2 $\\times$ pB1 + 3 $\\times$ pB2 + "
        "4 $\\times$ pC. 95\\% CI: percentile bootstrap of the mean (2,000 resamples, seed 42). "
        "A to C: share of pages per predicted level (\\%). Harder: B2 or C (\\%).",
        "Area: SDGR Annex I area assigned by the zero-shot categoriser.",
    ],
    colspec="lrrrrrrrr",
    size=r"\footnotesize\setlength{\tabcolsep}{4pt}",
)

# ── Figure: twin of fig6 (b5_figures.py), measure E instead of % harder ────
r6 = areas.reset_index(drop=True)
overall = float(tab.loc[tab.name == "All areas", "mean_E"].iloc[0])
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ypos = list(range(9)) + [y + 0.8 for y in range(9, 17)]
for y, r in zip(ypos, r6.itertuples()):
    col = SCOPE_COLOUR[r.scope]
    ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6)
    ax.scatter(r.mean_E, y, s=40, color=col, edgecolor="white", linewidth=0.8, zorder=3)
ax.axvline(overall, color="#777777", lw=0.8, ls="--")
# fig6 offsets its label by 0.8 points on a 0 to 100 axis; the same share of the 1 to 4 axis
ax.text(overall + 0.8 * 3 / 100, -0.9, f"All areas {overall:.2f}", fontsize=8, color="#555555",
        va="center")
ax.set_yticks(ypos, [f"{r.area}  {r.name} (n = {r.n:,})" for r in r6.itertuples()])
ax.invert_yaxis()
ax.set_xlim(1, 4)
ax.set_xticks([1, 2, 3, 4], ["A (1)", "B1 (2)", "B2 (3)", "C (4)"])
ax.set_xlabel("Mean expected CEFR index E (95% bootstrap CI)")
ax.xaxis.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", color=SCOPE_COLOUR[s], lw=1.6, label=lab)
                   for s, lab in [("citizen", "Citizen area (A to I)"),
                                  ("business", "Business area (J to Q)")]],
          loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False)
save_fig(fig, "ch5b_fig_mean_e_by_area")
