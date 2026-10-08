# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b5_figures.py
# Purpose: B5 figures 1 to 7 (four countries, classifier run cefr_20260817_231925).
#          Run b4_regression.py first (figure 7 reads its coefficient CSV).
# Pipeline step: Analysis: Chapter 5 figures. Also draws fig3 to fig5 and fig7b, which are not in the
#                thesis; fig4 reads tables/b5_fig4_tokens_E_de_at.csv when the model is absent, and
#                the script stops there (before fig6 and fig7a) if that file is missing.
# Thesis: Section 5.2 fig1_cefr_distribution_by_country, Section 5.5 fig2_leichte_sprache_pairs,
#         Section 5.6 fig6_share_demanding_by_area, Section 5.7 fig7a_forest_M3a_country_scope.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from common import (PROJECT_ROOT, RUN_DIR, TAB_DIR, COUNTRIES, COUNTRY_LABEL, COUNTRY_COLOUR,
                    CEFR, CEFR_COLOUR, CEFR_TEXT_COLOUR, CEFR_CMAP, SCOPE_COLOUR, GRID_COLOUR, LINE_GREY, CITIZEN_AREAS,
                    BUSINESS_AREAS, AREA_NAMES, ANALYSIS_DF, load_cefr, load_text, combined_csv, save_fig,
                    save_json, wilson)

cefr = load_cefr()
assert len(cefr) == 8997
results = {}

# ── Figure 1: CEFR distribution by country, 100% stacked ───────────────────
rows = []
for c in COUNTRIES + ["all"]:
    d = cefr if c == "all" else cefr[cefr.country == c]
    shares = d["pred_label"].value_counts(normalize=True).reindex(CEFR, fill_value=0) * 100
    rows.append({"group": "All portals" if c == "all" else COUNTRY_LABEL[c], "n": len(d),
                 **{lvl: shares[lvl] for lvl in CEFR}})
dist = pd.DataFrame(rows)
dist.to_csv(os.path.join(TAB_DIR, "b5_fig1_cefr_distribution.csv"), index=False, float_format="%.2f")
results["fig1"] = dist.to_dict(orient="records")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ypos = [0, 1, 2, 3, 4.6]
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
ax.axhline(3.8, color="#999999", lw=0.6, ls=(0, (4, 3)))
ax.legend(handles=[Patch(facecolor=CEFR_COLOUR[l], label=l) for l in CEFR],
          title="Predicted CEFR level", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          frameon=False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save_fig(fig, "fig1_cefr_distribution_by_country")

# ── Figure 2: Leichte Sprache vs standard pages ────────────────────────────
PAIRS = [  # standard URL, Leichte Sprache URL, English service name (from make_thesis_figures.py)
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/102711706",
     "https://verwaltung.bund.de/portal/DE/Produkt-Info-Stellen", "Product Contact Points"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/102837976",
     "https://verwaltung.bund.de/portal/DE/Netz-Werk_Einheitlicher_Ansprech-Partner",
     "Point of Single Contact Network"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/102837977",
     "https://verwaltung.bund.de/portal/DE/Produkt-Info-Stelle_fuer_das_Bau-Wesen",
     "Product Contact Point for Construction"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/102837978",
     "https://verwaltung.bund.de/portal/DE/Nationales_Beratungs-Zentrum_fuer_Berufs-Qualifikationen",
     "Advice Centre for Professional Qualifications"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/102837979",
     "https://verwaltung.bund.de/portal/DE/EU_Patienten", "EU Patients"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/102837980",
     "https://verwaltung.bund.de/portal/DE/EURES_Deutschland", "EURES Germany"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/103710701",
     "https://verwaltung.bund.de/portal/DE/Gleichbehandlungs-Stelle_EU-Arbeitnehmer",
     "Equal Treatment Office for EU Workers"),
    ("https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/105318912",
     "https://verwaltung.bund.de/portal/DE/Europaeisches_Verbraucher-Zentrum_Deutschland",
     "European Consumer Centre"),
]
de = cefr[cefr.country == "germany"].set_index("url")
prs = []
for s, l, name in PAIRS:
    if s not in de.index or l not in de.index:
        raise RuntimeError(f"Leichte Sprache pair missing: {name}")
    prs.append({"service": name, "std_label": de.loc[s, "pred_label"], "ls_label": de.loc[l, "pred_label"],
                "std_E": de.loc[s, "E"], "ls_E": de.loc[l, "E"],
                "std_raw_category": de.loc[s, "raw_category"], "ls_raw_category": de.loc[l, "raw_category"]})
prs = pd.DataFrame(prs)
prs["change_E"] = prs["std_E"] - prs["ls_E"]
prs = prs.sort_values("change_E", ascending=False).reset_index(drop=True)
prs.to_csv(os.path.join(TAB_DIR, "b5_fig2_leichte_sprache_pairs.csv"), index=False, float_format="%.4f")
results["fig2"] = {"pairs": prs.to_dict(orient="records"),
                   "mean_change_E": prs["change_E"].mean(),
                   "n_pairs_lower_E": int((prs["change_E"] > 0).sum()),
                   "n_pairs_lower_label": int((prs["std_label"].map(CEFR.index) >
                                               prs["ls_label"].map(CEFR.index)).sum())}

fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, r in prs.iterrows():
    ax.annotate("", xy=(r.ls_E, i), xytext=(r.std_E, i),
                arrowprops=dict(arrowstyle="-|>", color=LINE_GREY, lw=1.0,
                                shrinkA=6, shrinkB=6))
    # dot colour follows the expected CEFR index (1 = A, 4 = C)
    ax.scatter(r.std_E, i, s=60, facecolor="white", edgecolor=CEFR_CMAP((r.std_E - 1) / 3),
               linewidth=1.6, zorder=3)
    ax.scatter(r.ls_E, i, s=60, color=CEFR_CMAP((r.ls_E - 1) / 3), edgecolor="white", linewidth=0.8,
               zorder=3)
    ax.text(r.std_E + 0.07, i, r.std_label, va="center", ha="left", fontsize=8, color="#444444")
    ax.text(r.ls_E - 0.07, i, r.ls_label, va="center", ha="right", fontsize=8, color="#444444")
ax.set_yticks(range(len(prs)), prs["service"])
ax.invert_yaxis()
ax.set_xlim(0.8, 4.2)
ax.set_xticks([1, 2, 3, 4], ["1 (A)", "2 (B1)", "3 (B2)", "4 (C)"])
ax.set_xlabel("Expected CEFR index")
ax.xaxis.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=LINE_GREY,
           markeredgewidth=1.6, markersize=8, label="Standard page"),
    Line2D([], [], marker="o", ls="none", color=LINE_GREY, markersize=8,
           label="Leichte Sprache page")],
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save_fig(fig, "fig2_leichte_sprache_pairs")

# ── Figure 3: Violin of E, Germany vs Austria ──────────────────────────────
vals = {c: cefr.loc[cefr.country == c, "E"].to_numpy() for c in ["germany", "austria"]}
fig, ax = plt.subplots(figsize=(5.0, 4.0))
parts = ax.violinplot([vals["germany"], vals["austria"]], positions=[1, 2], widths=0.8,
                      showextrema=False, bw_method=0.15)
for body, c in zip(parts["bodies"], ["germany", "austria"]):
    body.set_facecolor(COUNTRY_COLOUR[c])
    body.set_edgecolor("none")
    body.set_alpha(0.55)
res3 = {}
for pos, c in zip([1, 2], ["germany", "austria"]):
    q1, med, q3 = np.percentile(vals[c], [25, 50, 75])
    ax.add_patch(plt.Rectangle((pos - 0.07, q1), 0.14, q3 - q1, facecolor="white",
                               edgecolor=COUNTRY_COLOUR[c], lw=1.2, zorder=3))
    ax.plot([pos - 0.07, pos + 0.07], [med, med], color=COUNTRY_COLOUR[c], lw=2.2, zorder=4)
    res3[c] = {"n": len(vals[c]), "median": med, "q1": q1, "q3": q3, "mean": vals[c].mean()}
mw = stats.mannwhitneyu(vals["germany"], vals["austria"], alternative="two-sided")
res3["mann_whitney_U"] = float(mw.statistic)
res3["mann_whitney_p"] = float(mw.pvalue)
results["fig3"] = res3
ax.set_xticks([1, 2], [f"Germany\n(n = {len(vals['germany']):,})",
                       f"Austria\n(n = {len(vals['austria']):,})"])
ax.set_yticks([1, 2, 3, 4], ["1 (A)", "2 (B1)", "3 (B2)", "4 (C)"])
ax.set_ylim(0.9, 4.1)
ax.set_ylabel("Expected CEFR index")
ax.yaxis.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
save_fig(fig, "fig3_violin_expected_cefr_de_at")

# ── Figure 4: E vs page length in classifier tokens (DE, AT) ───────────────
# Token counts need the scraped text and the classifier tokenizer, which are not shipped;
# without them the counts saved in tables/b5_fig4_tokens_E_de_at.csv are reused.
TOKENS_CSV = os.path.join(TAB_DIR, "b5_fig4_tokens_E_de_at.csv")
tok_dir = os.path.join(RUN_DIR, "mdeberta_cefr_multi")
sc = cefr[cefr.country.isin(["germany", "austria"])][["country", "url", "E"]]
if os.path.isdir(tok_dir) and os.path.exists(combined_csv("germany")):
    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained(tok_dir)
    text = load_text()
    sc = sc.merge(text[["url", "text_joined"]], on="url", how="left", validate="one_to_one")
    sc["tokens"] = [len(tok.encode(t, add_special_tokens=False)) for t in sc["text_joined"].str.strip()]
else:
    print("  scraped text or tokenizer not found; using saved token counts")
    sc = sc.merge(pd.read_csv(TOKENS_CSV)[["url", "tokens"]], on="url", how="left",
                  validate="one_to_one")
assert (sc["tokens"] > 0).all()


def running_median(x, y, n_bins=40, min_count=30):
    bins = np.logspace(np.log10(x.min()), np.log10(x.max() + 1), n_bins + 1)
    cx, cy = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_count:
            cx.append(np.sqrt(lo * hi))
            cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


res4 = {}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for c in ["germany", "austria"]:
    d = sc[sc.country == c]
    ax.scatter(d["tokens"], d["E"], s=7, alpha=0.25, color=COUNTRY_COLOUR[c], linewidths=0,
               rasterized=True)
    cx, cy = running_median(d["tokens"].to_numpy(), d["E"].to_numpy())
    ax.plot(cx, cy, color=COUNTRY_COLOUR[c], lw=2.2, zorder=5)
    rho, p = stats.spearmanr(d["tokens"], d["E"])
    res4[c] = {"n": len(d), "spearman_rho": float(rho), "p": float(p),
               "median_tokens": float(d["tokens"].median()),
               "n_running_median_bins": int(len(cx))}
results["fig4"] = res4
sc[["country", "url", "tokens", "E"]].to_csv(os.path.join(TAB_DIR, "b5_fig4_tokens_E_de_at.csv"),
                                             index=False, float_format="%.4f")
ax.set_xscale("log")
ax.set_xlabel("Page length (classifier tokens, log scale)")
ax.set_ylabel("Expected CEFR index")
ax.set_yticks([1, 2, 3, 4], ["1 (A)", "2 (B1)", "3 (B2)", "4 (C)"])
ax.set_ylim(0.9, 4.1)
ax.yaxis.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
hand = []
for c in ["germany", "austria"]:
    r = res4[c]
    hand.append(Line2D([], [], color=COUNTRY_COLOUR[c], lw=2.2,
                       label=f"{COUNTRY_LABEL[c]}: running median (rho = {r['spearman_rho']:.2f})"))
ax.legend(handles=hand, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save_fig(fig, "fig4_expected_cefr_vs_length_de_at")

# ── Figure 5: ECDF of E, DE / AT / IE ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.0))
res5 = {}
for c, ls in zip(["germany", "austria", "ireland"], ["-", "--", ":"]):
    x = np.sort(cefr.loc[cefr.country == c, "E"].to_numpy())
    yv = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, yv, where="post", color=COUNTRY_COLOUR[c], lw=1.8, ls=ls,
            label=f"{COUNTRY_LABEL[c]} (n = {len(x):,})")
    res5[c] = {"n": len(x), "median": float(np.median(x)),
               "share_E_le_2": float((x <= 2).mean()), "share_E_ge_3": float((x >= 3).mean())}
results["fig5"] = res5
ax.set_xlabel("Expected CEFR index")
ax.set_ylabel("Cumulative share of pages")
ax.set_xticks([1, 2, 3, 4], ["1 (A)", "2 (B1)", "3 (B2)", "4 (C)"])
ax.set_xlim(1, 4)
ax.set_ylim(0, 1.01)
ax.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False)
save_fig(fig, "fig5_ecdf_expected_cefr_de_at_ie")

# ── Figure 6: share of B2/C per SDGR area with 95% CI ──────────────────────
adf = pd.read_csv(ANALYSIS_DF)
m6 = adf[adf.scope.notna()]
areas = CITIZEN_AREAS + BUSINESS_AREAS
r6 = []
for a in areas:
    d = m6[m6.area == a]
    k, n = int(d.cefr_binary.sum()), len(d)
    lo, hi = wilson(k, n)
    r6.append({"area": a, "name": AREA_NAMES[a], "scope": "citizen" if a in CITIZEN_AREAS else "business",
               "n": n, "n_B2_C": k, "share": k / n, "lo": lo, "hi": hi})
r6 = pd.DataFrame(r6)
r6.to_csv(os.path.join(TAB_DIR, "b5_fig6_share_demanding_by_area.csv"), index=False, float_format="%.4f")
overall = m6.cefr_binary.mean()
results["fig6"] = {"overall_share": overall, "areas": r6.to_dict(orient="records")}

fig, ax = plt.subplots(figsize=(7.2, 5.6))
ypos = list(range(9)) + [y + 0.8 for y in range(9, 17)]
for y, r in zip(ypos, r6.itertuples()):
    col = SCOPE_COLOUR[r.scope]
    ax.plot([r.lo * 100, r.hi * 100], [y, y], color=col, lw=1.6)
    ax.scatter(r.share * 100, y, s=40, color=col, edgecolor="white", linewidth=0.8, zorder=3)
ax.axvline(overall * 100, color="#777777", lw=0.8, ls="--")
ax.text(overall * 100 + 0.8, -0.9, f"All areas {overall * 100:.0f}%", fontsize=8, color="#555555",
        va="center")
ax.set_yticks(ypos, [f"{r.area}  {r.name} (n = {r.n:,})" for r in r6.itertuples()])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Harder pages, B2 or C (%, 95% Wilson CI)")
ax.xaxis.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", color=SCOPE_COLOUR[s], lw=1.6, label=lab)
                   for s, lab in [("citizen", "Citizen area (A to I)"),
                                  ("business", "Business area (J to Q)")]],
          loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False)
save_fig(fig, "fig6_share_demanding_by_area")

# ── Figure 7: forest plots ─────────────────────────────────────────────────
coefs = pd.read_csv(os.path.join(TAB_DIR, "b4_coefficients_binary.csv"))


def forest(model, labels, colours, name, size, xlim):
    d = coefs[coefs.model == model].set_index("label").loc[labels].reset_index()
    fig, ax = plt.subplots(figsize=size)
    for i, r in d.iterrows():
        ax.plot([r.OR_lo, r.OR_hi], [i, i], color=colours[i], lw=1.6)
        ax.scatter(r.OR, i, s=42, color=colours[i], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(1, color="#555555", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ticks = [t for t in [0.1, 0.2, 0.25, 0.5, 1, 2, 4, 8] if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(d)), [f"{r.label}  {r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]"
                                  for r in d.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio for harder text, B2 or C (log scale, 95% CI)")
    ax.xaxis.grid(True, color=GRID_COLOUR, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return fig, ax


labs = ["Austria", "Netherlands", "Ireland", "Business"]
fig, ax = forest("M3a", labs, [COUNTRY_COLOUR["austria"], COUNTRY_COLOUR["netherlands"],
                               COUNTRY_COLOUR["ireland"], SCOPE_COLOUR["business"]],
                 "fig7a", (6.4, 2.6), (0.25, 8))
ax.set_xlabel("Odds ratio for harder text, B2 or C (log scale, 95% CI)\nReference: Germany, citizen scope")
save_fig(fig, "fig7a_forest_M3a_country_scope")

alabs = [f"{a}: {AREA_NAMES[a]}" for a in areas if a != "B"]
cols = [SCOPE_COLOUR["citizen" if l[0] in CITIZEN_AREAS else "business"] for l in alabs]
fig, ax = forest("M3b", alabs, cols, "fig7b", (7.2, 5.4), (0.5, 8))
ax.set_xlabel("Odds ratio for harder text, B2 or C (log scale, 95% CI)\nReference: area B, Work and retirement; "
              "model also adjusts for country")
ax.legend(handles=[Line2D([], [], marker="o", color=SCOPE_COLOUR[s], lw=1.6, label=lab)
                   for s, lab in [("citizen", "Citizen area"), ("business", "Business area")]],
          loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, frameon=False)
save_fig(fig, "fig7b_forest_M3b_areas")

save_json(results, "b5_figures.json")
for k in ["fig2", "fig3", "fig4", "fig5"]:
    print(k, {kk: vv for kk, vv in results[k].items() if kk != "pairs"})
print(prs.to_string())
print(r6.to_string())
