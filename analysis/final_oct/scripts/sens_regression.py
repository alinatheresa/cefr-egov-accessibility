# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/sens_regression.py
# Purpose: sensitivity of the primary binary logit (M3a, M3b) to the two data collection problems
#          found in data_collection_report.md: German leistung pages are short teasers, and the
#          Dutch corpus contains news and document pages. Same coding, references and estimation
#          rules as b4_regression.py. Reads existing outputs only (analysis_dataframe.csv and the
#          step 02 combined_<country>.csv files); nothing is re-scraped or re-classified.
#          Specifications: A main; B1 drop pages under 50 words (all countries); B2 drop German
#          leistung pages; C drop Dutch news and document pages; D = B1 and C.
# Pipeline step: Analysis: sensitivity specifications S1 to S4 (labelled B1, B2, C, D in the files).
#                The forest plots it also draws are not in the thesis.
# Thesis: Chapter 5 robustness table (tab:robustness-overview, via ch5b_analysis.py); full
#         outputs tab:sens-country-ors and tables/sens_*.csv.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf

from common import (TAB_DIR, RES_DIR, ANALYSIS_DF, COUNTRIES, COUNTRY_LABEL, COUNTRY_COLOUR,
                    CEFR, SCOPE_COLOUR, GRID_COLOUR, LINE_GREY, combined_csv, tex_table, fmt_p,
                    fmt_int, save_fig, save_json)

MIN_WORDS = 50      # rule B1, same for every country
SPARSE = 30         # flag threshold for cells

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(ANALYSIS_DF)
assert len(df) == 8997, len(df)
text = pd.concat([pd.read_csv(combined_csv(c), usecols=["category", "url", "word_count"])
                  .assign(src=c) for c in COUNTRIES], ignore_index=True)
df = df.merge(text, on="url", how="left", validate="one_to_one")
assert df["word_count"].notna().all() and (df["src"] == df["country"]).all()

df["teaser"] = (df["country"] == "germany") & (df["category"] == "leistung")
nl_seg = df["url"].str.extract(r"rijksoverheid\.nl/onderwerpen/[^/]+/([^/]+)")[0]
df["nl_newsdoc"] = (df["country"] == "netherlands") & nl_seg.isin(["nieuws", "documenten"])
df["nl_type"] = np.where(df["nl_newsdoc"], nl_seg, "rest")
assert df["teaser"].sum() == 1049 and df["nl_newsdoc"].sum() == 954

df["country"] = pd.Categorical(df["country"], categories=COUNTRIES)
df["y"] = df["cefr_binary"].astype(int)
main = df[df["scope"].notna()].copy()
main["scope"] = pd.Categorical(main["scope"], categories=["citizen", "business"])
assert len(main) == 7359

CTRY = "C(country, Treatment('germany'))"
SCP = "C(scope, Treatment('citizen'))"
AREA = "C(area, Treatment('B'))"

SPECS = {
    "A": ("A. Main specification", main),
    "B1": (f"B1. Pages under {MIN_WORDS} words dropped", main[main["word_count"] >= MIN_WORDS]),
    "B2": ("B2. German leistung pages dropped", main[~main["teaser"]]),
    "C": ("C. Dutch news and document pages dropped", main[~main["nl_newsdoc"]]),
    "D": ("D. B1 and C combined", main[(main["word_count"] >= MIN_WORDS) & ~main["nl_newsdoc"]]),
}
TERMS = {"Austria": f"{CTRY}[T.austria]", "Netherlands": f"{CTRY}[T.netherlands]",
         "Ireland": f"{CTRY}[T.ireland]", "Business": f"{SCP}[T.business]"}

results = {"min_words": MIN_WORDS, "sparse_threshold": SPARSE}

# ── 1. Descriptives ─────────────────────────────────────────────────────────


def dist(d):
    s = d["pred_label"].value_counts().reindex(CEFR, fill_value=0)
    out = {"n": int(len(d))}
    for lvl in CEFR:
        out[lvl] = int(s[lvl])
        out[f"{lvl}_pct"] = 100 * s[lvl] / len(d)
    out["B2_or_C_pct"] = 100 * (s["B2"] + s["C"]) / len(d)
    out["median_words"] = float(d["word_count"].median())
    return out


ger = df[df["country"] == "germany"]
d1 = pd.DataFrame([{"group": "German leistung pages (teasers)", **dist(ger[ger["teaser"]])},
                   {"group": "Other German pages", **dist(ger[~ger["teaser"]])},
                   {"group": "All German pages", **dist(ger)}])
nl = df[df["country"] == "netherlands"]
d3 = pd.DataFrame([{"group": "Dutch news pages (nieuws)", **dist(nl[nl["nl_type"] == "nieuws"])},
                   {"group": "Dutch document pages (documenten)",
                    **dist(nl[nl["nl_type"] == "documenten"])},
                   {"group": "Other Dutch pages", **dist(nl[~nl["nl_newsdoc"]])},
                   {"group": "All Dutch pages", **dist(nl)}])
d2 = (df.groupby(["country", "pred_label"], observed=True)["word_count"]
      .agg(n="size", median="median").reset_index())
d2["pred_label"] = pd.Categorical(d2["pred_label"], categories=CEFR, ordered=True)
d2 = d2.sort_values(["country", "pred_label"])
d1.to_csv(os.path.join(TAB_DIR, "sens_cefr_german_teasers.csv"), index=False, float_format="%.2f")
d3.to_csv(os.path.join(TAB_DIR, "sens_cefr_dutch_newsdocs.csv"), index=False, float_format="%.2f")
d2.to_csv(os.path.join(TAB_DIR, "sens_median_words_by_level.csv"), index=False)
print(d1.to_string(), "\n", d3.to_string(), "\n", d2.to_string(), sep="")
results["desc_german"] = d1.to_dict("records")
results["desc_dutch"] = d3.to_dict("records")
results["median_words_by_level"] = d2.astype({"pred_label": str}).to_dict("records")


def pct_row(r):
    return [f"{r[f'{l}_pct']:.1f}" for l in CEFR] + [f"{r['B2_or_C_pct']:.1f}"]


rows = [[r["group"], fmt_int(r["n"]), f"{r['median_words']:g}"] + pct_row(r)
        for _, r in pd.concat([d1, d3]).iterrows()]
med = d2.pivot(index="country", columns="pred_label", values="median").reindex(COUNTRIES)
cnt = d2.pivot(index="country", columns="pred_label", values="n").reindex(COUNTRIES)
rows += [[f"{COUNTRY_LABEL[c]}", fmt_int(cnt.loc[c].sum()), ""]
         + [f"{med.loc[c, l]:g} ({fmt_int(cnt.loc[c, l])})" for l in CEFR] + [""]
         for c in COUNTRIES]
tex_table(
    os.path.join(TAB_DIR, "sens_descriptives.tex"),
    caption="Predicted CEFR level of German teaser pages and Dutch news and document pages, "
            "and page length by predicted level",
    label="tab:sens-descriptives",
    header=["", "Pages", "Median words", "A", "B1", "B2", "C", "Harder"],
    rows=rows,
    midrules=(3, 7),
    notes=[
        "Upper two panels: share of pages per predicted level (\\%); harder: B2 or C. German teasers: the 1,049 "
        "leistung pages; the remaining 214 German pages include the 8 Leichte Sprache pages. "
        "Dutch news and documents: pages whose third path segment is nieuws or documenten "
        "(including topic news and document index pages).",
        "Lower panel: median words per page for each predicted level, number of pages in "
        "parentheses. All 8,997 pages with a CEFR label.",
        "Words: \\texttt{word\\_count} in \\texttt{combined\\_<country>.csv} (matches of "
        "\\texttt{\\textbackslash w+}).",
    ],
    colspec="lrrrrrrr",
    size=r"\small",
)

# ── 2. Refits ───────────────────────────────────────────────────────────────
coef_rows, sample_rows, warn_log = [], [], {}


def fit(model, rhs, data, spec):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = smf.logit(f"y ~ {rhs}", data).fit(disp=False, maxiter=500, method="newton")
    msgs = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in w})
    conv = bool(res.mle_retvals.get("converged", False))
    big = res.params[(res.params.abs() > 10) | (res.bse > 10)].index.tolist()
    warn_log[f"{model}_{spec}"] = {"warnings": msgs, "converged": conv,
                                   "large_coef_or_se_terms": big, "N": int(res.nobs),
                                   "aic": res.aic, "mcfadden_r2": 1 - res.llf / res.llnull}
    ci = res.conf_int()
    for lab, t in TERMS.items():
        if t not in res.params.index:
            continue
        coef_rows.append({"model": model, "spec": spec, "term": lab, "coef": res.params[t],
                          "se": res.bse[t], "p": res.pvalues[t], "OR": np.exp(res.params[t]),
                          "OR_lo": np.exp(ci.loc[t, 0]), "OR_hi": np.exp(ci.loc[t, 1]),
                          "N": int(res.nobs)})
    return res


for spec, (title, d) in SPECS.items():
    fit("M3a", f"{CTRY} + {SCP}", d, spec)
    fit("M3b", f"{CTRY} + {AREA}", d, spec)
    n_c = d["country"].value_counts().reindex(COUNTRIES)
    cy = pd.crosstab(d["country"], d["y"]).reindex(index=COUNTRIES, columns=[0, 1], fill_value=0)
    csy = pd.crosstab([d["country"], d["scope"]], d["y"]).reindex(columns=[0, 1], fill_value=0)
    area_n = d["area"].value_counts()
    sample_rows.append({
        "spec": spec, "title": title, "N": len(d), **{f"N_{c}": int(n_c[c]) for c in COUNTRIES},
        **{f"y0_{c}": int(cy.loc[c, 0]) for c in COUNTRIES},
        **{f"y1_{c}": int(cy.loc[c, 1]) for c in COUNTRIES},
        "country_outcome_cells_lt30": [f"{c} y={k}: {int(cy.loc[c, k])}" for c in COUNTRIES
                                       for k in (0, 1) if cy.loc[c, k] < SPARSE],
        "country_scope_outcome_cells_lt30": [f"{c}/{s} y={k}: {int(csy.loc[(c, s), k])}"
                                             for (c, s) in csy.index for k in (0, 1)
                                             if csy.loc[(c, s), k] < SPARSE],
        "areas_lt30": {a: int(n) for a, n in area_n.items() if n < SPARSE},
        "areas_absent": [a for a in "ABCDEFGHIJKLMNOPQ" if a not in area_n.index],
    })

coefs = pd.DataFrame(coef_rows)
samples = pd.DataFrame(sample_rows)

# Specification A must reproduce b4 exactly
b4 = pd.read_csv(os.path.join(TAB_DIR, "b4_coefficients_binary.csv"))
for m in ("M3a", "M3b"):
    for lab in TERMS:
        a = coefs[(coefs.model == m) & (coefs.spec == "A") & (coefs.term == lab)]
        if a.empty:
            continue
        ref = b4[(b4.model == m) & (b4.label == lab)].iloc[0]
        assert abs(a.iloc[0].coef - ref.coef) < 1e-5 and abs(a.iloc[0].se - ref.se) < 1e-5, (m, lab)
print("Specification A reproduces b4 M3a and M3b.")


def flagged(spec, term):
    """Country OR: flag if that country or Germany has < SPARSE pages in an outcome class.
    Business: flag if any scope x outcome cell (pooled over countries) has < SPARSE pages."""
    s = samples[samples.spec == spec].iloc[0]
    if term == "Business":
        d = SPECS[spec][1]
        return bool((pd.crosstab(d["scope"], d["y"]) < SPARSE).any().any())
    c = {"Austria": "austria", "Netherlands": "netherlands", "Ireland": "ireland"}[term]
    return any(s[f"y{k}_{x}"] < SPARSE for x in ("germany", c) for k in (0, 1))


coefs["flag_lt30"] = [flagged(r.spec, r.term) for r in coefs.itertuples()]
coefs.to_csv(os.path.join(TAB_DIR, "sens_country_ors.csv"), index=False, float_format="%.6g")
samples.to_csv(os.path.join(TAB_DIR, "sens_samples.csv"), index=False)
print(coefs.to_string())
print(samples.drop(columns=["title"]).to_string())
for k, v in warn_log.items():
    print(k, v)
results["coefficients"] = coefs.to_dict("records")
results["samples"] = samples.to_dict("records")
results["fit_log"] = warn_log

# ── 3. LaTeX table ──────────────────────────────────────────────────────────


def cell(model, spec, term):
    r = coefs[(coefs.model == model) & (coefs.spec == spec) & (coefs.term == term)]
    if r.empty:
        return "", ""
    r = r.iloc[0]
    mark = "$^\\dagger$" if r.flag_lt30 else ""
    return f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]{mark}", fmt_p(r.p)


rows, mids = [], []
for term in TERMS:
    mids.append(len(rows))
    rows.append([f"\\textbf{{{term}}}", "", "", "", ""])
    for spec, (title, _) in SPECS.items():
        a, ap = cell("M3a", spec, term)
        b, bp = cell("M3b", spec, term)
        rows.append([f"\\quad {title}", a, ap, b, bp])
mids.append(len(rows))
rows.append(["\\textbf{N (Germany / Austria / Netherlands / Ireland)}", "", "", "", ""])
for _, s in samples.iterrows():
    per = " / ".join(fmt_int(s[f"N_{c}"]) for c in COUNTRIES)
    rows.append([f"\\quad {s.title}", f"\\multicolumn{{4}}{{l}}{{{fmt_int(s.N)} ({per})}}"])
lt30 = samples.set_index("spec")
_b1 = SPECS["B1"][1]
n_de_bus_y0_b1 = int(((_b1["country"] == "germany") & (_b1["scope"] == "business")
                      & (_b1["y"] == 0)).sum())
tex_table(
    os.path.join(TAB_DIR, "sens_country_ors.tex"),
    caption="Sensitivity of the country and scope effects to short pages and to Dutch news "
            "and document pages",
    label="tab:sens-country-ors",
    header=["", "M3a OR [95\\% CI]", "p", "M3b OR [95\\% CI]", "p"],
    rows=rows,
    midrules=tuple(mids[1:]),
    notes=[
        "Binary logit of harder (predicted B2 or C, 1) versus easier (predicted A or B1, 0), coded, referenced and "
        "estimated as in Table~\\ref{tab:regression-main}: M3a country + scope (reference "
        "Germany, citizen); M3b country + 17 SDGR areas (reference Germany, area B; area "
        "coefficients not shown, so M3b has no Business row). Wald 95\\% confidence intervals.",
        f"A: estimation sample of the main analysis. B1: pages with fewer than {MIN_WORDS} words "
        "removed in every country. B2: the 1,049 German leistung pages (one- to three-sentence "
        "teasers) removed. C: the 954 Dutch pages under \\texttt{/onderwerpen/<topic>/nieuws/} or "
        "\\texttt{/documenten/} removed. D: B1 and C together. Only pages in the main estimation "
        "sample (citizen or business area) can be removed, so N falls by less than these counts.",
        f"$^\\dagger$ Fewer than {SPARSE} pages in one outcome class (easier, A or B1, or harder, B2 or C) for "
        "the country or for Germany in that sample; the estimate rests on few pages. In B1 "
        f"Germany keeps only {lt30.loc['B1', 'y0_germany']} easier pages "
        f"({n_de_bus_y0_b1} of them business pages), and the same {lt30.loc['D', 'y0_germany']} "
        "in D, so every country odds ratio in B1 and D is compared against these few pages.",
        "Words: \\texttt{word\\_count} in \\texttt{combined\\_<country>.csv} (matches of "
        "\\texttt{\\textbackslash w+}).",
    ],
    colspec="lrrrr",
    size=r"\small",
)

# ── 3b. Forest plots ────────────────────────────────────────────────────────
SPEC_MARK = {"A": "o", "B1": "s", "B2": "D", "C": "^", "D": "v"}
TERM_COLOUR = {"Austria": COUNTRY_COLOUR["austria"], "Netherlands": COUNTRY_COLOUR["netherlands"],
               "Ireland": COUNTRY_COLOUR["ireland"], "Business": SCOPE_COLOUR["business"]}


def forest(model, terms, size):
    d = coefs[coefs.model == model]
    fig, ax = plt.subplots(figsize=size)
    labels, i = [], 0
    for g, term in enumerate(terms):
        if g:
            ax.axhline(i - 0.5, color=GRID_COLOUR, lw=0.8)
        for spec in SPECS:
            r = d[(d.term == term) & (d.spec == spec)].iloc[0]
            col = TERM_COLOUR[term]
            ax.plot([r.OR_lo, r.OR_hi], [i, i], color=col, lw=1.6)
            ax.scatter(r.OR, i, s=42, marker=SPEC_MARK[spec], zorder=3, linewidth=1.0,
                       color="white" if r.flag_lt30 else col, edgecolor=col)
            labels.append(f"{term}, {spec}  {r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]")
            i += 1
    ax.axvline(1, color=LINE_GREY, lw=0.8, ls="--")
    ax.set_xscale("log")
    lo = min(0.25, d.OR_lo.min() * 0.8)
    hi = max(8, d.OR_hi.max() * 1.25)
    ax.set_xlim(lo, hi)
    ticks = [t for t in [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64] if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID_COLOUR, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker=SPEC_MARK[s], color=LINE_GREY, lw=0, label=s)
               for s in SPECS]
    handles.append(Line2D([], [], marker="o", color=LINE_GREY, markerfacecolor="white", lw=0,
                          label=f"open: outcome cell < {SPARSE}"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=6,
              frameon=False, handletextpad=0.2, columnspacing=0.9)
    return fig, ax


fig, ax = forest("M3a", list(TERMS), (6.8, 5.6))
ax.set_xlabel("Odds ratio for harder text, B2 or C (log scale, 95% CI)\nM3a, reference: Germany, citizen scope. "
              "A main; B1 < 50 words dropped;\nB2 German leistung dropped; C Dutch news/documents "
              "dropped; D B1 and C")
save_fig(fig, "sens_forest_M3a")
fig, ax = forest("M3b", ["Austria", "Netherlands", "Ireland"], (6.8, 4.4))
ax.set_xlabel("Odds ratio for harder text, B2 or C (log scale, 95% CI)\nM3b, reference: Germany, area B; "
              "model also adjusts for 17 SDGR areas")
save_fig(fig, "sens_forest_M3b")

save_json(results, "sens_regression.json")
