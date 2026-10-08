# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/ch5b_analysis.py
# Purpose: outputs for the restructured Chapter 5. Reads existing outputs only; nothing is
#          re-scraped, re-classified or re-categorised, and no model is refitted.
#          Part I (classifier output by country and URL path group):
#            1. CEFR by first-level URL path (table, figure)
#            2. Distribution of the expected CEFR index E by country (figure)
#            3. Classifier confidence by country (table)
#            4. Easier/harder shares with and without the German leistung pages (table)
#            5. Corpus overview and median words per predicted level (tables)
#            6. Example pages (ch5b_examples.md)
#          Part II (SDGR and regression, assembled from existing CSVs):
#            7. SDGR counts per country (table, from b1_descriptives.csv)
#            8. Robustness overview (table, from the b4, sens and ch5 coefficient CSVs)
#          Binary outcome: easier = A or B1, harder = B2 or C.
# Pipeline step: Analysis: Chapter 5 tables and figures by country and URL path (needs the page
#                texts: it joins title and text_joined for the example pages).
# Thesis: Section 5.2 ch5b_fig_expected_index_by_country; Section 5.3 ch5b_fig_cefr_by_path
#         and tab:cefr-by-path; Chapter 5 tab:corpus-overview, tab:confidence,
#         tab:easier-harder-groups, tab:words-by-level, tab:sdgr-counts,
#         tab:robustness-overview.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import (FINAL_DIR, TAB_DIR, COUNTRIES, COUNTRY_LABEL, COUNTRY_COLOUR, CEFR, CEFR_COLOUR,
                    CEFR_TEXT_COLOUR, GRID_COLOUR, LINE_GREY, load_cefr, combined_csv,
                    tex_table, tex_escape, fmt_int, save_fig, save_json)

EASY = ["A", "B1"]
SMALL_N = 30            # path rows with fewer pages are marked in the table
results = {}

# ── Data ────────────────────────────────────────────────────────────────────
corpus = pd.concat([pd.read_csv(combined_csv(c), dtype={"word_count": int}).fillna("")
                    .assign(country=c) for c in COUNTRIES], ignore_index=True)
assert len(corpus) == 8998 and corpus["url"].is_unique

cefr = load_cefr()
assert len(cefr) == 8997
cefr = cefr.merge(corpus[["url", "title", "word_count", "text_joined"]], on="url", how="left",
                  validate="one_to_one")
assert cefr["word_count"].notna().all()
cefr["easier"] = cefr["pred_label"].isin(EASY)

# ── First-level URL paths ───────────────────────────────────────────────────
# The paths included in the corpus, in the order and with the names of the annex tables
# (tables/dc_categories_<country>.tex). raw_category is the path without the language prefix;
# the 8 German Leichte Sprache pages are their own raw_category, shown under their path.
PATHS = {
    "germany": [("lagen", "de_lagen"), ("leistung", "de_leistung"),
                ("rechte_pflichten", "de_rechte_pflichten"),
                ("hilfs-problemloesungsdienste", "de_hilfs-problemloesungsdienste"),
                ("hilfs-problemloesungsdienste_ls",
                 "de_hilfs-problemloesungsdienste (LS)")],
    "austria": [("lebenslagen", "de_lebenslagen"), ("themen", "de_themen"),
                ("landingpages", "de_landingpages"),
                ("gesetzliche-neuerungen", "de_gesetzliche-neuerungen")],
    "netherlands": [("nl_onderwerpen", "nl_onderwerpen")],
    "ireland": [("campaigns", "en_campaigns"), ("policies", "en_policies"),
                ("services", "en_services")],
}
PATH_LABEL = {raw: lab for c in COUNTRIES for raw, lab in PATHS[c]}
LS_RAW = "hilfs-problemloesungsdienste_ls"
for c in COUNTRIES:
    assert set(cefr.loc[cefr.country == c, "raw_category"]) == {raw for raw, _ in PATHS[c]}, c
cefr["group"] = cefr["raw_category"].map(PATH_LABEL)
# Corpus counts per path as in the annex tables; labelled pages differ only for de_lagen
# (118 in the corpus, 117 labelled: the German page without body text).
n_lab = cefr.groupby("raw_category").size()
assert n_lab["lagen"] == 117 and n_lab["leistung"] == 1049 and n_lab["themen"] == 3633
assert n_lab["landingpages"] == 3 and n_lab[LS_RAW] == 8 and n_lab["services"] == 1280
GROUP_ORDER = {c: [lab for _, lab in PATHS[c]] for c in COUNTRIES}


def summarise(d):
    shares = d["pred_label"].value_counts(normalize=True).reindex(CEFR, fill_value=0) * 100
    return {"n": len(d), "median_words": float(d["word_count"].median()),
            **{f"pct_{l}": float(shares[l]) for l in CEFR},
            "pct_easier": float(100 * d["easier"].mean()),
            "pct_harder": float(100 * (~d["easier"]).mean()),
            "median_E": float(d["E"].median()),
            "median_confidence": float(d["pred_confidence"].median())}


def by_group(d_country, col, order, country):
    rows = []
    for g in order:
        m = d_country[col] == g
        s, rest = summarise(d_country[m]), d_country[~m]
        s.update({"country": country, "group": g,
                  "harder_diff_vs_rest_pp": s["pct_harder"] - 100 * (~rest["easier"]).mean(),
                  "medianE_diff_vs_rest": s["median_E"] - rest["E"].median()})
        rows.append(s)
    rows.append({**summarise(d_country), "country": country, "group": "All pages",
                 "harder_diff_vs_rest_pp": np.nan, "medianE_diff_vs_rest": np.nan})
    return rows


path_rows = []
for c in COUNTRIES:
    path_rows += by_group(cefr[cefr.country == c], "group", GROUP_ORDER[c], c)
paths = pd.DataFrame(path_rows)
cols = ["country", "group", "n", "median_words"] + [f"pct_{l}" for l in CEFR] + \
       ["pct_easier", "pct_harder", "median_E", "median_confidence", "harder_diff_vs_rest_pp",
        "medianE_diff_vs_rest"]
paths = paths[cols]
paths.assign(group=paths.group.str.replace("\\", "", regex=False)) \
     .to_csv(os.path.join(TAB_DIR, "ch5b_cefr_by_path.csv"), index=False, float_format="%.4f")
print(paths.to_string())
results["cefr_by_path"] = paths.to_dict("records")

# Groups that differ most from the rest of their portal (by percentage points harder)
diff = {}
for c in COUNTRIES:
    d = paths[(paths.country == c) & (paths.group != "All pages")].copy()
    d = d.reindex(d["harder_diff_vs_rest_pp"].abs().sort_values(ascending=False).index)
    diff[c] = d[["group", "n", "pct_harder", "harder_diff_vs_rest_pp", "median_E",
                 "medianE_diff_vs_rest"]].to_dict("records")
results["largest_differences_vs_rest"] = diff

rows, mids, i = [], [], 0
for c in COUNTRIES:
    d = paths[paths.country == c]
    if rows:
        mids.append(len(rows))
    rows.append([f"\\textbf{{{COUNTRY_LABEL[c]}}}"] + [""] * 9)
    for r in d.itertuples():
        name = r.group.replace("_", "\\_")
        if r.group.endswith("(LS)"):
            name = "\\quad " + name
        if r.group != "All pages" and r.n < SMALL_N:
            name += "$^\\dagger$"
        rows.append([f"\\quad {name}", fmt_int(r.n), f"{r.median_words:g}",
                     *[f"{getattr(r, 'pct_' + l):.1f}" for l in CEFR],
                     f"{r.pct_easier:.1f}", f"{r.pct_harder:.1f}", f"{r.median_E:.2f}"])
tex_table(
    os.path.join(TAB_DIR, "ch5b_cefr_by_path.tex"),
    caption="Predicted CEFR level by URL path group",
    label="tab:cefr-by-path",
    header=["", "n", "Words", "A", "B1", "B2", "C", "Easier", "Harder", "E"],
    rows=rows,
    midrules=tuple(mids),
    notes=[
        "Pages with a CEFR label (8,997). Words: median words per page. A to C: share of pages "
        "per predicted level (\\%). Easier: A or B1; harder: B2 or C (\\%). E: median expected "
        "CEFR index, "
        "1 $\\times$ pA + 2 $\\times$ pB1 + 3 $\\times$ pB2 + 4 $\\times$ pC.",
        "Rows: the first-level URL paths included in the corpus (\\texttt{raw\\_category}), named "
        "and ordered as in the data collection annex. The 8 Leichte Sprache versions of the "
        "German help-service pages are a separate category and are shown under their path "
        "(LS). "
        "de\\_lagen has 118 pages in the corpus; one has no body text and no CEFR label.",
        f"$^\\dagger$ Fewer than {SMALL_N} pages; the shares rest on few pages.",
        "Words: \\texttt{word\\_count} in \\texttt{combined\\_<country>.csv} (matches of "
        "\\texttt{\\textbackslash w+}).",
    ],
    colspec="lrrrrrrrrr",
    size=r"\footnotesize\setlength{\tabcolsep}{4pt}",
)

# ── Figure: CEFR by path group, one panel per country ───────────────────────
nrows = {c: (paths.country == c).sum() for c in COUNTRIES}
fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.8), sharex=True,
                         gridspec_kw={"height_ratios": [nrows[c] + 0.6 for c in COUNTRIES],
                                      "hspace": 0.32})
for ax, c in zip(axes, COUNTRIES):
    d = paths[paths.country == c].reset_index(drop=True)
    ypos = [i if r.group != "All pages" else i + 0.35 for i, r in d.iterrows()]
    for y, r in zip(ypos, d.itertuples()):
        left = 0
        for lvl in CEFR:
            w = getattr(r, f"pct_{lvl}")
            ax.barh(y, w, left=left, height=0.74, color=CEFR_COLOUR[lvl], edgecolor="white",
                    linewidth=0.8)
            if w >= 7:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=7.5,
                        color=CEFR_TEXT_COLOUR[lvl])
            left += w
        ax.text(101.5, y, f"n = {r.n:,}", ha="left", va="center", fontsize=8)
    ax.set_yticks(ypos, list(d.group))
    ax.axhline(ypos[-1] - 0.67, color="#999999", lw=0.6, ls=(0, (4, 3)))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_title(COUNTRY_LABEL[c], loc="left", fontsize=10, fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
axes[-1].set_xlabel("Share of pages (%)")
axes[0].legend(handles=[Patch(facecolor=CEFR_COLOUR[l], label=l) for l in CEFR],
               title="Predicted CEFR level", ncol=4, loc="lower right",
               bbox_to_anchor=(1.0, 1.15), frameon=False)
save_fig(fig, "ch5b_fig_cefr_by_path")

# ── Figure: distribution of E by country ────────────────────────────────────
groups = [(COUNTRY_LABEL[c], cefr.loc[cefr.country == c, "E"].to_numpy(), COUNTRY_COLOUR[c])
          for c in COUNTRIES] + [("All portals", cefr["E"].to_numpy(), "#888888")]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
pos = [1, 2, 3, 4, 5.3]
parts = ax.violinplot([g[1] for g in groups], positions=pos, widths=0.8, showextrema=False,
                      bw_method=0.15)
dist_E = {}
for body, (name, v, col), p in zip(parts["bodies"], groups, pos):
    body.set_facecolor(col)
    body.set_edgecolor("none")
    body.set_alpha(0.55)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(plt.Rectangle((p - 0.07, q1), 0.14, q3 - q1, facecolor="white", edgecolor=col,
                               lw=1.2, zorder=3))
    ax.plot([p - 0.07, p + 0.07], [med, med], color=col, lw=2.2, zorder=4)
    ax.text(p + 0.11, med, f"{med:.2f}", ha="left", va="center", fontsize=8.5, zorder=5)
    dist_E[name] = {"n": len(v), "median": med, "q1": q1, "q3": q3, "mean": float(v.mean())}
ax.axhline(2.5, color=LINE_GREY, lw=0.7, ls=(0, (2, 2)), zorder=1)
ax.text(4.65, 2.53, "B1/B2\nboundary", ha="center", va="bottom", fontsize=7.5, color=LINE_GREY)
ax.set_xticks(pos, [f"{g[0]}\n(n = {len(g[1]):,})" for g in groups])
ax.set_xlim(0.4, 6.0)
ax.set_yticks([1, 2, 3, 4], ["1 (A)", "2 (B1)", "3 (B2)", "4 (C)"])
ax.set_ylim(0.9, 4.1)
ax.set_ylabel("Expected CEFR index E")
ax.yaxis.grid(True, color=GRID_COLOUR, lw=0.6)
ax.set_axisbelow(True)
save_fig(fig, "ch5b_fig_expected_index_by_country")
print(pd.DataFrame(dist_E).T.to_string())
results["expected_index_by_country"] = dist_E
# fig3 of b5_figures.py gives the German and Austrian medians; they must agree
assert abs(dist_E["Germany"]["median"] - 2.5859) < 1e-3 and abs(dist_E["Austria"]["median"] - 3.47) < 5e-3

# ── Table: classifier confidence ────────────────────────────────────────────
conf_rows = []
for name, d in [(COUNTRY_LABEL[c], cefr[cefr.country == c]) for c in COUNTRIES] + [("All portals", cefr)]:
    q1, med, q3 = np.percentile(d["pred_confidence"], [25, 50, 75])
    conf_rows.append({"group": name, "n": len(d), "median_conf": med, "q1_conf": q1, "q3_conf": q3,
                      "pct_conf_lt_05": 100 * (d["pred_confidence"] < 0.5).mean(),
                      "pct_E_near_boundary": 100 * ((d["E"] - 2.5).abs() <= 0.5).mean()})
conf = pd.DataFrame(conf_rows)
conf.to_csv(os.path.join(TAB_DIR, "ch5b_confidence.csv"), index=False, float_format="%.4f")
print(conf.to_string())
results["confidence"] = conf.to_dict("records")
tex_table(
    os.path.join(TAB_DIR, "ch5b_confidence.tex"),
    caption="Classifier confidence by country",
    label="tab:confidence",
    header=["", "n", "Median confidence (IQR)", "Confidence $<$ 0.5 (\\%)",
            "E within 0.5 of 2.5 (\\%)"],
    rows=[[r.group, fmt_int(r.n), f"{r.median_conf:.2f} ({r.q1_conf:.2f} to {r.q3_conf:.2f})",
           f"{r.pct_conf_lt_05:.1f}", f"{r.pct_E_near_boundary:.1f}"] for r in conf.itertuples()],
    midrules=(4,),
    notes=[
        "Pages with a CEFR label. Confidence: the largest of the four predicted probabilities "
        "(the probability of the predicted level). With four levels it is at least 0.25.",
        "E: expected CEFR index, 1 $\\times$ pA + 2 $\\times$ pB1 + 3 $\\times$ pB2 + "
        "4 $\\times$ pC. E within 0.5 of 2.5 means 2.0 $\\le$ E $\\le$ 3.0, i.e. pages close to "
        "the line between easier (A or B1) and harder (B2 or C).",
    ],
    colspec="lrrrr",
)

# ── Table: easier and harder with and without the German leistung pages ────
ger, aut, nld, irl = (cefr[cefr.country == c] for c in COUNTRIES)
leistung = cefr.raw_category.eq("leistung")
eh_groups = [
    ("Germany", "All pages", ger), ("Germany", "Without de_leistung", ger[~leistung[ger.index]]),
    ("Austria", "All pages", aut), ("Netherlands", "All pages", nld), ("Ireland", "All pages", irl),
    ("All portals", "All pages", cefr), ("All portals", "Without de_leistung", cefr[~leistung]),
]
eh = pd.DataFrame([{"country": c, "pages": p, "n": len(d), "n_easier": int(d.easier.sum()),
                    "pct_easier": 100 * d.easier.mean(), "n_harder": int((~d.easier).sum()),
                    "pct_harder": 100 * (~d.easier).mean()} for c, p, d in eh_groups])
assert list(eh.n) == [1263, 214, 3736, 2410, 1588, 8997, 8997 - 1049]
eh.to_csv(os.path.join(TAB_DIR, "ch5b_easier_harder_groups.csv"), index=False, float_format="%.4f")
print(eh.to_string())
results["easier_harder_groups"] = eh.to_dict("records")
rows = [[r.country if r.pages == "All pages" else "", r.pages if r.pages == "All pages"
         else f"\\quad {r.pages}".replace("_", "\\_"), fmt_int(r.n), f"{fmt_int(r.n_easier)} ({r.pct_easier:.1f})",
         f"{fmt_int(r.n_harder)} ({r.pct_harder:.1f})"] for r in eh.itertuples()]
tex_table(
    os.path.join(TAB_DIR, "ch5b_easier_harder_groups.tex"),
    caption="Easier and harder pages by country, with and without the dominant page groups",
    label="tab:easier-harder-groups",
    header=["", "Pages", "n", "Easier, n (\\%)", "Harder, n (\\%)"],
    rows=rows,
    midrules=(2, 3, 4, 5),
    notes=[
        "Pages with a CEFR label. Easier: predicted A or B1; harder: predicted B2 or C. "
        "Descriptive shares; no model is fitted.",
        "de\\_leistung: the 1,049 German service pages under the first-level path leistung "
        "(one- to three-sentence teasers).",
    ],
    colspec="llrrr",
)

# ── Tables: corpus overview and median words per predicted level ────────────
b1 = pd.read_csv(os.path.join(TAB_DIR, "b1_descriptives.csv")).set_index("country")
ov_cols = [COUNTRY_LABEL[c] for c in COUNTRIES] + ["Total"]
words = {COUNTRY_LABEL[c]: corpus.loc[corpus.country == c, "word_count"] for c in COUNTRIES}
words["Total"] = corpus["word_count"]
wq = {k: np.percentile(v, [25, 50, 75]) for k, v in words.items()}
for k, ref in {"Germany": 26, "Austria": 249, "Netherlands": 275, "Ireland": 196.5}.items():
    assert wq[k][1] == ref, k        # as in tables/dc_page_length.tex
lt50 = {k: 100 * (v < 50).mean() for k, v in words.items()}
overview = pd.DataFrame({"pages": b1.loc[ov_cols, "pages_corpus"],
                         "pages_cefr_labelled": b1.loc[ov_cols, "pages_cefr_labelled"],
                         "words_median": [wq[k][1] for k in ov_cols],
                         "words_q1": [wq[k][0] for k in ov_cols],
                         "words_q3": [wq[k][2] for k in ov_cols],
                         "pct_under_50_words": [lt50[k] for k in ov_cols]})
overview.to_csv(os.path.join(TAB_DIR, "ch5b_corpus_overview.csv"), float_format="%.4f")
results["corpus_overview"] = overview.reset_index().to_dict("records")
print(overview.to_string())
tex_table(
    os.path.join(TAB_DIR, "ch5b_corpus_overview.tex"),
    caption="Corpus overview by country",
    label="tab:corpus-overview",
    header=[""] + ov_cols,
    rows=[["Pages in corpus"] + [fmt_int(overview.loc[k, "pages"]) for k in ov_cols],
          ["Pages with CEFR label"] + [fmt_int(overview.loc[k, "pages_cefr_labelled"]) for k in ov_cols],
          ["Words per page, median"] + [f"{overview.loc[k, 'words_median']:g}" for k in ov_cols],
          ["Words per page, IQR"] + [f"{overview.loc[k, 'words_q1']:g} to {overview.loc[k, 'words_q3']:g}"
                                     for k in ov_cols],
          ["Pages under 50 words (\\%)"] + [f"{overview.loc[k, 'pct_under_50_words']:.1f}" for k in ov_cols]],
    midrules=(2,),
    notes=[
        "Words: \\texttt{word\\_count} in \\texttt{combined\\_<country>.csv} (matches of "
        "\\texttt{\\textbackslash w+}), all corpus pages. One German page has no body text and "
        "therefore no CEFR label.",
        "Germany's 1,049 leistung pages have a median of 24 words; all of them have fewer than "
        "50 words (Table~\\ref{tab:cefr-by-path}).",
    ],
    colspec="lrrrrr",
)

med = cefr.pivot_table(index="country", columns="pred_label", values="word_count",
                       aggfunc="median").reindex(index=COUNTRIES, columns=CEFR)
cnt = cefr.pivot_table(index="country", columns="pred_label", values="word_count",
                       aggfunc="size").reindex(index=COUNTRIES, columns=CEFR)
ref = pd.read_csv(os.path.join(TAB_DIR, "sens_median_words_by_level.csv"))
for r in ref.itertuples():
    assert med.loc[r.country, r.pred_label] == r.median and cnt.loc[r.country, r.pred_label] == r.n
results["median_words_by_level"] = {c: {l: float(med.loc[c, l]) for l in CEFR} for c in COUNTRIES}
tex_table(
    os.path.join(TAB_DIR, "ch5b_median_words_by_level.tex"),
    caption="Median words per page by predicted CEFR level",
    label="tab:words-by-level",
    header=["", "A", "B1", "B2", "C"],
    rows=[[COUNTRY_LABEL[c]] + [f"{med.loc[c, l]:g} ({fmt_int(cnt.loc[c, l])})" for l in CEFR]
          for c in COUNTRIES],
    notes=[
        "Median words per page for each predicted level, number of pages in parentheses. All "
        "8,997 pages with a CEFR label. Same numbers as the lower panel of "
        "\\texttt{sens\\_descriptives.tex}.",
        "Words: \\texttt{word\\_count} in \\texttt{combined\\_<country>.csv} (matches of "
        "\\texttt{\\textbackslash w+}).",
    ],
    colspec="lrrrr",
)

# ── Examples ────────────────────────────────────────────────────────────────


def closest_to_median(d, exclude=()):
    d = d[~d.url.isin(exclude)]
    m = d["E"].median()
    d = d.assign(dist=(d["E"] - m).abs()).sort_values(["dist", "url"])
    return d.iloc[0], m


def first_words(text, n=80):
    w = text.split()
    return " ".join(w[:n]) + (" [...]" if len(w) > n else "")


ex, used = [], []
for c in COUNTRIES:
    r, m = closest_to_median(cefr[cefr.country == c])
    ex.append((f"{COUNTRY_LABEL[c]}: page closest to the country median",
               f"{COUNTRY_LABEL[c]} median E {m:.4f}, n = {(cefr.country == c).sum():,}", r))
    used.append(r.url)
lg = cefr[(cefr.country == "germany") & cefr.raw_category.eq("leistung")]
og = cefr[(cefr.country == "germany") & ~cefr.raw_category.eq("leistung")]
for title, d in [("Germany: leistung teaser", lg), ("Germany: other page (not leistung)", og)]:
    r, m = closest_to_median(d, exclude=used)
    ex.append((title, f"median E of this group {m:.4f}, n = {len(d):,}; closest page not already "
               f"shown above", r))
    used.append(r.url)

md = ["# Examples for Chapter 5 (ch5b)", "",
      "Source: classifier run cefr_20260817_231925 (`country_pipelines/*/03_cefr_output_*`), page "
      "text and word counts from `country_pipelines/*/02_scrape_output_*/combined_*.csv`, first-"
      "level URL paths as in `tables/ch5b_cefr_by_path.tex`. Script `scripts/ch5b_analysis.py`. "
      "E = 1 x pA + 2 x pB1 + 3 x pB2 + 4 x pC. Each page is the one whose E is closest to the "
      "median E of the stated group (ties broken by URL). Text: the first 80 words of the "
      "scraped body. `examples.md` is unchanged.", ""]
ex_json = []
for i, (head, basis, r) in enumerate(ex, 1):
    grp = r.group
    md += [f"## {i}. {head}", "",
           f"- URL: {r.url}",
           f"- Path group: {grp} (raw_category {r.raw_category})",
           f"- Title: {r.title}",
           f"- Predicted label: {r.pred_label} ({'easier' if r.easier else 'harder'})",
           f"- Probabilities: A {r.prob_A:.4f}, B1 {r.prob_B1:.4f}, B2 {r.prob_B2:.4f}, "
           f"C {r.prob_C:.4f}",
           f"- Expected CEFR index: {r.E:.4f} ({basis})",
           f"- Words: {r.word_count:,}", "",
           f"> {first_words(r.text_joined)}", ""]
    ex_json.append({"example": head, "url": r.url, "group": grp, "title": r.title,
                    "pred_label": r.pred_label, "E": r.E, "word_count": int(r.word_count),
                    **{f"prob_{l}": getattr(r, f"prob_{l}") for l in CEFR}})
with open(os.path.join(FINAL_DIR, "ch5b_examples.md"), "w") as f:
    f.write("\n".join(md))
print("  md     -> ch5b_examples.md")
results["examples"] = ex_json

# ── Part II: SDGR counts per country (from b1_descriptives.csv) ─────────────


def line(label, col, pct=None):
    return [label] + [fmt_int(b1.loc[k, col]) + (f" ({b1.loc[k, pct]:.1f})" if pct else "")
                      for k in ov_cols]


tex_table(
    os.path.join(TAB_DIR, "ch5b_sdgr_counts.tex"),
    caption="SDGR categorisation by country",
    label="tab:sdgr-counts",
    header=[""] + ov_cols,
    rows=[line("Pages in corpus", "pages_corpus"),
          line("Citizen (A to I)", "citizen_A_I"),
          line("Business (J to Q)", "business_J_Q"),
          line("UNCLEAR", "unclear"),
          line("\\quad of which deterministic rule", "unclear_rule"),
          line("OUT\\_OF\\_SCOPE", "out_of_scope"),
          ["Unclassified (\\%)"] + [f"{100 * b1.loc[k, 'unclassified'] / b1.loc[k, 'pages_corpus']:.1f}"
                                for k in ov_cols],
          line("Regression sample", "regression_sample")],
    midrules=(1, 7),
    notes=[
        "All corpus pages. Unclassified: UNCLEAR plus OUT\\_OF\\_SCOPE as a share of corpus pages "
        "(no page is NO\\_TEXT). The deterministic rule routes the 749 Austrian ``Leben in der "
        "Gemeinde/Stadt'' template pages to UNCLEAR without a model call.",
        "Regression sample: pages with a CEFR label and a citizen or business area. One German "
        "page without body text was assigned area I from its title but has no CEFR label.",
        "Same numbers as Table \\texttt{b1\\_descriptives.tex}.",
    ],
    colspec="lrrrrr",
)

# ── Part II: robustness overview (M3a, from existing coefficient CSVs) ──────
TERMS4 = ["Austria", "Netherlands", "Ireland", "Business"]
sens = pd.read_csv(os.path.join(TAB_DIR, "sens_country_ors.csv"))
sens = sens[sens.model == "M3a"]
b4s = pd.read_csv(os.path.join(TAB_DIR, "b4_sensitivity.csv"))
b4s = b4s.merge(pd.read_csv(os.path.join(TAB_DIR, "b4_model_fit.csv"))[["model", "N"]], on="model",
                how="left", validate="many_to_one")
lenc = pd.read_csv(os.path.join(TAB_DIR, "ch5_length_control.csv"))
rob = pd.read_csv(os.path.join(TAB_DIR, "b4_robustness_ols_ordinal.csv"))
rob = rob[(rob.model == "M3a") & (rob.method == "Ordinal logit")]


specs = [("Main specification (M3a)", sens[sens.spec == "A"], "OR", None),
         ("UNCLEAR as own scope", b4s[b4s.model == "M3a_ii"], "OR", None),
         ("\\quad without template pages", b4s[b4s.model == "M3a_iii"], "OR", None),
         ("S1. Under 50 words dropped", sens[sens.spec == "B1"], "OR", None),
         ("S2. German leistung dropped", sens[sens.spec == "B2"], "OR", None),
         ("S3. Dutch news, documents dropped", sens[sens.spec == "C"], "OR", None),
         ("S4. S1 and S3", sens[sens.spec == "D"], "OR", None),
         ("Length control, + log(words)", lenc[lenc.spec == "L"], "OR", None),
         ("Ordinal logit", rob, "estimate", None)]
ov_rows, ov_tex = [], []
for name, d, est, _ in specs:
    key = "term" if "term" in d.columns and d["term"].isin(TERMS4).any() else "label"
    d = d.set_index(key)
    lo, hi = ("lo", "hi") if est == "estimate" else ("OR_lo", "OR_hi")
    N = int(d["N"].iloc[0])
    row = {"spec": name.replace("\\quad ", "  "), "N": N}
    cells = []
    for t in TERMS4:
        if t not in d.index:
            cells.append("")
            continue
        r = d.loc[t]
        flag = "$^\\dagger$" if bool(r.get("flag_lt30", False)) else ""
        star = "*" if r.p < 0.05 else ""
        dp = 3 if r[hi] < 0.1 else 2     # three decimals where two would round the CI to 0
        cells.append(f"{r[est]:.{dp}f} [{r[lo]:.{dp}f}, {r[hi]:.{dp}f}]{star}{flag}")
        row.update({f"{t}_OR": r[est], f"{t}_lo": r[lo], f"{t}_hi": r[hi], f"{t}_p": r.p})
    ov_rows.append(row)
    ov_tex.append([name, fmt_int(N)] + cells)
robo = pd.DataFrame(ov_rows)
# cross-checks against the printed main and sensitivity results
assert robo.N.tolist() == [7359, 8391, 7642, 5406, 6314, 6614, 4769, 7359, 7359]
assert round(robo.loc[0, "Austria_OR"], 2) == 4.39 and round(robo.loc[4, "Austria_OR"], 2) == 1.09
assert round(robo.loc[7, "Austria_OR"], 2) == 0.54 and round(robo.loc[8, "Austria_OR"], 2) == 6.57
robo.to_csv(os.path.join(TAB_DIR, "ch5b_robustness_overview.csv"), index=False, float_format="%.6g")
print(robo.to_string())
results["robustness_overview"] = robo.to_dict("records")
tex_table(
    os.path.join(TAB_DIR, "ch5b_robustness_overview.tex"),
    caption="Robustness of the country and scope effects (M3a)",
    label="tab:robustness-overview",
    header=["", "N", "Austria", "Netherlands", "Ireland", "Business"],
    rows=ov_tex,
    midrules=(1, 3, 7),
    notes=[
        "Odds ratios [Wald 95\\% CI] of M3a (country + scope; reference Germany, citizen); "
        "* p $<$ 0.05. Outcome: harder (predicted B2 or C) versus easier (predicted A or B1), "
        "except the last row, which is a proportional-odds logit on A $<$ B1 $<$ B2 $<$ C "
        "(cumulative odds ratios; the Brant test rejects proportional odds). No model is "
        "refitted for this table; the values are copied from the full tables.",
        "UNCLEAR rows: Table~\\ref{tab:regression-sensitivity}, columns (ii) and (iii). "
        "S1 to S4: specifications B1, B2, C and D of Table~\\ref{tab:sens-country-ors}. "
        "Length control: Table~\\ref{tab:length-control}; adding log(words) changes the "
        "reference to Germany at equal page length, which the German pages hardly reach. "
        "Ordinal logit: Table~\\ref{tab:regression-robustness}.",
        "Three decimals where the upper confidence limit is below 0.1. "
        "$^\\dagger$ Fewer than 30 pages in one outcome class for the country or for Germany "
        "(Germany keeps 11 easier pages in S1 and S4).",
    ],
    colspec="lrrrrr",
    size=r"\scriptsize\setlength{\tabcolsep}{3pt}",
)

save_json(results, "ch5b_analysis.json")
