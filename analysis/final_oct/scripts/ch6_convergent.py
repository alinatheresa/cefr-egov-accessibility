# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/ch6_convergent.py
# Purpose: Chapter 6 convergent validity refinement. Per language, Spearman rho (with p) of the
#          predicted level (A=1 to C=4) and of the expected CEFR index E with mean sentence length
#          and with the Flesch score, for all pages, for pages with at least 3 sentences, and for
#          German without the de_leistung pages. Reads the existing per-page output of
#          analysis/readability/convergent_validity_flesch.py and the CEFR probabilities; nothing
#          is re-run or recomputed from text.
# Pipeline step: Analysis: convergent validity correlations.
# Thesis: Chapter 6, tab:convergent.

import os

import pandas as pd
from scipy.stats import spearmanr

from common import (PROJECT_ROOT, TAB_DIR, CEFR, load_cefr, save_json, tex_table, tex_escape,
                    fmt_int, fmt_p)

PER_PAGE = os.path.join(PROJECT_ROOT, "analysis", "readability", "readability_per_page.csv")
MIN_SENT = 3  # same threshold as MIN_SENTENCES_SENSITIVITY in convergent_validity_flesch.py

rd = pd.read_csv(PER_PAGE)
cefr = load_cefr()[["country", "raw_category", "url", "pred_label", "E"]]
df = rd.merge(cefr, on=["country", "url"], how="left", suffixes=("", "_cefr"), validate="1:1")

# Checks: every readability page has probabilities, the labels agree, and the per-page file
# covers every labelled page (the one German page without text has no label).
assert df["E"].notna().all(), "readability page without CEFR probabilities"
assert (df["pred_label"] == df["pred_label_cefr"]).all(), "label mismatch"
assert len(df) == len(cefr) == 8997, (len(df), len(cefr))
assert (df["cefr_ordinal"] == df["pred_label"].map({l: i + 1 for i, l in enumerate(CEFR)})).all()
leistung = (df["country"] == "germany") & (df["raw_category"] == "leistung")
assert leistung.sum() == 1049, leistung.sum()
print(f"Joined {len(df)} pages; de_leistung pages: {int(leistung.sum())}")

LANGS = [("german", "German"), ("dutch", "Dutch"), ("english", "English")]
SAMPLES = [("all", "All pages"), ("min3", f"At least {MIN_SENT} sentences"),
           ("no_leistung", "Without de_leistung")]
PREDICTORS = [("cefr_ordinal", "level"), ("E", "E")]
MEASURES = [("mean_sentence_length_words", "sentlen"), ("flesch", "flesch")]

records = []
for lang, lang_label in LANGS:
    d_lang = df[df["language"] == lang]
    for key, sample_label in SAMPLES:
        if key == "all":
            d = d_lang
        elif key == "min3":
            d = d_lang[d_lang["n_sentences"] >= MIN_SENT]
        else:
            if lang != "german":
                continue
            d = d_lang[~leistung.loc[d_lang.index]]
        rec = {"language": lang, "sample": key, "n": len(d),
               "n_germany": int((d["country"] == "germany").sum()),
               "n_leistung": int(leistung.loc[d.index].sum())}
        for pcol, pname in PREDICTORS:
            for mcol, mname in MEASURES:
                rho, p = spearmanr(d[pcol], d[mcol])
                rec[f"rho_{pname}_{mname}"] = float(rho)
                rec[f"p_{pname}_{mname}"] = float(p)
        records.append(rec)
        print(f"{lang_label:<8} {sample_label:<24} n={len(d):5d}  "
              + "  ".join(f"{pn}~{mn}: {rec[f'rho_{pn}_{mn}']:+.3f} (p={rec[f'p_{pn}_{mn}']:.2g})"
                          for _, pn in PREDICTORS for _, mn in MEASURES))

res = pd.DataFrame(records)
res.to_csv(os.path.join(TAB_DIR, "ch6_convergent.csv"), index=False, float_format="%.6g")

# Cross-check: level rho in the full samples equals the existing b5 Spearman table.
b5 = pd.read_csv(os.path.join(TAB_DIR, "b5_fig8_spearman_by_language.csv"))
for _, r in b5.iterrows():
    mname = "sentlen" if r["measure"] == "mean_sentence_length_words" else "flesch"
    mine = res[(res.language == r["language"]) & (res["sample"] == "all")].iloc[0]
    assert abs(mine[f"rho_level_{mname}"] - r["rho"]) < 1e-3, (r["language"], mname)
    assert mine["n"] == r["n"]
print("Check: level rho (all pages) equals tables/b5_fig8_spearman_by_language.csv")

save_json({"records": records, "min_sentences": MIN_SENT,
           "source_per_page": "analysis/readability/readability_per_page.csv"},
          "ch6_convergent.json")


def cell(r, pname, mname):
    return f"{r[f'rho_{pname}_{mname}']:.2f} ({fmt_p(r[f'p_{pname}_{mname}'])})"


rows = []
lang_label = dict(LANGS)
sample_label = dict(SAMPLES)
prev = None
midrules = []
for i, r in res.iterrows():
    if prev is not None and r["language"] != prev:
        midrules.append(i)
    rows.append([lang_label[r["language"]] if r["language"] != prev else "",
                 tex_escape(sample_label[r["sample"]]), fmt_int(r["n"]),
                 cell(r, "level", "sentlen"), cell(r, "level", "flesch"),
                 cell(r, "E", "sentlen"), cell(r, "E", "flesch")])
    prev = r["language"]

g = res.set_index(["language", "sample"])
notes = [
    "Spearman rank correlation with p in parentheses. Level: predicted CEFR level coded A = 1, "
    "B1 = 2, B2 = 3, C = 4. E: expected CEFR index, 1 $\\times$ pA + 2 $\\times$ pB1 + "
    "3 $\\times$ pB2 + 4 $\\times$ pC. Convergence means a positive correlation with sentence "
    "length and a negative one with the Flesch score (higher Flesch = easier).",
    "Sentence length: words per sentence. Flesch variants: Amstad (German), Douma (Dutch), Flesch "
    "(English); syllables use a vowel-cluster heuristic. Per-page values from "
    "\\texttt{convergent\\_validity\\_flesch.py}. German = Germany and Austria; Dutch = "
    "Netherlands; English = Ireland.",
    f"At least {MIN_SENT} sentences: pages with fewer detected sentences removed (German "
    f"{fmt_int(g.loc[('german', 'all'), 'n'] - g.loc[('german', 'min3'), 'n'])}, Dutch "
    f"{fmt_int(g.loc[('dutch', 'all'), 'n'] - g.loc[('dutch', 'min3'), 'n'])}, English "
    f"{fmt_int(g.loc[('english', 'all'), 'n'] - g.loc[('english', 'min3'), 'n'])}); "
    f"{fmt_int(g.loc[('german', 'min3'), 'n_leistung'])} de\\_leistung pages remain. "
    f"Without de\\_leistung: the {fmt_int(leistung.sum())} German service pages (one- to "
    "three-sentence teasers) removed.",
]
tex_table(os.path.join(TAB_DIR, "ch6_convergent.tex"),
          caption="Convergent validity: correlation of the classifier output with sentence length "
                  "and Flesch reading ease",
          label="tab:convergent",
          header=["Language", "Sample", "n", "Level, sentence length", "Level, Flesch",
                  "E, sentence length", "E, Flesch"],
          rows=rows, midrules=tuple(midrules), notes=notes, colspec="llrrrrr",
          size=r"\footnotesize\setlength{\tabcolsep}{4pt}")
