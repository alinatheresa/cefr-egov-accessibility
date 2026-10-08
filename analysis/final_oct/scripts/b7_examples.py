# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b7_examples.py
# Purpose: B7 examples.md. (1) In the SDGR area with the most pages in both Germany and
#          Austria (at least 30 each), the page closest to each country's median expected
#          CEFR index within that area. (2) The same across all areas. (3) Three pages the
#          author annotated as unclear that the zero-shot categoriser placed in a citizen area,
#          with the categoriser's reasoning (first three in annotation-sample order).
# Pipeline step: Analysis: selects the example pages closest to the median E (needs the page texts,
#                which are not shipped). Writes examples.md (page text, not shipped) and
#                results/b7_examples.json (URLs and E values).
# Thesis: Section 6.1 example pages.

import os
import pandas as pd

from common import (FINAL_DIR, AREA_NAMES, COUNTRY_LABEL, ANNOTATION_CSV, load_cefr, load_text,
                    load_sdg, scope_of, save_json)

cefr = load_cefr()
text = load_text()
sdg = load_sdg()
df = (cefr.merge(sdg[["url", "assigned_area", "assigned_reasoning"]], on="url", validate="one_to_one")
          .merge(text[["url", "title", "text_joined"]], on="url", validate="one_to_one"))


def first_words(t, n=150):
    w = str(t).split()
    return " ".join(w[:n]) + (" [...]" if len(w) > n else "")


def closest_to_median(d):
    med = d["E"].median()
    i = (d["E"] - med).abs().sort_values(kind="mergesort").index[0]
    return d.loc[i], med


def block(r, med, scope_note):
    return "\n".join([
        f"- URL: {r.url}",
        f"- Title: {r.title}",
        f"- SDGR area: {r.assigned_area} ({AREA_NAMES.get(r.assigned_area, r.assigned_area)})",
        f"- Predicted label: {r.pred_label}",
        f"- Probabilities: A {r.prob_A:.4f}, B1 {r.prob_B1:.4f}, B2 {r.prob_B2:.4f}, C {r.prob_C:.4f}",
        f"- Expected CEFR index: {r.E:.4f} ({scope_note} median {med:.4f})",
        "",
        f"> {first_words(r.text_joined)}",
        "",
    ])


lines = ["# Examples for the discussion", "",
         "Source: classifier run cefr_20260817_231925 (country_pipelines/*/03_cefr_output_*), SDGR areas "
         "from analysis/sdg/sdg_categorization_full.csv, page text from "
         "country_pipelines/*/02_scrape_output_*/combined_*.csv. Expected CEFR index = "
         "1 x pA + 2 x pB1 + 3 x pB2 + 4 x pC. Text is the first 150 words of the scraped body.", ""]
out = {}

# (1) one area
dea = df[df.country.isin(["germany", "austria"])]
counts = pd.crosstab(dea.assigned_area, dea.country)
counts = counts[[a in AREA_NAMES for a in counts.index]]
elig = counts[(counts.germany >= 30) & (counts.austria >= 30)]
area = elig.min(axis=1).sort_values(ascending=False, kind="mergesort").index[0]
lines += [f"## 1. Within one SDGR area: {area} ({AREA_NAMES[area]})", "",
          f"Eligible areas (at least 30 pages in both Germany and Austria): "
          + ", ".join(f"{a} (DE {counts.loc[a, 'germany']}, AT {counts.loc[a, 'austria']})"
                      for a in elig.index) + ". "
          f"Chosen: the area with the largest smaller count, {area}.", ""]
out["area_choice"] = {"area": area, "eligible": elig.to_dict(orient="index")}
for c in ["germany", "austria"]:
    d = df[(df.country == c) & (df.assigned_area == area)]
    r, med = closest_to_median(d)
    lines += [f"### {COUNTRY_LABEL[c]} (n = {len(d)} pages in area {area})", "",
              block(r, med, f"{COUNTRY_LABEL[c]} area {area}")]
    out[f"area_{c}"] = {"url": r.url, "E": r.E, "median": med}

# (2) across all areas
lines += ["## 2. Across all areas", ""]
for c in ["germany", "austria"]:
    d = df[df.country == c]
    r, med = closest_to_median(d)
    lines += [f"### {COUNTRY_LABEL[c]} (n = {len(d)} labelled pages)", "",
              block(r, med, f"{COUNTRY_LABEL[c]} all pages")]
    out[f"all_{c}"] = {"url": r.url, "E": r.E, "median": med}

# (3) manual unclear -> zero-shot citizen
ann = pd.read_csv(ANNOTATION_CSV, sep=None, engine="python", dtype=str).fillna("")
ann = ann.merge(sdg[["url", "assigned_area", "assigned_reasoning"]], on="url", how="left")
cand = ann[(ann.annotation.str.strip() == "unclear") & (ann.assigned_area.map(scope_of) == "citizen")]
lines += ["## 3. Pages annotated as unclear that the zero-shot categoriser placed in a citizen area", "",
          f"{len(cand)} of the 34 pages annotated unclear were placed in a citizen area. The first three "
          "in annotation-sample order are shown, with the reasoning stored in the corpus labels.", ""]
for i, r in enumerate(cand.head(3).itertuples(), 1):
    lines += [f"### Example {i} ({COUNTRY_LABEL[r.country]})", "",
              f"- URL: {r.url}", f"- Title: {r.title}",
              f"- Zero-shot area: {r.assigned_area} ({AREA_NAMES[r.assigned_area]})",
              f"- Zero-shot reasoning: \"{r.assigned_reasoning}\"", "",
              f"> {first_words(r.text)}", ""]
out["unclear_to_citizen"] = cand[["url", "country", "assigned_area"]].to_dict(orient="records")

txt = "\n".join(lines)
for bad in [chr(0x2013), chr(0x2014)]:
    if bad in txt:
        print(f"note: scraped page text contains dash character {bad!r}; left as quoted source text")
with open(os.path.join(FINAL_DIR, "examples.md"), "w") as f:
    f.write(txt)
save_json(out, "b7_examples.json")
print(out)
