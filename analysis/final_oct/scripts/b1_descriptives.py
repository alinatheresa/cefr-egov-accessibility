# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b1_descriptives.py
# Purpose: B1 descriptive counts per country and total (corpus, CEFR labels, SDGR scope,
#          UNCLEAR incl. deterministic rule, OUT_OF_SCOPE, NO_TEXT, CEFR distribution).
# Pipeline step: Analysis: descriptive counts.
# Thesis: Chapter 5 corpus and SDGR counts (b1_descriptives.csv, the source of
#         tab:sdgr-counts and tab:corpus-overview in ch5b_analysis.py).

import os
import pandas as pd

from common import (COUNTRIES, COUNTRY_LABEL, CORPUS_SIZE, CEFR, TAB_DIR, load_cefr,
                    load_sdg, scope_of, tex_table, fmt_int, save_json)

cefr_all = load_cefr(include_unlabelled=True)
sdg = load_sdg()

df = cefr_all.merge(sdg[["url", "assigned_area", "rule"]], on="url", how="left",
                    validate="one_to_one")
assert df["assigned_area"].notna().all(), "page without SDG label"
df["labelled"] = df["pred_label"].isin(CEFR)
df["scope"] = df["assigned_area"].map(scope_of)

rows = []
for c in COUNTRIES + ["total"]:
    d = df if c == "total" else df[df["country"] == c]
    lab = d[d["labelled"]]
    n = len(d)
    r = {
        "country": "Total" if c == "total" else COUNTRY_LABEL[c],
        "pages_corpus": n,
        "pages_cefr_labelled": len(lab),
        "citizen_A_I": int((d["scope"] == "citizen").sum()),
        "business_J_Q": int((d["scope"] == "business").sum()),
        "unclear": int((d["assigned_area"] == "UNCLEAR").sum()),
        "unclear_rule": int(((d["assigned_area"] == "UNCLEAR") & d["rule"]).sum()),
        "out_of_scope": int((d["assigned_area"] == "OUT_OF_SCOPE").sum()),
        "no_text": int((d["assigned_area"] == "NO_TEXT").sum()),
    }
    r["unclassified"] = r["unclear"] + r["out_of_scope"] + r["no_text"]
    r["pct_unclassified"] = 100 * r["unclassified"] / n
    for lvl in CEFR:
        r[f"n_{lvl}"] = int((lab["pred_label"] == lvl).sum())
        r[f"pct_{lvl}"] = 100 * r[f"n_{lvl}"] / len(lab)
    r["n_A_B1"] = r["n_A"] + r["n_B1"]
    r["n_B2_C"] = r["n_B2"] + r["n_C"]
    r["pct_A_B1"] = 100 * r["n_A_B1"] / len(lab)
    r["pct_B2_C"] = 100 * r["n_B2_C"] / len(lab)
    # pages entering the regression (CEFR label and citizen or business scope)
    r["regression_sample"] = int((lab["scope"].notna()).sum())
    rows.append(r)

out = pd.DataFrame(rows)
assert out.loc[out.country == "Total", "pages_corpus"].item() == sum(CORPUS_SIZE.values())
out.to_csv(os.path.join(TAB_DIR, "b1_descriptives.csv"), index=False, float_format="%.2f")
print(out.T.to_string())

# The one unlabelled page
unl = df[~df["labelled"]][["country", "url", "assigned_area"]]
save_json({"unlabelled_pages": unl.to_dict(orient="records")}, "b1_unlabelled_pages.json")
print(unl.to_string())

# LaTeX: countries as columns
cols = [r["country"] for r in rows]
def line(label, key, pct_key=None, dec=1):
    cells = [label]
    for r in rows:
        if pct_key:
            cells.append(f"{fmt_int(r[key])} ({r[pct_key]:.{dec}f})")
        else:
            cells.append(fmt_int(r[key]))
    return cells

body = [
    line("Pages in corpus", "pages_corpus"),
    line("Pages with CEFR label", "pages_cefr_labelled"),
    line("Citizen (A to I)", "citizen_A_I"),
    line("Business (J to Q)", "business_J_Q"),
    line("UNCLEAR", "unclear"),
    line(r"\quad of which deterministic rule", "unclear_rule"),
    line("OUT\\_OF\\_SCOPE", "out_of_scope"),
    line("NO\\_TEXT", "no_text"),
    ["Unclassified (\\%)"] + [f"{r['pct_unclassified']:.1f}" for r in rows],
    line("Regression sample", "regression_sample"),
    line("A", "n_A", "pct_A"),
    line("B1", "n_B1", "pct_B1"),
    line("B2", "n_B2", "pct_B2"),
    line("C", "n_C", "pct_C"),
    line("Easier (A or B1)", "n_A_B1", "pct_A_B1"),
    line("Harder (B2 or C)", "n_B2_C", "pct_B2_C"),
]
tex_table(
    os.path.join(TAB_DIR, "b1_descriptives.tex"),
    caption="Corpus, SDGR categorisation and predicted CEFR levels by country",
    label="tab:descriptives",
    header=[""] + cols,
    rows=body,
    midrules=(2, 10),
    notes=[
        "SDGR rows count all corpus pages. CEFR rows count pages with a CEFR label; "
        "percentages in parentheses are shares of labelled pages.",
        "One German page has no body text and therefore no CEFR label; the categoriser "
        "still assigned it area I from its title, so NO\\_TEXT is zero for every country.",
        "The deterministic rule routes Austrian municipal overview pages "
        "(``Leben in der Gemeinde'') to UNCLEAR without a model call.",
        "Unclassified is UNCLEAR plus OUT\\_OF\\_SCOPE plus NO\\_TEXT as a share of corpus pages. "
        "The regression sample contains labelled pages with a citizen or business area.",
        "CEFR labels: classifier run cefr\\_20260817\\_231925.",
    ],
    colspec="lrrrrr",
    size=r"\small",
)
