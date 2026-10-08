# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: make_collection_tables.py
# Purpose: rebuild the data collection tables from the shipped step 00 to 02 outputs and check
#          every table row (and the counts quoted in the table notes) against the shipped .tex
#          files. Reads files only; writes nothing. Exits with an error on the first mismatch.
# Pipeline step: Data collection, summary of steps 00 to 02 (all four countries).
# Thesis: Chapter 3 corpus construction funnel (tab:dc_corpus_funnel), rejected URLs in step 01
#         (tab:dc_rejection_reasons), page length (tab:dc_page_length); Annex URL path tables
#         (tab:dc_categories_<country>).

import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CP_DIR     = os.path.join(REPO_ROOT, "country_pipelines")
TAB_DIR    = os.path.join(REPO_ROOT, "analysis", "final_oct", "tables")

COUNTRIES = ["germany", "austria", "netherlands", "ireland"]
LABEL = {"germany": "Germany", "austria": "Austria", "netherlands": "Netherlands",
         "ireland": "Ireland"}
LANG = {"germany": "DE", "austria": "DE", "netherlands": "NL", "ireland": "EN"}

n_checked = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def fint(n):
    return f"{int(n):,}"


def f1(x):
    """One decimal, rounded half up; whole numbers without decimals."""
    d = Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return str(int(d)) if d == d.to_integral_value() else str(d)


def tex(s):
    return s.replace("_", r"\_")


def tex_rows(name):
    """Data rows of a shipped table: lines between \\begin{tabular} and \\end{tabular} that end
    in \\\\, without the header row."""
    lines = open(os.path.join(TAB_DIR, name), encoding="utf-8").read().splitlines()
    i, j = lines.index(next(l for l in lines if l.startswith(r"\begin{tabular}"))), \
        lines.index(r"\end{tabular}")
    rows = [l for l in lines[i + 1:j] if l.endswith(r"\\")]
    return rows[1:]


def tex_text(name):
    return open(os.path.join(TAB_DIR, name), encoding="utf-8").read()


def check(name, built):
    global n_checked
    shipped = tex_rows(name)
    if sorted(built) != sorted(shipped):
        print(f"MISMATCH in {name}")
        for r in sorted(set(built) - set(shipped)):
            print("  rebuilt only:", r)
        for r in sorted(set(shipped) - set(built)):
            print("  shipped only:", r)
        sys.exit(1)
    n_checked += len(built)
    print(f"  {name}: {len(built)} rows match")


def check_note(name, text):
    global n_checked
    if text not in tex_text(name):
        print(f"MISMATCH in {name}: note text not found: {text!r}")
        sys.exit(1)
    n_checked += 1


def summary_blocks(path):
    """Step 00/01 summary files hold several CSV blocks separated by blank lines."""
    blocks, cur = [], []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.strip():
            cur.append(line.split(","))
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


# ── Load the shipped outputs ──────────────────────────────────────────────────

disc, disc_total, sitemap_rows = {}, {}, {}
clean, rejects, corpus = {}, {}, {}
for c in COUNTRIES:
    sub = os.path.join("categories", "") if c == "ireland" else ""
    b = summary_blocks(os.path.join(CP_DIR, c, f"00_discover_output_{c}", sub,
                                    f"{c}_summary_overview.csv"))
    sitemap_rows[c] = b[0][1:]
    rows = [r for blk in b[1:] for r in blk if r[0] not in ("Language",)]
    disc[c] = {r[1]: int(r[2]) for r in rows if not r[0].startswith("TOTAL")}
    disc_total[c] = {r[0]: int(r[2]) for r in rows if r[0].startswith("TOTAL")}

    b = summary_blocks(os.path.join(CP_DIR, c, f"01_clean_output_{c}", f"{c}_summary_overview.csv"))
    head = b[0][0]
    clean[c] = pd.DataFrame(b[0][1:], columns=head)
    for col in head[2:]:
        clean[c][col] = clean[c][col].astype(int)

    rejects[c] = pd.read_csv(os.path.join(CP_DIR, c, f"01_clean_output_{c}", LANG[c],
                                          f"rejects_{LANG[c].lower()}_validation.csv"))
    corpus[c] = pd.read_csv(os.path.join(CP_DIR, c, f"02_scrape_output_{c}", f"combined_{c}.csv"))

cefr_n = sum(len(pd.read_csv(os.path.join(CP_DIR, c, f"03_cefr_output_{c}", f"{c}_cefr.csv"))
                 .query("pred_label in ['A', 'B1', 'B2', 'C']")) for c in COUNTRIES)

# Consistency between the step 01 summary and the step 01 reject files
for c in COUNTRIES:
    assert clean[c]["Rejected"].sum() == len(rejects[c]), c
    assert (clean[c]["Kept_Valid"] + clean[c]["Rejected"] == clean[c]["Total_Candidates"]).all(), c
    assert corpus[c]["url"].is_unique, c


# ── Corpus construction funnel ────────────────────────────────────────────────

print("Corpus construction funnel")
lang_total = {"germany": "TOTAL_DE", "austria": "TOTAL_DE", "netherlands": "TOTAL_NL",
              "ireland": "TOTAL_EN"}
rows, tot = [], [0] * 7
for c in COUNTRIES:
    discovered = disc_total[c][lang_total[c]]
    validated = clean[c]["Total_Candidates"].sum()
    rejected = clean[c]["Rejected"].sum()
    kept = clean[c]["Kept_Valid"].sum()
    ls = clean[c]["With_Leichte_Sprache"].sum() if "With_Leichte_Sprache" in clean[c] else 0
    final = len(corpus[c])
    assert kept + ls == final, c
    vals = [discovered, discovered - validated, validated, rejected, kept, ls, final]
    tot = [a + b for a, b in zip(tot, vals)]
    rows.append(" & ".join([LABEL[c]] + [fint(v) for v in vals]) + r" \\")
rows.append(" & ".join(["Total"] + [fint(v) for v in tot]) + r" \\")
check("dc_corpus_funnel.tex", rows)

empty = corpus["germany"].query("word_count == 0")
assert len(empty) == 1
check_note("dc_corpus_funnel.tex",
           f"Germany's sitemaps also list {fint(disc_total['germany']['TOTAL_EN'])} English URLs")
check_note("dc_corpus_funnel.tex",
           f"Ireland's sitemap lists {fint(disc_total['ireland']['TOTAL_GA'])} Irish-language URLs "
           f"and {fint(int(sitemap_rows['ireland'][0][1]))} entries in total")
check_note("dc_corpus_funnel.tex",
           f"One German page ({empty['url'].iloc[0].split('/de/')[1]}) is in the final corpus")
check_note("dc_corpus_funnel.tex", f"the analysed corpus has {fint(cefr_n)} pages")
assert all((corpus[c]["word_count"] >= 0).all() for c in COUNTRIES)


# ── Rejected URLs in step 01 ──────────────────────────────────────────────────

print("Rejected URLs in step 01")
REASONS = [("Region selector on page", "region_selector"),
           ("No sentence ending in . ! ?", "no_text"),
           ("Redirect to services listing", "services_listing"),
           ("Error template (regex)", "error_template"),
           ("HTTP 404", "404"),
           ("Request error or timeout", "get_error")]
counts = {c: rejects[c]["reason"].astype(str).value_counts() for c in COUNTRIES}
all_codes = set().union(*[set(v.index) for v in counts.values()])
assert all_codes <= {code for _, code in REASONS}, all_codes
rows = []
for label, code in REASONS:
    vals = [int(counts[c].get(code, 0)) for c in COUNTRIES]
    rows.append(" & ".join([label, rf"\texttt{{{tex(code)}}}"] + [fint(v) for v in vals + [sum(vals)]])
                + r" \\")
vals = [len(rejects[c]) for c in COUNTRIES]
rows.append(" & ".join(["Total rejected", ""] + [fint(v) for v in vals + [sum(vals)]]) + r" \\")
check("dc_rejection_reasons.tex", rows)
check_note("dc_rejection_reasons.tex",
           f"All {['zero', 'one', 'two', 'three', 'four', 'five'][int(counts['netherlands'].get('error_template', 0))]}"
           " Dutch cases are question-and-answer pages")
assert rejects["netherlands"].query("reason == 'error_template'")["url"]\
    .str.contains("/vraag-en-antwoord/").all()


# ── Page length ───────────────────────────────────────────────────────────────

print("Page length")


def length_row(label, d):
    w, s = d["word_count"], d["sentence_count"]
    q = lambda x: f"{f1(x.median())} ({f1(x.quantile(0.25))} to {f1(x.quantile(0.75))})"
    # percentages keep one decimal (10.0), except a full 100
    pct = lambda m: "100" if m.all() else str(Decimal(str(100 * m.mean()))
                                              .quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return " & ".join([label, fint(len(d)), q(w), q(s), f1(w.sum() / s.sum()),
                       pct(s < 3), pct(w < 50)]) + r" \\"


de = corpus["germany"]
rows = [length_row("Germany", de),
        length_row(r"\quad without Leichte Sprache", de[de.category != "hilfs-problemloesungsdienste_ls"]),
        length_row(r"\quad of which leistung", de[de.category == "leistung"])]
rows += [length_row(LABEL[c], corpus[c]) for c in ("austria", "netherlands", "ireland")]
check("dc_page_length.tex", rows)


# ── Annex URL path tables ─────────────────────────────────────────────────────

print("Annex URL path tables")
for c in COUNTRIES:
    lang = LANG[c].lower()
    cl = clean[c].copy()
    # step 01 names categories <lang>_<segment> (Ireland: without the prefix)
    cl["cat"] = cl["Category"].where(cl["Category"].str.startswith(f"{lang}_"), f"{lang}_" + cl["Category"])
    cl = cl.set_index("cat")
    final = corpus[c]["category"].value_counts()
    rows = []
    for cat, n in disc[c].items():
        if not cat.startswith(f"{lang}_"):
            continue
        if cat in cl.index:
            k = cl.loc[cat]
            seg = cat[len(lang) + 1:]
            fin = final.get(seg, final.get(cat, 0))
            assert fin == k["Kept_Valid"], (c, cat)
            assert n == k["Total_Candidates"], (c, cat)
            rows.append(" & ".join([tex(cat), fint(n), "included", fint(k["Kept_Valid"]),
                                    fint(k["Rejected"]), fint(fin)]) + r" \\")
        else:
            rows.append(" & ".join([tex(cat), fint(n), "excluded", "", "", ""]) + r" \\")
    if c == "germany":
        n_ls = int(final.get("hilfs-problemloesungsdienste_ls", 0))
        rows.append(" & ".join([r"hilfs-problemloesungsdienste\_ls (Leichte Sprache)", "",
                                "added in 01", "", "", fint(n_ls)]) + r" \\")
    rows.append(" & ".join(["Total", fint(sum(v for k, v in disc[c].items() if k.startswith(f"{lang}_"))),
                            "", fint(cl["Kept_Valid"].sum()), fint(cl["Rejected"].sum()),
                            fint(len(corpus[c]))]) + r" \\")
    check(f"dc_categories_{c}.tex", rows)

# Counts quoted in the notes of the Annex tables
de_reg = int(rejects["germany"].query("category == 'de_leistung' and reason == 'region_selector'").shape[0])
check_note("dc_categories_germany.tex",
           f"Of the {fint(clean['germany'].set_index('Category').loc['de_leistung', 'Rejected'])} rejected "
           f"leistung pages, {fint(de_reg)} were rejected because")
for k, v in disc["germany"].items():
    if k.startswith("en_"):
        check_note("dc_categories_germany.tex", f"{tex(k)} {fint(v)}")
at_excl = disc_total["austria"]["TOTAL_DE"] - clean["austria"]["Total_Candidates"].sum()
check_note("dc_categories_austria.tex", f"Excluded: {fint(at_excl)} URLs.")
ie_filt = disc_total["ireland"]["TOTAL_EN"] - clean["ireland"]["Total_Candidates"].sum()
check_note("dc_categories_ireland.tex", f"Path-filtered: {fint(ie_filt)} URLs.")
ie_rej = rejects["ireland"]
ie_nt = ie_rej[ie_rej.reason == "no_text"]
ie_index = ie_nt["url"].str.rstrip("/").str.split("/").str[-1].isin(["campaigns", "policies", "services"])
check_note("dc_categories_ireland.tex",
           f"Of the {fint(len(ie_rej))} rejected URLs, "
           f"{fint((ie_rej.reason == 'services_listing').sum())} redirected to an organisation's services "
           f"listing and {fint(ie_index.sum())} of the {fint(len(ie_nt))} no-text rejections are category "
           f"index pages")
check_note("dc_categories_ireland.tex",
           f"so {fint(disc['ireland']['ga_other'])} of them fall into ga\\_other")

# Composition of the Dutch corpus by the third path segment
nl = corpus["netherlands"]["url"].str.replace("https://www.rijksoverheid.nl", "", regex=False)
parts = nl.str.strip("/").str.split("/")
third = parts.str[2].fillna("")
kind = pd.Series("other", index=nl.index)
kind[parts.str.len() == 2] = "landing"
kind[third == "vraag-en-antwoord"] = "qa"
kind[third == "nieuws"] = "news"
kind[third == "documenten"] = "documents"
k = kind.value_counts()
check_note("dc_categories_netherlands.tex",
           f"{fint(k['qa'])} vraag-en-antwoord (question and answer), {fint(k['news'])} nieuws (news), "
           f"{fint(k['documents'])} documenten (documents), {fint(k['landing'])} topic landing pages and "
           f"{fint(k['other'])} other subpages")

print(f"\nAll checks passed: {n_checked} table rows and note counts reproduce the shipped .tex files.")
