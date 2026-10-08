# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: convergent_validity_flesch.py
# Purpose: convergent validity check — compare the fine-tuned CEFR classifier's labels
#          against a classical readability formula (language-appropriate Flesch variant)
#          on the same pages, per the Discussion's "include it and then argue" framing:
#          the two measures operationalise readability differently (ordinal CEFR judgement
#          vs. sentence-length/syllable-count arithmetic), so partial disagreement is the
#          expected finding, not a bug.
# Pipeline step: Readability measures per page (sentence length, syllables, Flesch variant per
#                language); writes readability_per_page.csv (needs the page texts).
# Thesis: Section 6.4.3 convergent validity, via b5_readability_and_copies.py and
#         ch6_convergent.py.
#
# Standalone script. Does not modify or import any other script in this repo.
#
# Syllable counting: no hyphenation-dictionary library (e.g. pyphen) is installed in this
# environment, so syllables are counted with a standard vowel-cluster heuristic (count
# maximal runs of consecutive vowel letters per word), the common approach when a proper
# per-language hyphenation dictionary isn't available. This is an approximation, not exact
# syllabification -- most under/over-counting affects diphthongs and diaeresis marks (see
# count_syllables_word() docstring). For higher precision, `pip install pyphen` (supports
# de/nl/en hyphenation dictionaries) and swap in pyphen.Pyphen(lang=...).inserted(word).count("-").

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUT_BOXPLOT_FLESCH = os.path.join(SCRIPT_DIR, "boxplot_flesch_by_cefr.png")
OUT_BOXPLOT_SENTLEN = os.path.join(SCRIPT_DIR, "boxplot_sentlen_by_cefr.png")
OUT_SUMMARY_CSV     = os.path.join(SCRIPT_DIR, "readability_summary_by_level.csv")
OUT_SPEARMAN_CSV    = os.path.join(SCRIPT_DIR, "readability_spearman_by_language.csv")
OUT_PER_PAGE_CSV    = os.path.join(SCRIPT_DIR, "readability_per_page.csv")  # bonus, not requested but cheap/useful

# New outputs (this revision): per-country Flesch facet + >=3-sentence sensitivity check
OUT_BOXPLOT_FLESCH_COUNTRY        = os.path.join(SCRIPT_DIR, "boxplot_flesch_by_cefr_country.png")
OUT_BOXPLOT_FLESCH_SENS           = os.path.join(SCRIPT_DIR, "boxplot_flesch_by_cefr_sensitivity.png")
OUT_BOXPLOT_FLESCH_COUNTRY_SENS   = os.path.join(SCRIPT_DIR, "boxplot_flesch_by_cefr_country_sensitivity.png")
OUT_BOXPLOT_SENTLEN_SENS          = os.path.join(SCRIPT_DIR, "boxplot_sentlen_by_cefr_sensitivity.png")
OUT_SPEARMAN_COUNTRY_CSV          = os.path.join(SCRIPT_DIR, "readability_spearman_by_country.csv")
OUT_SENSITIVITY_COMPARISON_CSV    = os.path.join(SCRIPT_DIR, "readability_sensitivity_comparison.csv")
MIN_SENTENCES_SENSITIVITY = 3  # sensitivity check excludes pages with fewer than this many detected sentences

# Country -> Flesch-formula note, for figure titles (Germany/Austria both use the German
# Amstad variant, since the formula is chosen by language, not by country)
COUNTRY_FORMULA_NOTE = {
    "germany": "German: Amstad", "austria": "German: Amstad",
    "netherlands": "Dutch: Douma", "ireland": "English: standard Flesch",
}

# Country order requested throughout: Germany, Austria, Netherlands, Ireland
COUNTRIES = ["germany", "austria", "netherlands", "ireland"]
COUNTRY_LABELS = {"germany": "Germany", "austria": "Austria",
                  "netherlands": "Netherlands", "ireland": "Ireland"}

# Language order requested throughout: German, Dutch, English
COUNTRY_TO_LANG = {"germany": "german", "austria": "german",
                   "netherlands": "dutch", "ireland": "english"}
LANG_ORDER   = ["german", "dutch", "english"]
LANG_LABELS  = {"german": "German", "dutch": "Dutch", "english": "English"}

CEFR_LEVELS  = ["A", "B1", "B2", "C"]
CEFR_ORDINAL = {"A": 1, "B1": 2, "B2": 3, "C": 4}  # matches CEFR_1234 in build_dataframe.py

# Repo-standard CEFR colour ramp (matches figures_and_tables.py / make_thesis_figures.py)
CEFR_COLORS = {"A": "#C0D5F1", "B1": "#6E9BD8", "B2": "#3F6FC4", "C": "#1B3985"}
# Paul Tol colourblind-safe qualitative palette for the 4 countries (new mapping — this
# script introduces the only country-colour legend in the repo; chosen to be visually
# distinct from both the CEFR blue ramp above and the citizen/business teal/amber scheme
# used in analysis/regression/figures_and_tables.py, so none of the three collide).
COUNTRY_COLORS = {"germany": "#4477AA", "austria": "#CCBB44",
                  "netherlands": "#EE6677", "ireland": "#66CCEE"}

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":       10,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype":    42,
})


def combined_csv(country: str) -> str:
    return os.path.join(PROJECT_ROOT, "country_pipelines", country,
                        f"02_scrape_output_{country}", f"combined_{country}.csv")


def cefr_csv(country: str) -> str:
    return os.path.join(PROJECT_ROOT, "country_pipelines", country,
                        f"03_cefr_output_{country}", f"{country}_cefr.csv")


# ─────────────────────────────────────────────────────────────────────
# Tokenisation (self-contained: sentences, words, syllables all derived
# from the same tokenisation so ASL/ASW stay internally consistent)
# ─────────────────────────────────────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")   # same convention already used in this
                                                  # repo's country_pipelines chunk_text()
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # Unicode letters only, no digits/underscore

def tokenize_sentences(text: str) -> list:
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts


def tokenize_words(text: str) -> list:
    return _WORD_RE.findall(text.lower())


# Vowel sets per language. Diphthongs (e.g. German "ei", "au"; Dutch "ij", "eu") are
# handled correctly by construction, since consecutive vowel letters form one contiguous
# cluster = one syllable. Known limitation: diaeresis marks in Dutch (e.g. "ï" in "ruïne")
# signal a syllable BREAK, not a vowel merge, but this heuristic still treats them as part
# of one contiguous cluster and under-counts such cases -- a standard limitation of
# vowel-cluster heuristics without a hyphenation dictionary.
_VOWELS = {
    "german":  set("aeiouyäöü"),
    "dutch":   set("aeiouyäëïöü"),
    "english": set("aeiouy"),
}
_VOWEL_RUN_RE_CACHE = {}

def _vowel_run_re(lang: str):
    if lang not in _VOWEL_RUN_RE_CACHE:
        chars = "".join(sorted(_VOWELS[lang]))
        _VOWEL_RUN_RE_CACHE[lang] = re.compile(f"[{re.escape(chars)}]+")
    return _VOWEL_RUN_RE_CACHE[lang]


def count_syllables_word(word: str, lang: str) -> int:
    """Vowel-cluster heuristic syllable count for one lowercase word (see module docstring
    for the accuracy caveat). English gets the classic silent-trailing-e adjustment
    (e.g. "state" -> 1, not 2); German/Dutch do not, since a word-final unstressed <e> is
    pronounced (schwa) in both languages, unlike English."""
    w = word.strip().lower()
    if not w:
        return 0
    n = len(_vowel_run_re(lang).findall(w))
    if lang == "english" and w.endswith("e") and not w.endswith("le") and n > 1:
        n -= 1
    return max(1, n)


# ─────────────────────────────────────────────────────────────────────
# Flesch Reading Ease — language-specific variants
# ASL = average sentence length (words/sentence); ASW = average syllables/word.
# ─────────────────────────────────────────────────────────────────────

def flesch_score(asl: float, asw: float, lang: str) -> float:
    if lang == "english":
        # Flesch (1948) — standard Flesch Reading Ease
        return 206.835 - 1.015 * asl - 84.6 * asw
    if lang == "german":
        # Amstad (1978) adaptation for German
        return 180.0 - 1.0 * asl - 58.5 * asw
    if lang == "dutch":
        # Douma (1960) adaptation for Dutch
        return 206.84 - 0.93 * asl - 77.0 * asw
    raise ValueError(f"No Flesch variant defined for language: {lang!r}")


def process_page_text(text: str, lang: str):
    """Returns (n_sentences, n_words, n_syllables, asl, asw, flesch) or None if the text
    has no usable sentences/words."""
    sentences = tokenize_sentences(text)
    if not sentences:
        return None
    words = tokenize_words(text)
    if not words:
        return None
    n_sentences = len(sentences)
    n_words     = len(words)
    n_syllables = sum(count_syllables_word(w, lang) for w in words)
    asl = n_words / n_sentences
    asw = n_syllables / n_words
    fre = flesch_score(asl, asw, lang)
    return n_sentences, n_words, n_syllables, asl, asw, fre


# ─────────────────────────────────────────────────────────────────────
# Load + join
# ─────────────────────────────────────────────────────────────────────

def load_and_join() -> pd.DataFrame:
    frames = []
    print("=== Loading + joining scraped text (Step 02) with classifier output (Step 03) ===")
    for c in COUNTRIES:
        text_path = combined_csv(c)
        clf_path  = cefr_csv(c)
        text_df = pd.read_csv(text_path, dtype=str)
        clf_df  = pd.read_csv(clf_path, dtype=str)
        print(f"\n  {COUNTRY_LABELS[c]}:")
        print(f"    combined_{c}.csv (scraped text): {len(text_df)} rows")
        print(f"    {c}_cefr.csv (classifier output): {len(clf_df)} rows")

        merged = text_df.merge(clf_df[["url", "pred_label"]], on="url", how="inner")
        n_text_unmatched = len(text_df) - text_df["url"].isin(clf_df["url"]).sum()
        n_clf_unmatched  = len(clf_df) - clf_df["url"].isin(text_df["url"]).sum()
        print(f"    after inner join on url: {len(merged)} rows matched "
              f"({n_text_unmatched} text rows had no classifier match, "
              f"{n_clf_unmatched} classifier rows had no text match)")

        merged = merged[merged["pred_label"].isin(CEFR_LEVELS)].copy()
        print(f"    after filtering to valid CEFR labels (A/B1/B2/C): {len(merged)} rows")

        merged["country"]  = c
        merged["language"] = COUNTRY_TO_LANG[c]
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  TOTAL across 4 countries after join + label filtering: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────
# Figure drawing (factored out so the new per-country facet and the
# sensitivity variants render with identical styling to the originals —
# same code path, just given different facet slices of the same 'flesch'/
# 'mean_sentence_length_words' values already computed in main()).
# ─────────────────────────────────────────────────────────────────────

def draw_flesch_figure(facet_frames, out_path, suptitle) -> None:
    """facet_frames: ordered list of (facet_label, sub_df) pairs, one per facet."""
    n = len(facet_frames)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (label, sub) in zip(axes, facet_frames):
        data = [sub.loc[sub["pred_label"] == lvl, "flesch"].values for lvl in CEFR_LEVELS]
        bp = ax.boxplot(data, positions=range(len(CEFR_LEVELS)), widths=0.6,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="#111111", linewidth=1.4),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#999999",
                                        markeredgecolor="none", alpha=0.5))
        for patch, lvl in zip(bp["boxes"], CEFR_LEVELS):
            patch.set_facecolor(CEFR_COLORS[lvl])
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.8)
        ns = [int((sub["pred_label"] == lvl).sum()) for lvl in CEFR_LEVELS]
        ax.set_xticks(range(len(CEFR_LEVELS)))
        ax.set_xticklabels([f"{lvl}\n(n={n_})" for lvl, n_ in zip(CEFR_LEVELS, ns)])
        ax.set_title(label, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#e8e8e8", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Flesch Reading Ease\n(language-appropriate variant)")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def draw_sentlen_figure(df: pd.DataFrame, out_path, suptitle) -> None:
    """Panel A = pooled across languages/countries. Panel B = per country, country order
    Germany/Austria/Netherlands/Ireland, shared y-axis (comparable units)."""
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.15)

    axA = fig.add_subplot(gs[0, 0])
    data_pooled = [df.loc[df["pred_label"] == lvl, "mean_sentence_length_words"].values
                  for lvl in CEFR_LEVELS]
    bpA = axA.boxplot(data_pooled, positions=range(len(CEFR_LEVELS)), widths=0.6,
                      patch_artist=True, showfliers=True,
                      medianprops=dict(color="#111111", linewidth=1.4),
                      flierprops=dict(marker="o", markersize=3, markerfacecolor="#999999",
                                      markeredgecolor="none", alpha=0.5))
    for patch, lvl in zip(bpA["boxes"], CEFR_LEVELS):
        patch.set_facecolor(CEFR_COLORS[lvl])
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    ns_pooled = [int((df["pred_label"] == lvl).sum()) for lvl in CEFR_LEVELS]
    axA.set_xticks(range(len(CEFR_LEVELS)))
    axA.set_xticklabels([f"{lvl}\n(n={n})" for lvl, n in zip(CEFR_LEVELS, ns_pooled)])
    axA.set_ylabel("Mean sentence length (words)")
    axA.set_title("Pooled (all countries)", fontsize=11)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.yaxis.grid(True, color="#e8e8e8", lw=0.5, zorder=0)
    axA.set_axisbelow(True)
    shared_ylim = axA.get_ylim()

    gsB = gs[0, 1].subgridspec(1, 4, wspace=0.08)
    for i, c in enumerate(COUNTRIES):
        axB = fig.add_subplot(gsB[0, i], sharey=axA)
        sub = df[df["country"] == c]
        data_c = [sub.loc[sub["pred_label"] == lvl, "mean_sentence_length_words"].values
                 for lvl in CEFR_LEVELS]
        bpB = axB.boxplot(data_c, positions=range(len(CEFR_LEVELS)), widths=0.6,
                          patch_artist=True, showfliers=True,
                          medianprops=dict(color="#111111", linewidth=1.2),
                          flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#999999",
                                          markeredgecolor="none", alpha=0.5))
        for patch in bpB["boxes"]:
            patch.set_facecolor(COUNTRY_COLORS[c])
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.85)
        ns_c = [int((sub["pred_label"] == lvl).sum()) for lvl in CEFR_LEVELS]
        axB.set_xticks(range(len(CEFR_LEVELS)))
        axB.set_xticklabels([f"{lvl}\n(n={n})" for lvl, n in zip(CEFR_LEVELS, ns_c)], fontsize=8)
        axB.set_title(COUNTRY_LABELS[c], fontsize=10)
        axB.spines["top"].set_visible(False)
        axB.spines["right"].set_visible(False)
        if i > 0:
            axB.tick_params(axis="y", labelleft=False)
            axB.spines["left"].set_visible(False)
        axB.yaxis.grid(True, color="#e8e8e8", lw=0.5, zorder=0)
        axB.set_axisbelow(True)
    axA.set_ylim(shared_ylim)

    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convergent validity: CEFR classifier vs. language-appropriate Flesch "
                    "Reading Ease (Amstad/Douma/standard) on the 4-country portal corpus."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N pages per country (for a quick test run).")
    args = parser.parse_args()

    df = load_and_join()

    if args.limit:
        df = pd.concat([g.head(args.limit) for _, g in df.groupby("country")], ignore_index=True)
        print(f"\n  --limit {args.limit}: reduced to {len(df)} rows total (test run)")

    print("\n=== Computing sentence/word/syllable stats and Flesch scores per page ===")
    print("  (self-contained tokenisation: sentences via .!? split, words via Unicode-letter "
          "regex, syllables via vowel-cluster heuristic — see module docstring for caveats)")

    records = []
    n_empty = 0
    for row in df.itertuples(index=False):
        text = getattr(row, "text_joined", "") or ""
        result = process_page_text(text, row.language)
        if result is None:
            n_empty += 1
            continue
        n_sentences, n_words, n_syllables, asl, asw, fre = result
        records.append({
            "country": row.country, "language": row.language, "url": row.url,
            "pred_label": row.pred_label, "cefr_ordinal": CEFR_ORDINAL[row.pred_label],
            "n_sentences": n_sentences, "n_words": n_words, "n_syllables": n_syllables,
            "mean_sentence_length_words": asl, "mean_syllables_per_word": asw,
            "flesch": fre,
        })
    print(f"  Dropped {n_empty} rows with no usable sentences/words (empty or unparseable text)")
    print(f"  Final analysable pages: {len(records)}")

    out = pd.DataFrame(records)
    out["country"]  = pd.Categorical(out["country"], categories=COUNTRIES, ordered=True)
    out["language"] = pd.Categorical(out["language"], categories=LANG_ORDER, ordered=True)
    out["pred_label"] = pd.Categorical(out["pred_label"], categories=CEFR_LEVELS, ordered=True)

    out.to_csv(OUT_PER_PAGE_CSV, index=False)
    print(f"\n  Per-page measures (bonus output, not explicitly requested) -> {OUT_PER_PAGE_CSV}")

    # ── N per CEFR level per language ────────────────────────────────────────
    print("\n=== N per CEFR level per language ===")
    n_table = pd.crosstab(out["language"], out["pred_label"]).reindex(
        index=LANG_ORDER, columns=CEFR_LEVELS)
    print(n_table.to_string())

    # ── Median + IQR of each measure per language x CEFR level ──────────────
    print("\n=== Median and IQR (Q1-Q3) of Flesch score and mean sentence length, "
          "per language x CEFR level ===")

    def iqr_stats(s: pd.Series) -> dict:
        return {"median": s.median(), "q1": s.quantile(0.25), "q3": s.quantile(0.75), "n": len(s)}

    summary_rows = []
    for lang in LANG_ORDER:
        for lvl in CEFR_LEVELS:
            sub = out[(out["language"] == lang) & (out["pred_label"] == lvl)]
            if sub.empty:
                continue
            f_stats = iqr_stats(sub["flesch"])
            s_stats = iqr_stats(sub["mean_sentence_length_words"])
            summary_rows.append({
                "language": LANG_LABELS[lang], "cefr_level": lvl, "n": f_stats["n"],
                "flesch_median": round(f_stats["median"], 2),
                "flesch_q1": round(f_stats["q1"], 2), "flesch_q3": round(f_stats["q3"], 2),
                "sentlen_median": round(s_stats["median"], 2),
                "sentlen_q1": round(s_stats["q1"], 2), "sentlen_q3": round(s_stats["q3"], 2),
            })
            print(f"  {LANG_LABELS[lang]:<8} {lvl:<3}  n={f_stats['n']:4d}  "
                  f"Flesch median={f_stats['median']:6.1f} (IQR {f_stats['q1']:.1f}-{f_stats['q3']:.1f})  "
                  f"SentLen median={s_stats['median']:5.1f} (IQR {s_stats['q1']:.1f}-{s_stats['q3']:.1f})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\n  Summary table -> {OUT_SUMMARY_CSV}")

    # ── Spearman correlation: readability measure vs. ordinal CEFR, per language ──
    print("\n=== Spearman correlation: readability measure vs. ordinal CEFR coding "
          "(A=1, B1=2, B2=3, C=4), within each language ===")
    corr_rows = []
    for lang in LANG_ORDER:
        sub = out[out["language"] == lang]
        for measure_col, measure_name in [("flesch", "Flesch"),
                                          ("mean_sentence_length_words", "Mean sentence length")]:
            rho, pval = spearmanr(sub["cefr_ordinal"], sub[measure_col])
            corr_rows.append({
                "language": LANG_LABELS[lang], "measure": measure_name,
                "n": len(sub), "spearman_rho": round(rho, 4), "p_value": pval,
            })
            print(f"  {LANG_LABELS[lang]:<8} {measure_name:<22} n={len(sub):4d}  "
                  f"rho={rho:+.4f}  p={pval:.4g}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_SPEARMAN_CSV, index=False)
    print(f"\n  Spearman correlation table -> {OUT_SPEARMAN_CSV}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1 — boxplot_flesch_by_cefr.png: Flesch by CEFR level, faceted by
    # language (variants are not comparable across languages, hence facets +
    # independent y-scales rather than one pooled panel). UNCHANGED from the
    # previous revision -- same data, same code path (now via draw_flesch_figure).
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Figure 1: boxplot_flesch_by_cefr.png ===")
    lang_facets = [(LANG_LABELS[lang], out[out["language"] == lang]) for lang in LANG_ORDER]
    draw_flesch_figure(
        lang_facets, OUT_BOXPLOT_FLESCH,
        "Flesch Reading Ease by CEFR level, per language "
        "(German: Amstad · Dutch: Douma · English: standard Flesch)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Figure 2 — boxplot_sentlen_by_cefr.png: mean sentence length by CEFR
    # level. Panel A = pooled across languages/countries. Panel B = per
    # country (comparable units, so shared y-axis across facets is fine).
    # UNCHANGED from the previous revision.
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Figure 2: boxplot_sentlen_by_cefr.png ===")
    draw_sentlen_figure(out, OUT_BOXPLOT_SENTLEN,
                        "Mean sentence length by CEFR level — pooled and per country")

    # ══════════════════════════════════════════════════════════════════════
    # NEW Figure — boxplot_flesch_by_cefr_country.png: same Flesch values as
    # Figure 1, faceted by COUNTRY instead of language (Germany/Austria split
    # out of the pooled "German" facet). Country order: Germany, Austria,
    # Netherlands, Ireland. No new computation -- same per-page 'flesch'
    # column, different grouping for display.
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== New figure: boxplot_flesch_by_cefr_country.png ===")
    country_facets = [
        (f"{COUNTRY_LABELS[c]}\n({COUNTRY_FORMULA_NOTE[c]})", out[out["country"] == c])
        for c in COUNTRIES
    ]
    draw_flesch_figure(
        country_facets, OUT_BOXPLOT_FLESCH_COUNTRY,
        "Flesch Reading Ease by CEFR level, per country "
        "(Germany/Austria: Amstad · Netherlands: Douma · Ireland: standard Flesch)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Sensitivity check: exclude pages with < 3 detected sentences (very
    # short pages, where "mean sentence length" is just the whole-page word
    # count and can be an unrepresentative outlier). No new measures -- same
    # 'flesch' / 'mean_sentence_length_words' columns, filtered subset only.
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n=== Sensitivity check: excluding pages with < {MIN_SENTENCES_SENSITIVITY} "
          f"detected sentences ===")
    out_sens = out[out["n_sentences"] >= MIN_SENTENCES_SENSITIVITY].copy()
    n_dropped_total = len(out) - len(out_sens)
    print(f"  Dropped {n_dropped_total} / {len(out)} pages overall "
          f"({n_dropped_total / len(out) * 100:.1f}%); {len(out_sens)} remain")
    print("\n  Dropped per country:")
    dropped_per_country = {}
    for c in COUNTRIES:
        sub_all = out[out["country"] == c]
        n_drop_c = int((sub_all["n_sentences"] < MIN_SENTENCES_SENSITIVITY).sum())
        dropped_per_country[c] = n_drop_c
        print(f"    {COUNTRY_LABELS[c]:<12} {n_drop_c:5d} / {len(sub_all):5d} dropped "
              f"({n_drop_c / len(sub_all) * 100:5.1f}%)")

    # ── Per-country Spearman correlations, both measures, FULL data (also
    #    answers "print the per-country Spearman correlations for both
    #    measures") ──────────────────────────────────────────────────────
    print(f"\n=== Spearman correlation vs. ordinal CEFR, per COUNTRY "
          f"(Germany/Austria split), full data ===")
    corr_country_rows = []
    for c in COUNTRIES:
        sub = out[out["country"] == c]
        for measure_col, measure_name in [("flesch", "Flesch"),
                                          ("mean_sentence_length_words", "Mean sentence length")]:
            rho, pval = spearmanr(sub["cefr_ordinal"], sub[measure_col])
            corr_country_rows.append({
                "country": COUNTRY_LABELS[c], "measure": measure_name,
                "n": len(sub), "spearman_rho": round(rho, 4), "p_value": pval,
            })
            print(f"  {COUNTRY_LABELS[c]:<12} {measure_name:<22} n={len(sub):4d}  "
                  f"rho={rho:+.4f}  p={pval:.4g}")
    corr_country_df = pd.DataFrame(corr_country_rows)
    corr_country_df.to_csv(OUT_SPEARMAN_COUNTRY_CSV, index=False)
    print(f"\n  Per-country Spearman correlation table -> {OUT_SPEARMAN_COUNTRY_CSV}")

    # ── Comparison table: Spearman correlations with vs. without the
    #    sensitivity filter, per country, both measures ──────────────────
    print(f"\n=== Sensitivity comparison: Spearman correlations with vs. without pages "
          f"< {MIN_SENTENCES_SENSITIVITY} sentences, per country ===")
    sens_rows = []
    for c in COUNTRIES:
        sub_all  = out[out["country"] == c]
        sub_sens = out_sens[out_sens["country"] == c]
        for measure_col, measure_name in [("flesch", "Flesch"),
                                          ("mean_sentence_length_words", "Mean sentence length")]:
            rho_all, p_all = spearmanr(sub_all["cefr_ordinal"], sub_all[measure_col])
            if len(sub_sens) >= 2 and sub_sens[measure_col].nunique() > 1:
                rho_sens, p_sens = spearmanr(sub_sens["cefr_ordinal"], sub_sens[measure_col])
            else:
                rho_sens, p_sens = float("nan"), float("nan")
            sens_rows.append({
                "country": COUNTRY_LABELS[c], "measure": measure_name,
                "n_all": len(sub_all), "rho_all": round(rho_all, 4), "p_all": p_all,
                "n_filtered": len(sub_sens), "rho_filtered": round(rho_sens, 4) if rho_sens == rho_sens else np.nan,
                "p_filtered": p_sens, "n_dropped": dropped_per_country[c],
            })
            print(f"  {COUNTRY_LABELS[c]:<12} {measure_name:<22} "
                  f"rho_all={rho_all:+.4f} (n={len(sub_all)})   "
                  f"rho_filtered={rho_sens:+.4f} (n={len(sub_sens)})   "
                  f"dropped={dropped_per_country[c]}")

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_SENSITIVITY_COMPARISON_CSV, index=False)
    print(f"\n  Sensitivity comparison table -> {OUT_SENSITIVITY_COMPARISON_CSV}")

    # ══════════════════════════════════════════════════════════════════════
    # Sensitivity-filtered versions of each figure (existing two figures are
    # NOT modified -- these are new, separately-named files).
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Sensitivity-filtered figures ===")
    lang_facets_sens = [(LANG_LABELS[lang], out_sens[out_sens["language"] == lang])
                        for lang in LANG_ORDER]
    draw_flesch_figure(
        lang_facets_sens, OUT_BOXPLOT_FLESCH_SENS,
        f"Flesch Reading Ease by CEFR level, per language — sensitivity: "
        f"pages with ≥{MIN_SENTENCES_SENSITIVITY} detected sentences only "
        f"(German: Amstad · Dutch: Douma · English: standard Flesch)",
    )
    country_facets_sens = [
        (f"{COUNTRY_LABELS[c]}\n({COUNTRY_FORMULA_NOTE[c]})", out_sens[out_sens["country"] == c])
        for c in COUNTRIES
    ]
    draw_flesch_figure(
        country_facets_sens, OUT_BOXPLOT_FLESCH_COUNTRY_SENS,
        f"Flesch Reading Ease by CEFR level, per country — sensitivity: "
        f"pages with ≥{MIN_SENTENCES_SENSITIVITY} detected sentences only "
        f"(Germany/Austria: Amstad · Netherlands: Douma · Ireland: standard Flesch)",
    )
    draw_sentlen_figure(
        out_sens, OUT_BOXPLOT_SENTLEN_SENS,
        f"Mean sentence length by CEFR level — pooled and per country — sensitivity: "
        f"pages with ≥{MIN_SENTENCES_SENSITIVITY} detected sentences only",
    )

    print("\n=== Done ===")
    print("Expect heavily overlapping boxes and modest |rho| here -- Flesch (syllable/")
    print("sentence-length arithmetic) and the CEFR classifier (learned holistic judgement)")
    print("are different operationalisations of readability, so partial disagreement is the")
    print("expected convergent-validity finding, not an error.")


if __name__ == "__main__":
    main()
