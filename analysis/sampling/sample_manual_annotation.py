# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: sample_manual_annotation.py
# Purpose: draw a stratified random sample of pages from the 4-country corpus (DE/AT/NL/IE)
#          for manual SDGR Annex I area annotation, plus a codebook of the label set.
# Pipeline step: Draws the stratified 1% manual annotation sample (90 pages, seed 42) and writes
#                the SDGR codebook (needs the page texts).
# Thesis: Section 3.6.3 (manual annotation used to validate the SDGR labels).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import math
import csv

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Portugal is excluded -- this thesis's classifier and country re-run now cover DE/AT/NL/IE only.
COUNTRIES = ["germany", "austria", "netherlands", "ireland"]
SEED      = 42
FRACTION  = 0.01

OUT_SAMPLE_CSV = os.path.join(SCRIPT_DIR, "manual_annotation_sample.csv")
OUT_CODEBOOK   = os.path.join(SCRIPT_DIR, "sdgr_codebook.md")

SDG_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "analysis", "sdg", "sdg_categorize_full.py")


def combined_csv(country: str) -> str:
    return os.path.join(
        PROJECT_ROOT, "country_pipelines", country,
        f"02_scrape_output_{country}", f"combined_{country}.csv",
    )


# ─────────────────────────────────────────────────────────────────────
# Codebook: parsed straight out of sdg_categorize_full.py's SYSTEM_PROMPT so the
# codebook can never drift from what the zero-shot classifier actually uses.
# ─────────────────────────────────────────────────────────────────────

def load_sdg_module():
    sys.path.insert(0, os.path.dirname(SDG_SCRIPT_PATH))
    import sdg_categorize_full as sdg  # noqa: E402  (import after sys.path tweak, by design)
    return sdg


def parse_area_descriptions(system_prompt: str) -> dict:
    """Full per-area descriptions live only in the prompt's 'Available areas:' block
    (the AREAS dict itself only has short labels). Backslash line continuations in the
    source collapse each area's text onto one logical line at runtime, so each area is
    exactly one line in this block, starting with 'X -- '."""
    marker = "Available areas:\n"
    idx = system_prompt.index(marker) + len(marker)
    block = system_prompt[idx:]
    descriptions = {}
    for line in block.split("\n"):
        m = re.match(r"^([A-Q]) -- (.+)$", line.strip())
        if m:
            descriptions[m.group(1)] = m.group(2).strip()
    return descriptions


def parse_bullet(system_prompt: str, key: str) -> str:
    """Extract the one-line '- KEY: ...' bullet definition for OUT_OF_SCOPE / UNCLEAR."""
    m = re.search(rf"^- {key}: (.+)$", system_prompt, re.MULTILINE)
    return m.group(1).strip() if m else ""


def write_codebook(sdg) -> None:
    area_desc = parse_area_descriptions(sdg.SYSTEM_PROMPT)
    oos_def   = parse_bullet(sdg.SYSTEM_PROMPT, "OUT_OF_SCOPE")
    unc_def   = parse_bullet(sdg.SYSTEM_PROMPT, "UNCLEAR")

    lines = [
        "# SDGR Annex I area codebook",
        "",
        "Codes for the `annotation` column in `manual_annotation_sample.csv`. Definitions are",
        f"reproduced verbatim from `{os.path.relpath(SDG_SCRIPT_PATH, PROJECT_ROOT)}` "
        "(the zero-shot classifier's prompt) so the manual annotation guide always matches",
        "what the automated classifier is actually instructed to do.",
        "",
        "| Code | Label | Description |",
        "|---|---|---|",
    ]
    for code in sdg.AREAS:  # dict preserves A..Q insertion order
        label = sdg.AREAS[code]
        desc  = area_desc.get(code, "")
        lines.append(f"| {code} | {label} | {desc} |")
    lines.append(f"| UNCLEAR | — | {unc_def} |")
    lines.append(f"| OUT_OF_SCOPE | — | {oos_def} |")
    lines.append("")

    with open(OUT_CODEBOOK, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Codebook -> {OUT_CODEBOOK}  ({len(sdg.AREAS)} areas + UNCLEAR + OUT_OF_SCOPE)")


# ─────────────────────────────────────────────────────────────────────
# Proportional-to-size allocation, largest-remainder method so per-country
# counts sum exactly to n_target (plain floor/round would not guarantee that).
# ─────────────────────────────────────────────────────────────────────

def allocate_largest_remainder(sizes: dict, n_target: int) -> dict:
    total = sum(sizes.values())
    raw = {c: n_target * n / total for c, n in sizes.items()}
    floor_alloc = {c: math.floor(v) for c, v in raw.items()}
    remainder = n_target - sum(floor_alloc.values())
    # give the leftover units to the countries with the largest fractional remainder
    order = sorted(sizes, key=lambda c: raw[c] - floor_alloc[c], reverse=True)
    for c in order[:remainder]:
        floor_alloc[c] += 1
    return floor_alloc


def main():
    print("=== Manual annotation sample (SDGR Annex I areas, DE/AT/NL/IE) ===")

    dfs = {}
    for c in COUNTRIES:
        path = combined_csv(c)
        df = pd.read_csv(path, dtype=str).fillna("")
        dfs[c] = df
        print(f"  {c:<12}: {len(df):5d} pages  ({path})")

    sizes = {c: len(df) for c, df in dfs.items()}
    total = sum(sizes.values())
    n_target = round(total * FRACTION)
    print(f"\n  Total corpus (4 countries): {total}")
    print(f"  Sample fraction: {FRACTION*100:.0f}%  ->  n = round({total} x {FRACTION}) = {n_target}")

    alloc = allocate_largest_remainder(sizes, n_target)
    print("\n  Per-country allocation (proportional to size, largest-remainder rounding):")
    for c in COUNTRIES:
        print(f"    {c:<12}: {alloc[c]:3d}  ({100*alloc[c]/n_target:.1f}% of sample, "
              f"{100*sizes[c]/total:.1f}% of corpus)")
    assert sum(alloc.values()) == n_target, "allocation must sum exactly to n_target"

    sampled_frames = []
    for c in COUNTRIES:
        s = dfs[c].sample(n=alloc[c], random_state=SEED)
        sampled_frames.append(s.assign(country=c))
    sample = pd.concat(sampled_frames, ignore_index=True)

    out = pd.DataFrame({
        "url":        sample["url"],
        "country":    sample["country"],
        "title":      sample["title"],
        "text":       sample["text_joined"],
        "annotation": "",
    })
    # Shuffle across countries so annotation order doesn't cue the country/language
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out.to_csv(OUT_SAMPLE_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n  Sample -> {OUT_SAMPLE_CSV}  ({len(out)} rows, columns: {list(out.columns)})")

    n_empty = (out["text"].str.strip() == "").sum()
    if n_empty:
        print(f"  WARNING: {n_empty} sampled rows have empty text (unannotatable) -- check the source corpus.")

    sdg = load_sdg_module()
    write_codebook(sdg)

    print("\nDone. Fill in the 'annotation' column with one of the codebook codes for each row,")
    print("then run eval_zeroshot_areas.py against the annotated file.")


if __name__ == "__main__":
    main()
