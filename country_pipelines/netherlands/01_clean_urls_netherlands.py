# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 01_clean_urls_netherlands.py
# Purpose: filter and deduplicate discovered URLs from the Dutch government portal
# Pipeline step: Data collection, step 01 (path filter and URL validation), Netherlands.
# Thesis: Chapter 3: kept and rejected URLs (tab:dc_corpus_funnel, tab:dc_rejection_reasons);
#         Annex URL path table (tab:dc_categories_netherlands).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Step 00 input root and NL folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_ROOT = os.path.join(SCRIPT_DIR, "00_discover_output_netherlands")
IN_NL = os.path.join(IN_ROOT, "NL")

# Step 01 output root and NL folder
OUT_ROOT = os.path.join(SCRIPT_DIR, "01_clean_output_netherlands")
OUT_NL = os.path.join(OUT_ROOT, "NL")

# Files from Step 00 to read
IN_ALL_NL = os.path.join(IN_NL, "netherlands_all_sitemap_urls_NL.csv")

# Files written by this script
OUT_SUMMARY = os.path.join(OUT_ROOT, "netherlands_summary_overview.csv")
OUT_ALL_NL_CLEAN = os.path.join(OUT_NL, "netherlands_all_sitemap_urls_NL_clean.csv")
OUT_REJECTS = os.path.join(OUT_NL, "rejects_nl_validation.csv")
OUT_CHART_NL = os.path.join(OUT_NL, "netherlands_category_counts_NL.png")

# Only citizen-facing topic content — mirrors other countries' exclusion of
# organizational/news/directory pages (AT: de_orgsearch, PT: pt_entidades/pt_noticias).
ALLOWED_NL_CATEGORIES = {"nl_onderwerpen"}


def out_cat_path(cat: str) -> str:
    """Return the output path for a cleaned category CSV."""
    return os.path.join(OUT_NL, f"netherlands_urls_{cat}_clean.csv")


# Networking parameters
UA = "Mozilla/5.0 (compatible; NL-URLCleaner/1.0; research use)"
TIMEOUT = 12
DELAY = 0.25  # polite pacing

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# -------------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------------

ERROR_PATTERNS = [
    re.compile(r"\b404\b"),
    re.compile(r"pagina niet gevonden", re.IGNORECASE),
    re.compile(r"page not found", re.IGNORECASE),
    re.compile(r"niet gevonden", re.IGNORECASE),
]

# Selectors used to remove boilerplate and navigation elements
NON_CONTENT_SELECTORS = [
    "nav", "header", "footer", "aside",
    ".breadcrumb", ".toc",
    ".cookie", ".cookie-banner",
]

ADDRESS_LIKE = re.compile(
    r"(\bTel\.|\bTelefon\b|\bFax\b|@|www\.|http|straat\b|str\.\b|\bpostcode\b|\bPostbus\b|\bE-mail\b)",
    flags=re.IGNORECASE,
)

DIGIT_HEAVY = re.compile(r"(?:\d[\d\s./-]{5,})")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _looks_like_error_html(text: str) -> bool:
    """Identify typical error templates in HTML content."""
    snippet = text[:4000]
    return any(rx.search(snippet) for rx in ERROR_PATTERNS)


def _is_html(resp: requests.Response) -> bool:
    """Check whether the HTTP response is HTML."""
    ctype = resp.headers.get("Content-Type", "")
    return "text/html" in ctype.lower()


def _extract_clean_text_for_count(html: str) -> str:
    """
    Remove navigation and boilerplate elements and keep potential prose
    blocks for a CEFR-style word count.
    rijksoverheid.nl uses standard HTML semantics (main / article).
    """
    soup = BeautifulSoup(html, "html.parser")

    main = (
        soup.find("main")
        or soup.find("article")
        or soup
    )

    for sel in NON_CONTENT_SELECTORS:
        for el in main.select(sel):
            el.decompose()

    # Drop links
    for a in main.find_all("a"):
        a.decompose()

    blocks = []
    # Paragraph-like content: p / h2 / h3 / h4 / li
    for tag in main.find_all(["p", "h2", "h3", "h4", "li"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            blocks.append(txt)

    text = "\n".join(blocks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def _to_full_sentences(text: str) -> list[str]:
    """
    Apply simple heuristics to keep only full sentences suitable for
    downstream CEFR analysis. Sentences must end in sentence-final
    punctuation and must not match typical address or contact patterns.
    """
    kept = []
    for block in text.split("\n"):
        for s in SENT_SPLIT.split(block):
            s = s.strip()
            if not s:
                continue
            if not re.search(r"[.!?]$", s):
                continue
            if ADDRESS_LIKE.search(s):
                continue
            kept.append(s)
    return kept


def _cefr_word_count(html: str) -> int:
    """
    Extract visible content, filter to full sentences, and return the
    total number of words in those sentences.
    """
    cleaned = _extract_clean_text_for_count(html)
    sentences = _to_full_sentences(cleaned)
    page_text = " ".join(sentences)
    words = len(page_text.split())
    return words


def check_url(url: str) -> tuple[bool, str]:
    """
    Return (is_valid, reason). A URL is considered valid if it returns a 2xx
    HTML response and contains at least one CEFR-style sentence (word count > 0).
    """
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=TIMEOUT)
    except Exception:
        return False, "head_error"

    if r.status_code == 404:
        return False, "404"
    if r.status_code >= 400:
        return False, f"http_{r.status_code}"

    # Fetch full content for HTML and CEFR checks
    try:
        g = SESSION.get(url, allow_redirects=True, timeout=TIMEOUT)
    except Exception:
        return False, "get_error"

    if g.status_code >= 400:
        return False, f"http_{g.status_code}"
    if not _is_html(g):
        return False, "non_html"
    if _looks_like_error_html(g.text):
        return False, "error_template"

    # CEFR-style content check
    words = _cefr_word_count(g.text)
    if words == 0:
        return False, "no_text"

    return True, "ok"


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def ensure_dirs():
    """Create the output directory for Dutch-language content."""
    os.makedirs(OUT_NL, exist_ok=True)


def read_rows(csv_path: str) -> list[dict]:
    """
    Read source, URL, and lastmod fields from a Step 00 CSV file.
    Step 00 used 'source' (crawl label) instead of 'sitemap'.
    """
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "source": (row.get("source") or "").strip(),
                "url": (row.get("url") or "").strip(),
                "lastmod": (row.get("lastmod") or "").strip(),
            })
    return rows


def write_rows(csv_path: str, rows: list[dict]) -> None:
    """Write source, URL, and lastmod fields to a CSV file."""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "url", "lastmod"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------------------------------------------------------
# Chart writer (NL only)
# -------------------------------------------------------------------------

def write_nl_chart(counts_dict: dict[str, int], out_path: str, title: str) -> None:
    """Visualise the distribution of cleaned URLs across NL categories."""
    if not counts_dict:
        return
    plt.figure(figsize=(12, 6))
    labels = list(counts_dict.keys())
    vals = list(counts_dict.values())
    plt.bar(labels, vals)
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of URLs")
    mv = max(vals) if vals else 0
    for i, v in enumerate(vals):
        plt.text(i, v + (mv * 0.01 if mv else 1), str(v), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved {out_path}")


# -------------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------------

def main(selected: list[str] | None, delay: float):
    """
    Validate responsiveness and CEFR-style content of NL URLs in the
    selected categories and write cleaned files, a merged NL list, and a
    summary table.
    """
    ensure_dirs()

    # Discover all category CSV files written by Step 00
    all_csv_paths = sorted(glob.glob(os.path.join(IN_NL, "netherlands_urls_nl_*.csv")))
    cat_to_path: dict[str, str] = {}
    for path in all_csv_paths:
        basename = os.path.basename(path)
        # Skip merged and already-cleaned files
        if basename.startswith("netherlands_all_") or basename.endswith("_clean.csv"):
            continue
        # netherlands_urls_{cat}.csv  →  cat
        cat = basename[len("netherlands_urls_"):-len(".csv")]
        cat_to_path[cat] = path

    if selected:
        scope = [c for c in selected if c in cat_to_path and c in ALLOWED_NL_CATEGORIES]
    else:
        scope = [c for c in cat_to_path if c in ALLOWED_NL_CATEGORIES]

    if not scope:
        print("No category CSV files found in", IN_NL)
        return

    # Read category CSVs from Step 00
    cat_rows: dict[str, list[dict]] = {}
    for cat in scope:
        path = cat_to_path[cat]
        rows = read_rows(path)
        rows = sorted(rows, key=lambda r: (r["source"], r["url"]))  # deterministic
        cat_rows[cat] = rows
        print(f"[input] {cat}: {len(rows)} candidates from {os.path.basename(path)}")

    # Validation loop
    rejects_f = open(OUT_REJECTS, "w", encoding="utf-8", newline="")
    rej_writer = csv.DictWriter(rejects_f, fieldnames=["category", "url", "reason"])
    rej_writer.writeheader()

    cleaned_per_cat: dict[str, list[dict]] = {c: [] for c in scope}
    kept_urls_set: set[str] = set()

    for cat in scope:
        kept = 0
        rejected = 0
        for i, row in enumerate(cat_rows[cat], start=1):
            url = row["url"]
            ok, reason = check_url(url)
            if ok:
                cleaned_per_cat[cat].append(row)
                kept_urls_set.add(url)
                kept += 1
            else:
                rej_writer.writerow({"category": cat, "url": url, "reason": reason})
                rejected += 1
            if i % 50 == 0:
                print(f"[progress] {cat}: {i}/{len(cat_rows[cat])}")
            time.sleep(delay)
        print(f"[validated] {cat}: kept {kept}, rejected {rejected}")

    rejects_f.close()

    # Write cleaned category CSVs
    for cat in scope:
        out_path = out_cat_path(cat)
        write_rows(out_path, cleaned_per_cat[cat])
        print(f"[output] {cat}: {len(cleaned_per_cat[cat])} → {os.path.basename(out_path)}")

    # Build cleaned merged NL list from Step 00 all-NL by intersecting kept URLs
    all_nl_rows = read_rows(IN_ALL_NL)
    cleaned_all_nl = [r for r in all_nl_rows if r["url"] in kept_urls_set]
    cleaned_all_nl = sorted(cleaned_all_nl, key=lambda r: (r["source"], r["url"]))
    write_rows(OUT_ALL_NL_CLEAN, cleaned_all_nl)
    print(f"[output] merged NL: {len(cleaned_all_nl)} → {os.path.basename(OUT_ALL_NL_CLEAN)}")

    # Prepare NL-only cleaned counts for chart and summary
    counts_nl: dict[str, int] = {c: len(cleaned_per_cat[c]) for c in scope}

    # Chart for cleaned NL categories
    write_nl_chart(
        counts_nl,
        OUT_CHART_NL,
        "URL counts per NL category (cleaned; rijksoverheid.nl)"
    )

    # Summary: categories only, including rejects
    total_nl_clean = sum(counts_nl.values())
    with open(OUT_SUMMARY, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Language", "Category", "Kept_Valid", "Rejected", "Total_Candidates"])
        for cat in sorted(scope):
            kept = len(cleaned_per_cat[cat])
            total = len(cat_rows[cat])
            rej = total - kept
            w.writerow(["NL", cat, kept, rej, total])
        w.writerow([])
        w.writerow(["TOTAL_NL", "", total_nl_clean, "", sum(len(cat_rows[c]) for c in scope)])

    print("=== Cleaning complete ===")
    print(f"Outputs in: {OUT_NL}")
    print(f"Summary written to: {OUT_SUMMARY}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate and clean NL URLs from Step 00 for selected categories and write cleaned NL outputs."
    )
    ap.add_argument(
        "--categories",
        nargs="*",
        help="Optional subset of categories to clean. Defaults to all predefined NL categories."
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=DELAY,
        help="Delay between requests in seconds (default 0.25)."
    )
    args = ap.parse_args()
    main(args.categories, args.delay)
