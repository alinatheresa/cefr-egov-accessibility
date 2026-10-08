# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 01_clean_urls_ireland.py
# Purpose: filter and deduplicate discovered URLs from the Irish government portal
# Pipeline step: Data collection, step 01 (path filter and URL validation), Ireland.
# Thesis: Chapter 3: kept and rejected URLs (tab:dc_corpus_funnel, tab:dc_rejection_reasons);
#         Annex URL path table (tab:dc_categories_ireland).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
import re
import time
from typing import Dict, List
from urllib.parse import urlparse, parse_qs  # URL-based pattern detection

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Step 00 input root and EN folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_ROOT = os.path.join(SCRIPT_DIR, "00_discover_output_ireland")
IN_EN = os.path.join(IN_ROOT, "categories", "EN")

# Step 01 output root and EN folder
OUT_ROOT = os.path.join(SCRIPT_DIR, "01_clean_output_ireland")
OUT_EN = os.path.join(OUT_ROOT, "EN")

# Files from Step 00 to read
IN_ALL_EN = os.path.join(IN_EN, "ireland_all_sitemap_urls_EN.csv")

# Files written by this script
OUT_SUMMARY = os.path.join(OUT_ROOT, "ireland_summary_overview.csv")
OUT_ALL_EN_CLEAN = os.path.join(OUT_EN, "ireland_all_sitemap_urls_EN_clean.csv")
OUT_REJECTS = os.path.join(OUT_EN, "rejects_en_validation.csv")
OUT_CHART_EN = os.path.join(OUT_EN, "ireland_category_counts_EN.png")

# Restrict processing to these EN categories
ALLOWED_EN_CATEGORIES = {"campaigns", "policies", "services"}

# Networking parameters
UA = "Mozilla/5.0 (compatible; IE-URLCleaner/1.0; research use)"
TIMEOUT = 12
DELAY = 0.25  # polite pacing

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# -------------------------------------------------------------------------
# Validation helpers (HTTP / HTML)
# -------------------------------------------------------------------------

ERROR_PATTERNS = [
    re.compile(r"\b404\b"),
    re.compile(r"page not found", re.IGNORECASE),
    re.compile(r"not found", re.IGNORECASE),
]

SOFT_404_STRINGS = [
    "the page you are looking for does not exist",
    "we cannot find the page you are looking for",
]

def _looks_like_error_html(text: str) -> bool:
    """Identify typical error or soft-404 templates in HTML content."""
    snippet = text[:4000]
    if any(rx.search(snippet) for rx in ERROR_PATTERNS):
        return True
    return any(s.lower() in snippet.lower() for s in SOFT_404_STRINGS)

def _is_govie_404_url(final_url: str) -> bool:
    """Detect explicit 404-style URLs after redirects on gov.ie."""
    u = (final_url or "").lower().split("?")[0].rstrip("/")
    return u.endswith("/404") or u.endswith("/page-not-found")

def _is_services_listing(url: str) -> bool:
    """
    Detect gov.ie EN services search or listing pages with organisation
    filters, which list multiple services rather than presenting a single
    service description.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = (parsed.path or "").rstrip("/").lower()
    if path != "/en/services":
        return False
    qs = parse_qs(parsed.query)
    return "organisation" in qs

def _is_html(resp: requests.Response) -> bool:
    """Check whether the HTTP response is HTML or XHTML."""
    ctype = resp.headers.get("Content-Type", "")
    ctype_lower = ctype.lower()
    return ("text/html" in ctype_lower) or ("application/xhtml+xml" in ctype_lower)

# -------------------------------------------------------------------------
# CEFR-style content validation (aligned with Austria)
# -------------------------------------------------------------------------

# Selectors used to remove boilerplate and navigation elements
NON_CONTENT_SELECTORS = [
    "nav", "header", "footer", "aside",
    ".breadcrumb", ".cookie", ".cookie-banner",
]

ADDRESS_LIKE = re.compile(
    r"(\bTel\.|\bTelephone\b|\bPhone\b|\bFax\b|@|www\.|http|street\b|str\.\b|\bPostcode\b|\bE-Mail\b|\bemail\b)",
    flags=re.IGNORECASE,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_clean_text_for_count(html: str) -> str:
    """
    Remove navigation and boilerplate elements and keep potential prose
    blocks for a CEFR-style word count.
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
    HTML response, is not an explicit or soft 404, and contains at least one
    CEFR-style sentence (word count > 0). Organisation-based services search
    listings on gov.ie are excluded.
    """
    # HEAD probe
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=TIMEOUT)
    except Exception:
        return False, "head_error"

    if _is_services_listing(r.url):
        return False, "services_listing"

    if r.status_code == 404:
        return False, "404"
    if r.status_code >= 400:
        return False, f"http_{r.status_code}"

    if _is_govie_404_url(r.url):
        return False, "redirect_to_404"

    # Ensure HTML and fetch full content
    try:
        if _is_html(r):
            g = SESSION.get(url, allow_redirects=True, timeout=TIMEOUT)
        else:
            g = SESSION.get(url, allow_redirects=True, timeout=TIMEOUT)
    except Exception:
        return False, "get_error"

    if g.status_code >= 400:
        return False, f"http_{g.status_code}"
    if _is_govie_404_url(g.url):
        return False, "redirect_to_404"
    if not _is_html(g):
        return False, "non_html"
    if _looks_like_error_html(g.text):
        return False, "soft_404_template"

    # CEFR-style content check
    words = _cefr_word_count(g.text)
    if words == 0:
        return False, "no_text"

    return True, "ok"

# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create the output directory for English-language content."""
    os.makedirs(OUT_EN, exist_ok=True)

def read_rows(csv_path: str) -> List[Dict[str, str]]:
    """Read sitemap, URL, and lastmod fields from a CSV file."""
    rows: List[Dict[str, str]] = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "sitemap": (row.get("sitemap") or "").strip(),
                "url": (row.get("url") or "").strip(),
                "lastmod": (row.get("lastmod") or "").strip(),
            })
    return rows

def write_rows(csv_path: str, rows: List[Dict[str, str]]) -> None:
    """Write sitemap, URL, and lastmod fields to a CSV file."""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sitemap", "url", "lastmod"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def out_cat_path_en(cat_tail: str) -> str:
    """Return the output path for a cleaned EN category file."""
    return os.path.join(OUT_EN, f"ireland_urls_en_{cat_tail}_clean.csv")

# -------------------------------------------------------------------------
# Chart writer (EN only)
# -------------------------------------------------------------------------

def write_en_chart(counts_dict: Dict[str, int], out_path: str, title: str) -> None:
    """Visualise the distribution of cleaned URLs across EN categories."""
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
# Category discovery (EN)
# -------------------------------------------------------------------------

def discover_en_categories(selected: list[str] | None) -> list[str]:
    """
    Discover EN category tails from Step 00 and restrict them to the
    allowed set (campaigns, policies, services).
    """
    discovered = []
    for p in glob.glob(os.path.join(IN_EN, "ireland_urls_en_*.csv")):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "ireland_all_sitemap_urls_EN":
            continue
        tail = name.replace("ireland_urls_en_", "", 1)
        if tail == "other":
            continue
        discovered.append(tail)

    discovered = sorted({c for c in discovered if c in ALLOWED_EN_CATEGORIES})

    if selected:
        subset = [c for c in selected if c in discovered]
        return subset
    return discovered

# -------------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------------

def main(selected: list[str] | None, delay: float):
    """
    Validate responsiveness and CEFR-style content of IE-EN URLs in the
    selected categories and write cleaned files, a merged EN list, and a
    summary table.
    """
    ensure_dirs()

    # Determine categories to process
    scope = discover_en_categories(selected)
    if not scope:
        print("No EN categories selected or discovered. Nothing to do.")
        return

    # Read category CSVs from Step 00
    cat_rows: Dict[str, List[Dict[str, str]]] = {}
    for cat in scope:
        in_path = os.path.join(IN_EN, f"ireland_urls_en_{cat}.csv")
        rows = read_rows(in_path)
        rows = sorted(rows, key=lambda r: (r["sitemap"], r["url"]))  # deterministic
        cat_rows[cat] = rows
        print(f"[input] {cat}: {len(rows)} candidates from {os.path.basename(in_path)}")

    # Validation loop
    with open(OUT_REJECTS, "w", encoding="utf-8", newline="") as rejects_f:
        rej_writer = csv.DictWriter(rejects_f, fieldnames=["category", "url", "reason"])
        rej_writer.writeheader()

        cleaned_per_cat: Dict[str, List[Dict[str, str]]] = {c: [] for c in scope}
        kept_urls_set: set[str] = set()

        for cat in scope:
            kept = 0
            rejected = 0
            rows = cat_rows[cat]
            for i, row in enumerate(rows, start=1):
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
                    print(f"[progress] {cat}: {i}/{len(rows)}")
                time.sleep(delay)
            print(f"[validated] {cat}: kept {kept}, rejected {rejected}")

    # Write cleaned category CSVs
    for cat in scope:
        out_path = out_cat_path_en(cat)
        write_rows(out_path, cleaned_per_cat[cat])
        print(f"[output] {cat}: {len(cleaned_per_cat[cat])} → {os.path.basename(out_path)}")

    # Build cleaned merged EN list from Step 00 all-EN by intersecting kept URLs
    all_en_rows = read_rows(IN_ALL_EN)
    cleaned_all_en = [r for r in all_en_rows if r["url"] in kept_urls_set]
    cleaned_all_en = sorted(cleaned_all_en, key=lambda r: (r["sitemap"], r["url"]))
    write_rows(OUT_ALL_EN_CLEAN, cleaned_all_en)
    print(f"[output] merged EN: {len(cleaned_all_en)} → {os.path.basename(OUT_ALL_EN_CLEAN)}")

    # Prepare EN-only cleaned counts for chart and summary
    counts_en: Dict[str, int] = {c: len(cleaned_per_cat[c]) for c in scope}

    # Chart for cleaned EN categories
    write_en_chart(
        counts_en,
        OUT_CHART_EN,
        "URL counts per EN category (cleaned; gov.ie)"
    )

    # Cleaned overview: categories only, including rejects (no sitemap listing)
    total_en_clean = sum(counts_en.values())
    with open(OUT_SUMMARY, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Language", "Category", "Kept_Valid", "Rejected", "Total_Candidates"])
        for cat in sorted(scope):
            kept = len(cleaned_per_cat[cat])
            total = len(cat_rows[cat])
            rej = total - kept
            w.writerow(["EN", cat, kept, rej, total])
        w.writerow([])
        w.writerow(["TOTAL_EN", "", total_en_clean, "", sum(len(cat_rows[c]) for c in scope)])

    print("=== Cleaning complete ===")
    print(f"Outputs in: {OUT_EN}")
    print(f"Summary written to: {OUT_SUMMARY}")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate and clean IE EN URLs from Step 00 for selected categories and write cleaned EN outputs."
    )
    ap.add_argument(
        "--categories",
        nargs="*",
        help="Optional subset of EN category tails to clean, e.g., campaigns policies services."
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=DELAY,
        help="Delay between requests in seconds (default 0.25)."
    )
    args = ap.parse_args()
    main(args.categories, args.delay)
