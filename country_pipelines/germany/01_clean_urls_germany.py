# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 01_clean_urls_germany.py
# Purpose: filter and deduplicate discovered URLs from the German federal service portal
# Pipeline step: Data collection, step 01 (path filter and URL validation), Germany.
# Thesis: Chapter 3: kept and rejected URLs (tab:dc_corpus_funnel, tab:dc_rejection_reasons);
#         Annex URL path table (tab:dc_categories_germany).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import time
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Step 00 input root and DE folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_ROOT = os.path.join(SCRIPT_DIR, "00_discover_output_germany")
IN_DE = os.path.join(IN_ROOT, "DE")

# Step 01 output root and DE folder
OUT_ROOT = os.path.join(SCRIPT_DIR, "01_clean_output_germany")
OUT_DE = os.path.join(OUT_ROOT, "DE")

# Files from Step 00 to read
IN_ALL_DE = os.path.join(IN_DE, "germany_all_sitemap_urls_DE.csv")

# Files written by this script
OUT_SUMMARY = os.path.join(OUT_ROOT, "germany_summary_overview.csv")
OUT_ALL_DE_CLEAN = os.path.join(OUT_DE, "germany_all_sitemap_urls_DE_clean.csv")
OUT_REJECTS = os.path.join(OUT_DE, "rejects_de_validation.csv")
OUT_CHART_DE = os.path.join(OUT_DE, "germany_category_counts_DE.png")

# Category scope: DE categories selected for CEFR analysis
CATEGORIES = [
    "de_hilfs-problemloesungsdienste",
    "de_lagen",
    "de_leistung",
    "de_rechte_pflichten",
]

# Map each category to its Step 00 filename
CAT_TO_INPUT_FILE = {
    "de_hilfs-problemloesungsdienste": os.path.join(IN_DE, "germany_urls_de_hilfs-problemloesungsdienste.csv"),
    "de_lagen": os.path.join(IN_DE, "germany_urls_de_lagen.csv"),
    "de_leistung": os.path.join(IN_DE, "germany_urls_de_leistung.csv"),
    "de_rechte_pflichten": os.path.join(IN_DE, "germany_urls_de_rechte_pflichten.csv"),
}

# Output filenames for cleaned category CSVs
def out_cat_path(cat: str) -> str:
    return os.path.join(OUT_DE, f"germany_urls_{cat}_clean.csv")


# Output filenames for Leichte Sprache URL mappings per category
def out_cat_ls_path(cat: str) -> str:
    return os.path.join(OUT_DE, f"germany_urls_{cat}_ls.csv")


# Networking parameters
UA = "Mozilla/5.0 (compatible; TU-Berlin-CEFR-Scraper/2.5-multicat)"
TIMEOUT = 25
DELAY = 0.25

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# -------------------------------------------------------------------------
# Validation helpers (HTTP / HTML)
# -------------------------------------------------------------------------

ERROR_PATTERNS = [
    re.compile(r"\b404\b"),
    re.compile(r"seite nicht gefunden", re.IGNORECASE),
    re.compile(r"page not found", re.IGNORECASE),
    re.compile(r"fehler\s*[\d]{3}", re.IGNORECASE),
]

SOFT_404_STRINGS = [
    "Die angeforderte Seite wurde nicht gefunden",
    "the page you are looking for does not exist",
]

# Region-selection stubs ("Region auswählen") should be excluded
REGION_SELECTOR_STRINGS = [
    "<lv-region-opener-ce",
    "region auswählen",
    "lv-region_opener",
]


def _has_region_selector(text: str) -> bool:
    """Detect region-selection stubs that do not contain service content."""
    snippet = text.lower()
    return any(s in snippet for s in REGION_SELECTOR_STRINGS)


def _looks_like_error_html(text: str) -> bool:
    """Identify typical error or soft-404 templates in HTML content."""
    snippet = text[:4000]
    if any(rx.search(snippet) for rx in ERROR_PATTERNS):
        return True
    return any(s.lower() in snippet.lower() for s in SOFT_404_STRINGS)


def _is_explicit_404_url(final_url: str) -> bool:
    """Detect explicit 404 URLs after redirects."""
    u = (final_url or "").lower().split("?")[0].rstrip("/")
    return u.endswith("/404")


def _is_html(resp: requests.Response) -> bool:
    """Check whether the HTTP response is HTML or XHTML."""
    ctype = resp.headers.get("Content-Type", "")
    ctype_lower = ctype.lower()
    return ("text/html" in ctype_lower) or ("application/xhtml+xml" in ctype_lower)


def check_url(url: str) -> tuple[bool, str, str]:
    """
    Return (is_valid, reason, html). A URL is considered technically valid if
    it returns a 2xx HTML response, is not an explicit or soft 404, and
    provides HTML content for subsequent CEFR-style content checks.
    """
    try:
        g = SESSION.get(url, allow_redirects=True, timeout=TIMEOUT)
    except Exception:
        return False, "get_error", ""
    if g.status_code >= 400:
        return False, f"http_{g.status_code}", ""
    if _is_explicit_404_url(g.url):
        return False, "redirect_to_404", ""
    if not _is_html(g):
        return False, "non_html", ""
    if _looks_like_error_html(g.text):
        return False, "soft_404_template", ""
    return True, "ok", g.text


# -------------------------------------------------------------------------
# CEFR-style content validation (aligned with Austria)
# -------------------------------------------------------------------------

# Selectors used to remove boilerplate and navigation elements
NON_CONTENT_SELECTORS = [
    "nav",
    "header",
    "footer",
    "aside",
    ".breadcrumb",
    ".cookie",
    ".cookie-banner",
]

ADDRESS_LIKE = re.compile(
    r"(\bTel\. |\bTelefon\b|\bFax\b|@|www\.|http|straße\b|str\.\b|\bPLZ\b|\bPostfach\b|\bE-Mail\b)",
    flags=re.IGNORECASE,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_clean_text_for_count(html: str) -> str:
    """
    Remove navigation and boilerplate elements and keep potential prose
    blocks for a CEFR-style word count.
    """
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main") or soup.find("article") or soup

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


# -------------------------------------------------------------------------
# Leichte Sprache detection
# -------------------------------------------------------------------------


def find_leichtesprache_url(html: str, base_url: str) -> str:
    """
    Detect a link to a corresponding Leichte Sprache page, if present.
    Returns the absolute URL or an empty string if no candidate is found.
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()
        context = a.find_parent(["p", "div", "section"])
        ctx_text = context.get_text(" ", strip=True).lower() if context else ""
        if re.search(r"leichte[rs]?\s+sprache", text) or re.search(r"leichte[rs]?\s+sprache", ctx_text):
            abs_url = urljoin(base_url, a["href"])
            if abs_url not in candidates:
                candidates.append(abs_url)
    for u in candidates:
        if any(x in u for x in ["/leichte", "/Leichte", "Leichte", "leichte"]):
            return u
    return candidates[0] if candidates else ""


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------


def ensure_dirs():
    """Create the output directory for German-language content."""
    os.makedirs(OUT_DE, exist_ok=True)


def read_rows(csv_path: str) -> list[dict]:
    """Read sitemap, URL, and lastmod fields from a CSV file."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "sitemap": (row.get("sitemap") or "").strip(),
                    "url": (row.get("url") or "").strip(),
                    "lastmod": (row.get("lastmod") or "").strip(),
                }
            )
    return rows


def write_rows(csv_path: str, rows: list[dict]) -> None:
    """Write sitemap, URL, and lastmod fields to a CSV file."""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sitemap", "url", "lastmod"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_ls_rows(csv_path: str, rows: list[dict]) -> None:
    """Write original URL and Leichte Sprache URL pairs to a CSV file."""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["url", "ls_url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------------------------------------------------------
# Chart writer (DE only)
# -------------------------------------------------------------------------


def write_de_chart(counts_dict: dict[str, int], out_path: str, title: str) -> None:
    """Visualise the distribution of cleaned URLs across DE categories."""
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
    Validate responsiveness and CEFR-style content of DE URLs in the
    selected categories and write cleaned files, Leichte Sprache mappings,
    a merged DE list, and a summary table.
    """
    ensure_dirs()

    # Determine categories to process
    scope = CATEGORIES if not selected else [c for c in selected if c in CATEGORIES]
    if not scope:
        print("No categories selected. Nothing to do.")
        return

    # Read category CSVs from Step 00
    cat_rows: dict[str, list[dict]] = {}
    for cat in scope:
        path = CAT_TO_INPUT_FILE[cat]
        rows = read_rows(path)
        rows = sorted(rows, key=lambda r: (r["sitemap"], r["url"]))  # deterministic
        cat_rows[cat] = rows
        print(f"[input] {cat}: {len(rows)} candidates from {os.path.basename(path)}")

    # Validation loop
    rejects_f = open(OUT_REJECTS, "w", encoding="utf-8", newline="")
    rej_writer = csv.DictWriter(rejects_f, fieldnames=["category", "url", "reason"])
    rej_writer.writeheader()

    cleaned_per_cat: dict[str, list[dict]] = {c: [] for c in scope}
    ls_mappings_per_cat: dict[str, list[dict]] = {c: [] for c in scope}
    kept_urls_set: set[str] = set()
    ls_counts: dict[str, int] = {c: 0 for c in scope}

    for cat in scope:
        kept = 0
        rejected = 0
        for i, row in enumerate(cat_rows[cat], start=1):
            url = row["url"]
            ok, reason, html = check_url(url)
            if ok:
                if _has_region_selector(html):
                    rej_writer.writerow({"category": cat, "url": url, "reason": "region_selector"})
                    rejected += 1
                else:
                    words = _cefr_word_count(html)
                    if words > 0:
                        cleaned_per_cat[cat].append(row)
                        kept_urls_set.add(url)
                        kept += 1

                        ls_url = find_leichtesprache_url(html, url)
                        if ls_url:
                            ls_mappings_per_cat[cat].append({"url": url, "ls_url": ls_url})
                            ls_counts[cat] += 1
                    else:
                        rej_writer.writerow({"category": cat, "url": url, "reason": "no_text"})
                        rejected += 1
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

    # Write Leichte Sprache mapping CSVs (per category)
    for cat in scope:
        ls_rows = ls_mappings_per_cat[cat]
        if ls_rows:
            out_ls = out_cat_ls_path(cat)
            write_ls_rows(out_ls, ls_rows)
            print(f"[ls_output] {cat}: {len(ls_rows)} → {os.path.basename(out_ls)}")

    # Build cleaned merged DE list from Step 00 all-DE by intersecting kept URLs
    all_de_rows = read_rows(IN_ALL_DE)
    cleaned_all_de = [r for r in all_de_rows if r["url"] in kept_urls_set]
    cleaned_all_de = sorted(cleaned_all_de, key=lambda r: (r["sitemap"], r["url"]))
    write_rows(OUT_ALL_DE_CLEAN, cleaned_all_de)
    print(f"[output] merged DE: {len(cleaned_all_de)} → {os.path.basename(OUT_ALL_DE_CLEAN)}")

    # Prepare DE-only cleaned counts for chart and summary
    counts_de: dict[str, int] = {c: len(cleaned_per_cat[c]) for c in scope}

    # Chart for cleaned DE categories
    write_de_chart(
        counts_de,
        OUT_CHART_DE,
        "URL counts per DE category (cleaned; verwaltung.bund.de)",
    )

    # Cleaned overview: categories only, including rejects (no sitemap listing)
    total_de_clean = sum(counts_de.values())
    total_candidates = sum(len(cat_rows[c]) for c in scope)
    total_ls = sum(ls_counts[c] for c in scope)
    with open(OUT_SUMMARY, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Language",
                "Category",
                "Kept_Valid",
                "Rejected",
                "Total_Candidates",
                "With_Leichte_Sprache",
            ]
        )
        for cat in sorted(scope):
            kept = len(cleaned_per_cat[cat])
            total = len(cat_rows[cat])
            rej = total - kept
            w.writerow(["DE", cat, kept, rej, total, ls_counts[cat]])
        w.writerow([])
        w.writerow(["TOTAL_DE", "", total_de_clean, "", total_candidates, total_ls])

    print("=== Cleaning complete ===")
    print(f"Outputs in: {OUT_DE}")
    print(f"Summary written to: {OUT_SUMMARY}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate and clean DE URLs from Step 00 for selected categories and write cleaned DE outputs."
    )
    ap.add_argument(
        "--categories",
        nargs="*",
        help="Optional subset of categories to clean. Defaults to the predefined DE categories.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=DELAY,
        help="Delay between requests in seconds (default 0.25).",
    )
    args = ap.parse_args()
    main(args.categories, args.delay)
