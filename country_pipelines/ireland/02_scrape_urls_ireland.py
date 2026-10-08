# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 02_scrape_urls_ireland.py
# Purpose: scrape and extract text content from cleaned Irish government URLs
# Pipeline step: Data collection, step 02 (text extraction from static HTML), Ireland.
# Thesis: Section 3.4.3 (text extraction); final corpus in tab:dc_corpus_funnel, page length in
#         tab:dc_page_length. Writes combined_<country>.csv, the input of all later steps.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (compatible: IrelandScrapeEN/0.2; +https://example.org)"
TIMEOUT = 30

# CSS selectors used to exclude non-content elements
EXCLUDE_PARENT_SELECTORS = [
    "header", "nav", "footer", "aside", "form", "[role='navigation']",
    ".site-header", ".site-footer", ".breadcrumb", ".breadcrumbs",
    ".govuk-breadcrumbs", ".cookie-banner", ".cookie", ".share", ".social",
]

# Heuristics for filtering out contact and address-style strings
ADDRESS_PATTERNS = [
    r"\bTel\.?\b",
    r"\bPhone\b",
    r"\bFax\b",
    r"\bEmail\b",
    r"\bDublin\s+\d+\b",
    r"\bOpening hours\b",
    r"\bHow to apply\b$",
    r"^\d{1,2}\s*[A-Za-z]{3}\s*\d{4}$",
]

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def pr(msg: str) -> None:
    """Print a log message and flush stdout."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def normalize_ws(s: str) -> str:
    """Normalise whitespace in a string."""
    return re.sub(r"\s+", " ", s).strip()

def sentence_split(text: str) -> List[str]:
    """
    Split a block of text into candidate sentences using simple
    punctuation-based heuristics.
    """
    parts = re.split(
        r'(?<=[\.\!\?])\s+(?=[“"(\[]?[A-Z0-9])',
        text,
        flags=re.UNICODE,
    )
    out: List[str] = []
    for p in parts:
        p = p.strip()
        # Minimum length constraint removed: keep all candidates that end with sentence punctuation.
        if re.search(r"[\.!\?]$", p):
            out.append(p)
    return out

def looks_like_contact_or_address(s: str) -> bool:
    """Return True if a string resembles a contact or address block."""
    for pat in ADDRESS_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def extract_lists_as_sentences(main_soup) -> List[str]:
    """
    For each ul/ol, find the nearest preceding intro text and join li items
    into one coherent sentence per list, so bullet lists are meaningful for
    CEFR classification.
    """
    results: List[str] = []
    for lst in main_soup.find_all(["ul", "ol"]):
        items: List[str] = []
        for li in lst.find_all("li", recursive=False):
            txt = normalize_ws(li.get_text(" ", strip=True))
            if len(txt.split()) < 2:
                continue
            if looks_like_contact_or_address(txt):
                continue
            if re.search(r"[.!?]$", txt):
                results.append(txt)
            else:
                items.append(txt)
        if not items:
            continue
        intro = ""
        prev = lst.find_previous_sibling(["p", "h1", "h2", "h3", "h4"])
        if prev:
            intro = normalize_ws(prev.get_text(" ", strip=True)).rstrip(":")
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            joined = "; ".join(chunk) + "."
            sentence = f"{intro}: {joined}" if (intro and i == 0) else joined
            results.append(sentence)
    return results


def extract_meaningful_sentences(html_text: str) -> List[str]:
    """
    Remove boilerplate elements, extract text from paragraph-like tags,
    and keep only linguistically meaningful sentences.
    """
    soup = BeautifulSoup(html_text, "lxml")
    
    # Remove decorative icons that break sentence extraction
    for icon in soup.select("span[aria-hidden='true']"):
        icon.decompose()

    for sel in EXCLUDE_PARENT_SELECTORS:
        for bad in soup.select(sel):
            bad.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: List[str] = []

    # Paragraph-like content: p / h2 / h3 / h4 (li handled via list grouping below)
    for tag in main.find_all(["p", "h2", "h3", "h4"]):
        txt = normalize_ws(tag.get_text(" ", strip=True))
        if txt:
            blocks.append(txt)

    blocks.extend(extract_lists_as_sentences(main))

    candidate_sents: List[str] = []
    for b in blocks:
        for s in sentence_split(b):
            if not looks_like_contact_or_address(s):
                candidate_sents.append(s)

    seen = set()
    uniq: List[str] = []
    for s in candidate_sents:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def fetch_html(url: str, session: requests.Session) -> Optional[str]:
    """Fetch HTML for a given URL or return None on failure or non-HTML."""
    try:
        r = session.get(
            url,
            headers={"User-Agent": UA},
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        if r.ok and "text/html" in r.headers.get("Content-Type", "").lower():
            return r.text
    except Exception:
        return None
    return None

def get_title(html_text: str) -> str:
    """Extract the document title from an HTML page, if available."""
    try:
        soup = BeautifulSoup(html_text, "lxml")
        return soup.title.get_text(strip=True) if soup.title else ""
    except Exception:
        return ""

# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create a directory if it does not yet exist."""
    os.makedirs(path, exist_ok=True)

def read_category_urls(csv_path: str) -> List[str]:
    """Read and deduplicate URL strings from a category CSV file."""
    urls: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            u = row.get("url") or row.get("uri") or ""
            u = u.strip()
            if u:
                urls.append(u)
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

# -------------------------------------------------------------------------
# Category-level scraping and aggregation
# -------------------------------------------------------------------------

def process_category(
    cat_name: str,
    url_csv: str,
    out_root: str,
    session: requests.Session,
    page_id_start: int,
    sentences_for_cefr: List[Dict[str, str]],
    global_sentences: List[str],
) -> Tuple[Dict[str, object], int]:
    """
    Process all URLs for a single EN category, extract meaningful sentences,
    and write per-category CSV/TXT outputs into a dedicated subfolder.

    Returns a statistics dictionary and the next page_id to be used.
    """
    urls = read_category_urls(url_csv)
    if not urls:
        pr(f"[WARN] No URLs in {url_csv}")
        stats = {
            "category": cat_name,
            "attempted": 0,
            "success": 0,
            "failures": 0,
            "total_sentences": 0,
            "total_words": 0,
            "sentence_counts": [],
            "word_counts": [],
        }
        return stats, page_id_start

    cat_slug = cat_name
    cat_dir = os.path.join(out_root, cat_slug)
    ensure_dir(cat_dir)

    per_cat_csv = os.path.join(cat_dir, f"ireland_{cat_slug}.csv")
    per_cat_txt = os.path.join(cat_dir, f"ireland_{cat_slug}_combined.txt")

    fields = ["category", "url", "title", "sentence_count", "word_count", "text_joined"]
    success_rows: List[Dict[str, str]] = []
    combined_sentences: List[str] = []

    attempted = 0
    success = 0
    failure = 0
    total_sentences = 0
    total_words = 0
    sentence_counts: List[int] = []
    word_counts: List[int] = []

    page_id = page_id_start

    for u in urls:
        attempted += 1
        html = fetch_html(u, session)
        if not html:
            failure += 1
            continue

        sents = extract_meaningful_sentences(html)
        if not sents:
            sents = []

        joined = " ".join(sents)
        wc = len(re.findall(r"\w+", joined, flags=re.UNICODE))
        title = get_title(html)

        sentence_count = len(sents)
        total_sentences += sentence_count
        total_words += wc
        sentence_counts.append(sentence_count)
        word_counts.append(wc)

        success_rows.append(
            {
                "category": cat_slug,
                "url": u,
                "title": title,
                "sentence_count": str(sentence_count),
                "word_count": str(wc),
                "text_joined": joined,
            }
        )

        combined_sentences.extend(sents)
        global_sentences.extend(sents)
        success += 1

        # Populate sentence-level CEFR dataset
        for sid, s in enumerate(sents, start=1):
            sent_word_count = len(s.split())
            char_count = len(s)
            sentences_for_cefr.append(
                {
                    "country": "IE",
                    "language": "EN",
                    "category": cat_slug,
                    "url": u,
                    "title": title,
                    "page_id": str(page_id),
                    "sentence_id": str(sid),
                    "sentence": s,
                    "word_count": str(sent_word_count),
                    "char_count": str(char_count),
                }
            )

        page_id += 1

    if success_rows:
        with open(per_cat_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in success_rows:
                w.writerow(row)

        with open(per_cat_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_sentences))

    pr(
        f"[OK] {cat_slug}: attempted={attempted}, "
        f"success={success}, failures={failure}"
    )

    stats = {
        "category": cat_slug,
        "attempted": attempted,
        "success": success,
        "failures": failure,
        "total_sentences": total_sentences,
        "total_words": total_words,
        "sentence_counts": sentence_counts,
        "word_counts": word_counts,
    }
    return stats, page_id

# -------------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------------

def main() -> None:
    """
    Scrape linguistically meaningful content for all EN categories selected
    in Step 01, and write per-category, combined, and sentence-level outputs.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Step 02 for Ireland (EN). Scrape meaningful text from all URLs "
            "per category and prepare CEFR-ready outputs."
        )
    )
    ap.add_argument(
        "--in-root",
        default=os.path.join(SCRIPT_DIR, "01_clean_output_ireland", "EN"),
        help="Input folder with ireland_urls_en_<category>_clean.csv files.",
    )
    ap.add_argument(
        "--out-root",
        default=os.path.join(SCRIPT_DIR, "02_scrape_output_ireland"),
        help="Output root folder for scraped EN content.",
    )
    args = ap.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    ensure_dir(out_root)

    # Discover EN category CSVs (accept both clean and non-clean names, but strip _clean)
    all_csvs = sorted(glob.glob(os.path.join(in_root, "ireland_urls_en_*.csv")))
    csvs: List[Tuple[str, str]] = []
    for path in all_csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in ("ireland_all_sitemap_urls_EN", "ireland_all_sitemap_urls_EN_clean"):
            continue
        cat = name.replace("ireland_urls_en_", "", 1)
        if cat in {"other", "other_clean"}:
            continue
        if cat.endswith("_clean"):
            cat = cat[:-6]
        csvs.append((cat, path))

    if not csvs:
        pr("No EN category CSVs found to process.")
        return

    pr(f"Found {len(csvs)} EN categories to process.")
    for cat, path in csvs:
        pr(f"  - {cat} ← {os.path.basename(path)}")

    # Renamed output file: combined_ireland.csv
    combined_csv = os.path.join(out_root, "combined_ireland.csv")
    summary_csv = os.path.join(out_root, "ireland_scrape_summary.csv")
    global_txt = os.path.join(out_root, "all_categories_ireland_combined.txt")
    sentences_cefr_csv = os.path.join(out_root, "sentences_for_cefr_ireland.csv")

    fields = ["category", "url", "title", "sentence_count", "word_count", "text_joined"]
    combined_rows: List[Dict[str, str]] = []
    all_sentences: List[str] = []
    sentences_for_cefr: List[Dict[str, str]] = []
    stats_per_category: List[Dict[str, object]] = []

    session = requests.Session()
    page_id_counter = 1

    # Per-category scraping
    for cat, csv_path in csvs:
        stats, page_id_counter = process_category(
            cat,
            csv_path,
            out_root,
            session,
            page_id_counter,
            sentences_for_cefr,
            all_sentences,
        )
        stats_per_category.append(stats)

        per_cat_csv = os.path.join(out_root, cat, f"ireland_{cat}.csv")
        if os.path.exists(per_cat_csv):
            with open(per_cat_csv, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    combined_rows.append(row)

    # Combined CSV (one row per URL)
    if combined_rows:
        with open(combined_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in combined_rows:
                w.writerow(row)
        pr(f"Wrote combined CSV → {combined_csv}")

    # Global TXT with all sentences (one per line)
    if all_sentences:
        with open(global_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(all_sentences))
        pr(f"Wrote combined TXT → {global_txt}")

    # Sentence-level CEFR input CSV
    if sentences_for_cefr:
        cefr_fields = [
            "country",
            "language",
            "category",
            "url",
            "title",
            "page_id",
            "sentence_id",
            "sentence",
            "word_count",
            "char_count",
        ]
        with open(sentences_cefr_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cefr_fields)
            w.writeheader()
            for rec in sentences_for_cefr:
                w.writerow(rec)
        pr(f"Wrote CEFR input CSV → {sentences_cefr_csv}")

    # Summary statistics with averages and medians
    if stats_per_category:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "category",
                    "attempted",
                    "success",
                    "failures",
                    "total_sentences",
                    "total_words",
                    "avg_sentences_per_url",
                    "median_sentences_per_url",
                    "avg_words_per_url",
                    "median_words_per_url",
                    "avg_words_per_sentence",
                ]
            )

            total_attempted = 0
            total_success = 0
            total_failures = 0
            total_sentences = 0
            total_words = 0
            all_sentence_counts: List[int] = []
            all_word_counts: List[int] = []

            for stats in sorted(stats_per_category, key=lambda s: s["category"]):
                cat = stats["category"]
                attempted = int(stats["attempted"])
                success = int(stats["success"])
                failures = int(stats["failures"])
                t_sents = int(stats["total_sentences"])
                t_words = int(stats["total_words"])
                s_counts: List[int] = stats["sentence_counts"]  # type: ignore
                w_counts: List[int] = stats["word_counts"]      # type: ignore

                total_attempted += attempted
                total_success += success
                total_failures += failures
                total_sentences += t_sents
                total_words += t_words
                all_sentence_counts.extend(s_counts)
                all_word_counts.extend(w_counts)

                if success > 0:
                    avg_sents = t_sents / success
                    avg_words = t_words / success
                    med_sents = statistics.median(s_counts) if s_counts else 0.0
                    med_words = statistics.median(w_counts) if w_counts else 0.0
                else:
                    avg_sents = 0.0
                    avg_words = 0.0
                    med_sents = 0.0
                    med_words = 0.0

                if t_sents > 0:
                    avg_w_per_sent = total_words / total_sentences
                else:
                    avg_w_per_sent = 0.0

                w.writerow(
                    [
                        cat,
                        attempted,
                        success,
                        failures,
                        t_sents,
                        t_words,
                        f"{avg_sents:.3f}",
                        f"{med_sents:.3f}",
                        f"{avg_words:.3f}",
                        f"{med_words:.3f}",
                        f"{avg_w_per_sent:.3f}",
                    ]
                )

            # Aggregate "ALL" row
            if total_success > 0:
                all_avg_sents = total_sentences / total_success
                all_avg_words = total_words / total_success
            else:
                all_avg_sents = 0.0
                all_avg_words = 0.0

            all_med_sents = statistics.median(all_sentence_counts) if all_sentence_counts else 0.0
            all_med_words = statistics.median(all_word_counts) if all_word_counts else 0.0
            if total_sentences > 0:
                all_avg_w_per_sent = total_words / total_sentences
            else:
                all_avg_w_per_sent = 0.0

            w.writerow([])
            w.writerow(
                [
                    "ALL",
                    total_attempted,
                    total_success,
                    total_failures,
                    total_sentences,
                    total_words,
                    f"{all_avg_sents:.3f}",
                    f"{all_med_sents:.3f}",
                    f"{all_avg_words:.3f}",
                    f"{all_med_words:.3f}",
                    f"{all_avg_w_per_sent:.3f}",
                ]
            )

        pr(f"Wrote summary → {summary_csv}")

    pr("Done.")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()