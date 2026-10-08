# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 00_discover_urls_austria.py
# Purpose: crawl and collect sitemap URLs from the Austrian government portal (oesterreich.gv.at)
# Pipeline step: Data collection, step 00 (URL discovery), Austria.
# Thesis: Chapter 3: discovered URLs in the corpus construction funnel (tab:dc_corpus_funnel);
#         Annex URL path table (tab:dc_categories_austria).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import re
from collections import defaultdict
from urllib.parse import urlparse
from datetime import datetime

import matplotlib.pyplot as plt
import requests
from lxml import etree

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SITE_ROOT = "https://www.oesterreich.gv.at"
ROBOTS_TXT = f"{SITE_ROOT}/robots.txt"
UA = "Mozilla/5.0 (compatible; AT-SitemapAnalyzer/1.3; +https://tu-berlin.example)"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "00_discover_output_austria")

# Regular expression for language and category segments
CAT_RE = re.compile(
    r"^https?://[^/]+/(?P<lang>de|en)/(?P<cat>lebenslagen|themen|landingpages|lexicon|orgsearch|formsearch|hilfe|Gesetzliche-Neuerungen)/",
    re.IGNORECASE,
)

# Explicit category ordering for reporting
CAT_LABELS_ORDER = [
    "de_lebenslagen", "de_themen", "de_landingpages", "de_lexicon",
    "de_orgsearch", "de_formsearch", "de_hilfe", "de_gesetzliche-neuerungen",
    "de_other",
    "en_lebenslagen", "en_themen", "en_landingpages", "en_lexicon",
    "en_orgsearch", "en_formsearch", "en_other",
    "other",
]

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def fetch(url: str) -> requests.Response:
    """Download a URL using a custom user agent."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r

def fetch_xml(url: str):
    """Download and parse XML content, including optional gzip compression."""
    r = fetch(url)
    content = r.content
    if url.lower().endswith(".gz"):
        import gzip
        content = gzip.decompress(content)
    return etree.fromstring(content, parser=etree.XMLParser(recover=True))

def normalize_lastmod(raw: str) -> str:
    """
    Normalize lastmod to 'YYYY-MM-DD' (date only).
    Falls back to the original string if parsing fails.
    """
    if not raw:
        return ""
    raw = raw.strip()
    patterns = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for p in patterns:
        try:
            dt = datetime.strptime(raw, p)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        date_part = raw.split("T", 1)[0]
        return datetime.strptime(date_part, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return raw

def discover_sitemaps() -> list[str]:
    """Retrieve sitemap URLs from robots.txt."""
    sitemaps = []
    try:
        resp = fetch(ROBOTS_TXT)
        sitemap_lines = re.findall(r"(?i)^\s*Sitemap:\s*(\S+)\s*$", resp.text, flags=re.MULTILINE)
        for sm in sitemap_lines:
            if sm.strip().lower().endswith((".xml", ".xml.gz")):
                sitemaps.append(sm.strip())
    except Exception:
        # If robots.txt cannot be accessed, return the list collected so far
        pass
    return sorted(set(sitemaps))

def parse_sitemap(sitemap_url: str) -> list[tuple[str, str]]:
    """
    Parse a sitemap XML and return a list of (url, lastmod) tuples.
    Handles both index sitemaps and regular urlset sitemaps.
    """
    root = fetch_xml(sitemap_url)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    idx = root.findall(".//sm:sitemap/sm:loc", ns)
    if idx:
        urls = []
        for loc_el in idx:
            child = (loc_el.text or "").strip()
            if child:
                urls.extend(parse_sitemap(child))
        return urls

    urls = []
    for u in root.findall(".//sm:url", ns):
        loc = u.findtext("sm:loc", default="", namespaces=ns).strip()
        lastmod_raw = u.findtext("sm:lastmod", default="", namespaces=ns).strip()
        if loc:
            urls.append((loc, normalize_lastmod(lastmod_raw)))
    return urls

def categorise_url(url: str) -> str:
    """
    Assign a URL to a language-specific category based on the path pattern.
    If no category is recognised but a language segment exists, assign *_other.
    """
    m = CAT_RE.search(url)
    if m:
        lang = m.group("lang").lower()
        cat = m.group("cat").lower()
        return f"{lang}_{cat}"
    try:
        path = urlparse(url).path
        segs = [s for s in path.split("/") if s]
        if segs and segs[0].lower() in {"de", "en"}:
            return f"{segs[0].lower()}_other"
    except Exception:
        pass
    return "other"

def lang_of(cat: str) -> str:
    """Return the language code ('de', 'en', or 'other') for a given category."""
    if cat.startswith("de_"):
        return "de"
    if cat.startswith("en_"):
        return "en"
    return "other"

def ensure_root():
    """Create the main output directory if it does not yet exist."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

def ensure_lang_dir(lang_code: str):
    """Create and return the language-specific output directory."""
    folder = os.path.join(OUTPUT_ROOT, lang_code.upper())
    os.makedirs(folder, exist_ok=True)
    return folder

def write_bar_chart(counts_dict: dict[str, int], lang_code: str, title: str):
    """Visualise the distribution of URLs across categories for one language."""
    if not counts_dict:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(list(counts_dict.keys()), list(counts_dict.values()))
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of URLs")
    plt.xticks(rotation=90, ha="center", fontsize=8)
    max_val = max(counts_dict.values())
    for i, (k, v) in enumerate(counts_dict.items()):
        plt.text(i, v + (max_val * 0.01), str(v), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out_chart = os.path.join(
        OUTPUT_ROOT,
        lang_code.upper(),
        f"austria_category_counts_{lang_code.upper()}.png"
    )
    plt.savefig(out_chart, dpi=220)
    plt.close()
    print(f"Saved {out_chart}")

# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def main():
    """Main execution function for sitemap discovery and analysis."""
    # Overview of main workflow:
    # 1) Discover sitemap URLs for the Austrian portal from robots.txt.
    # 2) Parse all sitemaps, normalise lastmod values, and collect unique URLs.
    # 3) Categorise URLs by language (DE/EN) and high-level portal category.
    # 4) Write per-category CSV files and merged language-specific CSVs.
    # 5) Generate URL count charts and a summary CSV for descriptive analysis.
    
    ensure_root()
    print(f"Discovering sitemaps for {SITE_ROOT}")

    submaps = discover_sitemaps()

    if not submaps:
        print("No sitemaps discovered.")
        return

    print(f"Found {len(submaps)} sitemap(s):")
    for s in submaps:
        print(f"  - {s}")

    all_entries = []
    sitemap_counts = []

    for sm in submaps:
        try:
            print(f"\nParsing {sm}")
            entries = parse_sitemap(sm)
            print(f"  {len(entries)} URLs found")
            sitemap_counts.append((sm, len(entries)))
            for (loc, lastmod) in entries:
                all_entries.append((sm, loc, lastmod))
        except Exception as e:
            print(f"  Warning: Could not parse {sm}: {e}")
            sitemap_counts.append((sm, 0))

    # Deduplicate URLs across all sitemaps
    seen: dict[str, tuple[str, str]] = {}
    for sm, loc, lastmod in all_entries:
        if loc not in seen:
            seen[loc] = (sm, lastmod)

    print(f"\nTotal unique URLs: {len(seen)}")

    # Assign URLs to categories
    per_cat: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for loc, (sm, lastmod) in seen.items():
        per_cat[categorise_url(loc)].append((sm, loc, lastmod))

    counts_de, counts_en = {}, {}

    # Write per-category CSVs in a fixed category order
    for cat in CAT_LABELS_ORDER:
        rows = per_cat.get(cat, [])
        if not rows:
            continue
        language = lang_of(cat)
        if language == "de":
            folder = ensure_lang_dir("DE")
            target = os.path.join(folder, f"austria_urls_{cat}.csv")
            counts_de[cat] = len(rows)
        elif language == "en":
            folder = ensure_lang_dir("EN")
            target = os.path.join(folder, f"austria_urls_{cat}.csv")
            counts_en[cat] = len(rows)
        else:
            target = os.path.join(OUTPUT_ROOT, f"austria_urls_{cat}.csv")

        with open(target, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sitemap", "url", "lastmod"])
            w.writerows(rows)
        print(f"  Wrote {len(rows)} URLs → {target}")

    total_de = sum(counts_de.values())
    total_en = sum(counts_en.values())

    # Merged CSV and bar chart for German content
    if total_de > 0:
        folder = ensure_lang_dir("DE")
        merged_de = os.path.join(folder, "austria_all_sitemap_urls_DE.csv")
        merged_rows = []
        for cat, rows in per_cat.items():
            if cat.startswith("de_"):
                merged_rows.extend(rows)
        with open(merged_de, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sitemap", "url", "lastmod"])
            w.writerows(merged_rows)
        print(f"  Merged {len(merged_rows)} URLs → {merged_de}")
        write_bar_chart({k: counts_de[k] for k in counts_de}, "DE", "URL counts per DE category (oesterreich.gv.at)")

    # Merged CSV and bar chart for English content
    if total_en > 0:
        folder = ensure_lang_dir("EN")
        merged_en = os.path.join(folder, "austria_all_sitemap_urls_EN.csv")
        merged_rows = []
        for cat, rows in per_cat.items():
            if cat.startswith("en_"):
                merged_rows.extend(rows)
        with open(merged_en, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sitemap", "url", "lastmod"])
            w.writerows(merged_rows)
        print(f"  Merged {len(merged_rows)} URLs → {merged_en}")
        write_bar_chart({k: counts_en[k] for k in counts_en}, "EN", "URL counts per EN category (oesterreich.gv.at)")
    else:
        print("No EN URLs detected — EN folder and outputs were not created.")

    # Summary overview across all sitemaps and categories
    summary_path = os.path.join(OUTPUT_ROOT, "austria_summary_overview.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sitemap URL", "Number of URLs Found"])
        for sm, cnt in sitemap_counts:
            w.writerow([sm, cnt])
        w.writerow([])
        w.writerow(["Language", "Category", "Count"])
        for cat, cnt in counts_de.items():
            w.writerow(["DE", cat, cnt])
        for cat, cnt in counts_en.items():
            w.writerow(["EN", cat, cnt])
        w.writerow([])
        w.writerow(["TOTAL_DE", "", total_de])
        w.writerow(["TOTAL_EN", "", total_en])
        w.writerow(["TOTAL_ALL", "", total_de + total_en])

    print(f"\nSummary written to {summary_path}")
    print(f"DE total: {total_de} | EN total: {total_en} | All: {total_de + total_en}")

if __name__ == "__main__":
    main()