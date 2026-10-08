# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 00_discover_urls_germany.py
# Purpose: crawl and collect sitemap URLs from the German federal service portal (verwaltung.bund.de)
# Pipeline step: Data collection, step 00 (URL discovery), Germany.
# Thesis: Chapter 3: discovered URLs in the corpus construction funnel (tab:dc_corpus_funnel);
#         Annex URL path table (tab:dc_categories_germany).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from lxml import etree
import csv
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SITE_ROOT = "https://verwaltung.bund.de"
ROBOTS_TXT = f"{SITE_ROOT}/robots.txt"
UA = "Mozilla/5.0 (compatible; SitemapAnalyzer/1.0; +https://example.org)"

# Main output directory corresponding to this step in the thesis workflow
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "00_discover_output_germany")

# Language subfolders
LANG_FOLDERS = {"de": "DE", "en": "EN"}

# URL pattern definitions to categorise sitemap entries
CAT_PATTERNS = {
    "de_lagen": "https://verwaltung.bund.de/leistungsverzeichnis/de/lagen/",
    "en_lagen": "https://verwaltung.bund.de/leistungsverzeichnis/en/lagen/",
    "de_rechte_pflichten": "https://verwaltung.bund.de/leistungsverzeichnis/de/rechte-und-pflichten/",
    "en_rechte_pflichten": "https://verwaltung.bund.de/leistungsverzeichnis/en/rechte-und-pflichten/",
    "de_hilfs-problemloesungsdienste": "https://verwaltung.bund.de/leistungsverzeichnis/de/hilfs-problemloesungsdienste/",
    "en_hilfs-problemloesungsdienste": "https://verwaltung.bund.de/leistungsverzeichnis/en/hilfs-problemloesungsdienste/",
    "de_leistung": "https://verwaltung.bund.de/leistungsverzeichnis/de/leistung/",
    "en_leistung": "https://verwaltung.bund.de/leistungsverzeichnis/en/leistung/",
}

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def fetch(url):
    """Download a URL using a custom user agent."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r


def fetch_xml(url):
    """Download and parse XML content."""
    r = fetch(url)
    return etree.fromstring(r.content, parser=etree.XMLParser(recover=True))


def discover_sitemaps():
    """Retrieve all sitemap URLs from robots.txt."""
    sitemaps = []
    try:
        resp = fetch(ROBOTS_TXT)
        sitemap_lines = re.findall(r"(?i)^\s*Sitemap:\s*(\S+)\s*$", resp.text, flags=re.MULTILINE)
        for sm in sitemap_lines:
            if sm.strip().lower().endswith((".xml", ".xml.gz")):
                sitemaps.append(sm.strip())
    except Exception:
        # In case of connection or parsing issues, return the list as collected so far
        pass

    sitemaps = [s for s in sitemaps if s.lower().endswith((".xml", ".xml.gz"))]
    return sorted(set(sitemaps))


def parse_sitemap(sitemap_url):
    """Parse a sitemap XML and return a list of URLs with metadata."""
    root = fetch_xml(sitemap_url)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Check if this is an index sitemap
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
        lastmod = u.findtext("sm:lastmod", default="", namespaces=ns).strip()
        if loc:
            urls.append((loc, lastmod))
    return urls


def categorise_url(url: str) -> str:
    """Assign a URL to one of the predefined category patterns."""
    for key, prefix in CAT_PATTERNS.items():
        if url.startswith(prefix):
            return key
    return "other"


def language_from_category(cat: str) -> str:
    """Determine language (DE/EN) based on category label."""
    if cat.startswith("de_"):
        return "de"
    elif cat.startswith("en_"):
        return "en"
    return "other"


def ensure_dirs():
    """Create the output directory and its language subfolders."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for lang in LANG_FOLDERS.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, lang), exist_ok=True)


def write_bar_chart(counts_dict, lang, title):
    """Create a bar chart visualising the number of URLs per category."""
    if not counts_dict:
        return  # nothing to plot; avoids max() on empty sequence
    plt.figure(figsize=(10, 6))
    plt.bar(list(counts_dict.keys()), list(counts_dict.values()))
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of URLs")

    # Rotate and shrink x-axis labels for readability (vertical orientation)
    plt.xticks(rotation=90, ha="center", fontsize=8)

    max_val = max(counts_dict.values())
    for i, (k, v) in enumerate(counts_dict.items()):
        plt.text(i, v + (max_val * 0.01), str(v), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_chart = os.path.join(OUTPUT_ROOT, LANG_FOLDERS[lang], f"germany_category_counts_{lang.upper()}.png")
    plt.savefig(out_chart, dpi=220)
    plt.close()
    print(f"Saved {out_chart}")


# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def main():
    """Main execution function for sitemap discovery and analysis."""
    # Overview of main workflow:
    # 1) Discover sitemap URLs from robots.txt.
    # 2) Parse all sitemaps and collect unique URLs.
    # 3) Categorise URLs by content type and language.
    # 4) Write per-category and per-language CSV files.
    # 5) Generate URL count charts and a summary CSV for reporting.
    
    ensure_dirs()
    print(f"Discovering sitemaps for {SITE_ROOT}")

    submaps = discover_sitemaps()

    # Ensure index children are also listed (so summary shows all three sitemaps)
    expanded = set(submaps)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    for sm in list(submaps):
        try:
            root = fetch_xml(sm)
            idx = root.findall(".//sm:sitemap/sm:sitemap/sm:loc", ns)
            if not idx:
                idx = root.findall(".//sm:sitemap/sm:loc", ns)
            for loc_el in idx:
                child = (loc_el.text or "").strip()
                if child and child.lower().endswith((".xml", ".xml.gz")):
                    expanded.add(child)
        except Exception:
            # Skip sitemaps that cannot be fetched or parsed
            pass
    submaps = sorted(expanded)

    if not submaps:
        print("No sitemaps discovered.")
        return

    print(f"Found {len(submaps)} sitemap(s):")
    for s in submaps:
        print("  -", s)

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

    # Deduplicate by URL
    seen = {}
    for sm, loc, lastmod in all_entries:
        if loc not in seen:
            seen[loc] = (sm, lastmod)
    print(f"\nTotal unique URLs: {len(seen)}")

    # Categorise each unique URL
    per_cat = defaultdict(list)
    for loc, (sm, lastmod) in seen.items():
        cat = categorise_url(loc)
        per_cat[cat].append((sm, loc, lastmod))

    # Write per-category CSVs into language-specific folders
    counts_de, counts_en = {}, {}
    for cat, rows in per_cat.items():
        lang = language_from_category(cat)
        if lang == "de":
            target = os.path.join(OUTPUT_ROOT, "DE", f"germany_urls_{cat}.csv")
            counts_de[cat] = len(rows)
        elif lang == "en":
            target = os.path.join(OUTPUT_ROOT, "EN", f"germany_urls_{cat}.csv")
            counts_en[cat] = len(rows)
        else:
            target = os.path.join(OUTPUT_ROOT, f"germany_urls_{cat}.csv")

        with open(target, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sitemap", "url", "lastmod"])
            w.writerows(rows)
        print(f"  Wrote {len(rows)} URLs → {target}")

    # Create combined CSVs per language
    for lang, folder in LANG_FOLDERS.items():
        merged_rows = []
        for cat, rows in per_cat.items():
            if cat.startswith(lang + "_"):
                merged_rows.extend(rows)
        merged_path = os.path.join(OUTPUT_ROOT, folder, f"germany_all_sitemap_urls_{lang.upper()}.csv")
        with open(merged_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sitemap", "url", "lastmod"])
            w.writerows(merged_rows)
        print(f"  Merged {len(merged_rows)} URLs → {merged_path}")

    # Visualise category distributions
    write_bar_chart(counts_de, "de", "URL counts per DE category (verwaltung.bund.de)")
    write_bar_chart(counts_en, "en", "URL counts per EN category (verwaltung.bund.de)")

    # Summary file with sitemap- and category-level statistics
    total_de = sum(counts_de.values())
    total_en = sum(counts_en.values())
    total_all = total_de + total_en

    summary_path = os.path.join(OUTPUT_ROOT, "germany_summary_overview.csv")
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
        w.writerow(["TOTAL_ALL", "", total_all])

    print(f"\nSummary written to {summary_path}")
    print(f"DE total: {total_de} | EN total: {total_en} | All: {total_all}")


if __name__ == "__main__":
    main()