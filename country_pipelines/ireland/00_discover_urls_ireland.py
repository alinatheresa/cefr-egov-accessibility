# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 00_discover_urls_ireland.py
# Purpose: crawl and collect sitemap URLs from the Irish government portal (gov.ie)
# Pipeline step: Data collection, step 00 (URL discovery), Ireland.
# Thesis: Chapter 3: discovered URLs in the corpus construction funnel (tab:dc_corpus_funnel);
#         Annex URL path table (tab:dc_categories_ireland).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import re
import unicodedata
from collections import defaultdict
from urllib.parse import urlparse, unquote

import matplotlib.pyplot as plt
import requests
from lxml import etree

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SITE_ROOT = "https://www.gov.ie"
ROBOTS_TXT = f"{SITE_ROOT}/robots.txt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (compatible; GovieSitemapAnalyzer/2.0; +https://example.org)"

# Language detection based on first path segment
LANG_RE = re.compile(r"^/(?P<lang>en|ga)(/|$)", re.IGNORECASE)

# Department segment pattern (publisher slugs)
DEPT_SEGMENT_RE = re.compile(
    r"^/(?:en|ga)/(?P<seg>(?:department(?:-of)?-[^/]+|an-roinn-[^/]+|roinn-[^/]+))/",
    re.IGNORECASE,
)

# Category segment patterns (biographies, campaigns, collections, etc.)
# Matches:
#   /en/[department]/biographies/[slug]/
#   /en/[department]/campaigns/[slug]/
#   ...
#   /ga/[department]/services/[slug]/
#   /ga/[department]/speeches/[slug]/
CATEGORY_SEGMENT_RE = re.compile(
    r"^/(?:en|ga)/[^/]+/(?P<seg>(?:"
    r"biographies|campaigns|collections|consultations|"
    r"oireachtas-enquiries|organisation-information|policies|policy-information|"
    r"press-releases|publications|roles|services|speeches|"
    r"circulars|forms|news|intreo-centres|foreshore-notices|"
    r"school-inspection-reports|schools|early-years-service-inspection-report|"
    r"early-years-services|youthreach-centres|aquaculture-licences|"
    r"colaiste-gaeilge-inspection-reports|youthreach-centre-inspection-report|"
    r"colaisti-gaeilge|coroners|coroner-case|ogp-frameworks"
    r"))/",
    re.IGNORECASE,
)

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def fetch(url: str):
    """Download a URL using a custom user agent."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r

def fetch_xml(url: str):
    """Download and parse XML content."""
    r = fetch(url)
    return etree.fromstring(r.content, parser=etree.XMLParser(recover=True))

def discover_sitemaps():
    """Retrieve all sitemap URLs from robots.txt."""
    sitemaps = []
    try:
        resp = fetch(ROBOTS_TXT)
        sitemap_lines = re.findall(
            r"(?i)^\s*Sitemap:\s*(\S+)\s*$", resp.text, flags=re.MULTILINE
        )
        for sm in sitemap_lines:
            if sm.strip().lower().endswith((".xml", ".xml.gz")):
                sitemaps.append(sm.strip())
    except Exception:
        # If robots.txt cannot be processed, return the current list
        pass
    sitemaps = [s for s in sitemaps if s.lower().endswith((".xml", ".xml.gz"))]
    return sorted(set(sitemaps))

def parse_sitemap(sitemap_url: str):
    """
    Parse a sitemap XML and return a list of (url, lastmod).
    Handles both index sitemaps and regular urlset sitemaps.
    """
    root = fetch_xml(sitemap_url)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Index sitemap → recurse
    idx = root.findall(".//sm:sitemap/sm:loc", ns)
    if idx:
        urls = []
        for loc_el in idx:
            child = (loc_el.text or "").strip()
            if child:
                try:
                    urls.extend(parse_sitemap(child))
                except Exception:
                    # Skip problematic child sitemaps
                    pass
        return urls

    # URL set → extract
    urls = []
    for u in root.findall(".//sm:url", ns):
        loc = u.findtext("sm:loc", default="", namespaces=ns).strip()
        lastmod = u.findtext("sm:lastmod", default="", namespaces=ns).strip()
        if loc:
            urls.append((loc, lastmod))
    return urls

def detect_language(url: str) -> str:
    """Return EN or GA based on the first path segment, UNKNOWN otherwise."""
    try:
        path = urlparse(url).path
        m = LANG_RE.match(path)
        if m:
            return m.group("lang").upper()
    except Exception:
        pass
    return "UNKNOWN"

# ------------------------- department helpers ------------------------------

def extract_department_slug(url: str) -> str:
    """Extract department slug for EN/GA or return 'other_publishers'."""
    try:
        path = urlparse(url).path
        m = DEPT_SEGMENT_RE.search(path)
        if m:
            return m.group("seg").lower()
    except Exception:
        pass
    return "other_publishers"

def make_department_base(dept_slug: str) -> str:
    """Return department base without language suffix."""
    return "other" if dept_slug == "other_publishers" else dept_slug

# -------------------------- category helpers -------------------------------

def extract_category_slug(url: str) -> str:
    """
    Extract category segment for EN/GA URLs based on the given structures:
      /en/[department]/biographies/[slug]/
      /en/[department]/campaigns/[slug]/
      ...
      /ga/[department]/services/[slug]/
      /ga/[department]/speeches/[slug]/
      /en/irish-treaty-series/[slug]/
      /en/irish-government-economic-and-evaluation-service-igees/[slug]/
    Returns 'other' if no category segment matches.
    """
    try:
        path = urlparse(url).path
        m = CATEGORY_SEGMENT_RE.search(path)
        if m:
            return m.group("seg").lower()

        # Special-case categories where the first non-language segment
        # itself represents the category.
        segments = [seg for seg in path.split("/") if seg]
        if segments and segments[0] in ("en", "ga") and len(segments) >= 2:
            first_seg = segments[1].lower()
            if first_seg in {
                "irish-treaty-series",
                "irish-government-economic-and-evaluation-service-igees",
            }:
                return first_seg
    except Exception:
        pass
    return "other"

def make_category_base(cat_slug: str) -> str:
    """Return normalized category base without language suffix."""
    return cat_slug or "other"

# ---------------------- shared display helper ------------------------------

def display_from_base(base: str) -> str:
    """Return a human-readable label for charts and summaries."""
    if base == "other":
        return "other"
    dec = unquote(base)
    return unicodedata.normalize("NFC", dec)

# -------------------------------------------------------------------------
# CSV, plotting, and summary helpers
# -------------------------------------------------------------------------

def write_category_csvs(base_to_rows: dict, lang_folder: str, prefix: str, filename_map):
    """
    Write one CSV per base key (department or category).
    filename_map(base) controls how the base is turned into a filename.
    """
    for base, rows in sorted(base_to_rows.items(), key=lambda kv: kv[0]):
        filename_tail = filename_map(base)
        out_csv = os.path.join(lang_folder, f"{prefix}_{filename_tail}.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sitemap", "url", "lastmod"])
            w.writerows(rows)
        print(f"  Wrote {len(rows)} URLs → {out_csv}")

def write_master(rows, lang_folder, lang_code):
    """Write merged CSV with all URLs per language."""
    merged_path = os.path.join(lang_folder, f"ireland_all_sitemap_urls_{lang_code}.csv")
    with open(merged_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sitemap", "url", "lastmod"])
        w.writerows(rows)
    print(f"  Merged {len(rows)} URLs → {merged_path}")

def plot_counts_departments(folder: str, title: str, outfile: str, base_to_rows: dict):
    """Bar chart for departments (top 20 by count)."""
    items = [(base, len(rows)) for base, rows in base_to_rows.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [display_from_base(c) for c, _ in items[:20]]
    values = [v for _, v in items[:20]]

    if not labels:
        print(f"No data to plot for {title}")
        return

    plt.figure(figsize=(16, 9))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of URLs")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    ymax = max(values) if values else 0
    if ymax > 0:
        plt.ylim(0, ymax * 1.12)
    for i, v in enumerate(values):
        offset = (0.02 * ymax) if ymax else 1
        plt.text(i, v + offset, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(folder, outfile)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved {out_path}")

def plot_counts_categories(folder: str, title: str, outfile: str, base_to_rows: dict):
    """Bar chart for categories (all, alphabetical)."""
    labels = []
    values = []
    for base, rows in sorted(base_to_rows.items(), key=lambda kv: kv[0]):
        labels.append(display_from_base(base))
        values.append(len(rows))

    if not labels:
        print(f"No data to plot for {title}")
        return

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Number of URLs")
    ymax = max(values) if values else 0
    for i, v in enumerate(values):
        offset = (0.02 * ymax) if ymax else 1
        plt.text(i, v + offset, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(folder, outfile)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved {out_path}")

def write_summary(summary_path: str, sitemap_counts, en_dict: dict, ga_dict: dict):
    """Write ireland_summary_overview.csv for one perspective."""
    total_en = sum(len(v) for v in en_dict.values())
    total_ga = sum(len(v) for v in ga_dict.values())

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sitemap URL", "Number of URLs Found"])
        for sm, cnt in sorted(sitemap_counts, key=lambda x: x[0]):
            w.writerow([sm, cnt])
        w.writerow([])
        w.writerow(["Language", "Category", "Count"])
        for base in sorted(en_dict.keys()):
            w.writerow(["EN", f"en_{display_from_base(base)}", len(en_dict[base])])
        for base in sorted(ga_dict.keys()):
            w.writerow(["GA", f"ga_{display_from_base(base)}", len(ga_dict[base])])
        w.writerow(["TOTAL_EN", "", total_en])
        w.writerow(["TOTAL_GA", "", total_ga])
        w.writerow(["TOTAL_ALL", "", total_en + total_ga])

    print(f"\nSummary written to {summary_path}")
    print(f"EN total: {total_en} | GA total: {total_ga} | All: {total_en + total_ga}")

# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def main():
    """Main execution function for sitemap discovery and analysis."""
    # Overview of main workflow:
    # 1) Discover sitemap URLs from the Irish portal robots.txt.
    # 2) Parse all sitemaps and collect unique URLs for both EN and GA content.
    # 3) Classify URLs simultaneously by department (publisher) and by content category.
    # 4) For each perspective, write per-category CSV files, merged language-specific CSVs, and bar charts.
    # 5) Create summary CSV files that report sitemap counts and language/category distributions for both perspectives.
    
    # Root output folders
    out_root = os.path.join(SCRIPT_DIR, "00_discover_output_ireland")
    dept_root = os.path.join(out_root, "departments")
    cat_root = os.path.join(out_root, "categories")

    # Departments: EN/GA subfolders
    dept_en = os.path.join(dept_root, "EN")
    dept_ga = os.path.join(dept_root, "GA")

    # Categories: EN/GA subfolders
    cat_en = os.path.join(cat_root, "EN")
    cat_ga = os.path.join(cat_root, "GA")

    for path in [dept_en, dept_ga, cat_en, cat_ga]:
        os.makedirs(path, exist_ok=True)

    print(f"Discovering sitemaps for {SITE_ROOT}")
    submaps = discover_sitemaps()
    if not submaps:
        print("No sitemaps discovered.")
        return
    print(f"Found {len(submaps)} sitemap(s):")
    for s in submaps:
        print(f"  - {s}")

    # Collect and deduplicate URLs across all sitemaps
    sitemap_counts = []
    all_entries = []
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

    seen = {}
    for sm, loc, lastmod in all_entries:
        if loc not in seen:
            seen[loc] = (sm, lastmod)
    print(f"\nTotal unique URLs: {len(seen)}")

    # Language-level collections (same for both perspectives)
    en_all, ga_all = [], []

    # Departments perspective
    en_by_dept, ga_by_dept = defaultdict(list), defaultdict(list)

    # Categories perspective
    en_by_cat, ga_by_cat = defaultdict(list), defaultdict(list)

    # Classify once per URL into both department and category buckets
    for loc, (sm, lastmod) in seen.items():
        lang = detect_language(loc)
        if lang not in ("EN", "GA"):
            continue

        row = (sm, loc, lastmod)

        if lang == "EN":
            en_all.append(row)
        else:
            ga_all.append(row)

        # Departments
        dept_slug = extract_department_slug(loc)
        dept_base = make_department_base(dept_slug)

        # Categories
        cat_slug = extract_category_slug(loc)
        cat_base = make_category_base(cat_slug)

        if lang == "EN":
            en_by_dept[dept_base].append(row)
            en_by_cat[cat_base].append(row)
        else:
            ga_by_dept[dept_base].append(row)
            ga_by_cat[cat_base].append(row)

    # ------------------------------------------------------------------
    # Write CSVs for departments perspective
    # ------------------------------------------------------------------

    print("\nWriting department-based CSVs (departments/...)")
    write_category_csvs(
        en_by_dept,
        dept_en,
        "ireland_urls_en",
        filename_map=lambda b: b,
    )
    write_category_csvs(
        ga_by_dept,
        dept_ga,
        "ireland_urls_ga",
        filename_map=display_from_base,
    )

    write_master(en_all, dept_en, "EN")
    write_master(ga_all, dept_ga, "GA")

    plot_counts_departments(
        dept_en,
        "Ireland – Department counts (EN)",
        "ireland_category_counts_EN.png",
        en_by_dept,
    )
    plot_counts_departments(
        dept_ga,
        "Ireland – Department counts (GA)",
        "ireland_category_counts_GA.png",
        ga_by_dept,
    )

    dept_summary_csv = os.path.join(dept_root, "ireland_summary_overview.csv")
    write_summary(dept_summary_csv, sitemap_counts, en_by_dept, ga_by_dept)

    # ------------------------------------------------------------------
    # Write CSVs for categories perspective
    # ------------------------------------------------------------------

    print("\nWriting category-based CSVs (categories/...)")
    write_category_csvs(
        en_by_cat,
        cat_en,
        "ireland_urls_en",
        filename_map=lambda b: b,
    )
    write_category_csvs(
        ga_by_cat,
        cat_ga,
        "ireland_urls_ga",
        filename_map=lambda b: b,
    )

    write_master(en_all, cat_en, "EN")
    write_master(ga_all, cat_ga, "GA")

    plot_counts_categories(
        cat_en,
        "Ireland – Category counts (EN)",
        "ireland_category_counts_EN.png",
        en_by_cat,
    )
    plot_counts_categories(
        cat_ga,
        "Ireland – Category counts (GA)",
        "ireland_category_counts_GA.png",
        ga_by_cat,
    )

    cat_summary_csv = os.path.join(cat_root, "ireland_summary_overview.csv")
    write_summary(cat_summary_csv, sitemap_counts, en_by_cat, ga_by_cat)

    print("\nDone. Outputs written to 00_discover_output_ireland/departments and /categories.")

if __name__ == "__main__":
    main()