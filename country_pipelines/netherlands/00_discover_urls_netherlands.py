# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 00_discover_urls_netherlands.py
# Purpose: crawl and collect URLs from the Dutch government portal (rijksoverheid.nl)
# Pipeline step: Data collection, step 00 (URL discovery), Netherlands.
# Thesis: Chapter 3: discovered URLs in the corpus construction funnel (tab:dc_corpus_funnel);
#         Annex URL path table (tab:dc_categories_netherlands).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import re
import time
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SITE_ROOT    = "https://www.rijksoverheid.nl"
SITEMAP_URL  = "https://www.rijksoverheid.nl/sitemap"
UA           = "Mozilla/5.0 (compatible; NL-SitemapAnalyzer/1.0; +https://example.org)"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT  = os.path.join(SCRIPT_DIR, "00_discover_output_netherlands")
REQUEST_DELAY = 0.5   # seconds between requests (be polite)
TIMEOUT       = 20

# Section on /sitemap whose links are theme index pages (need a Level 2 fetch).
THEME_SECTIONS = {"onderwerpen"}

# Sections to skip entirely (links excluded from all output).
SKIP_SECTIONS = {"documenten"}

# Path prefixes that are never useful content
SKIP_PREFIXES = ("/binaries/", "/error/", "/sitemap", "/zoeken", "/rss",
                 "/feeds/", "/stats/", "/documenten")

# File extensions to skip
SKIP_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xls", ".zip", ".tar", ".gz",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".mp4", ".mp3", ".wav", ".ogg", ".css", ".js",
}

# -------------------------------------------------------------------------
# HTTP helpers
# -------------------------------------------------------------------------

def fetch_html(url: str, session: requests.Session) -> str | None:
    """Fetch a URL and return its HTML text, or None on failure."""
    try:
        r = session.get(
            url,
            headers={"User-Agent": UA},
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        if r.ok and "text/html" in r.headers.get("Content-Type", "").lower():
            return r.text
    except Exception as exc:
        print(f"  [WARN] {url}: {exc}")
    return None


def normalise_url(url: str) -> str:
    """Strip fragment, query string, and trailing slash for deduplication."""
    url = url.split("#")[0].split("?")[0].rstrip("/")
    return url or SITE_ROOT


def is_internal(url: str) -> bool:
    """Return True if the URL belongs to rijksoverheid.nl."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc in ("www.rijksoverheid.nl", "rijksoverheid.nl", "")
    except Exception:
        return False


def should_skip(url: str) -> bool:
    """Return True if the URL should be excluded from results."""
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        if any(path.startswith(p) for p in SKIP_PREFIXES):
            return True
        ext = os.path.splitext(path)[1]
        if ext in SKIP_EXTENSIONS:
            return True
        # Skip English-language pages (/en/ prefix) – NL only
        if path.startswith("/en/") or path == "/en":
            return True
    except Exception:
        return True
    return False


def first_path_segment(url: str) -> str:
    """Return the first non-empty path segment (e.g. 'onderwerpen')."""
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    return parts[0] if parts else ""


def cat_key_for_url(url: str) -> str:
    """Return the category key derived from the first path segment."""
    seg = first_path_segment(url)
    return f"nl_{seg}" if seg else "nl_other"



# -------------------------------------------------------------------------
# Sitemap parsing (Level 1)
# -------------------------------------------------------------------------

def parse_sitemap_page(html: str) -> dict[str, list[str]]:
    """
    Parse https://www.rijksoverheid.nl/sitemap and return a dict:
        { section_label_lower: [absolute_url, ...] }

    The sitemap page uses the structure:
        <ul class="sitemap three-columns">
          <li class="column">
            <a href="/onderwerpen">Onderwerpen</a>   ← section heading
            <ul class="common">
              <li><a href="/onderwerpen/themas/economie">Economie</a></li>
              ...
            </ul>
          </li>
          ...
        </ul>
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body or soup

    sections: dict[str, list[str]] = {}

    sitemap_ul = main.find("ul", class_=lambda c: c and "sitemap" in c)
    if not sitemap_ul:
        sitemap_ul = main.find("ul")
    if not sitemap_ul:
        return sections

    for col in sitemap_ul.find_all("li", recursive=False):
        heading_a = col.find("a", recursive=False)
        if not heading_a:
            continue
        section_label = heading_a.get_text(strip=True).lower()
        if not section_label:
            continue

        # Skip documenten entirely
        if any(skip in section_label for skip in SKIP_SECTIONS):
            continue

        links: list[str] = []
        nested_ul = col.find("ul")
        if nested_ul:
            for li in nested_ul.find_all("li", recursive=False):
                a = li.find("a")
                if not a:
                    continue
                href = a.get("href", "").strip()
                if not href or href.startswith(("mailto:", "javascript:", "#")):
                    continue
                abs_url = normalise_url(urljoin(SITE_ROOT, href))
                if is_internal(abs_url) and not should_skip(abs_url):
                    links.append(abs_url)

        if links:
            sections[section_label] = links

    return sections


# -------------------------------------------------------------------------
# Theme page parsing (Level 2 – Onderwerpen theme pages)
# -------------------------------------------------------------------------

def fetch_theme_subtopics(theme_url: str, session: requests.Session) -> list[str]:
    """
    Fetch an Onderwerpen theme page (e.g. /onderwerpen/themas/economie)
    and return the subtopic content URLs found on it (/onderwerpen/<slug>).
    Theme index pages (/themas/) are excluded.
    """
    html = fetch_html(theme_url, session)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    for sel in ["nav", "header", "footer", "aside", "[role='navigation']"]:
        for bad in soup.select(sel):
            bad.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    urls: list[str] = []
    seen: set[str] = set()

    for a in main.find_all("a", href=True):
        href = a["href"].strip()
        abs_url = normalise_url(urljoin(SITE_ROOT, href))
        if not is_internal(abs_url) or should_skip(abs_url):
            continue
        path = urlparse(abs_url).path.lower()
        if not path.startswith("/onderwerpen/"):
            continue
        if "/themas/" in path or path == "/onderwerpen":
            continue
        if abs_url not in seen:
            seen.add(abs_url)
            urls.append(abs_url)

    return urls


# -------------------------------------------------------------------------
# Deeper link collection (Level 3 – follow links within same section)
# -------------------------------------------------------------------------

def fetch_deeper_links(
    page_url: str,
    path_prefix: str,
    session: requests.Session,
) -> list[str]:
    """
    Fetch `page_url` and return all internal links whose path starts with
    `path_prefix` (e.g. '/onderwerpen/'). Navigation/footer are stripped
    first so only main-content links are returned.
    """
    html = fetch_html(page_url, session)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    for sel in ["nav", "header", "footer", "aside", "[role='navigation']"]:
        for bad in soup.select(sel):
            bad.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    urls: list[str] = []
    seen: set[str] = set()

    for a in main.find_all("a", href=True):
        href = a["href"].strip()
        abs_url = normalise_url(urljoin(SITE_ROOT, href))
        if not is_internal(abs_url) or should_skip(abs_url):
            continue
        path = urlparse(abs_url).path.lower()
        if not path.startswith(path_prefix):
            continue
        if abs_url not in seen:
            seen.add(abs_url)
            urls.append(abs_url)

    return urls


# -------------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------------

def ensure_lang_dir(lang_code: str) -> str:
    """Create and return the language-specific output subdirectory."""
    folder = os.path.join(OUTPUT_ROOT, lang_code.upper())
    os.makedirs(folder, exist_ok=True)
    return folder


def write_bar_chart(counts_dict: dict, lang_code: str, title: str) -> None:
    """Visualise URL counts per category as a bar chart PNG."""
    if not counts_dict:
        return
    plt.figure(figsize=(max(10, len(counts_dict) * 0.8), 6))
    plt.bar(list(counts_dict.keys()), list(counts_dict.values()))
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of URLs")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    max_val = max(counts_dict.values()) if counts_dict.values() else 1
    for i, (_, v) in enumerate(counts_dict.items()):
        plt.text(i, v + (max_val * 0.01), str(v), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out_chart = os.path.join(
        OUTPUT_ROOT,
        lang_code.upper(),
        f"netherlands_category_counts_{lang_code.upper()}.png",
    )
    plt.savefig(out_chart, dpi=220)
    plt.close()
    print(f"  Saved chart → {out_chart}")


# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def main() -> None:
    """
    Main execution:
    1) Fetch /sitemap and parse section headings + links.
       Documenten section is skipped entirely.
    2) Onderwerpen: each sitemap link is a theme page; fetch it to get subtopic
       URLs (Level 2). Then fetch each subtopic URL to collect deeper links
       within /onderwerpen/ (Level 3). All go into category 'nl_onderwerpen'.
    3) Other sections (Actueel, Ministeries, Regering, …): use sitemap links
       directly as Level 2, then fetch each to collect deeper links within the
       same top-level prefix (Level 3). Category = 'nl_<section>'.
    4) Write per-category NL CSV files, merged NL CSV, bar chart, and summary.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    session = requests.Session()

    # ── Level 1: fetch and parse the HTML sitemap ─────────────────────────
    print(f"Fetching sitemap: {SITEMAP_URL}")
    sitemap_html = fetch_html(SITEMAP_URL, session)
    if not sitemap_html:
        print("ERROR: Could not fetch sitemap. Aborting.")
        return

    sections = parse_sitemap_page(sitemap_html)
    print(f"Found {len(sections)} sections on sitemap: {list(sections.keys())}")

    # per_cat: { cat_key: { url: source_label } }  (dict preserves insertion order, deduplicates)
    per_cat: dict[str, dict[str, str]] = defaultdict(dict)

    for section_label, section_urls in sections.items():
        section_clean = section_label.strip().lower()
        is_theme_section = any(ts in section_clean for ts in THEME_SECTIONS)

        if is_theme_section:
            # ── Level 2 (Onderwerpen): theme page → subtopics ──────────────
            print(f"\n[Onderwerpen] {len(section_urls)} theme(s) to fetch:")
            level2_urls: list[str] = []

            for theme_url in section_urls:
                print(f"  Level 2 – theme: {theme_url}")
                subtopics = fetch_theme_subtopics(theme_url, session)
                print(f"    → {len(subtopics)} subtopic URL(s)")
                level2_urls.extend(subtopics)
                time.sleep(REQUEST_DELAY)

            # Deduplicate Level 2
            level2_unique = list(dict.fromkeys(level2_urls))
            print(f"  Level 2 total unique: {len(level2_unique)}")

            # ── Level 3: fetch each subtopic page for deeper /onderwerpen/ links
            cat_key = "nl_onderwerpen"
            source_l2 = "sitemap-onderwerpen-level2"
            source_l3 = "sitemap-onderwerpen-level3"
            path_prefix = "/onderwerpen/"

            for i, sub_url in enumerate(level2_unique, 1):
                per_cat[cat_key].setdefault(sub_url, source_l2)
                deeper = fetch_deeper_links(sub_url, path_prefix, session)
                for u in deeper:
                    per_cat[cat_key].setdefault(u, source_l3)
                if i % 10 == 0:
                    print(f"    Level 3 progress: {i}/{len(level2_unique)} subtopics processed")
                time.sleep(REQUEST_DELAY)

            print(f"  [nl_onderwerpen] total after Level 3: {len(per_cat[cat_key])}")

        else:
            # ── Level 2 (direct section): sitemap links are the content URLs
            first_seg = first_path_segment(section_urls[0]) if section_urls else section_clean
            cat_key = f"nl_{first_seg}" if first_seg else f"nl_{re.sub(r'[^a-z0-9]+', '-', section_clean).strip('-')}"
            path_prefix = f"/{first_seg}/"
            source_l2 = f"sitemap-section:{section_clean}"
            source_l3 = f"sitemap-section-level3:{section_clean}"

            print(f"\n[{section_label}] {len(section_urls)} direct URL(s) → category '{cat_key}'")

            for url in section_urls:
                per_cat[cat_key].setdefault(url, source_l2)

            # ── Level 3: follow deeper links within same section
            for i, url in enumerate(section_urls, 1):
                deeper = fetch_deeper_links(url, path_prefix, session)
                for u in deeper:
                    per_cat[cat_key].setdefault(u, source_l3)
                time.sleep(REQUEST_DELAY)

            print(f"  [{cat_key}] total after Level 3: {len(per_cat[cat_key])}")

    # ── Write per-category CSV files ──────────────────────────────────────
    print("\n--- Writing per-category CSV files ---")
    counts_nl: dict[str, int] = {}
    folder = ensure_lang_dir("NL")

    for cat in sorted(per_cat.keys()):
        url_source_map = per_cat[cat]
        if not url_source_map:
            continue

        target = os.path.join(folder, f"netherlands_urls_{cat}.csv")
        counts_nl[cat] = len(url_source_map)

        with open(target, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "url", "lastmod"])
            for url, src in url_source_map.items():
                w.writerow([src, url, ""])
        print(f"  {cat}: {len(url_source_map)} URL(s) → {os.path.basename(target)}")

    # ── Merged NL CSV ─────────────────────────────────────────────────────
    total_nl = sum(counts_nl.values())

    if total_nl > 0:
        merged_nl = os.path.join(folder, "netherlands_all_sitemap_urls_NL.csv")
        seen_merged: set[str] = set()
        with open(merged_nl, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "url", "lastmod"])
            for cat in sorted(per_cat.keys()):
                for url, src in per_cat[cat].items():
                    if url not in seen_merged:
                        seen_merged.add(url)
                        w.writerow([src, url, ""])
        print(f"\nMerged NL CSV: {len(seen_merged)} unique URL(s) → {merged_nl}")
        write_bar_chart(counts_nl, "NL", "URL counts per NL category (rijksoverheid.nl)")
    else:
        print("No NL URLs discovered.")

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_ROOT, "netherlands_summary_overview.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Number of URLs Found"])
        w.writerow(["sitemap:rijksoverheid.nl/sitemap", total_nl])
        w.writerow([])
        w.writerow(["Language", "Category", "Count"])
        for cat, cnt in sorted(counts_nl.items()):
            w.writerow(["NL", cat, cnt])
        w.writerow([])
        w.writerow(["TOTAL_NL", "", total_nl])

    print(f"\nSummary written → {summary_path}")
    print(f"NL total: {total_nl}")
    print("Done.")


if __name__ == "__main__":
    main()
