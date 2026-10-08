# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 02_scrape_urls_germany.py
# Purpose: scrape and extract text content from cleaned German government URLs
# Pipeline step: Data collection, step 02 (text extraction from static HTML), Germany. The Playwright
#                fallback did not run when the corpus was collected.
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
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Optional: Playwright for shadow-DOM accordion expansion (if installed)
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (compatible; GermanyScrapeDE/0.2; TU-Berlin-CEFR-Scraper)"
TIMEOUT = 25
RETRY = 3

# CSS selectors used to exclude non-content elements
EXCLUDE_SELECTORS = [
    "header", "nav", "footer", "form", "aside",
    ".breadcrumb", ".feedback", ".cookie", ".navigation", ".menu",
]

# Heuristics for filtering out boilerplate / meta text
DROP_PATTERNS = [
    r"\bImpressum\b",
    r"\bDatenschutz\b",
    r"Fragen\s*&\s*Antworten",
]

# Minimum length for non-question sentences (characters)
MIN_CHARS = 40

# Markers where the useful page content typically ends
CUTOFF_MARKERS = [
    "Sagen Sie uns Ihre Meinung!",
    "Meinung sagen",
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
            if any(re.search(p, txt, re.IGNORECASE) for p in DROP_PATTERNS):
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


def fetch_with_retry(url: str, session: requests.Session) -> Optional[requests.Response]:
    """
    Fetch a URL with simple retry logic.

    Returns a successful HTML response object or None if the request
    fails repeatedly or does not return HTML.
    """
    for i in range(RETRY):
        try:
            r = session.get(
                url,
                headers={"User-Agent": UA},
                timeout=TIMEOUT,
                allow_redirects=True,
            )
            if (
                r.status_code == 200
                and "text/html" in r.headers.get("Content-Type", "").lower()
            ):
                return r
        except requests.RequestException:
            pass
        time.sleep(1 + i)
    return None


def is_ls_portal_url(url: str) -> bool:
    """
    Heuristic to detect Leichte-Sprache portal pages whose main content
    is delivered via the CMS article API.

    Example:
        https://verwaltung.bund.de/portal/DE/Produkt-Info-Stellen
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return (
        parsed.netloc.endswith("verwaltung.bund.de")
        and parsed.path.startswith("/portal/DE/")
    )


def fetch_ls_article_html(url: str, session: requests.Session) -> Optional[str]:
    """
    For Leichte-Sprache portal pages (…/portal/DE/<slug>), call the CMS
    article API and return the HTML content string from the JSON response.

    The API response has the following relevant structure:

        {
          "data": [
            {
              "content": "<p>…</p>",
              ...
            }
          ],
          "meta": { ... }
        }

    If the URL does not match the expected pattern or the API call fails,
    return None so that the caller can fall back to the normal HTML.
    """
    if not is_ls_portal_url(url):
        return None

    try:
        parsed = urlparse(url)
        m = re.match(r"^/portal/DE/([^/?#]+)", parsed.path)
        if not m:
            return None
        slug = m.group(1)

        api_url = f"https://{parsed.netloc}/portal/cms/api/DE/article/{slug}"
        headers = {
            "User-Agent": UA,
            "Accept": "application/json, text/plain, */*",
            "Referer": url,
        }
        r = session.get(api_url, headers=headers, timeout=TIMEOUT)
        if r.status_code != 200:
            return None

        data = r.json()
        items = data.get("data") or []
        if not isinstance(items, list) or not items:
            return None

        first = items[0]
        content_html = (first.get("content") or "").strip()
        if not content_html:
            return None
        return content_html
    except Exception:
        return None


def fetch_html_with_playwright(url: str) -> Optional[str]:
    """
    Fetch rendered HTML for a URL using Playwright as a fallback for pages
    that are partially or fully JavaScript-rendered. This is only used for
    non-Leichte-Sprache URLs when the standard requests-based fetch does
    not return usable HTML.
    """
    if sync_playwright is None:
        return None

    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page(user_agent=UA)
            page.goto(url, timeout=90000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None

# -------------------------------------------------------------------------
# Dropdown / shadow-DOM extraction (if available)
# -------------------------------------------------------------------------

def extract_dropdown_content(url: str) -> List[str]:
    """
    Expand accordion-style components implemented via shadow DOM on
    verwaltung.bund.de and extract their textual content as sentence-like
    strings. For Leichte Sprache pages, also extract the main article body
    content rendered inside shadow DOM (e.g. data-testid="article_page_content").

    If Playwright is not available, an empty list is returned.

    Note: this function is skipped for Leichte-Sprache portal URLs when the
    CMS article API is used instead, to avoid unnecessary browser overhead.
    """
    if sync_playwright is None:
        return []

    JS_EXPAND_AND_DUMP = r"""
    (() => {
      const visitShadow = (root, fn) => {
        const q = [root];
        const seen = new Set([root]);
        while (q.length) {
          const r = q.shift();
          fn(r);
          const nodes = r.querySelectorAll ? r.querySelectorAll('*') : [];
          for (const el of nodes) {
            if (el.shadowRoot && !seen.has(el.shadowRoot)) {
              seen.add(el.shadowRoot);
              q.push(el.shadowRoot);
            }
          }
        }
      };

      const sleep = (ms) => new Promise(res => setTimeout(res, ms));

      return (async () => {
        const accHosts = [];
        const articleBodies = []

        visitShadow(document, (r) => {
          if (!r.querySelectorAll) return;

          // Accordion components (original behaviour)
          for (const el of r.querySelectorAll('lv-accordion-ce, lv-accordion-textmodule-ce')) {
            accHosts.push(el);
          }

          // Main article body on Leichte-Sprache pages (inside shadow DOM)
          for (const el of r.querySelectorAll('[data-testid="article_page_content"]')) {
            articleBodies.push(el);
          }
        });

        // Expand accordion hosts
        for (const host of accHosts) {
          const sr = host.shadowRoot;
          if (!sr) continue;
          const btn = sr.querySelector('button,[role="button"]');
          if (btn) { try { btn.click(); } catch(e) {} }
          await sleep(120);
        }
        await sleep(500);

        const out = [];
        const splitSentences = (t) => {
          const norm = (t || "").replace(/\s+/g, " ").trim();
          if (!norm) return [];
          return norm.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
        };

        // Extract text from accordion content
        for (const host of accHosts) {
          const sr = host.shadowRoot;
          if (!sr) continue;

          const btn = sr.querySelector('button,[role="button"]');
          const title = (btn ? (btn.innerText || btn.textContent || "") : "").trim();

          let content = "";
          const slots = sr.querySelectorAll('slot');
          if (slots.length) {
            const parts = [];
            for (const s of slots) {
              const nodes = s.assignedNodes ? s.assignedNodes({flatten: true}) : [];
              for (const n of nodes) {
                if (n.nodeType === Node.TEXT_NODE) {
                  const t = (n.textContent || "").trim();
                  if (t) parts.push(t);
                } else if (n.nodeType === Node.ELEMENT_NODE) {
                  const t = (n.innerText || n.textContent || "").trim();
                  if (t) parts.push(t);
                }
              }
            }
            content = parts.join(" ");
          } else {
            content = (sr.innerText || sr.textContent || "");
          }

          if (title) {
            const re = new RegExp(
              "^\\s*" + title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\s*:?\\s*",
              "i"
            );
            content = content.replace(re, "");
          }

          content = content.replace(/\s+/g, " ").trim();
          if (content) {
            for (const s of splitSentences(content)) {
              if (s.endswith(".") or s.endswith("!") or s.endswith("?")):
                out.push(s);
            }
          }

          // Stop after contact blocks (usually not linguistically relevant)
          if (/^\s*Kontakt\s*$/i.test(title)) {
            break;
          }
        }

        // Extract main article body text from Leichte-Sprache pages
        for (const container of articleBodies) {
          const raw = (container.innerText || container.textContent || "").trim();
          if (!raw) continue;
          const sents = splitSentences(raw);
          for (const s of sents) {
            if (s.endswith(".") or s.endswith("!") or s.endswith("?")):
              out.push(s);
          }
        }

        return out;
      })();
    })();
    """

    dropdown_sentences: List[str] = []
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=90000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            data = page.evaluate(JS_EXPAND_AND_DUMP)
            if isinstance(data, list):
                dropdown_sentences = data
            browser.close()
    except Exception:
        return []
    return dropdown_sentences

# -------------------------------------------------------------------------
# Extract linguistically meaningful sentences
# -------------------------------------------------------------------------

def extract_meaningful_sentences(html: str, url: str) -> List[str]:
    """
    Remove navigation and boilerplate elements, integrate visible content
    and accordion text, and return linguistically meaningful German
    sentences suitable for CEFR analysis.

    Logic:
    - Do not merge all text into one large string.
    - Work at the level of individual text nodes.
    - For each node, split locally into candidate sentences.
    - Keep only candidates that already end with '.', '!' or '?'.
    - Apply the same rule to dropdown / shadow-DOM content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Drop standard non-content containers
    for sel in EXCLUDE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    main = soup.find("main") or soup

    # Extract bullet lists as coherent grouped sentences before text-node pass
    list_sentences: List[str] = extract_lists_as_sentences(main)

    # Expand simple button-based accordions in the static DOM
    for btn in main.find_all("button"):
        collected = []
        for sib in btn.next_siblings:
            if getattr(sib, "name", None) == "button":
                break
            if isinstance(sib, str):
                t = normalize_ws(sib)
                if t:
                    collected.append(t)
            elif hasattr(sib, "get_text"):
                t = normalize_ws(sib.get_text(" ", strip=True))
                if t:
                    collected.append(t)
        if collected:
            merged = " ".join(collected).strip()
            btn.insert_after(BeautifulSoup(merged, "html.parser"))

    # Collect sentences from remaining visible text nodes without merging
    dom_sentences: List[str] = []
    for el in main.find_all(string=True):
        parent_name = el.parent.name
        if parent_name in ["script", "style", "noscript"]:
            continue
        txt = normalize_ws(el)
        if not txt:
            continue
        if any(re.search(p, txt, flags=re.IGNORECASE) for p in DROP_PATTERNS):
            continue
        if txt in ["Startseite", "Alle Lebenslagen", "Kontakt", "Mein Bereich"]:
            continue

        # Split this individual text node into local candidate sentences
        parts = re.split(r"(?<=[.!?])\s+", txt)
        for part in parts:
            s = part.strip()
            if not s:
                continue
            if s.endswith((".", "!", "?")):
                dom_sentences.append(s)

    # Apply cutoff markers at sentence level: stop once a marker appears
    trimmed_dom: List[str] = []
    cutoff_reached = False
    for s in dom_sentences:
        if any(marker in s for marker in CUTOFF_MARKERS):
            cutoff_reached = True
            break
        trimmed_dom.append(s)
    if cutoff_reached:
        dom_sentences = trimmed_dom

    # Dropdown / shadow-DOM sentences (already split and filtered there)
    dropdown_sentences: List[str] = []
    if not is_ls_portal_url(url):
        dropdown_sentences = extract_dropdown_content(url)

    # Combine and de-duplicate while preserving order
    # Apply cutoff markers to list sentences as well
    trimmed_list: List[str] = []
    for s in list_sentences:
        if any(marker in s for marker in CUTOFF_MARKERS):
            break
        trimmed_list.append(s)

    seen = set()
    uniq: List[str] = []
    for s in dom_sentences + dropdown_sentences + trimmed_list:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def get_title(html_text: str) -> str:
    """Extract the document title from an HTML page, if available."""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(" ", strip=True)
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
    """
    Read and deduplicate URL strings from a German category CSV file.

    For LS mapping files (filenames ending in "_ls.csv"), the ls_url column
    is preferred over url/loc so that the Leichte Sprache target pages are
    scraped instead of the original source URLs.
    """
    urls: List[str] = []
    filename = os.path.basename(csv_path)
    use_ls_column = filename.endswith("_ls.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if use_ls_column:
                u = row.get("ls_url") or row.get("url") or row.get("loc") or ""
            else:
                u = row.get("url") or row.get("loc") or ""
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
    rejects_accum: List[Dict[str, str]],
    page_id_start: int,
    sentences_for_cefr: List[Dict[str, str]],
    global_sentences: List[str],
) -> Tuple[Dict[str, object], int]:
    """
    Process all URLs for a single DE category, extract meaningful sentences,
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

    per_cat_csv = os.path.join(cat_dir, f"germany_{cat_slug}.csv")
    per_cat_txt = os.path.join(cat_dir, f"germany_{cat_slug}_combined.txt")

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

        # Detect whether this is a Leichte-Sprache portal URL
        is_ls = is_ls_portal_url(u)

        # 1) Try standard HTML fetch via requests
        html: str = ""
        resp = fetch_with_retry(u, session)
        if resp is not None:
            html = resp.text

        # 2) For Leichte-Sprache portal pages, prefer the CMS article body
        ls_html = fetch_ls_article_html(u, session)
        if ls_html:
            html = ls_html

        # 3) For non-LS URLs: if there is still no HTML, try Playwright fallback
        if not html and not is_ls:
            html_pl = fetch_html_with_playwright(u)
            if html_pl:
                html = html_pl

        # 4) If we still have no usable HTML, mark as reject
        if not html:
            rejects_accum.append(
                {"category": cat_slug, "url": u, "reason": "fetch_error_or_non_html"}
            )
            failure += 1
            continue

        sents = extract_meaningful_sentences(html, u)
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
                    "country": "DE",
                    "language": "DE",
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
    Scrape linguistically meaningful content for all DE categories selected
    in Step 01, and write per-category, combined, and sentence-level outputs.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Step 02 for Germany (DE). Scrape meaningful German text from all "
            "URLs per category and prepare CEFR-ready outputs."
        )
    )
    ap.add_argument(
        "--in-root",
        default=os.path.join(SCRIPT_DIR, "01_clean_output_germany", "DE"),
        help="Input folder with germany_urls_de_<category>_clean.csv files.",
    )
    ap.add_argument(
        "--out-root",
        default=os.path.join(SCRIPT_DIR, "02_scrape_output_germany"),
        help="Output root folder for scraped DE content.",
    )
    args = ap.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    ensure_dir(out_root)

    # Discover DE category CSVs (accept both clean and non-clean names, but strip _clean)
    all_csvs = sorted(glob.glob(os.path.join(in_root, "germany_urls_de_*.csv")))
    csvs: List[Tuple[str, str]] = []
    for path in all_csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in ("germany_all_sitemap_urls_DE", "germany_all_sitemap_urls_DE_clean"):
            continue
        cat = name.replace("germany_urls_de_", "", 1)
        if cat in {"other", "other_clean"}:
            continue
        if cat.endswith("_clean"):
            cat = cat[:-6]
        csvs.append((cat, path))

    if not csvs:
        pr("No DE category CSVs found to process.")
        return

    pr(f"Found {len(csvs)} DE categories to process.")
    for cat, path in csvs:
        pr(f"  - {cat} ← {os.path.basename(path)}")

    # Output artefacts (analogous naming to 02_scrape_urls_ireland.py)
    combined_csv = os.path.join(out_root, "combined_germany.csv")
    summary_csv = os.path.join(out_root, "germany_scrape_summary.csv")
    global_txt = os.path.join(out_root, "all_categories_germany_combined.txt")
    sentences_cefr_csv = os.path.join(out_root, "sentences_for_cefr_germany.csv")

    # Additional rejects file (DE only, to document crawling failures)
    rejected_dir = os.path.join(out_root, "rejected")
    ensure_dir(rejected_dir)
    rejects_csv = os.path.join(rejected_dir, "germany_scrape_rejects.csv")

    fields = ["category", "url", "title", "sentence_count", "word_count", "text_joined"]
    combined_rows: List[Dict[str, str]] = []
    all_sentences: List[str] = []
    rejects_accum: List[Dict[str, str]] = []
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
            rejects_accum,
            page_id_counter,
            sentences_for_cefr,
            all_sentences,
        )
        stats_per_category.append(stats)

        per_cat_csv = os.path.join(out_root, cat, f"germany_{cat}.csv")
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

    # Summary statistics with averages and medians (per category + ALL)
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
                    avg_w_per_sent = t_words / t_sents
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

    # Persist rejects information (DE only)
    with open(rejects_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "url", "reason"])
        w.writeheader()
        for rec in rejects_accum:
            w.writerow(rec)
    pr(f"Wrote rejects → {rejects_csv}")

    pr("Done.")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
