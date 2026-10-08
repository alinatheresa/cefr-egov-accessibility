# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: sdg_categorize_full.py
# Purpose: zero-shot EU Single Digital Gateway Annex I categorisation of all scraped government pages using a local Ollama model
# Pipeline step: Older SDGR categoriser (single-token prompt). Not used for the corpus labels and not
#                part of the thesis results. Kept only as a module: sample_manual_annotation.py imports
#                it to build the codebook, and b2_unread_text.py reads its prompt. A full run writes
#                to sdg_categorization_full.csv in this folder and would overwrite the corpus labels.
# Thesis: None directly.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, re, json, time, subprocess, sys, argparse
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL          = "qwen2.5:7b"
OLLAMA_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 50}

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# Scraped texts: <project root>/country_pipelines/<country>/02_scrape_output_<country>/
BASE_DATA   = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "country_pipelines")
OUT_CSV     = os.path.join(SCRIPT_DIR, "sdg_categorization_full.csv")

COUNTRIES   = ["germany", "austria", "ireland", "netherlands"]
MAX_CHARS   = 6000
SNIPPET_LEN = 150
TIMING_CHECKPOINT = 25

OUT_FIELDS = ["country", "raw_category", "url", "word_count",
              "assigned_area", "assigned_label", "snippet"]

# ── EU SDG Annex I labels ──────────────────────────────────────────────────────
AREAS = {
    "A": "Travel within the EU",
    "B": "Work and retirement",
    "C": "Vehicles",
    "D": "Residence in another EU country",
    "E": "Education or traineeship",
    "F": "Healthcare",
    "G": "Citizens and family rights",
    "H": "Consumer rights",
    "I": "Protection of personal data",
    "J": "Starting, running, closing a business",
    "K": "Employees (employer-side)",
    "L": "Taxes",
    "M": "Goods",
    "N": "Services and professional qualifications",
    "O": "Funding a business",
    "P": "Public contracts",
    "Q": "Health and safety at work",
}

VALID_AREAS = set(AREAS.keys()) | {"OUT_OF_SCOPE", "UNCLEAR"}

# Enum schema for Ollama structured output. Constrains the model to exactly the
# allowed set so it cannot return free text, labels, or explanations.
ENUM_VALUES = list(AREAS.keys()) + ["OUT_OF_SCOPE", "UNCLEAR"]
FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "area": {
            "type": "string",
            "enum": ENUM_VALUES,
        }
    },
    "required": ["area"],
    "additionalProperties": False,
}

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You classify pages from a national government portal into one EU Single Digital \
Gateway Annex I information area. The page may be in German, Dutch, English or \
Portuguese -- read it in its own language, do not translate.

Read and weigh the ENTIRE page before deciding. Do NOT classify based on the first \
topic or first paragraph alone. If different parts of the page cover different areas, \
judge which single area the page as a WHOLE is about.

Some pages -- especially local/municipal pages -- list several unrelated topics \
together (e.g. traffic, dogs, waste, snow clearing, refuse, contact details). \
Such pages have no single dominant area: answer UNCLEAR. \
Do NOT pick the first topic mentioned.

These areas concern citizens' and businesses' RIGHTS, OBLIGATIONS, and ADMINISTRATIVE \
PROCEDURES. If a page is only loosely or superficially related to an area -- for \
example recreational activities (such as cycling or hiking in nature), general \
background, or local notices -- and is NOT about a person's rights, obligations, or \
an administrative procedure, answer OUT_OF_SCOPE (or UNCLEAR if it is government \
content but fits no single area). Do NOT force a page into the nearest area just \
because a keyword matches (e.g. a page about riding a bicycle in a forest is NOT \
area C "Vehicles").

Reply with ONLY a single token -- one letter A-Q, OUT_OF_SCOPE, or UNCLEAR:
- A single letter A-Q: the page clearly fits that one Annex I area.
- OUT_OF_SCOPE: the page is NOT citizen-facing rights/procedures/services content \
(e.g. news, press releases, organisation or contact directories, \
internal or institutional government documents, recreational content).
- UNCLEAR: the page IS citizen-facing government content but does not clearly fit \
any single A-Q area, spans multiple unrelated areas with no dominant one, \
or has too little text to judge.

Do not force a weak guess into a letter -- use UNCLEAR instead. \
Reply with NOTHING else: no label, no punctuation, no explanation.

Available areas:
A -- Travel: identity and travel documents (ID card, passport, visa); rights of \
passengers travelling by plane, train, ship, or bus, and of people buying travel \
packages; assistance for travellers with reduced mobility; rules on transporting \
animals, plants, alcohol, tobacco, and other goods; rules on calls, electronic \
messages, and data while travelling.
B -- Work and retirement: seeking and taking up employment; recognition of \
qualifications for employment; taxation related to work; liability and mandatory \
insurance linked to employment; terms and conditions of employment (working hours, \
paid leave, holidays, overtime, health checks, termination, dismissal, redundancy); \
equal treatment and non-discrimination at work, equal pay; workplace health and safety \
obligations; social security rights and obligations, including pensions.
C -- Vehicles: registering a motor vehicle; acquiring and renewing a driving licence; \
mandatory motor vehicle insurance; buying and selling a motor vehicle; traffic rules \
and requirements for drivers, including use of road infrastructure (vignette, tolls, \
emission stickers).
D -- Residence: moving home and registering residence; purchasing, selling, owning, or \
using property (including taxation and secondary residences); participating in \
municipal and European elections; residence cards and registration requirements; \
naturalisation and citizenship; rules applicable in the case of death, including \
repatriation of remains.
E -- Education or traineeship: the education system (early childhood, primary, \
secondary, higher, and adult education); volunteering; traineeships; research as part \
of an education programme.
F -- Healthcare: getting medical treatment; buying prescribed pharmaceuticals; health \
insurance rules, including the European Health Insurance Card; access to public \
preventive healthcare; emergency numbers (112, 116); moving to a residential care home.
G -- Citizens' and family rights: birth, custody, parental responsibility, surrogacy, \
adoption, child maintenance; marriage, civil and registered partnerships, separation, \
divorce, marital property, cohabitants' rights, including same-sex couples; gender \
recognition; succession and inheritance, including related tax; parental child \
abduction.
H -- Consumer rights: buying goods, digital content, or services (including financial \
services), online or in person; holding a bank account; connection to utilities (gas, \
electricity, water, waste disposal, telecoms, internet); payments and credit transfers; \
consumer rights, guarantees, dispute resolution, and compensation; safety of consumer \
products; renting a motor vehicle.
I -- Protection of personal data: exercising data subjects' rights regarding the \
protection of personal data.
J -- Starting, running, and closing a business: registering, changing the legal form \
of, or closing a business; intellectual property (patents, trademarks, designs, \
licences); fair and transparent commercial practices; online payment facilities for \
selling goods/services; contract law and late-payment rules; insolvency and \
liquidation; credit insurance; mergers and selling a business; civil liability of \
company directors; processing of personal data by businesses.
K -- Employees (employer-side): terms of employment set by law (working hours, leave, \
overtime, health checks, termination, dismissals, redundancies); social security \
obligations (registering as an employer, registering employees, notifying end of \
contract, paying contributions, pensions); employing workers, posting of workers, \
residency requirements for workers; equal treatment and equal pay; staff \
representation.
L -- Taxes: VAT (rules, rates, exemptions, registration, payment, refunds); excise \
duties (rules, rates, exemptions, registration, payment, refunds); customs duties and \
import taxes; customs procedures for imports and exports; other taxes (payment, rates, \
tax returns).
M -- Goods: obtaining CE marking; product rules and requirements; standards, technical \
specifications, and product certification; mutual recognition of products; \
classification, labelling, and packaging of hazardous chemicals; distance and \
off-premises selling obligations; defective products, guarantees, and redress; \
certification and labels (EMAS, energy labels, Eco-design, EU Eco-label); recycling \
and waste management.
N -- Services (business): acquiring licences, authorisations, or permits to start and \
operate a business; notifying authorities of business activities; recognition of \
professional qualifications, including vocational education and training.
O -- Funding a business: access to finance and funding programmes and grants; \
initiatives for entrepreneurs (exchanges, mentoring programmes, and similar).
P -- Public contracts: participating in public tenders (rules and procedures); \
submitting bids in response to public calls for tender; reporting irregularities in \
the tender process.
Q -- Health and safety at work: health and safety obligations for different types of \
activity, including risk prevention, information, and training.\
"""


def build_user_text(row):
    title = str(row.get("title", "") or "").strip()
    body  = str(row.get("text_joined", "") or "").strip()
    text  = (f"Page title: {title}\n\n{body}").strip() if title else body
    return text[:MAX_CHARS]


def parse_reply(raw):
    """
    Parse model reply into (assigned_area, assigned_label).

    The reply is expected to be a single token. Handles both plain-text
    responses and JSON objects from Ollama structured output.
    Label is ALWAYS derived from the AREAS dict -- never from model text.
    Returns (None, None) if the reply cannot be parsed to a valid area.
    """
    text = raw.strip()

    # Structured-output path: JSON object {"area": "A"} or plain JSON string "A"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            text = str(obj.get("area", "")).strip()
        elif isinstance(obj, str):
            text = obj.strip()
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip code fences, quotes, stray punctuation, then uppercase
    text = re.sub(r"^```.*?```$", "", text, flags=re.DOTALL).strip()
    text = text.strip("`\"'").strip().upper()

    if text == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE", ""
    if text == "UNCLEAR":
        return "UNCLEAR", ""

    # Accept a bare valid letter, or a letter followed by anything (ignore the rest)
    if len(text) == 1 and text in AREAS:
        return text, AREAS[text]

    return None, None   # caller will retry once, then default to UNCLEAR


def classify_page(client, user_text):
    """
    Call the model, parse the reply. On an unparse-able response, retry once,
    then default to UNCLEAR. No PARSE_ERROR bucket -- these were scraped as
    in-scope content so UNCLEAR is the honest fallback.
    """
    for attempt in range(2):
        raw  = call_model(client, user_text)
        area, label = parse_reply(raw)
        if area is not None:
            return area, label
        if attempt == 0:
            print(f"    [unparse-able: {raw[:60]!r}, retrying once...]", flush=True)
    return "UNCLEAR", ""


def call_model(client, user_text, max_retries=3):
    """
    Send user_text to the model. Tries the structured-output format schema
    (Ollama 0.3+) first; falls back gracefully if the installed version does
    not support the format parameter.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat(
                    model=MODEL,
                    messages=messages,
                    format=FORMAT_SCHEMA,
                    options=OLLAMA_OPTIONS,
                )
            except TypeError:
                # Older Ollama Python package does not accept format=
                resp = client.chat(
                    model=MODEL,
                    messages=messages,
                    options=OLLAMA_OPTIONS,
                )
            return resp["message"]["content"].strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/{max_retries-1} after {wait}s: {exc}]",
                      flush=True)
                time.sleep(wait)
            else:
                raise


def load_done_urls(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("url") or "").strip()
            if u:
                done.add(u)
    return done


def print_summary(csv_path):
    if not os.path.exists(csv_path):
        print("No output file found.")
        return

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_rows  = len(df)
    total_oos   = (df["assigned_area"] == "OUT_OF_SCOPE").sum()
    total_unc   = (df["assigned_area"] == "UNCLEAR").sum()
    total_notxt = (df["assigned_area"] == "NO_TEXT").sum()

    non_scope = {"OUT_OF_SCOPE", "UNCLEAR", "NO_TEXT"}

    for country in COUNTRIES:
        cdf = df[df["country"] == country]
        if len(cdf) == 0:
            continue
        n = len(cdf)
        print(f"\n── {country.upper()} (n={n})")

        # In-scope area distribution
        in_scope = cdf[~cdf["assigned_area"].isin(non_scope)]
        if len(in_scope):
            print("  In-scope area distribution:")
            for area, cnt in in_scope["assigned_area"].value_counts().items():
                label = AREAS.get(area, area)
                print(f"    {area}  {cnt:4d}  {label}")

        oos = (cdf["assigned_area"] == "OUT_OF_SCOPE").sum()
        unc = (cdf["assigned_area"] == "UNCLEAR").sum()
        print(f"  OUT_OF_SCOPE:  {oos:4d} ({100*oos/n:.1f}%)")
        print(f"  UNCLEAR:       {unc:4d} ({100*unc/n:.1f}%)")
        print(f"  Median word count: {cdf['word_count'].median():.0f}")

        # Top-15 raw categories by UNCLEAR count
        unc_by_cat = (
            cdf[cdf["assigned_area"] == "UNCLEAR"]["raw_category"]
            .value_counts()
            .head(15)
        )
        if len(unc_by_cat):
            print(f"  Top raw_categories by UNCLEAR count:")
            for cat, cnt in unc_by_cat.items():
                cat_total = (cdf["raw_category"] == cat).sum()
                pct = 100 * cnt / cat_total if cat_total else 0
                print(f"    {cat:<45}  UNCLEAR={cnt:4d} / {cat_total:4d} ({pct:.0f}%)")

    # Overall
    print(f"\n── OVERALL (n={total_rows})")
    print(f"  OUT_OF_SCOPE:  {total_oos:5d} ({100*total_oos/total_rows:.1f}%)")
    print(f"  UNCLEAR:       {total_unc:5d} ({100*total_unc/total_rows:.1f}%)")
    if total_notxt:
        print(f"  NO_TEXT:       {total_notxt:5d} ({100*total_notxt/total_rows:.1f}%)")
    in_scope_total = total_rows - total_oos - total_unc - total_notxt
    print(f"  In-scope (A-Q):{in_scope_total:5d} ({100*in_scope_total/total_rows:.1f}%)")

    # Portugal pt_inicio content check
    pt     = df[df["country"] == "portugal"]
    inicio = pt[pt["raw_category"] == "pt_inicio"]
    if len(inicio):
        n_i  = len(inicio)
        med  = inicio["word_count"].median()
        u200 = (inicio["word_count"] < 200).sum()
        print(f"\n── PORTUGAL pt_inicio content check (n={n_i})")
        print(f"  Median word count:       {med:.0f}")
        print(f"  Under 200 words:         {u200}/{n_i} ({100*u200/n_i:.1f}%)")
        print("  Area distribution:")
        for area, cnt in inicio["assigned_area"].value_counts().items():
            label = AREAS.get(area, area)
            print(f"    {area}  {cnt:4d}  {label}")


def main():
    import ollama as ollama_lib

    parser = argparse.ArgumentParser(description="SDG zero-shot categorisation.")
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Classify only N random pages per country (random_state=42). "
             "Writes to sdg_categorization_sample_N.csv instead of the full file.",
    )
    cli = parser.parse_args()

    # ── Verify Ollama is reachable and model is available ──────────────────────
    print(f"Checking Ollama + model '{MODEL}'...")
    try:
        models_resp = ollama_lib.list()
        available   = [m["model"] for m in models_resp.get("models", [])]
        available_base = [m.split(":")[0] for m in available]
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama ({exc}).")
        print("Make sure Ollama is running: `ollama serve`")
        sys.exit(1)

    model_base = MODEL.split(":")[0]
    if MODEL not in available and model_base not in available_base:
        print(f"Model '{MODEL}' is not pulled.")
        print(f"Run:  ollama pull {MODEL}")
        sys.exit(1)
    print(f"  Model '{MODEL}' is available.")

    try:
        v = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        ollama_server_version = v.stdout.strip()
    except Exception:
        ollama_server_version = "unknown"
    try:
        import ollama as _ol
        ollama_pkg_version = getattr(_ol, "__version__", "unknown")
    except Exception:
        ollama_pkg_version = "unknown"

    print(f"  Ollama server: {ollama_server_version}")
    print(f"  ollama Python package: {ollama_pkg_version}")
    print()

    # ── Column names ───────────────────────────────────────────────────────────
    print("Columns and text sources per country:")
    for country in COUNTRIES:
        path = os.path.join(BASE_DATA, country,
                            f"02_scrape_output_{country}", f"combined_{country}.csv")
        with open(path, encoding="utf-8") as f:
            cols = next(csv.reader(f))
        text_src = "title + text_joined" if "title" in cols else "text_joined"
        print(f"  {country:<12}: {cols}")
        print(f"               text source: {text_src}")
    print()

    # ── Load data (and optionally sample) ──────────────────────────────────────
    country_frames = {}
    for country in COUNTRIES:
        path = os.path.join(BASE_DATA, country,
                            f"02_scrape_output_{country}", f"combined_{country}.csv")
        df = pd.read_csv(path, dtype=str).fillna("")
        df["word_count"] = pd.to_numeric(df.get("word_count", 0),
                                         errors="coerce").fillna(0).astype(int)
        if cli.sample is not None:
            df = df.sample(n=min(cli.sample, len(df)), random_state=42).reset_index(drop=True)
        country_frames[country] = df

    out_csv = (os.path.join(SCRIPT_DIR, f"sdg_categorization_sample_{cli.sample}.csv")
               if cli.sample is not None else OUT_CSV)

    total_rows = sum(len(df) for df in country_frames.values())
    mode_label = f"SAMPLE ({cli.sample} per country)" if cli.sample is not None else "FULL"
    print(f"Mode:                    {mode_label}")
    print(f"Total pages to classify: {total_rows}")
    print(f"Output CSV:              {out_csv}")
    print()

    # ── Resume ─────────────────────────────────────────────────────────────────
    done_urls = load_done_urls(out_csv)
    if done_urls:
        print(f"Resuming: {len(done_urls)} URLs already classified, skipping them.")

    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    out_fh = open(out_csv, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=OUT_FIELDS)
    if write_header:
        writer.writeheader()
        out_fh.flush()

    t_start        = time.time()
    new_classified = 0
    timing_done    = False

    try:
        for country in COUNTRIES:
            df = country_frames[country]
            n_todo = sum(1 for u in df["url"] if u not in done_urls)
            if n_todo == 0:
                print(f"── {country.upper()}: all {len(df)} rows already done, skipping.")
                continue
            print(f"── {country.upper()}: {len(df)} total, {n_todo} to classify...")

            for _, row in df.iterrows():
                url = str(row.get("url", "")).strip()
                if url in done_urls:
                    continue

                user_text = build_user_text(row)
                snippet   = user_text.replace("\n", " ")[:SNIPPET_LEN]

                if not user_text.strip():
                    area, label = "NO_TEXT", ""
                else:
                    area, label = classify_page(ollama_lib, user_text)

                rec = {
                    "country":        country,
                    "raw_category":   str(row.get("category", "")),
                    "url":            url,
                    "word_count":     int(row.get("word_count", 0)),
                    "assigned_area":  area,
                    "assigned_label": label,
                    "snippet":        snippet,
                }
                writer.writerow(rec)
                out_fh.flush()
                done_urls.add(url)
                new_classified += 1

                if new_classified == TIMING_CHECKPOINT and not timing_done:
                    elapsed  = time.time() - t_start
                    avg_sec  = elapsed / new_classified
                    remaining = total_rows - len(done_urls)
                    print(f"\n  Timing after {TIMING_CHECKPOINT} pages:")
                    print(f"    Elapsed:         {elapsed:.1f}s")
                    print(f"    Avg per page:    {avg_sec:.2f}s")
                    print(f"    Pages remaining: {remaining}")
                    print(f"    Projected time:  {remaining*avg_sec/60:.0f} min  "
                          f"({remaining*avg_sec/3600:.1f}h)")
                    print(f"  Continuing... (Ctrl-C to pause and resume later)\n",
                          flush=True)
                    timing_done = True

                if new_classified % 100 == 0:
                    elapsed  = time.time() - t_start
                    avg_sec  = elapsed / new_classified
                    remaining = total_rows - len(done_urls)
                    print(f"  [{country}] {len(done_urls)}/{total_rows} done  "
                          f"avg {avg_sec:.2f}s/page  "
                          f"~{remaining*avg_sec/60:.0f} min left",
                          flush=True)

            print(f"  {country} complete.")

    except KeyboardInterrupt:
        print(f"\nInterrupted. {len(done_urls)} URLs classified so far.")
        print("Re-run to resume from where it stopped.")
    finally:
        out_fh.close()

    elapsed_total = time.time() - t_start
    print(f"\nClassified {new_classified} new pages in {elapsed_total/60:.1f} min.")
    print(f"\nMethods info:")
    print(f"  MODEL:                 {MODEL}")
    print(f"  MAX_CHARS:             {MAX_CHARS}")
    print(f"  Ollama server version: {ollama_server_version}")
    print(f"  ollama Python package: {ollama_pkg_version}")
    print(f"  temperature=0, seed=42, num_predict=50")

    print_summary(out_csv)


if __name__ == "__main__":
    main()
