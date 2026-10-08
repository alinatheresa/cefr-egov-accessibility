# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b2_unread_text.py
# Purpose: B2 unread text. (1) Character truncation in the zero-shot categoriser
#          (build_user_text cuts title+text to MAX_CHARS=6000). (2) Prompt length in qwen2.5
#          tokens against the Ollama default context, using the tokenizer embedded in the
#          locally installed qwen2.5:7b GGUF. (3) Ollama log search. Reads files only; makes
#          no model calls.
# Pipeline step: Analysis: text not seen by the zero-shot categoriser (needs the page texts, the
#                qwen2.5 tokenizer and the local Ollama logs).
# Thesis: Chapter 5, tab:unread-text.
#
# Usage: python b2_unread_text.py --qwen-tokenizer <dir with tokenizer.json>
#        The tokenizer dir is produced once by extracting it from the Ollama GGUF blob with
#        transformers (AutoTokenizer.from_pretrained(dir, gguf_file=...)); see RESULTS notes.

import os
import re
import sys
import glob
import argparse
import importlib.util

import numpy as np
import pandas as pd

from common import (PROJECT_ROOT, COUNTRIES, COUNTRY_LABEL, TAB_DIR, RES_DIR, load_text,
                    load_sdg, tex_table, fmt_int, save_json)

ap = argparse.ArgumentParser()
ap.add_argument("--qwen-tokenizer", required=True)
ap.add_argument("--ollama-logs", default=os.path.expanduser("~/.ollama/logs"))
args = ap.parse_args()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The corpus labels carry an assigned_reasoning column and deterministic-rule rows, so they
# were produced by the reasoning version of the script (analysis/regression/). The older
# analysis/sdg/ version is the one eval_zeroshot_areas.py imported for the 90-page re-run.
sdg_full = load_module(os.path.join(PROJECT_ROOT, "analysis", "regression",
                                    "sdg_categorize_full.py"), "sdg_reasoning")
sdg_old = load_module(os.path.join(PROJECT_ROOT, "analysis", "sdg",
                                   "sdg_categorize_full.py"), "sdg_single_token")
MAX_CHARS = sdg_full.MAX_CHARS
assert MAX_CHARS == 6000
print("OLLAMA_OPTIONS (corpus run script):", sdg_full.OLLAMA_OPTIONS)
print("num_ctx set:", "num_ctx" in sdg_full.OLLAMA_OPTIONS)

# ── 1. Character truncation ──────────────────────────────────────────────────
text = load_text()
sdg = load_sdg()
text = text.merge(sdg[["url", "assigned_area", "rule"]], on="url", how="left",
                  validate="one_to_one")


def full_user_text(row):
    title = str(row["title"] or "").strip()
    body = str(row["text_joined"] or "").strip()
    return (f"Page title: {title}\n\n{body}").strip() if title else body


text["user_full"] = text.apply(full_user_text, axis=1)
text["len_full"] = text["user_full"].str.len()
text["len_seen"] = text["len_full"].clip(upper=MAX_CHARS)
text["len_body"] = text["text_joined"].str.strip().str.len()
text["model_call"] = ~text["rule"] & (text["len_full"] > 0)


def char_stats(d):
    return {
        "n_pages": len(d),
        "pct_pages_over_6000": 100 * (d["len_full"] > MAX_CHARS).mean(),
        "n_pages_over_6000": int((d["len_full"] > MAX_CHARS).sum()),
        "pct_chars_unseen": 100 * (1 - d["len_seen"].sum() / d["len_full"].sum()),
        "median_chars": float(d["len_full"].median()),
        "median_body_chars": float(d["len_body"].median()),
        "p90_chars": float(d["len_full"].quantile(0.9)),
    }


rows = []
for c in COUNTRIES + ["total"]:
    d = text if c == "total" else text[text["country"] == c]
    r = {"country": "Total" if c == "total" else COUNTRY_LABEL[c]}
    r.update(char_stats(d))
    m = char_stats(d[d["model_call"]])
    r.update({f"modelcalls_{k}": v for k, v in m.items()})
    rows.append(r)
chars = pd.DataFrame(rows)
chars.to_csv(os.path.join(TAB_DIR, "b2_unread_text_chars.csv"), index=False, float_format="%.2f")
print(chars.T.to_string())

# ── 2. Token length with the qwen2.5 tokenizer ──────────────────────────────
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(args.qwen_tokenizer)


def prompt_tokens(system_prompt, user_text):
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return len(ids)


# 2a. Check the method against the Ollama log: the 26 August validation re-run sent the 90
#     annotation pages with the older single-token prompt; the server logged task.n_tokens.
logs = sorted(glob.glob(os.path.join(args.ollama_logs, "server*.log")))
logged = []
for p in logs:
    with open(p, errors="replace") as f:
        for line in f:
            m = re.search(r"new prompt, n_ctx_slot = (\d+), n_keep = (\d+), task.n_tokens = (\d+)", line)
            if m:
                logged.append({"log": os.path.basename(p), "n_ctx": int(m.group(1)),
                               "n_keep": int(m.group(2)), "n_tokens": int(m.group(3))})
logged = pd.DataFrame(logged)
print(f"\nLogged prompts: {len(logged)}; logs: {[os.path.basename(p) for p in logs]}")

ann = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis", "sampling",
                               "manual_annotation_sample_ANNOTATED.csv"),
                  sep=None, engine="python", dtype=str).fillna("")
ann_tok = []
for _, r in ann.iterrows():
    title = r["title"].strip()
    body = r["text"].strip()
    ut = (f"Page title: {title}\n\n{body}").strip() if title else body
    ann_tok.append(prompt_tokens(sdg_old.SYSTEM_PROMPT, ut[:6000]))
check = {}
if len(logged) >= 90:
    lg = logged[logged["log"] == "server-2.log"]["n_tokens"].to_numpy()
    if len(lg) == 90:
        diff = lg - np.array(ann_tok)
        check = {"n_requests": 90, "n_exact_match": int((diff == 0).sum()),
                 "diff_min": int(diff.min()), "diff_max": int(diff.max()),
                 "logged_max": int(lg.max()), "n_ctx": int(logged["n_ctx"].iloc[-1]),
                 "n_keep": int(logged["n_keep"].iloc[-1])}
print("Tokenizer check against log:", check)

# 2b. Corpus prompts (reasoning prompt) for all pages that went to the model
sys_tokens = prompt_tokens(sdg_full.SYSTEM_PROMPT, "")
sys_tokens_old = prompt_tokens(sdg_old.SYSTEM_PROMPT, "")
print(f"System prompt + template, empty page: corpus prompt={sys_tokens}, older prompt={sys_tokens_old}")

mc = text[text["model_call"]].copy()
mc["user_seen"] = mc["user_full"].str[:MAX_CHARS]
mc["prompt_tokens"] = [prompt_tokens(sdg_full.SYSTEM_PROMPT, u) for u in mc["user_seen"]]
mc["page_tokens_seen"] = [len(tok.encode(u, add_special_tokens=False)) for u in mc["user_seen"]]
NUM_PREDICT = sdg_full.OLLAMA_OPTIONS["num_predict"]

tok_rows = []
for c in COUNTRIES + ["total"]:
    d = mc if c == "total" else mc[mc["country"] == c]
    t = d["prompt_tokens"]
    tok_rows.append({
        "country": "Total" if c == "total" else COUNTRY_LABEL[c],
        "n_model_calls": len(d),
        "median_prompt_tokens": float(t.median()),
        "max_prompt_tokens": int(t.max()),
        "n_over_4096": int((t > 4096).sum()),
        "pct_over_4096": 100 * (t > 4096).mean(),
        "n_over_4096_minus_num_predict": int((t > 4096 - NUM_PREDICT).sum()),
        "pct_over_4096_minus_num_predict": 100 * (t > 4096 - NUM_PREDICT).mean(),
        "n_over_2048": int((t > 2048).sum()),
        "pct_over_2048": 100 * (t > 2048).mean(),
    })
toks = pd.DataFrame(tok_rows)
toks.to_csv(os.path.join(TAB_DIR, "b2_prompt_tokens.csv"), index=False, float_format="%.2f")
print(toks.T.to_string())

# Full 6,000-character page: tokens for a page that hits the cut, per language
full_pages = mc[mc["len_full"] >= MAX_CHARS]
per_lang = (full_pages.groupby("country")["page_tokens_seen"]
            .agg(["count", "median", "min", "max"]).reindex(COUNTRIES))
print("\nTokens of a 6,000-character user text (pages at the cut):\n", per_lang)

# ── 3. Log search ───────────────────────────────────────────────────────────
hits_trunc_prompt, versions = [], []
for p in sorted(glob.glob(os.path.join(args.ollama_logs, "*.log"))):
    with open(p, errors="replace") as f:
        txt = f.read()
    hits_trunc_prompt += [(os.path.basename(p), l) for l in txt.splitlines()
                          if "truncating input prompt" in l]
    first = re.search(r"time=(\S+)", txt)
    ver = re.findall(r"version=(\d+\.\d+\.\d+)", txt)
    ctx = re.findall(r"default_num_ctx=(\d+)", txt)
    tr1 = len(re.findall(r"truncated = 1", txt))
    versions.append({"log": os.path.basename(p), "first_timestamp": first.group(1) if first else "",
                     "versions": sorted(set(ver)), "default_num_ctx": sorted(set(ctx)),
                     "n_truncated_1": tr1})
print("\n'truncating input prompt' hits:", len(hits_trunc_prompt))
for v in versions:
    print(v)

save_json({
    "max_chars": MAX_CHARS,
    "ollama_options": sdg_full.OLLAMA_OPTIONS,
    "num_ctx_set": "num_ctx" in sdg_full.OLLAMA_OPTIONS,
    "system_prompt_tokens_with_template": sys_tokens,
    "system_prompt_tokens_older_prompt": sys_tokens_old,
    "tokenizer_check_against_log": check,
    "tokens_6000_char_page_by_country": per_lang.reset_index().to_dict(orient="records"),
    "log_truncating_input_prompt_hits": len(hits_trunc_prompt),
    "logs": versions,
}, "b2_unread_text.json")

# LaTeX table
body = []
for r, t in zip(rows, tok_rows):
    body.append([
        r["country"],
        f"{r['median_chars']:,.0f}",
        f"{r['pct_pages_over_6000']:.1f}",
        f"{r['pct_chars_unseen']:.1f}",
        fmt_int(t["n_model_calls"]),
        f"{t['median_prompt_tokens']:,.0f}",
        f"{t['pct_over_4096_minus_num_predict']:.1f}",
        f"{t['pct_over_4096']:.1f}",
    ])
tex_table(
    os.path.join(TAB_DIR, "b2_unread_text.tex"),
    caption="Text not seen by the zero-shot categoriser",
    label="tab:unread-text",
    header=["", "Median chars", r"$>$6,000 chars (\%)", r"Chars unseen (\%)",
            "Model calls", "Median prompt tokens", r"$>$3,840 tokens (\%)",
            r"$>$4,096 tokens (\%)"],
    rows=body,
    midrules=(4,),
    notes=[
        "Character columns use all corpus pages and the categoriser input string "
        "(``Page title: '' plus title plus body), which build\\_user\\_text() cuts at 6,000 characters. "
        "Chars unseen is the share of all input characters beyond the cut.",
        "Token columns cover pages sent to the model (excluding the 749 deterministic-rule pages) and "
        "count the full chat prompt (system prompt, template and the cut page) with the qwen2.5 "
        "tokenizer embedded in the local qwen2.5:7b model file.",
        f"4,096 is the Ollama default context logged on this machine; 3,840 leaves room for "
        f"num\\_predict = {NUM_PREDICT} generated tokens.",
    ],
    colspec="lrrrrrrr",
    size=r"\small",
)
