# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: 00_prepare_data.py
# Purpose: load and clean the CEFR data, build the stratified split, tokenize, and save what the next steps need.
# Pipeline step: Classifier, step 00 (UniversalCEFR data preparation; German, English, Dutch).
# Thesis: Section 4.2.1 (training data; Portuguese removed). Tables tab:clf-data and
#         tab:clf-classes are built from 00_prepare_output/*.json by b6_classifier_tables.py.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
import hashlib
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# C1/C2 are merged into C and A1/A2 into A -- the finer distinctions are not
# consistent across datasets and we only need 4 levels for this classifier
LABELS = ["A", "B1", "B2", "C"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

LANG2ID = {"de": 0, "en": 1, "nl": 2}

# Each language group has its own HuggingFace paths within UniversalCEFR
DE_SUBSETS = ["UniversalCEFR/merlin_de", "UniversalCEFR/elg_cefr_de"]
EN_SUBSETS = ["UniversalCEFR/elg_cefr_en", "UniversalCEFR/icle500_en", "UniversalCEFR/cambridge_exams_en"]
NL_SUBSETS = ["UniversalCEFR/elg_cefr_nl"]

# Different datasets use different column names for the actual text --
# this tries common names in priority order before falling back to any string column
def pick_text_field(row: dict) -> str | None:
    txt = None
    for key in ("text", "sentence", "content", "value"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            txt = row[key]
            break
    if txt is None:
        for _, v in row.items():
            if isinstance(v, str) and v.strip():
                txt = v
                break
            if isinstance(v, list) and any(isinstance(i, str) for i in v):
                txt = " ".join([i for i in v if isinstance(i, str)])
                break
    if isinstance(txt, list):
        txt = " ".join([t for t in txt if isinstance(t, str)])
    if isinstance(txt, str):
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def normalize_label(y: str | None, lang: str | None = None) -> str | None:
    # Collapse CEFR to 4 labels for ALL languages, keeping every rating. Any plus variant maps
    # to its base bin (A1+/A2+ -> A, B1+ -> B1, B2+ -> B2, C1+/C2+ -> C); A1/A2 -> A; C1/C2 -> C.
    # lang stays in the signature for compatibility but is no longer used.
    if not isinstance(y, str):
        return None
    y = y.strip().upper()
    base = y[:-1] if y.endswith("+") else y
    if base in {"A1", "A2"}:
        return "A"
    if base == "B1":
        return "B1"
    if base == "B2":
        return "B2"
    if base in {"C1", "C2"}:
        return "C"
    return base if base in LABELS else None

# Each UniversalCEFR subset has slightly different schemas and noise --
# this normalises them all into the same (text, cefr_level) format
def load_and_clean(name: str, lang: str | None = None, _stats: dict | None = None) -> DatasetDict:
    ds = load_dataset(name)
    _n_raw  = sum(len(ds[s]) for s in ds)
    _n_plus = sum(
        1 for _s in ds.values() for _row in _s
        if isinstance(_row.get("cefr_level", _row.get("label", "")), str)
        and "+" in _row.get("cefr_level", _row.get("label", ""))
    )
    def _map(batch):
        n = len(next(iter(batch.values())))
        out_text, out_level = [], []
        for i in range(n):
            row = {k: batch[k][i] for k in batch}
            txt = pick_text_field(row)
            lab = normalize_label(row.get("cefr_level", row.get("label", None)), lang=lang)
            out_text.append(txt)
            out_level.append(lab)
        return {"text": out_text, "cefr_level": out_level}
    ds = ds.map(_map, batched=True)
    def _ok(x):
        return isinstance(x.get("text"), str) and x["text"].strip() and (x.get("cefr_level") in LABELS)
    ds = ds.filter(_ok)
    _n_clean = sum(len(ds[s]) for s in ds)
    print(f"    {name}: downloaded={_n_raw}  plus_level_labels={_n_plus}  kept={_n_clean}  removed_NA_or_invalid={_n_raw - _n_clean}")
    if _stats is not None:
        _stats[name] = {"n_raw": _n_raw, "n_plus": _n_plus,
                        "n_clean": _n_clean, "n_removed_na": _n_raw - _n_clean}
    return ds

def add_lang_column(dset: Dataset, lang_code: str) -> Dataset:
    if "lang" in dset.column_names:
        dset = dset.remove_columns("lang")
    return dset.add_column("lang", [lang_code] * len(dset))

# Stratifying by (language, label) together ensures every split has proportional
# coverage of all four languages at all four CEFR levels -- a plain random split
# would risk e.g. NL being mostly in train with barely any in val/test
def stratified_3way_split(dset: Dataset, seed=42, train=0.8, val=0.1, test=0.1) -> DatasetDict:
    assert abs(train + val + test - 1.0) < 1e-9
    labels = list(dset["cefr_level"])
    langs = list(dset["lang"])
    strata = [f"{l}|{c}" for l, c in zip(langs, labels)]
    idx_all = np.arange(len(dset))
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=(1.0 - train), random_state=seed, stratify=strata
    )
    temp_keys = [strata[i] for i in idx_temp]
    val_prop = val / (val + test)
    idx_val_rel, idx_test_rel = train_test_split(
        np.arange(len(idx_temp)), test_size=(1.0 - val_prop), random_state=seed, stratify=temp_keys
    )
    idx_val = [int(idx_temp[i]) for i in idx_val_rel]
    idx_test = [int(idx_temp[i]) for i in idx_test_rel]
    return DatasetDict({
        "train": dset.select(list(map(int, idx_train))),
        "validation": dset.select(idx_val),
        "test": dset.select(idx_test),
    })

def aggregate_labels(dset: Dataset, drop_full_spread: bool = False):
    # Collapse to exactly one row per unique (text, lang) using the ordinal median of all
    # raters' 4-class labels (A=0..C=3). Even rater counts take the lower (simpler) central
    # level. This resolves multi-rater disagreement AND removes every exact duplicate, so each
    # text survives once with a single label and no copy can leak across splits.
    # drop_full_spread=True additionally drops texts whose raters span the whole scale (A..C)
    # with no majority (the maximal-disagreement ties); default keeps them on the median.
    ord_ = {l: i for i, l in enumerate(LABELS)}
    inv = {i: l for i, l in enumerate(LABELS)}
    buckets = defaultdict(list)
    for txt, lang, lab in zip(dset["text"], dset["lang"], dset["cefr_level"]):
        buckets[(txt, lang)].append(ord_[lab])

    _n_single       = sum(1 for v in buckets.values() if len(v) == 1)
    _n_dup_same     = sum(1 for v in buckets.values() if len(v) > 1 and len(set(v)) == 1)
    _n_dup_conflict = sum(1 for v in buckets.values() if len(v) > 1 and len(set(v)) > 1)
    _extra_same     = sum(len(v) - 1 for v in buckets.values() if len(v) > 1 and len(set(v)) == 1)
    _extra_conflict = sum(len(v) - 1 for v in buckets.values() if len(v) > 1 and len(set(v)) > 1)
    print(f"  Dedup breakdown (unique text+lang keys: {len(buckets)}):")
    print(f"    Appeared once, single rating:              {_n_single:5d}")
    print(f"    Appeared >1x, identical label (exact dup): {_n_dup_same:5d}  extra rows removed:    {_extra_same}")
    print(f"    Appeared >1x, conflicting labels (median): {_n_dup_conflict:5d}  extra rows collapsed:  {_extra_conflict}")
    texts, langs, labels = [], [], []
    n_dropped_full = 0
    for (txt, lang), ords in buckets.items():
        ords = sorted(ords)
        n = len(ords)
        if drop_full_spread and (ords[-1] - ords[0] == 3):
            counts = Counter(ords); top = max(counts.values())
            if sum(1 for v in counts.values() if v == top) > 1:
                n_dropped_full += 1
                continue
        mid = ords[n // 2] if n % 2 == 1 else ords[n // 2 - 1]
        texts.append(txt); langs.append(lang); labels.append(inv[mid])

    agg = Dataset.from_dict({"text": texts, "lang": langs, "cefr_level": labels})

    keys = list(zip(agg["text"], agg["lang"]))
    assert len(keys) == len(set(keys)), \
        f"aggregate_labels produced {len(keys) - len(set(keys))} duplicate (text, lang) keys"

    stats = {"rows_in": len(dset), "rows_out": len(agg),
             "unique_keys": len(set(keys)), "dropped_full_spread": n_dropped_full}
    return agg, stats

def count_by_lang_and_label(ds: Dataset) -> dict:
    by = defaultdict(lambda: {k: 0 for k in LABELS})
    for lang, lab in zip(ds["lang"], ds["cefr_level"]):
        by[lang][lab] += 1
    return dict(by)

# Packed mode uses a sliding window -- long texts produce multiple overlapping
# chunks instead of being cut off, which keeps more of the content for training
def preprocess_batch(batch, tokenizer, max_len: int, stride: int, indices=None):
    out = {"input_ids": [], "attention_mask": [], "labels": [], "sample_id": [], "lang_id": []}
    texts = batch["text"]
    labels = batch["cefr_level"]
    langs = batch.get("lang", [""] * len(texts))
    for i, (txt, y) in enumerate(zip(texts, labels)):
        base_idx = indices[i] if indices is not None else i
        lid = LANG2ID.get(langs[i], -1)
        enc = tokenizer(
            txt, truncation=True, padding="max_length", max_length=max_len,
            return_overflowing_tokens=True, stride=stride
        )
        n = len(enc["input_ids"])
        out["input_ids"].extend(enc["input_ids"])
        out["attention_mask"].extend(enc["attention_mask"])
        out["labels"].extend([label2id[y]] * n)
        out["sample_id"].extend([int(base_idx)] * n)
        out["lang_id"].extend([lid] * n)
    return out

# B1 and B2 have more examples than A and C, so we upweight the rarer classes
# to stop the model from just getting good at the common ones
def compute_class_weights(train_ds: Dataset) -> dict:
    counts = Counter(train_ds["cefr_level"])
    total = sum(counts.values())
    weights = {lbl: (total / max(1, counts.get(lbl, 0))) for lbl in LABELS}
    return weights

def per_row_sampler_weights(tokenized_train: Dataset, label_weights: dict) -> torch.Tensor:
    labels = list(tokenized_train["labels"])
    arr = np.array([label_weights[LABELS[l]] for l in labels], dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32)

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare CEFR data and tokenization caches.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_prepare_output"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=== CEFR Data Preparation ===")
    print(f"Timestamp:            {ts}")
    print(f"Seed:                 {args.seed}")
    print(f"Tokenizer model:      {args.model}")
    print(f"Max length:           {args.max_len}")
    print(f"Stride:               {args.stride}")
    print("Subsets (DE):         " + ", ".join(DE_SUBSETS))
    print("Subsets (EN):         " + ", ".join(EN_SUBSETS))
    print("Subsets (NL):         " + ", ".join(NL_SUBSETS))

    # 1) Load and clean subsets (language code passed so normalize_label can apply language-specific rules)
    print("\nLoading and standardizing UniversalCEFR subsets...")
    _prep_stats: dict = {}
    de_parts = [load_and_clean(n, lang="de", _stats=_prep_stats)["train"] for n in DE_SUBSETS]
    en_parts = []
    for n in EN_SUBSETS:
        ds = load_and_clean(n, lang="en", _stats=_prep_stats)
        if "train" in ds and len(ds["train"]) > 0:
            en_parts.append(ds["train"])

    print("\n  Cross-dataset text overlap within language groups:")
    for _lang_name, _parts in [("DE", de_parts), ("EN", en_parts)]:
        if len(_parts) > 1:
            _text_sets = [set(p["text"]) for p in _parts]
            _overlap = sum(len(_text_sets[i] & _text_sets[j])
                          for i in range(len(_text_sets)) for j in range(i + 1, len(_text_sets)))
            print(f"    {_lang_name}: texts appearing in more than one subset = {_overlap}")
        else:
            print(f"    {_lang_name}: only 1 subset, no cross-dataset check needed")
    print(f"    NL: only 1 subset, no cross-dataset check needed")
    de = concatenate_datasets(de_parts) if de_parts else None
    en = concatenate_datasets(en_parts) if en_parts else None

    de = add_lang_column(de, "de")
    if en is not None:
        en = add_lang_column(en, "en")

    nl_parts = [load_and_clean(n, lang="nl", _stats=_prep_stats)["train"] for n in NL_SUBSETS]

    # Rater agreement computed from pipeline-normalised text.
    # Using parts (before concatenation) preserves per-dataset attribution.
    _ord_map = {"A": 0, "B1": 1, "B2": 2, "C": 3}
    _rater_agree: dict = {}
    for _pname, _part in [(NL_SUBSETS[0], nl_parts[0])]:
        _buckets: dict = defaultdict(list)
        for _t, _lb in zip(_part["text"], _part["cefr_level"]):
            _buckets[_t].append(_ord_map[_lb])
        _apart = Counter(max(v) - min(v) for v in _buckets.values())
        _rcnt  = Counter(len(v)           for v in _buckets.values())
        _rater_agree[_pname] = {
            "n_texts":    len(_buckets),
            "total_rows": sum(len(v) for v in _buckets.values()),
            "apart":      {str(k): int(v) for k, v in sorted(_apart.items())},
            "rater_dist": {str(k): int(v) for k, v in sorted(_rcnt.items())},
        }
    _prep_stats["rater_agreement"] = _rater_agree

    # EN within-dataset exact-duplicate counts (same text, same label, within one EN subset)
    _en_within: dict = {}
    for _pname, _part in zip(EN_SUBSETS, en_parts):
        _texts = list(_part["text"])
        _en_within[_pname] = len(_texts) - len(set(_texts))
    _prep_stats["en_within_dataset_dups"] = _en_within

    nl = concatenate_datasets(nl_parts) if nl_parts else None
    nl = add_lang_column(nl, "nl")

    parts = [d for d in (de, en, nl) if d is not None]
    all_ds = concatenate_datasets(parts)

    print(f"\nTotals after cleaning (before aggregation):")
    print(f"  DE: {len(de):5d}  EN: {len(en) if en is not None else 0:5d}  NL: {len(nl) if nl is not None else 0:5d}  ALL: {len(all_ds):5d}")

    print("\nAggregating multi-rater rows to one label per unique (text, lang)...")
    all_ds, agg_stats = aggregate_labels(all_ds, drop_full_spread=False)
    print(f"  rows {agg_stats['rows_in']} -> {agg_stats['rows_out']} unique texts"
          f" (dropped_full_spread={agg_stats['dropped_full_spread']})")
    per_lang = Counter(all_ds["lang"])
    print("  unique texts per language:", dict(sorted(per_lang.items())))

    # 2) Split 80/10/10 stratified by (lang, label)
    print("\nConstructing stratified 80/10/10 split by language and label...")
    split = stratified_3way_split(all_ds, seed=args.seed, train=0.8, val=0.1, test=0.1)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
    })
    print(f"Split sizes: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")

    def _split_keys(ds):
        return set(zip(ds["text"], ds["lang"]))
    k_tr, k_va, k_te = _split_keys(dataset["train"]), _split_keys(dataset["validation"]), _split_keys(dataset["test"])
    assert not (k_tr & k_va) and not (k_tr & k_te) and not (k_va & k_te), \
        "text leak across splits detected after aggregation"
    print("Cross-split leak check passed: train/val/test share 0 texts.")

    # 3) Class counts per language and per split
    print("\nLabel counts (merged C) by split and language:")
    counts_summary = {"by_split": {}, "overall": {}}
    for split_name in ("train", "validation", "test"):
        c = Counter(dataset[split_name]["cefr_level"])
        c_lang = count_by_lang_and_label(dataset[split_name])
        counts_summary["by_split"][split_name] = {
            "total": sum(c.values()),
            "by_label": {k: int(c.get(k, 0)) for k in LABELS},
            "by_lang_label": c_lang
        }
        print(f"  {split_name}: " + " ".join([f"{k}={c.get(k,0)}" for k in LABELS]))
    all_c = Counter(all_ds["cefr_level"])
    counts_summary["overall"] = {
        "total": int(sum(all_c.values())),
        "by_label": {k: int(all_c.get(k, 0)) for k in LABELS}
    }
    with open(os.path.join(args.out_dir, "label_counts_mergedC.json"), "w") as f:
        json.dump(counts_summary, f, indent=2, ensure_ascii=False)

    # 4) Save reproducible splits with stable sha256 hashes (Python's built-in hash() is
    # not stable across interpreter restarts, so sha256 over the text is used instead)
    print("\nSaving split indices and metadata...")
    splits_manifest = {
        "timestamp": ts,
        "seed": args.seed,
        "model": args.model,
        "max_len": args.max_len,
        "stride": args.stride,
        "labels": LABELS,
        "sizes": {k: len(v) for k, v in dataset.items()},
        "agg": agg_stats,
    }
    def row_hash(text: str, lang: str, lab: str) -> str:
        raw = f"{text}||{lang}||{lab}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    all_hashes = [row_hash(t, l, y) for t, l, y in zip(all_ds["text"], all_ds["lang"], all_ds["cefr_level"])]
    def hashes_for(split_ds):
        return [row_hash(t, l, y) for t, l, y in zip(split_ds["text"], split_ds["lang"], split_ds["cefr_level"])]
    splits_manifest["hashes"] = {
        "all": all_hashes,
        "train": hashes_for(dataset["train"]),
        "validation": hashes_for(dataset["validation"]),
        "test": hashes_for(dataset["test"]),
    }
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(splits_manifest, f, indent=2)

    # Save dedup_stats.json — structured record of every cleaning step for thesis verification
    _en_per_set   = [set(p["text"]) for p in en_parts]
    _en_all_texts = set(t for s in _en_per_set for t in s)
    _en_cross     = sum(len(s) for s in _en_per_set) - len(_en_all_texts)

    # Use short dataset names (strip "UniversalCEFR/" prefix) so verify_stats lookups work.
    # Also exclude the two helper keys that were stored inside _prep_stats.
    _rater_short  = {k.split("/")[-1]: v
                     for k, v in _prep_stats.get("rater_agreement", {}).items()}
    _within_short = {k.split("/")[-1]: v
                     for k, v in _prep_stats.get("en_within_dataset_dups", {}).items()}
    _per_ds_clean: dict = {}
    for _k, _v in _prep_stats.items():
        if _k in ("rater_agreement", "en_within_dataset_dups"):
            continue
        _short = _k.split("/")[-1]
        _entry = dict(_v)
        # n_final = post-all-dedup count (what the thesis calls "Final")
        if _short in _rater_short:              # multi-rater: final = unique texts
            _entry["n_final"] = _rater_short[_short]["n_texts"]
        elif _short in _within_short:           # EN: final = post-NA minus within-dups
            _entry["n_final"] = _entry["n_clean"] - _within_short.get(_short, 0)
        else:
            _entry["n_final"] = _entry["n_clean"]
        _per_ds_clean[_short] = _entry

    _en_short_names = [s.split("/")[-1] for s in EN_SUBSETS]
    _en_pairwise: dict = {}
    for _i in range(len(_en_per_set)):
        for _j in range(_i + 1, len(_en_per_set)):
            _key = f"{_en_short_names[_i]}+{_en_short_names[_j]}"
            _en_pairwise[_key] = len(_en_per_set[_i] & _en_per_set[_j])
    if len(_en_per_set) == 3:
        _en_pairwise["all_three"] = len(_en_per_set[0] & _en_per_set[1] & _en_per_set[2])

    _dedup_stats = {
        "per_dataset":           _per_ds_clean,
        "en_within_dups":        _within_short,
        "en_cross_dataset_dups": _en_cross,
        "en_pairwise_overlaps":  _en_pairwise,
        "rater_agreement":       _rater_short,
        "agg_stats":             agg_stats,
    }
    with open(os.path.join(args.out_dir, "dedup_stats.json"), "w") as f:
        json.dump(_dedup_stats, f, indent=2)
    print(f"  Dedup stats -> {os.path.join(args.out_dir, 'dedup_stats.json')}")

    # 5) Per-language truncation analysis -- hard numbers on how many docs exceed 512 tokens.
    # This justifies using packing (sliding window) instead of plain truncation.
    print(f"\nPer-language truncation analysis (limit={args.max_len} tokens):")
    tok_probe = AutoTokenizer.from_pretrained(args.model)
    for split_name in ("train", "validation", "test"):
        split_ds = dataset[split_name]
        lang_lengths: dict[str, list[int]] = defaultdict(list)
        for txt, lang in zip(split_ds["text"], split_ds["lang"]):
            n_tok = len(tok_probe(txt, add_special_tokens=True)["input_ids"])
            lang_lengths[lang].append(n_tok)
        print(f"  [{split_name}]")
        for lang_code in sorted(lang_lengths):
            lengths = lang_lengths[lang_code]
            n_total = len(lengths)
            n_trunc = sum(1 for l in lengths if l > args.max_len)
            avg_excess = (
                sum(l - args.max_len for l in lengths if l > args.max_len) / n_trunc
                if n_trunc else 0.0
            )
            pcts = np.percentile(lengths, [50, 90, 99]).tolist()
            print(
                f"    {lang_code}: n={n_total:4d}  truncated={n_trunc:3d}"
                f" ({n_trunc/n_total*100:5.1f}%)  avg_excess={avg_excess:6.1f} tok"
                f"  P50={pcts[0]:.0f}  P90={pcts[1]:.0f}  P99={pcts[2]:.0f}"
            )

    # 6) Tokenize with sliding window (packed) and save
    print("\nTokenizing and writing packed dataset to disk...")
    tokenizer = tok_probe
    save_tokenizer_dir = os.path.join(args.out_dir, "tokenizer")
    tokenizer.save_pretrained(save_tokenizer_dir)

    packed_dir = os.path.join(args.out_dir, "tokenized_packed")
    os.makedirs(packed_dir, exist_ok=True)
    tok_packed = DatasetDict({
        split_name: dataset[split_name].map(
            lambda b, idx=None: preprocess_batch(b, tokenizer, args.max_len, args.stride, indices=idx),
            batched=True,
            remove_columns=dataset[split_name].column_names,
            with_indices=True,
        )
        for split_name in ("train", "validation", "test")
    })
    tok_packed.save_to_disk(packed_dir)
    print(f"  Saved packed DatasetDict to {packed_dir}")

    # 7) Class weights and sampler weights (computed on document-level train split)
    print("\nComputing class weights on the train split...")
    class_w = compute_class_weights(dataset["train"])
    with open(os.path.join(args.out_dir, "class_weights.json"), "w") as f:
        json.dump(class_w, f, indent=2)
    print("  Class weights:", {k: round(v, 3) for k, v in class_w.items()})

    # Diagnostic: print and save the language x label count matrix and candidate alternative weights.
    # This is for thesis reporting only -- the training loss still uses label-only weights.
    print("\nComputing weight diagnostics...")
    lang_label_matrix = count_by_lang_and_label(dataset["train"])
    train_total = sum(sum(v.values()) for v in lang_label_matrix.values())
    lang_totals = {lang: sum(v.values()) for lang, v in lang_label_matrix.items()}
    n_langs = len(lang_totals)

    # Cap prevents extreme values in tiny cells (e.g. English A2 has very few examples
    # and without a cap its weight would dominate the gradient)
    cap = 5.0

    raw_lang_w = {lang: train_total / max(1, c) for lang, c in lang_totals.items()}
    capped_lang_w = {lang: min(w, cap) for lang, w in raw_lang_w.items()}
    sum_capped = sum(capped_lang_w.values())
    norm_lang_w = {lang: round(w / sum_capped * n_langs, 4) for lang, w in capped_lang_w.items()}

    capped_joint = {}
    for lang, counts in lang_label_matrix.items():
        for lbl in LABELS:
            raw = train_total / max(1, counts.get(lbl, 0))
            capped_joint[f"{lang}|{lbl}"] = round(min(raw, cap), 4)

    diagnostics = {
        "note": (
            "Label-only weights (class_weights.json) are used for the training loss. "
            "Language and joint weights here are candidates for future ablations only. "
            "English A2 has very few examples, so a cap of 5.0 is applied to prevent "
            "extreme weight values in tiny language-by-label cells."
        ),
        "lang_label_counts_train": {
            lang: {lbl: int(lang_label_matrix[lang].get(lbl, 0)) for lbl in LABELS}
            for lang in sorted(lang_label_matrix)
        },
        "label_only_weights": {k: round(v, 4) for k, v in class_w.items()},
        "candidate_language_weights_normalized_capped": norm_lang_w,
        "candidate_joint_weights_capped": capped_joint,
    }
    diag_path = os.path.join(args.out_dir, "weight_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Diagnostics saved to {diag_path}")
    print("  Language x label counts (train):")
    for lang in sorted(lang_label_matrix):
        print(f"    {lang}: " + "  ".join(f"{lbl}={lang_label_matrix[lang].get(lbl, 0):4d}" for lbl in LABELS))
    print("  Label-only weights (used in loss):", {k: round(v, 3) for k, v in class_w.items()})
    print("  Candidate language weights (normalized, capped at 5.0):", norm_lang_w)

    print("\nComputing per-row balanced sampler weights for packed dataset...")
    w_packed = per_row_sampler_weights(tok_packed["train"], class_w)
    torch.save(w_packed, os.path.join(args.out_dir, "balanced_sampler_weights_packed.pt"))

    # 8) Summary
    print("\nSummary")
    print(f"  Splits saved to:              {os.path.join(args.out_dir, 'splits.json')}")
    print(f"  Label counts saved to:        {os.path.join(args.out_dir, 'label_counts_mergedC.json')}")
    print(f"  Tokenizer saved to:           {save_tokenizer_dir}")
    print(f"  Packed dataset:               {packed_dir}")
    print(f"  Class weights:                {os.path.join(args.out_dir, 'class_weights.json')}")
    print(f"  Weight diagnostics:           {diag_path}")
    print(f"  Sampler weights (packed):     {os.path.join(args.out_dir, 'balanced_sampler_weights_packed.pt')}")
    print("\nDone.")

if __name__ == "__main__":
    main()
