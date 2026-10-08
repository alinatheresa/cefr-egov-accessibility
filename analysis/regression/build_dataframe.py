# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: build_dataframe.py
# Purpose: assemble the per-page analysis dataframe and print the descriptives.
# Pipeline step: Analysis: joins the CEFR labels and the SDGR areas into analysis_dataframe.csv.
# Thesis: Input to all Chapter 5 regressions (estimation sample N = 7,359).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataframe.py  —  STEPS 0, 1, 2
Assembles the per-page analysis dataframe and prints all descriptives.

CEFR source : fine-tuned mDeBERTa  (run cefr_20260817_231925 / mdeberta_cefr_multi, DE/EN/NL)
              country_pipelines/*/03_cefr_output_*/*_cefr.csv  (pred_label column)
Page type   : analysis/sdg/sdg_categorization_full.csv  (assigned_area column, SDGR Annex I)
Join key    : url

Outputs:
  analysis/regression/analysis_dataframe.csv
"""

import os
import sys
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

CLF_FILES = {
    "germany":     os.path.join(PROJECT_ROOT, "country_pipelines", "germany",
                                "03_cefr_output_germany",     "germany_cefr.csv"),
    "austria":     os.path.join(PROJECT_ROOT, "country_pipelines", "austria",
                                "03_cefr_output_austria",     "austria_cefr.csv"),
    "netherlands": os.path.join(PROJECT_ROOT, "country_pipelines", "netherlands",
                                "03_cefr_output_netherlands", "netherlands_cefr.csv"),
    "ireland":     os.path.join(PROJECT_ROOT, "country_pipelines", "ireland",
                                "03_cefr_output_ireland",     "ireland_cefr.csv"),
    # Portugal dropped -- classifier retrained on DE/EN/NL only; the 4-country pipeline
    # covers germany/austria/netherlands/ireland.
}
SDG_PATH = os.path.join(PROJECT_ROOT, "analysis", "sdg", "sdg_categorization_full.csv")
OUT_PATH = os.path.join(HERE, "analysis_dataframe.csv")

# ── Constants ──────────────────────────────────────────────────────────────
CEFR_LABELS   = ["A", "B1", "B2", "C"]
CITIZEN_AREAS = set("ABCDEFGHI")
BUSINESS_AREAS = set("JKLMNOPQ")

CEFR_1234 = {"A": 1, "B1": 2, "B2": 3, "C": 4}
CEFR_1248 = {"A": 1, "B1": 2, "B2": 4, "C": 8}
CEFR_BIN  = {"A": 0, "B1": 0, "B2": 1, "C": 1}   # 0=easy, 1=hard

COUNTRY_ORDER = ["germany", "austria", "netherlands", "ireland"]

# ══════════════════════════════════════════════════════════════════════════
# STEP 0  —  SOURCE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 0  —  SOURCE VERIFICATION")
print("=" * 70)

sdg_raw = pd.read_csv(SDG_PATH)
print(f"\nSDG CSV columns : {list(sdg_raw.columns)}")

cefr_like = [c for c in sdg_raw.columns
             if any(x in c.lower() for x in ["cefr", "pred_label", "level"])]
if cefr_like:
    sys.exit(f"\nFATAL — sdg CSV contains a CEFR-like column: {cefr_like}. "
             "Check your data sources before continuing.")

print("✓  No CEFR column in sdg_categorization_full.csv")
print(f"   assigned_area sample : {sorted(sdg_raw['assigned_area'].dropna().unique())}")
print(f"   assigned_label sample: {sdg_raw['assigned_label'].dropna().unique()[:4].tolist()}")
print("\n✓  CEFR will be read exclusively from the fine-tuned classifier CSVs")
print("   Model : cefr_20260817_231925 / mdeberta_cefr_multi  (deberta-v2, 4-label, DE/EN/NL)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1  —  LOAD & JOIN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1  —  ASSEMBLE ANALYSIS DATAFRAME")
print("=" * 70)

# 1a. Load classifier predictions
clf_frames = []
for country, path in CLF_FILES.items():
    df_c = pd.read_csv(path)
    before = len(df_c)
    df_c = df_c[df_c["pred_label"].isin(CEFR_LABELS)].copy()
    df_c["country"] = country
    clf_frames.append(df_c[["country", "url", "pred_label"]])
    print(f"  {country:<12}: {before:>5} rows loaded, {len(df_c):>5} with valid pred_label")

clf = pd.concat(clf_frames, ignore_index=True)
print(f"\n  Total classifier rows : {len(clf):,}")

# 1b. Load SDG areas
sdg = (sdg_raw[["url", "assigned_area", "assigned_label"]]
       .rename(columns={"assigned_area": "area", "assigned_label": "area_label"}))

# 1c. Join on url (left join from classifier side)
df = clf.merge(sdg, on="url", how="left")
n_matched   = df["area"].notna().sum()
n_unmatched = df["area"].isna().sum()

print(f"\n  Join (left, CEFR ← SDG area):")
print(f"    Matched   : {n_matched:,}")
print(f"    Unmatched : {n_unmatched:,}")

if n_unmatched > 0:
    miss = df[df["area"].isna()].groupby("country").size()
    print(f"\n  ⚠ Unmatched by country:\n{miss.to_string()}")
    print("  → Dropping unmatched rows.")
    df = df[df["area"].notna()].copy()
else:
    print("  ✓ 100% URL match — no rows dropped from join.")

# 1d. Build derived variables
df["cefr_ord"]    = pd.Categorical(df["pred_label"],
                                   categories=CEFR_LABELS, ordered=True)
df["cefr_1234"]   = df["pred_label"].map(CEFR_1234).astype(float)
df["cefr_1248"]   = df["pred_label"].map(CEFR_1248).astype(float)
df["cefr_binary"] = df["pred_label"].map(CEFR_BIN).astype(float)

def assign_scope(area):
    if area in CITIZEN_AREAS:  return "citizen"
    if area in BUSINESS_AREAS: return "business"
    return None                # UNCLEAR / OUT_OF_SCOPE

df["scope"] = df["area"].apply(assign_scope)

# Scope exclusion report
n_no_scope = df["scope"].isna().sum()
print(f"\n  Scope variable — rows excluded (UNCLEAR / OUT_OF_SCOPE): {n_no_scope:,}")
excl = df[df["scope"].isna()].groupby(["country", "area"]).size().reset_index(name="n")
print(excl.to_string(index=False))
print(f"  Rows usable for scope models: {len(df) - n_no_scope:,}")

# 1e. Save
col_order = ["country", "url", "pred_label", "cefr_ord", "cefr_1234", "cefr_1248",
             "cefr_binary", "area", "area_label", "scope"]
df[col_order].to_csv(OUT_PATH, index=False)
print(f"\n✓  Saved → {OUT_PATH}  ({len(df):,} rows × {len(col_order)} columns)")

print("\nHead (first 5 rows):")
print(df[col_order].head().to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# STEP 2  —  DESCRIPTIVES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2  —  DESCRIPTIVES")
print("=" * 70)

COUNTRIES = [c for c in COUNTRY_ORDER if c in df["country"].unique()]

# 2.1 Overall CEFR distribution
print("\n2.1  CEFR level counts — OVERALL")
overall = df["pred_label"].value_counts().reindex(CEFR_LABELS, fill_value=0)
overall_pct = (overall / overall.sum() * 100).round(1)
summary_overall = pd.DataFrame({"n": overall, "pct": overall_pct})
print(summary_overall.to_string())
for lvl, row in summary_overall.iterrows():
    if row["n"] < 100:
        print(f"  ⚠ FLAG: {lvl} is very sparse overall (n={row['n']}) — "
              "justifies binary DV and warrants caution for ordinal model.")

# 2.2 CEFR by country
print("\n2.2  CEFR counts by country")
ct_country = (pd.crosstab(df["country"], df["pred_label"])
              .reindex(index=COUNTRIES, columns=CEFR_LABELS, fill_value=0))
ct_country["Total"] = ct_country.sum(axis=1)
print(ct_country.to_string())

# Flag sparse within-country cells
for ctry in COUNTRIES:
    for lvl in CEFR_LABELS:
        n_cell = ct_country.loc[ctry, lvl]
        if n_cell < 30:
            print(f"  ⚠ FLAG: {ctry} × {lvl} has only n={n_cell} — sparse cell.")

# 2.3 CEFR by SDGR area
print("\n2.3  CEFR counts by SDGR area")
all_areas = sorted(df["area"].unique())
ct_area = (pd.crosstab(df["area"], df["pred_label"])
           .reindex(index=all_areas, columns=CEFR_LABELS, fill_value=0))
ct_area["Total"] = ct_area.sum(axis=1)
ct_area["scope"] = [assign_scope(a) if assign_scope(a) else "excluded" for a in ct_area.index]
print(ct_area.to_string())
sparse_areas = ct_area[ct_area["Total"] < 30].index.tolist()
if sparse_areas:
    print(f"\n  ⚠ FLAG: areas with Total < 30: {sparse_areas}")
    print("  → Coefficients for these areas will be unreliable; flag in thesis.")

# 2.4 Country × scope
print("\n2.4  Country × scope cell counts  (UNCLEAR/OUT_OF_SCOPE excluded)")
df_scope = df[df["scope"].notna()].copy()
ct_cs = pd.crosstab(df_scope["country"], df_scope["scope"],
                    margins=True, margins_name="Total")
ct_cs = ct_cs.reindex(COUNTRIES + ["Total"])
print(ct_cs.to_string())

sparse_cs = [(c, s) for c in COUNTRIES
             for s in ["citizen", "business"]
             if ct_cs.loc[c, s] < 30]
if sparse_cs:
    print(f"  ⚠ FLAG — country×scope cells < 30: {sparse_cs}")

# 2.5 Country × area (top-level summary; full table saved to CSV)
print("\n2.5  Country × area cell counts (selected; full table in analysis_dataframe.csv)")
ct_ca = pd.crosstab(df["country"], df["area"]).reindex(COUNTRIES)
ct_ca["Total"] = ct_ca.sum(axis=1)
print(ct_ca.to_string())
sparse_ca = [(c, a) for c in COUNTRIES
             for a in all_areas
             if ct_ca.loc[c, a] < 30]
print(f"\n  Country×area cells < 30: {len(sparse_ca)} of {len(COUNTRIES)*len(all_areas)}")
if sparse_ca:
    print("  (these predictors will be numerically unstable — flagged for thesis)")
    for c, a in sparse_ca:
        print(f"    {c} × {a}: n={ct_ca.loc[c, a]}")

print("\n" + "=" * 70)
print("build_dataframe.py COMPLETE — run run_models.py for the regression grid.")
print("=" * 70)
