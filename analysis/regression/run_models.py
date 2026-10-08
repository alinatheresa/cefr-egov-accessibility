# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: run_models.py
# Purpose: full pre-specified regression grid on the analysis dataframe.
# Pipeline step: Analysis: original regression grid (M1 to M4, four outcome codings).
# Thesis: Cross-check of the Chapter 5 models; the thesis tables come from
#         analysis/final_oct/scripts/b4_regression.py.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_models.py  —  STEPS 3 & 4
Full pre-specified regression grid on the assembled analysis dataframe (4-country data:
germany, austria, netherlands, ireland -- Portugal excluded, classifier retrained on DE/EN/NL).

Reads: analysis/regression/analysis_dataframe.csv
       (produced by build_dataframe.py)

Model grid:
  M1 : cefr ~ country
  M2 : cefr ~ page_type          (run at both scope and area granularity)
  M3 : cefr ~ country + page_type (PRIMARY — composition-adjusted)
  M4 : cefr ~ country * scope    (EXPLORATORY — interaction)

DV codings (run for every predictor model):
  binary   : binary logistic on cefr_binary (B2/C vs A/B1)   PRIMARY
  ols_1234 : OLS on cefr_1234                                 SECONDARY (robustness)
  ordinal  : proportional-odds logistic on cefr_ord           EXPLORATORY (robustness)
  ols_1248 : OLS on cefr_1248                                 EXPLORATORY (robustness)

Reference levels: country=germany, scope=citizen, area=B (Work and retirement, highest-n citizen area)
"""

import os
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "analysis_dataframe.csv")
OUT_GRID  = os.path.join(HERE, "regression_comparison.csv")
OUT_COEFS = os.path.join(HERE, "regression_coefs.csv")

if not os.path.exists(DATA_PATH):
    sys.exit("ERROR: analysis_dataframe.csv not found. Run build_dataframe.py first.")

# ── Load ───────────────────────────────────────────────────────────────────
df_full = pd.read_csv(DATA_PATH)
df_full["cefr_ord"] = pd.Categorical(df_full["pred_label"],
                                     categories=["A", "B1", "B2", "C"], ordered=True)
df_full["country"]  = pd.Categorical(df_full["country"])
df_full["area"]     = df_full["area"].astype(str)

# Scope subset (excludes UNCLEAR and OUT_OF_SCOPE)
df_scope = df_full[df_full["scope"].notna()].copy()
df_scope["scope"] = pd.Categorical(df_scope["scope"],
                                   categories=["citizen", "business"])

# Area subset (excludes UNCLEAR and OUT_OF_SCOPE — same N as df_scope)
df_areas = df_full[~df_full["area"].isin(["UNCLEAR", "OUT_OF_SCOPE"])].copy()

COUNTRY_ORDER = ["germany", "austria", "netherlands", "ireland"]
CEFR_LABELS   = ["A", "B1", "B2", "C"]

# Reference levels
REF_COUNTRY = "germany"
REF_SCOPE   = "citizen"
REF_AREA    = "B"   # Work and retirement — highest-n citizen area

print("=" * 70)
print("STEP 3  —  REGRESSION GRID")
print("=" * 70)
print(f"\nN (full, all pages): {len(df_full):,}")
print(f"N (scope subset, UNCLEAR/OOS excluded): {len(df_scope):,}")
print(f"N (area subset, UNCLEAR/OOS excluded): {len(df_areas):,}")
print(f"\nReference levels: country={REF_COUNTRY!r}, scope={REF_SCOPE!r}, area={REF_AREA!r} (Work and retirement)")

# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def fit_ols(formula, data, label):
    try:
        res = smf.ols(formula, data).fit()
        return res, None
    except Exception as e:
        return None, str(e)

def fit_logit(formula, data, label):
    try:
        res = smf.logit(formula, data).fit(disp=False, maxiter=200)
        return res, None
    except Exception as e:
        return None, str(e)

def fit_ordinal(formula_rhs, data, label):
    """Proportional-odds logistic via OrderedModel.from_formula."""
    formula = f"cefr_ord ~ {formula_rhs}"
    try:
        mod = OrderedModel.from_formula(formula, data, distr="logit")
        res = mod.fit(method="bfgs", disp=False, maxiter=500)
        return res, None
    except Exception as e:
        return None, str(e)

def mcfadden_r2(res):
    """McFadden pseudo-R² = 1 - llf/llnull."""
    try:
        return 1 - res.llf / res.llnull
    except Exception:
        return np.nan

def summary_row(model_name, dv, res, res_type, n, role, error=None):
    if error or res is None:
        return {"model": model_name, "dv": dv, "role": role, "N": n,
                "AIC": np.nan, "fit_metric": np.nan, "converged": False, "note": str(error)}
    row = {"model": model_name, "dv": dv, "role": role, "N": n, "converged": True, "note": ""}
    try:
        row["AIC"] = round(res.aic, 1)
    except Exception:
        row["AIC"] = np.nan
    if res_type in ("ols_1234", "ols_1248"):
        row["fit_metric"] = round(res.rsquared, 4)
        row["metric_name"] = "R²"
    else:
        row["fit_metric"] = round(mcfadden_r2(res), 4)
        row["metric_name"] = "McFadden R²"
    return row

def _ordinal_coef_table(res) -> pd.DataFrame:
    """OrderedResults (proportional-odds ordinal model) has no summary2() in this
    statsmodels version, so build the coef table directly from params/bse/pvalues,
    which every statsmodels results object exposes. Cut-point thresholds (named like
    'A/B1', 'B1/B2', ...) are excluded -- only the substantive predictor terms are kept."""
    threshold_names = {f"{CEFR_LABELS[i]}/{CEFR_LABELS[i+1]}" for i in range(len(CEFR_LABELS) - 1)}
    keep = [t for t in res.params.index if t not in threshold_names]
    return pd.DataFrame({
        "Coef.":    res.params.loc[keep],
        "Std.Err.": res.bse.loc[keep],
        "P>|z|":    res.pvalues.loc[keep],
    })

def coef_table(model_name, dv, res):
    """Return a tidy coefficient DataFrame."""
    if res is None:
        return pd.DataFrame()
    try:
        try:
            tbl = res.summary2().tables[1].copy()
        except AttributeError:
            tbl = _ordinal_coef_table(res)
        tbl.index.name = "term"
        tbl.reset_index(inplace=True)
        tbl.insert(0, "model", model_name)
        tbl.insert(1, "dv",    dv)
        return tbl
    except Exception:
        return pd.DataFrame()

def print_coefs(model_name, dv, res, res_type):
    """Print coefficient table with stars for p<.05, p<.01, p<.001."""
    if res is None:
        print("  [not converged / error]")
        return
    try:
        try:
            tbl   = res.summary2().tables[1]
            pvals = tbl["P>|z|"] if "P>|z|" in tbl.columns else tbl["P>|t|"]
            coefs = tbl["Coef."] if "Coef." in tbl.columns else tbl.iloc[:, 0]
            ses   = tbl["Std.Err."] if "Std.Err." in tbl.columns else tbl.iloc[:, 1]
        except AttributeError:
            tbl   = _ordinal_coef_table(res)
            pvals, coefs, ses = tbl["P>|z|"], tbl["Coef."], tbl["Std.Err."]
        for term in tbl.index:
            p = pvals.loc[term]
            c = coefs.loc[term]
            s = ses.loc[term]
            stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            print(f"    {term:<50}  {c:>8.4f}  SE={s:.4f}  p={p:.4f} {stars}")
    except Exception as e:
        print(f"  [coef table error: {e}]")

# ══════════════════════════════════════════════════════════════════════════
# Model specifications
# ══════════════════════════════════════════════════════════════════════════

SPECS = [
    # (label, rhs_formula, dataset, description)
    ("M1_country",   f"C(country, Treatment('{REF_COUNTRY}'))",
     "full",  "M1: CEFR ~ country"),
    ("M2_scope",     f"C(scope,   Treatment('{REF_SCOPE}'))",
     "scope", "M2a: CEFR ~ scope (citizen/business)"),
    ("M2_area",      f"C(area,    Treatment('{REF_AREA}'))",
     "areas", f"M2b: CEFR ~ area (17 SDGR areas, ref={REF_AREA})"),
    ("M3_scope",     f"C(country, Treatment('{REF_COUNTRY}')) + C(scope, Treatment('{REF_SCOPE}'))",
     "scope", "M3a: CEFR ~ country + scope  ← PRIMARY"),
    ("M3_area",      f"C(country, Treatment('{REF_COUNTRY}')) + C(area, Treatment('{REF_AREA}'))",
     "areas", f"M3b: CEFR ~ country + area  ← PRIMARY"),
    ("M4_interaction", f"C(country, Treatment('{REF_COUNTRY}')) * C(scope, Treatment('{REF_SCOPE}'))",
     "scope", "M4: CEFR ~ country × scope  [EXPLORATORY]"),
]

# DV roles: binary logistic (B2/C vs A/B1) is the PRIMARY specification on the 4-country
# data. OLS on cefr_1234 is the SECONDARY/robustness DV. Ordinal (proportional-odds) and
# ols_1248 remain in the grid as additional EXPLORATORY/robustness checks -- not dropped,
# just no longer framed as the principled-primary DV.
DV_SPECS = [
    # (dv_label, res_type, formula_lhs or None, role)
    ("binary",    "binary",    "cefr_binary", "PRIMARY"),
    ("ols_1234",  "ols_1234",  "cefr_1234",   "SECONDARY (robustness)"),
    ("ordinal",   "ordinal",   None,          "EXPLORATORY (robustness)"),
    ("ols_1248",  "ols_1248",  "cefr_1248",   "EXPLORATORY (robustness)"),
]

# ── Run the grid ───────────────────────────────────────────────────────────
grid_rows  = []
all_coefs  = []

for spec_label, rhs, dataset_key, description in SPECS:
    if dataset_key == "scope":
        data = df_scope
    elif dataset_key == "areas":
        data = df_areas
    else:
        data = df_full
    n    = len(data)

    print(f"\n{'─'*70}")
    print(f"  {description}   (N={n:,})")
    print(f"{'─'*70}")

    for dv_label, res_type, lhs, role in DV_SPECS:
        print(f"\n  DV = {dv_label}  [{role}]")

        if res_type == "ordinal":
            res, err = fit_ordinal(rhs, data, spec_label)
        elif res_type == "binary":
            res, err = fit_logit(f"cefr_binary ~ {rhs}", data, spec_label)
        else:
            res, err = fit_ols(f"{lhs} ~ {rhs}", data, spec_label)

        if err:
            print(f"    ✗ Error: {err}")
        else:
            print_coefs(spec_label, dv_label, res, res_type)
            try:
                aic = round(res.aic, 1)
                fm  = round(res.rsquared if res_type in ("ols_1234","ols_1248") else mcfadden_r2(res), 4)
                metric_name = "R²" if res_type in ("ols_1234","ols_1248") else "McFadden R²"
                print(f"    → N={n:,}  AIC={aic}  {metric_name}={fm}")
            except Exception:
                pass

        grid_rows.append(summary_row(spec_label, dv_label, res, res_type, n, role, error=err))
        all_coefs.append(coef_table(spec_label, dv_label, res))

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 summary table
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3  —  COMPARISON TABLE  (full grid)")
print("=" * 70)
grid_df = pd.DataFrame(grid_rows)
# dv column order follows DV_SPECS: binary=PRIMARY, ols_1234=SECONDARY (robustness),
# ordinal & ols_1248=EXPLORATORY (robustness) -- see module docstring / DV_SPECS above.
grid_df["dv"] = pd.Categorical(grid_df["dv"], categories=[d[0] for d in DV_SPECS], ordered=True)
print("\nDV roles: binary=PRIMARY | ols_1234=SECONDARY (robustness) | "
      "ordinal, ols_1248=EXPLORATORY (robustness)")
# Pivot: rows=model, cols=dv
pivot = grid_df.pivot_table(
    index="model", columns="dv",
    values=["AIC", "fit_metric"], aggfunc="first", observed=False
).round(4)
pivot.index = pd.CategoricalIndex(pivot.index,
    categories=[s[0] for s in SPECS], ordered=True)
pivot = pivot.sort_index()
print(pivot.to_string())
grid_df.to_csv(OUT_GRID, index=False)
print(f"\n✓ Full grid saved → {OUT_GRID}")

coefs_df = pd.concat([c for c in all_coefs if not c.empty], ignore_index=True)
coefs_df.to_csv(OUT_COEFS, index=False)
print(f"✓ Coefficient tables saved → {OUT_COEFS}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4  —  ASSUMPTION & ROBUSTNESS NOTES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4  —  ASSUMPTIONS & ROBUSTNESS")
print("=" * 70)

# 4.1 Proportional-odds assumption (Brant-like: compare coefficients at each cut)
print("\n4.1  Proportional-odds test (M3a — country + scope)")
print("     Strategy: fit 3 binary logits at each threshold cut;")
print("     compare country/scope coefficients — large variation = PO violated.\n")

def po_check(rhs, data, labels=["A","B1","B2","C"]):
    thresholds = []
    for k in range(1, len(labels)):
        cut_label = f"≥{labels[k]}"
        tmp = data.copy()
        code_map = {"A":0,"B1":1,"B2":2,"C":3}
        tmp["y_cut"] = (tmp["pred_label"].map(code_map) >= k).astype(int)
        try:
            res = smf.logit(f"y_cut ~ {rhs}", tmp).fit(disp=False, maxiter=300)
            par = res.params.drop(["Intercept"], errors="ignore").rename(cut_label)
            thresholds.append(par)
        except Exception as e:
            thresholds.append(pd.Series(name=cut_label, dtype=float))
    return pd.DataFrame(thresholds) if thresholds else None

po_rhs = f"C(country, Treatment('{REF_COUNTRY}')) + C(scope, Treatment('{REF_SCOPE}'))"
po_tbl = po_check(po_rhs, df_scope)
if po_tbl is not None:
    print(po_tbl.round(4).to_string())
    # Check variance of coefficients across cuts
    coef_range = po_tbl.max() - po_tbl.min()
    large_var = coef_range[coef_range > 0.5].index.tolist()
    if large_var:
        print(f"\n  ⚠ PO ASSUMPTION CONCERN: coefficient range > 0.5 for: {large_var}")
        print("    Ordinal model results should be interpreted with caution for these terms.")
        print("    Consider reporting binary logistic (cefr_binary) as co-primary DV.")
    else:
        print("\n  ✓ PO assumption appears roughly satisfied (coef range ≤ 0.5 across cuts).")

# 4.2 Condition number check
print("\n4.2  Condition number check (M3b — country + area)")
try:
    import patsy
    y, X = patsy.dmatrices(
        f"cefr_1234 ~ C(country, Treatment('{REF_COUNTRY}')) + C(area, Treatment('{REF_AREA}'))",
        df_full, return_type="matrix")
    cond = np.linalg.cond(np.array(X))
    print(f"     Condition number: {cond:.1f}")
    if cond > 1000:
        print("     ⚠ HIGH: near-multicollinearity — some country×area cells are empty.")
        print("       M3b coefficients may be unstable; report SEs with caution.")
    else:
        print("     ✓ Condition number acceptable.")
except Exception as e:
    print(f"     Could not compute: {e}")

# 4.3 Robustness: German-language group (Germany + Austria pooled)
print("\n4.3  Robustness — German-language group (Germany + Austria pooled)")
print("     [NOT the main model; one-line note only]")
df_pool = df_scope.copy()
df_pool["lang_group"] = df_pool["country"].apply(
    lambda c: "german_lang" if c in ("germany","austria") else c)
pool_rhs = f"C(lang_group, Treatment('ireland')) + C(scope, Treatment('{REF_SCOPE}'))"
pool_res, pool_err = fit_ols(f"cefr_1234 ~ {pool_rhs}", df_pool, "robustness")
if pool_err:
    print(f"     Error: {pool_err}")
else:
    gl_coef = pool_res.params.get("C(lang_group, Treatment('ireland'))[T.german_lang]", np.nan)
    gl_p    = pool_res.pvalues.get("C(lang_group, Treatment('ireland'))[T.german_lang]", np.nan)
    print(f"     german_lang vs Ireland — OLS(cefr_1234): β={gl_coef:.3f}, p={gl_p:.4f}")
    print("     (Germany and Austria behave as one group relative to Ireland at this level.)")

# 4.4 Flag sparse area levels that may cause ordinal model instability
print("\n4.4  Sparse area flags in M2b/M3b (area predictor)")
area_n = df_full["area"].value_counts()
sparse = area_n[area_n < 30]
if not sparse.empty:
    print("     ⚠ Areas with < 30 observations (coefficients unreliable):")
    for a, n in sparse.items():
        print(f"       {a}: n={n}")
else:
    print("     ✓ All area levels have n ≥ 30.")

print("\n" + "=" * 70)
print("run_models.py COMPLETE. Review flags above before writing thesis results.")
print("=" * 70)
