# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b4_regression.py
# Purpose: B4 final regression. Binary logistic (B2/C = 1 vs A/B1 = 0) is primary.
#          M1, M2a, M2b, M3a, M3b, M4 on one estimation sample (citizen or business scope,
#          N = 7,359) so AIC is comparable; M1 on all labelled pages reported as descriptive.
#          Diagnostics (VIF/GVIF, cell counts, convergence and separation), sensitivity specs
#          (UNCLEAR kept; UNCLEAR kept without deterministic-rule pages) and appendix
#          robustness (OLS on 1-2-4-8, proportional-odds logit with Brant test).
#          Input: analysis/regression/analysis_dataframe.csv (built from run cefr_20260817_231925).
# Pipeline step: Analysis: final binary logistic regressions, diagnostics, UNCLEAR-as-own-scope
#                sensitivity, OLS and ordinal robustness.
# Thesis: Section 5.7 (tab:regression-main); robustness table (via ch5b_analysis.py); Annex
#         tab:regression-m3b-full, tab:regression-grid, tab:regression-m4,
#         tab:regression-sensitivity, tab:regression-robustness, tab:cells-country-area.

import os
import re
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (TAB_DIR, RES_DIR, ANALYSIS_DF, COUNTRIES, COUNTRY_LABEL, AREA_NAMES,
                    CITIZEN_AREAS, BUSINESS_AREAS, load_sdg, tex_table, fmt_p, fmt_int,
                    save_json)

Z = stats.norm.ppf(0.975)

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(ANALYSIS_DF)
assert len(df) == 8997, len(df)
sdg = load_sdg()
df = df.merge(sdg[["url", "rule"]], on="url", how="left", validate="one_to_one")
assert df["rule"].notna().all()
df["rule"] = df["rule"].astype(bool)

df["country"] = pd.Categorical(df["country"], categories=COUNTRIES)
df["y"] = df["cefr_binary"].astype(int)
df["cefr_ord"] = pd.Categorical(df["pred_label"], categories=["A", "B1", "B2", "C"], ordered=True)

main = df[df["scope"].notna()].copy()
main["scope"] = pd.Categorical(main["scope"], categories=["citizen", "business"])
assert len(main) == 7359

unc = df[df["area"] != "OUT_OF_SCOPE"].copy()
unc["scope3"] = pd.Categorical(unc["scope"].fillna("unclear"),
                               categories=["citizen", "business", "unclear"])
unc_norule = unc[~unc["rule"]].copy()

CTRY = "C(country, Treatment('germany'))"
SCP = "C(scope, Treatment('citizen'))"
SCP3 = "C(scope3, Treatment('citizen'))"
AREA = "C(area, Treatment('B'))"


def label_term(t):
    if t == "Intercept":
        return "Intercept"
    parts = []
    for piece in t.split(":"):
        m = re.search(r"C\((\w+),.*\)\[T\.([^\]]+)\]", piece)
        if not m:
            parts.append(piece)
            continue
        var, lvl = m.groups()
        if var == "country":
            parts.append(COUNTRY_LABEL[lvl])
        elif var in ("scope", "scope3"):
            parts.append(lvl.capitalize())
        elif var == "area":
            parts.append(f"{lvl}: {AREA_NAMES.get(lvl, lvl)}" if lvl != "UNCLEAR" else "UNCLEAR")
        else:
            parts.append(lvl)
    return " x ".join(parts)


fit_rows, coef_rows, warn_log = [], [], {}


def fit_logit(name, rhs, data, sample, role):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = smf.logit(f"y ~ {rhs}", data).fit(disp=False, maxiter=500, method="newton")
    msgs = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in w})
    conv = bool(res.mle_retvals.get("converged", False))
    big = res.params[(res.params.abs() > 10) | (res.bse > 10)].index.tolist()
    warn_log[name] = {"warnings": msgs, "converged": conv, "large_coef_or_se_terms": big}
    fit_rows.append({"model": name, "sample": sample, "role": role, "N": int(res.nobs),
                     "k": int(res.df_model) + 1, "loglik": res.llf, "ll_null": res.llnull,
                     "mcfadden_r2": 1 - res.llf / res.llnull, "aic": res.aic, "bic": res.bic,
                     "converged": conv, "n_warnings": len(msgs)})
    ci = res.conf_int()
    for t in res.params.index:
        coef_rows.append({"model": name, "sample": sample, "term": t, "label": label_term(t),
                          "coef": res.params[t], "se": res.bse[t], "z": res.tvalues[t],
                          "p": res.pvalues[t], "OR": np.exp(res.params[t]),
                          "OR_lo": np.exp(ci.loc[t, 0]), "OR_hi": np.exp(ci.loc[t, 1])})
    return res


print("=" * 70, "\nPrimary grid (binary logit, same estimation sample)\n", "=" * 70, sep="")
SPECS = [
    ("M1", CTRY), ("M2a", SCP), ("M2b", AREA), ("M3a", f"{CTRY} + {SCP}"),
    ("M3b", f"{CTRY} + {AREA}"), ("M4", f"{CTRY} * {SCP}"),
]
res = {}
for name, rhs in SPECS:
    res[name] = fit_logit(name, rhs, main, "main (N=7,359)", "primary grid")
res["M1_all"] = fit_logit("M1_all", CTRY, df, "all labelled (N=8,997)", "descriptive")

# Sensitivity
res["M3a_ii"] = fit_logit("M3a_ii", f"{CTRY} + {SCP3}", unc, "UNCLEAR kept", "sensitivity ii")
res["M3b_ii"] = fit_logit("M3b_ii", f"{CTRY} + {AREA}", unc, "UNCLEAR kept", "sensitivity ii")
res["M3a_iii"] = fit_logit("M3a_iii", f"{CTRY} + {SCP3}", unc_norule,
                           "UNCLEAR kept, no rule pages", "sensitivity iii")
res["M3b_iii"] = fit_logit("M3b_iii", f"{CTRY} + {AREA}", unc_norule,
                           "UNCLEAR kept, no rule pages", "sensitivity iii")

fits = pd.DataFrame(fit_rows)
coefs = pd.DataFrame(coef_rows)
fits.to_csv(os.path.join(TAB_DIR, "b4_model_fit.csv"), index=False, float_format="%.4f")
coefs.to_csv(os.path.join(TAB_DIR, "b4_coefficients_binary.csv"), index=False, float_format="%.6g")
print(fits.to_string())
for k, v in warn_log.items():
    print(k, v)


def get(model, label):
    r = coefs[(coefs.model == model) & (coefs.label == label)]
    return None if r.empty else r.iloc[0]


def or_cell(model, label):
    r = get(model, label)
    if r is None:
        return "", ""
    return f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]", fmt_p(r.p)


def fit_of(model):
    return fits[fits.model == model].iloc[0]


# ── Main table: country + scope (M3a) and country + areas (M3b) ─────────────
rows = []
for lab in ["Austria", "Netherlands", "Ireland", "Business"]:
    a, ap = or_cell("M3a", lab)
    b, bp = or_cell("M3b", lab)
    rows.append([lab, a, ap, b if b else "", bp if bp else ""])
rows.append(["17 SDGR areas (16 dummies)", "no", "", "yes", ""])
fa, fb = fit_of("M3a"), fit_of("M3b")
rows.append(["N", fmt_int(fa.N), "", fmt_int(fb.N), ""])
rows.append(["McFadden $R^2$", f"{fa.mcfadden_r2:.3f}", "", f"{fb.mcfadden_r2:.3f}", ""])
rows.append(["AIC", f"{fa.aic:,.1f}", "", f"{fb.aic:,.1f}", ""])
tex_table(
    os.path.join(TAB_DIR, "b4_main_table.tex"),
    caption="Binary logistic regression of harder text (B2 or C) on country and content type",
    label="tab:regression-main",
    header=["", "M3a OR [95\\% CI]", "p", "M3b OR [95\\% CI]", "p"],
    rows=rows,
    midrules=(4,),
    notes=[
        "Outcome: harder (predicted B2 or C, 1) versus easier (predicted A or B1, 0). Odds ratios with Wald 95\\% "
        "confidence intervals. References: Germany; citizen scope (M3a); area B, Work and "
        "retirement (M3b).",
        "Estimation sample: pages with a CEFR label and a citizen (A to I) or business (J to Q) "
        "area; 1,032 UNCLEAR and 606 OUT\\_OF\\_SCOPE pages are excluded. Area coefficients of M3b "
        "are in the appendix.",
    ],
    colspec="lrrrr",
)

# ── Full M3b table (appendix) ───────────────────────────────────────────────
rows = []
m3b = coefs[coefs.model == "M3b"]
order = (["Intercept"] + [COUNTRY_LABEL[c] for c in COUNTRIES[1:]]
         + [f"{a}: {AREA_NAMES[a]}" for a in CITIZEN_AREAS + BUSINESS_AREAS if a != "B"])
cnt_area = main["area"].value_counts()
for lab in order:
    r = m3b[m3b.label == lab].iloc[0]
    n_txt = ""
    if ":" in lab:
        n_txt = fmt_int(cnt_area[lab.split(":")[0]])
    rows.append([lab, n_txt, f"{r.coef:.3f}", f"{r.se:.3f}",
                 f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]", fmt_p(r.p)])
tex_table(
    os.path.join(TAB_DIR, "b4_M3b_full.tex"),
    caption="Full results of model M3b (country and SDGR area)",
    label="tab:regression-m3b-full",
    header=["Term", "n", "$\\beta$", "SE", "OR [95\\% CI]", "p"],
    rows=rows,
    midrules=(1, 4, 12),
    notes=[
        f"Binary logit of harder (B2 or C) versus easier (A or B1), N = {fmt_int(fb.N)}, McFadden $R^2$ = {fb.mcfadden_r2:.3f}, "
        f"AIC = {fb.aic:,.1f}. References: Germany and area B, Work and retirement "
        f"(n = {fmt_int(cnt_area['B'])}).",
        "n: pages in the estimation sample with that area. Areas A to I are citizen areas, "
        "J to Q business areas.",
    ],
    colspec="lrrrrr",
    size=r"\small",
)

# ── Model grid (appendix) ──────────────────────────────────────────────────
desc = {"M1": "Country", "M2a": "Scope", "M2b": "17 areas", "M3a": "Country + scope",
        "M3b": "Country + 17 areas", "M4": "Country x scope",
        "M1_all": "Country (all labelled pages, descriptive)"}
rows = []
for m in ["M1", "M2a", "M2b", "M3a", "M3b", "M4", "M1_all"]:
    f = fit_of(m)
    rows.append([m.replace("_all", ""), desc[m], fmt_int(f.N), str(f.k),
                 f"{f.mcfadden_r2:.3f}", f"{f.aic:,.1f}"])
tex_table(
    os.path.join(TAB_DIR, "b4_model_grid.tex"),
    caption="Model comparison, binary logistic regression",
    label="tab:regression-grid",
    header=["Model", "Predictors", "N", "k", "McFadden $R^2$", "AIC"],
    rows=rows,
    midrules=(6,),
    notes=["Binary logit of harder (B2 or C) versus easier (A or B1). "
           "Models M1 to M4 share one estimation sample, so their AIC values are comparable. "
           "The last row refits M1 on all 8,997 labelled pages (including UNCLEAR and "
           "OUT\\_OF\\_SCOPE) and is descriptive only; its AIC is not comparable.",
           "k: number of estimated parameters including the intercept."],
    colspec="llrrrr",
)

# M4 interaction terms
rows = []
for t in coefs[coefs.model == "M4"].itertuples():
    if t.label == "Intercept":
        continue
    rows.append([t.label, f"{t.OR:.2f} [{t.OR_lo:.2f}, {t.OR_hi:.2f}]", fmt_p(t.p)])
f4 = fit_of("M4")
tex_table(
    os.path.join(TAB_DIR, "b4_M4_interaction.tex"),
    caption="Exploratory interaction model M4 (country x scope)",
    label="tab:regression-m4",
    header=["Term", "OR [95\\% CI]", "p"],
    rows=rows,
    notes=[f"Binary logit, N = {fmt_int(f4.N)}, McFadden $R^2$ = {f4.mcfadden_r2:.3f}, "
           f"AIC = {f4.aic:,.1f}. Country main effects refer to citizen pages; interaction terms "
           "are ratios of odds ratios (business versus citizen in that country relative to Germany)."],
    colspec="lrr",
)

# ── Diagnostics ─────────────────────────────────────────────────────────────
print("\n== Diagnostics ==")


def vif_table(rhs, data, name):
    X = patsy.dmatrix(rhs, data, return_type="dataframe")
    out = []
    for j, col in enumerate(X.columns):
        if col == "Intercept":
            continue
        out.append({"model": name, "term": label_term(col),
                    "VIF": variance_inflation_factor(X.values, j)})
    # generalised VIF per factor (Fox and Monette 1992)
    Xn = X.drop(columns="Intercept")
    R = np.corrcoef(Xn.values, rowvar=False)
    det_r = np.linalg.det(R)
    factors = {}
    for i, col in enumerate(Xn.columns):
        factors.setdefault(col.split("[")[0], []).append(i)
    g = []
    for fac, idx in factors.items():
        rest = [i for i in range(len(Xn.columns)) if i not in idx]
        gvif = np.linalg.det(R[np.ix_(idx, idx)]) * np.linalg.det(R[np.ix_(rest, rest)]) / det_r
        g.append({"model": name, "factor": fac, "df": len(idx), "GVIF": gvif,
                  "GVIF_1_2df": gvif ** (1 / (2 * len(idx)))})
    return pd.DataFrame(out), pd.DataFrame(g)


v1, g1 = vif_table(f"{CTRY} + {SCP}", main, "M3a")
v2, g2 = vif_table(f"{CTRY} + {AREA}", main, "M3b")
vif = pd.concat([v1, v2])
gvif = pd.concat([g1, g2])
vif.to_csv(os.path.join(TAB_DIR, "b4_vif.csv"), index=False, float_format="%.3f")
gvif.to_csv(os.path.join(TAB_DIR, "b4_gvif.csv"), index=False, float_format="%.3f")
print(vif.to_string(), "\n", gvif.to_string())

cs = pd.crosstab(main["country"], main["scope"]).reindex(COUNTRIES)
cs.to_csv(os.path.join(TAB_DIR, "b4_cells_country_scope.csv"))
cs_y = pd.crosstab([main["country"], main["scope"]], main["y"])
cs_y.columns = ["A_or_B1", "B2_or_C"]
cs_y.to_csv(os.path.join(TAB_DIR, "b4_cells_country_scope_outcome.csv"))
ca = pd.crosstab(main["country"], main["area"]).reindex(index=COUNTRIES,
                                                        columns=CITIZEN_AREAS + BUSINESS_AREAS,
                                                        fill_value=0)
ca.to_csv(os.path.join(TAB_DIR, "b4_cells_country_area.csv"))
area_y = pd.crosstab(main["area"], main["y"]).reindex(CITIZEN_AREAS + BUSINESS_AREAS)
area_y.columns = ["A_or_B1", "B2_or_C"]
area_y.to_csv(os.path.join(TAB_DIR, "b4_cells_area_outcome.csv"))
small_ca = [(COUNTRY_LABEL[c], a, int(ca.loc[c, a])) for c in COUNTRIES for a in ca.columns
            if ca.loc[c, a] < 10]
small_cs = [(c, s, int(cs.loc[c, s])) for c in COUNTRIES for s in cs.columns if cs.loc[c, s] < 10]
small_cs_y = [(i, col, int(cs_y.loc[i, col])) for i in cs_y.index for col in cs_y.columns
              if cs_y.loc[i, col] < 10]
print("country x scope cells < 10:", small_cs)
print("country x scope x outcome cells < 10:", small_cs_y)
print("country x area cells < 10:", small_ca)

rows = []
for c in COUNTRIES:
    rows.append([COUNTRY_LABEL[c]] + [(f"\\textbf{{{v}}}" if v < 10 else str(v))
                                      for v in ca.loc[c].tolist()])
rows.append(["Total"] + [str(v) for v in ca.sum().tolist()])
tex_table(
    os.path.join(TAB_DIR, "b4_cells_country_area.tex"),
    caption="Pages per country and SDGR area in the estimation sample",
    label="tab:cells-country-area",
    header=[""] + CITIZEN_AREAS + BUSINESS_AREAS,
    rows=rows,
    midrules=(4,),
    notes=[f"Estimation sample, N = {fmt_int(len(main))}. Cells with fewer than 10 pages in bold "
           f"({len(small_ca)} of {ca.size} cells). M3b has no country x area interaction, so "
           "small cells do not enter as separate parameters."],
    colspec="l" + "r" * 17,
    size=r"\scriptsize\setlength{\tabcolsep}{3pt}",
)

# ── Sensitivity table (country ORs across specs) ────────────────────────────
sens_rows = []
for lab in ["Austria", "Netherlands", "Ireland", "Business", "Unclear"]:
    row = [lab]
    for m in ["M3a", "M3a_ii", "M3a_iii", "M3b", "M3b_ii", "M3b_iii"]:
        r = get(m, lab if lab != "Unclear" or m.startswith("M3a") else "UNCLEAR")
        row.append("" if r is None else
                   f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]{'*' if r.p < 0.05 else ''}")
    sens_rows.append(row)
sens_rows.append(["N"] + [fmt_int(fit_of(m).N) for m in
                          ["M3a", "M3a_ii", "M3a_iii", "M3b", "M3b_ii", "M3b_iii"]])
sens_rows.append(["McFadden $R^2$"] + [f"{fit_of(m).mcfadden_r2:.3f}" for m in
                                       ["M3a", "M3a_ii", "M3a_iii", "M3b", "M3b_ii", "M3b_iii"]])
tex_table(
    os.path.join(TAB_DIR, "b4_sensitivity.tex"),
    caption="Sensitivity of the country and scope odds ratios to the treatment of UNCLEAR pages",
    label="tab:regression-sensitivity",
    header=["", "M3a (i)", "M3a (ii)", "M3a (iii)", "M3b (i)", "M3b (ii)", "M3b (iii)"],
    rows=sens_rows,
    midrules=(5,),
    notes=[
        "Binary logit of harder (B2 or C) versus easier (A or B1): odds ratios with Wald 95\\% CIs; "
        "* p $<$ 0.05. M3a: country + scope; "
        "M3b: country + 17 SDGR areas (area coefficients not shown).",
        "(i) Main specification: UNCLEAR and OUT\\_OF\\_SCOPE pages excluded. (ii) UNCLEAR pages kept "
        "as their own scope category (M3a) or area level (M3b). (iii) As (ii) without the 749 "
        "Austrian ``Leben in der Gemeinde'' template pages routed to UNCLEAR by the deterministic rule.",
        "AIC is not comparable across columns because the samples differ.",
    ],
    colspec="lrrrrrr",
    size=r"\scriptsize\setlength{\tabcolsep}{3pt}",
)
sens_csv = coefs[coefs.model.isin(["M3a", "M3a_ii", "M3a_iii", "M3b", "M3b_ii", "M3b_iii"]) &
                 coefs.label.isin(["Austria", "Netherlands", "Ireland", "Business", "Unclear",
                                   "UNCLEAR"])]
sens_csv.to_csv(os.path.join(TAB_DIR, "b4_sensitivity.csv"), index=False, float_format="%.6g")

# ── (iv) OLS on 1-2-4-8 and proportional-odds logit with Brant test ─────────
print("\n== Robustness (iv) ==")
rob_rows = []


def ols_fit(name, rhs):
    r = smf.ols(f"cefr_1248 ~ {rhs}", main).fit()
    ci = r.conf_int()
    for t in r.params.index:
        rob_rows.append({"model": name, "method": "OLS 1-2-4-8", "term": t, "label": label_term(t),
                         "estimate": r.params[t], "lo": ci.loc[t, 0], "hi": ci.loc[t, 1],
                         "p": r.pvalues[t], "N": int(r.nobs), "fit": r.rsquared, "aic": r.aic})
    return r


def ord_fit(name, rhs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = OrderedModel.from_formula(f"cefr_ord ~ {rhs}", main, distr="logit")
        r = mod.fit(method="bfgs", disp=False, maxiter=2000)
    conv = bool(r.mle_retvals.get("converged", False))
    warn_log[f"{name}_ordinal"] = {"warnings": sorted({str(x.message)[:160] for x in w}),
                                   "converged": conv}
    # McFadden against thresholds-only model: thresholds reproduce the marginal distribution
    p_marg = main["cefr_ord"].value_counts(normalize=True)
    llnull = float((main["cefr_ord"].map(p_marg).astype(float).apply(np.log)).sum())
    ci = r.conf_int()
    for t in r.params.index:
        if "/" in t:
            continue
        rob_rows.append({"model": name, "method": "Ordinal logit", "term": t,
                         "label": label_term(t), "estimate": np.exp(r.params[t]),
                         "lo": np.exp(ci.loc[t, 0]), "hi": np.exp(ci.loc[t, 1]),
                         "p": r.pvalues[t], "N": int(r.nobs),
                         "fit": 1 - r.llf / llnull, "aic": r.aic})
    return r


def brant(rhs, data):
    """Brant (1990) Wald test of the proportional-odds assumption, following the R brant
    package: separate binary logits at each cut, stacked slope covariance, omnibus and
    per-factor tests."""
    X = patsy.dmatrix(rhs, data, return_type="dataframe")
    cols = list(X.columns)
    Xv = X.values
    yk = data["cefr_ord"].cat.codes.to_numpy()
    J = 4
    K = Xv.shape[1] - 1
    betas, vars_, pis = [], [], []
    for m in range(1, J):
        yb = (yk >= m).astype(int)
        r = sm.Logit(yb, Xv).fit(disp=False, maxiter=500, method="newton")
        betas.append(r.params)
        vars_.append(r.cov_params())
        pis.append(r.predict(Xv))
    V = np.zeros(((J - 1) * K, (J - 1) * K))
    for m in range(J - 1):
        V[m * K:(m + 1) * K, m * K:(m + 1) * K] = vars_[m][1:, 1:]
    for m in range(J - 2):
        for l in range(m + 1, J - 1):
            wm = pis[m] * (1 - pis[m])
            wl = pis[l] * (1 - pis[l])
            wml = pis[l] - pis[m] * pis[l]
            A = np.linalg.inv(Xv.T @ (Xv * wm[:, None]))
            B = Xv.T @ (Xv * wml[:, None])
            C = np.linalg.inv(Xv.T @ (Xv * wl[:, None]))
            blk = (A @ B @ C)[1:, 1:]
            V[m * K:(m + 1) * K, l * K:(l + 1) * K] = blk
            V[l * K:(l + 1) * K, m * K:(m + 1) * K] = blk.T
    bstar = np.concatenate([b[1:] for b in betas])

    def wald(idx):
        k = len(idx)
        rows_ = []
        for i in range(J - 2):
            row = np.zeros((k, (J - 1) * K))
            for j, c in enumerate(idx):
                row[j, c] = 1.0
                row[j, (i + 1) * K + c] = -1.0
            rows_.append(row)
        D = np.vstack(rows_)
        d = D @ bstar
        stat = float(d @ np.linalg.solve(D @ V @ D.T, d))
        df_ = k * (J - 2)
        return stat, df_, float(stats.chi2.sf(stat, df_))

    out = [("Omnibus",) + wald(list(range(K)))]
    factors = {}
    for i, c in enumerate(cols[1:]):
        factors.setdefault(c.split("[")[0], []).append(i)
    for fac, idx in factors.items():
        name = {"C(country, Treatment('germany'))": "Country",
                "C(scope, Treatment('citizen'))": "Scope",
                "C(area, Treatment('B'))": "Area"}.get(fac, fac)
        out.append((name,) + wald(idx))
    for i, c in enumerate(cols[1:]):
        out.append((label_term(c),) + wald([i]))
    # cut-wise coefficients for transparency
    cut = pd.DataFrame([b[1:] for b in betas], columns=[label_term(c) for c in cols[1:]],
                       index=["B1 or above", "B2 or above", "C"])
    return pd.DataFrame(out, columns=["term", "chi2", "df", "p"]), cut


ols_fit("M3a", f"{CTRY} + {SCP}")
ols_fit("M3b", f"{CTRY} + {AREA}")
ord_a = ord_fit("M3a", f"{CTRY} + {SCP}")
ord_b = ord_fit("M3b", f"{CTRY} + {AREA}")
rob = pd.DataFrame(rob_rows)
rob.to_csv(os.path.join(TAB_DIR, "b4_robustness_ols_ordinal.csv"), index=False, float_format="%.6g")
br_a, cut_a = brant(f"{CTRY} + {SCP}", main)
br_b, cut_b = brant(f"{CTRY} + {AREA}", main)
br_a.insert(0, "model", "M3a")
br_b.insert(0, "model", "M3b")
br = pd.concat([br_a, br_b])
br.to_csv(os.path.join(TAB_DIR, "b4_brant_test.csv"), index=False, float_format="%.6g")
cut_a.to_csv(os.path.join(TAB_DIR, "b4_cutwise_logits_M3a.csv"), float_format="%.4f")
cut_b.to_csv(os.path.join(TAB_DIR, "b4_cutwise_logits_M3b.csv"), float_format="%.4f")
print(br.to_string())
print(cut_a.round(3).to_string())


def rob_cell(model, method, lab):
    r = rob[(rob.model == model) & (rob.method == method) & (rob.label == lab)]
    if r.empty:
        return ""
    r = r.iloc[0]
    return f"{r.estimate:.2f} [{r.lo:.2f}, {r.hi:.2f}]{'*' if r.p < 0.05 else ''}"


rows = []
for lab in ["Austria", "Netherlands", "Ireland", "Business"]:
    rows.append([lab] + [rob_cell(m, meth, lab) for meth in ["OLS 1-2-4-8", "Ordinal logit"]
                         for m in ["M3a", "M3b"]])


def rob_fit(model, method, key):
    r = rob[(rob.model == model) & (rob.method == method)].iloc[0]
    return r[key]


rows.append(["N"] + [fmt_int(rob_fit(m, meth, "N")) for meth in ["OLS 1-2-4-8", "Ordinal logit"]
                     for m in ["M3a", "M3b"]])
rows.append(["$R^2$ / McFadden $R^2$"] + [f"{rob_fit(m, meth, 'fit'):.3f}" for meth in
                                          ["OLS 1-2-4-8", "Ordinal logit"] for m in ["M3a", "M3b"]])
oa = br_a[br_a.term == "Omnibus"].iloc[0]
ob = br_b[br_b.term == "Omnibus"].iloc[0]
rows.append(["Brant omnibus $\\chi^2$ (df)", "", "",
             f"{oa.chi2:,.1f} ({int(oa.df)}), p {fmt_p(oa.p)}",
             f"{ob.chi2:,.1f} ({int(ob.df)}), p {fmt_p(ob.p)}"])
tex_table(
    os.path.join(TAB_DIR, "b4_robustness_ols_ordinal.tex"),
    caption="Robustness: OLS on a 1-2-4-8 coding and proportional-odds logit",
    label="tab:regression-robustness",
    header=["", "OLS M3a $\\beta$", "OLS M3b $\\beta$", "Ordinal M3a OR", "Ordinal M3b OR"],
    rows=rows,
    midrules=(4,),
    notes=[
        "Same estimation sample as the main models. OLS outcome: A = 1, B1 = 2, B2 = 4, C = 8; "
        "coefficients with 95\\% CIs. Ordinal: proportional-odds logit on A $<$ B1 $<$ B2 $<$ C; "
        "cumulative odds ratios with 95\\% CIs. * p $<$ 0.05. Area coefficients not shown.",
        "Brant test: Wald test that slopes are equal across the three cumulative logits; a "
        "significant result means the proportional-odds assumption does not hold.",
    ],
    colspec="lrrrr",
    size=r"\small",
)

save_json({"warnings": warn_log,
           "small_cells_country_area_lt10": small_ca,
           "small_cells_country_scope_lt10": small_cs,
           "small_cells_country_scope_outcome_lt10": small_cs_y,
           "samples": {"main": len(main), "all_labelled": len(df), "unclear_kept": len(unc),
                       "unclear_kept_no_rule": len(unc_norule),
                       "rule_pages_labelled": int(df["rule"].sum())}},
          "b4_diagnostics.json")
print("\nSamples:", len(main), len(df), len(unc), len(unc_norule))
