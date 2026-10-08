# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b3_zeroshot_validation.py
# Purpose: B3 validation of the zero-shot SDGR categoriser against the manual scope
#          annotations (90 pages). Primary comparison: the corpus labels that enter the
#          regression (analysis/sdg/sdg_categorization_full.csv). Secondary: the cached
#          26 August re-run (analysis/sampling/zeroshot_area_predictions.csv), which used the
#          older single-token prompt and is what zeroshot_eval_results.json reports.
# Pipeline step: Validation of the SDGR categoriser against the 90 manual annotations.
# Thesis: Section 3.6.3: tab:zeroshot-validation, b3_confusion_manual_vs_zeroshot. The table
#         file also has rows for the 26 August re-run (older prompt, read from
#         analysis/sampling/zeroshot_area_predictions.csv); the thesis reports only the
#         corpus-label rows and the majority and random baselines.

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from common import (TAB_DIR, ANNOTATION_CSV, ZS_CACHE_CSV, load_sdg, scope_of, wilson,
                    tex_table, save_fig, save_json, CM_CMAP, cm_text_colour)

SEED = 42
N_BOOT = 2000
L3 = ["citizen", "business", "unclear"]
L2 = ["citizen", "business"]

ann = pd.read_csv(ANNOTATION_CSV, sep=None, engine="python", dtype=str).fillna("")
ann["annotation"] = ann["annotation"].str.strip()
assert len(ann) == 90 and set(ann["annotation"]) <= set(L3), ann["annotation"].value_counts()
sdg = load_sdg()
cache = pd.read_csv(ZS_CACHE_CSV, dtype=str)
df = (ann.merge(sdg[["url", "assigned_area", "assigned_reasoning", "rule"]], on="url",
                how="left", validate="one_to_one")
         .merge(cache, on="url", how="left", validate="one_to_one"))
assert df["assigned_area"].notna().all() and df["zs_area"].notna().all()


def collapse(a):
    return scope_of(a) or "unclear"


df["zs_corpus"] = df["assigned_area"].map(collapse)
df["zs_cache"] = df["zs_area"].map(collapse)


def macro_f1(y, p, labels):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def boot_ci(y, p, labels, rng):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[b] = macro_f1(y[idx], p[idx], labels)
    return np.percentile(vals, [2.5, 97.5])


def evaluate(y, p, labels):
    y, p = list(y), list(p)
    rng = np.random.default_rng(SEED)
    k = sum(a == b for a, b in zip(y, p))
    lo, hi = wilson(k, len(y))
    f = macro_f1(y, p, labels)
    flo, fhi = boot_ci(y, p, labels, rng)
    return {"n": len(y), "correct": k, "accuracy": k / len(y), "acc_lo": lo, "acc_hi": hi,
            "macro_f1": f, "f1_lo": flo, "f1_hi": fhi}


def baselines(y, labels):
    y = list(y)
    top = Counter(y).most_common(1)[0][0]
    maj = evaluate(y, [top] * len(y), labels)
    maj["label"] = top
    # Random baseline: draws from the empirical manual label distribution (as in
    # eval_zeroshot_areas.py); reported as the mean and 2.5/97.5 percentiles over N_BOOT draws.
    labs, cnts = zip(*Counter(y).items())
    probs = np.array(cnts, float) / sum(cnts)
    rng = np.random.default_rng(SEED)
    accs, f1s = [], []
    for _ in range(N_BOOT):
        r = rng.choice(labs, size=len(y), p=probs)
        accs.append(accuracy_score(y, r))
        f1s.append(macro_f1(y, r, labels))
    rnd = {"n": len(y), "accuracy": float(np.mean(accs)),
           "acc_lo": float(np.percentile(accs, 2.5)), "acc_hi": float(np.percentile(accs, 97.5)),
           "macro_f1": float(np.mean(f1s)),
           "f1_lo": float(np.percentile(f1s, 2.5)), "f1_hi": float(np.percentile(f1s, 97.5))}
    return maj, rnd


results = {}
for src in ["zs_corpus", "zs_cache"]:
    res = {}
    # (a) 3-class
    res["3class"] = evaluate(df["annotation"], df[src], L3)
    # (b) strict 2-class: rows manually labelled unclear removed; a zero-shot 'unclear'
    #     on a remaining row counts as an error (it is neither citizen nor business).
    d2 = df[df["annotation"] != "unclear"]
    res["2class"] = evaluate(d2["annotation"], d2[src], L2)
    # (b') variant used by eval_zeroshot_areas.py: also drop rows where zero-shot says unclear
    d2b = d2[d2[src] != "unclear"]
    res["2class_both_dropped"] = evaluate(d2b["annotation"], d2b[src], L2)
    cm = confusion_matrix(df["annotation"], df[src], labels=L3)
    res["confusion_3x3"] = cm.tolist()
    mu = df[df["annotation"] == "unclear"]
    res["manual_unclear_to_citizen"] = int((mu[src] == "citizen").sum())
    res["manual_unclear_to_business"] = int((mu[src] == "business").sum())
    res["manual_unclear_to_unclear"] = int((mu[src] == "unclear").sum())
    results[src] = res

maj3, rnd3 = baselines(df["annotation"], L3)
d2 = df[df["annotation"] != "unclear"]
maj2, rnd2 = baselines(d2["annotation"], L2)
d2b = d2[d2["zs_corpus"] != "unclear"]
maj2b, rnd2b = baselines(d2b["annotation"], L2)
results["baselines"] = {"3class": {"majority": maj3, "random": rnd3},
                        "2class": {"majority": maj2, "random": rnd2},
                        "2class_both_dropped_corpus": {"majority": maj2b, "random": rnd2b}}
results["agreement_corpus_vs_cache"] = {
    "area_exact": int((df["assigned_area"] == df["zs_area"]).sum()),
    "scope": int((df["zs_corpus"] == df["zs_cache"]).sum()), "n": len(df)}
results["n_rule_pages_in_sample"] = int(df["rule"].sum())
results["manual_counts"] = df["annotation"].value_counts().to_dict()
# zero-shot 'unclear' split into UNCLEAR vs OUT_OF_SCOPE (corpus labels)
results["corpus_unclear_breakdown"] = df.loc[df["zs_corpus"] == "unclear", "assigned_area"].value_counts().to_dict()
save_json(results, "b3_zeroshot_validation.json")

for k, v in results.items():
    print(k, v)

# Confusion-matrix CSV
cm = pd.DataFrame(results["zs_corpus"]["confusion_3x3"], index=[f"manual_{l}" for l in L3],
                  columns=[f"zeroshot_{l}" for l in L3])
cm.to_csv(os.path.join(TAB_DIR, "b3_confusion_manual_vs_zeroshot.csv"))
pd.DataFrame(results["zs_cache"]["confusion_3x3"], index=[f"manual_{l}" for l in L3],
             columns=[f"zeroshot_{l}" for l in L3]).to_csv(
    os.path.join(TAB_DIR, "b3_confusion_manual_vs_cached_rerun.csv"))

# Validation CSV
recs = []
for task, key in [("3-class", "3class"), ("2-class strict", "2class")]:
    for name, m in [("Zero-shot (corpus labels)", results["zs_corpus"][key]),
                    ("Majority class", results["baselines"][key]["majority"]),
                    ("Random (label distribution)", results["baselines"][key]["random"]),
                    ("Zero-shot (26 Aug re-run, older prompt)", results["zs_cache"][key])]:
        recs.append({"task": task, "method": name, **{k: m[k] for k in
                     ["n", "accuracy", "acc_lo", "acc_hi", "macro_f1", "f1_lo", "f1_hi"]}})
pd.DataFrame(recs).to_csv(os.path.join(TAB_DIR, "b3_zeroshot_validation.csv"), index=False,
                          float_format="%.4f")


rows = []
for r in recs:
    rows.append([r["task"], r["method"], str(r["n"]),
                 f"{r['accuracy']:.2f} [{r['acc_lo']:.2f}, {r['acc_hi']:.2f}]",
                 f"{r['macro_f1']:.2f} [{r['f1_lo']:.2f}, {r['f1_hi']:.2f}]"])
c = results["zs_corpus"]
tex_table(
    os.path.join(TAB_DIR, "b3_zeroshot_validation.tex"),
    caption="Validation of the zero-shot SDGR categoriser against manual scope annotations",
    label="tab:zeroshot-validation",
    header=["Task", "Method", "n", "Accuracy [95\\% CI]", "Macro-F1 [95\\% CI]"],
    rows=rows,
    midrules=(4,),
    notes=[
        "Manual annotation of a stratified 1\\% sample (90 pages; 35 citizen, 21 business, "
        "34 unclear). Zero-shot areas are collapsed to scope: A to I citizen, J to Q business, "
        "UNCLEAR and OUT\\_OF\\_SCOPE unclear.",
        "3-class: citizen, business, unclear. 2-class strict: pages manually labelled unclear "
        f"removed (n = {c['2class']['n']}); a zero-shot unclear on a remaining page counts as an error.",
        "Accuracy CI: Wilson. Macro-F1 CI: percentile bootstrap, 2,000 resamples, seed 42. "
        "Random baseline: mean and 2.5th to 97.5th percentile over 2,000 draws from the manual label "
        "distribution (seed 42).",
        "Corpus labels are the labels used in the regression. The 26 August re-run reclassified the "
        "sample with the older single-token prompt; it agrees with the corpus labels on "
        f"{results['agreement_corpus_vs_cache']['scope']} of 90 scopes.",
    ],
    colspec="llrrr",
    size=r"\small",
)

# 3x3 confusion table (LaTeX)
cmv = results["zs_corpus"]["confusion_3x3"]
rows = [[f"Manual {l}"] + [str(x) for x in cmv[i]] + [str(sum(cmv[i]))] for i, l in enumerate(L3)]
rows.append(["Total"] + [str(sum(cmv[i][j] for i in range(3))) for j in range(3)] + ["90"])
tex_table(
    os.path.join(TAB_DIR, "b3_confusion_manual_vs_zeroshot.tex"),
    caption="Manual scope annotation versus zero-shot scope (corpus labels)",
    label="tab:zeroshot-confusion",
    header=["", "Zero-shot citizen", "Zero-shot business", "Zero-shot unclear", "Total"],
    rows=rows,
    midrules=(3,),
    notes=["Rows: manual annotation. Columns: zero-shot area collapsed to scope "
           "(unclear includes UNCLEAR and OUT\\_OF\\_SCOPE). Diagonal cells are agreements.",
           f"Of the 34 pages annotated unclear, the categoriser placed "
           f"{c['manual_unclear_to_citizen']} in a citizen area and "
           f"{c['manual_unclear_to_business']} in a business area."],
    colspec="lrrrr",
)

# Figure
fig, ax = plt.subplots(figsize=(4.6, 3.8))
arr = np.array(cmv)
ax.imshow(arr, cmap=CM_CMAP, vmin=0, vmax=arr.max())
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                color=cm_text_colour(arr[i, j], arr), fontsize=11)
ax.set_xticks(range(3), [l.capitalize() for l in L3])
ax.set_yticks(range(3), [l.capitalize() for l in L3])
ax.set_xlabel("Zero-shot scope (corpus labels)")
ax.set_ylabel("Manual annotation")
for s in ax.spines.values():
    s.set_visible(False)
save_fig(fig, "b3_confusion_manual_vs_zeroshot")
