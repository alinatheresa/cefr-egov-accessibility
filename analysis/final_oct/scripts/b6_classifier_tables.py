# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b6_classifier_tables.py
# Purpose: B6 classifier chapter tables for run cefr_20260817_231925 (DE/EN/NL retrain).
#          Sources: classifier/00_prepare_output/*.json, the run's metrics.json and
#          run_manifest.json, and the run's confusion matrices. Per-document predictions were
#          not saved by 02_train_final.py, so per-language results use the window-level
#          confusion matrices (PNG) transcribed below; the transcription is checked against
#          the per-language macro-F1 values and the overall matrix in metrics.json.
# Pipeline step: Analysis: classifier tables from the run metadata.
# Thesis: Chapter 4: tab:clf-data, tab:clf-classes, tab:clf-weights, tab:clf-metrics,
#         tab:clf-prf, tab:clf-f1-lang, tab:clf-errors.

import os
import json

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, RUN_DIR, TAB_DIR, CEFR, tex_table, save_json, fmt_int

PREP = os.path.join(PROJECT_ROOT, "classifier", "00_prepare_output")
metrics = json.load(open(os.path.join(RUN_DIR, "metrics.json")))
manifest = json.load(open(os.path.join(RUN_DIR, "run_manifest.json")))
dedup = json.load(open(os.path.join(PREP, "dedup_stats.json")))
labels = json.load(open(os.path.join(PREP, "label_counts_mergedC.json")))
cw_json = json.load(open(os.path.join(PREP, "class_weights.json")))

# Rows = true A, B1, B2, C; columns = predicted A, B1, B2, C.
# Source PNGs: <run>/trials/linear_bal0/confusion_matrix_{split}_{lang}.png (window level),
# <run>/confusion_matrix_{split}_{window,doc}.png.
CM = {
    ("test", "de"): [[46, 2, 0, 0], [14, 22, 6, 0], [1, 6, 30, 5], [0, 1, 9, 13]],
    ("test", "en"): [[18, 0, 0, 0], [6, 42, 6, 1], [0, 11, 60, 11], [0, 0, 22, 102]],
    ("test", "nl"): [[10, 7, 1, 0], [7, 38, 10, 0], [1, 14, 20, 3], [0, 1, 5, 3]],
    ("val", "de"): [[45, 4, 0, 0], [15, 22, 5, 0], [2, 5, 29, 6], [0, 0, 6, 17]],
    ("val", "en"): [[18, 1, 0, 0], [7, 27, 19, 2], [1, 10, 55, 19], [0, 1, 16, 122]],
    ("val", "nl"): [[11, 8, 0, 0], [7, 37, 11, 0], [0, 15, 23, 0], [0, 0, 6, 2]],
}
CM_ALL = {
    ("test", "window"): [[74, 9, 1, 0], [27, 102, 22, 1], [2, 31, 110, 19], [0, 2, 36, 118]],
    ("test", "doc"): [[74, 9, 1, 0], [26, 85, 20, 0], [2, 23, 81, 15], [0, 2, 24, 51]],
    ("val", "doc"): [[74, 13, 0, 0], [29, 77, 24, 1], [3, 25, 76, 16], [0, 1, 18, 56]],
}


def prf(cm):
    cm = np.array(cm, float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return prec, rec, f1


def summary(cm):
    cm = np.array(cm)
    n = cm.sum()
    i, j = np.indices(cm.shape)
    dist = np.abs(i - j)
    acc = np.trace(cm) / n
    p, r, f = prf(cm)
    # quadratic weighted kappa
    w = (i - j) ** 2 / 9
    exp = np.outer(cm.sum(1), cm.sum(0)) / n
    qwk = 1 - (w * cm).sum() / (w * exp).sum()
    off = cm * (dist > 0)
    offd = off.copy()
    np.fill_diagonal(offd, 0)
    top = np.unravel_index(offd.argmax(), offd.shape)
    return {"n": int(n), "accuracy": acc, "macro_f1": f.mean(), "qwk": qwk,
            "within_one": (cm * (dist <= 1)).sum() / n,
            "errors": int(off.sum()), "errors_1": int((cm * (dist == 1)).sum()),
            "errors_2plus": int((cm * (dist >= 2)).sum()),
            "top_confusion": f"{CEFR[top[0]]} as {CEFR[top[1]]} ({offd[top]})"}


# ── Transcription check ─────────────────────────────────────────────────────
check = {}
for split, key in [("test", "test"), ("val", "validation")]:
    for lang in ["de", "en", "nl"]:
        got = prf(CM[(split, lang)])[2].mean()
        exp = metrics["window_level"][key]["per_lang_f1"][lang]
        check[f"{split}_{lang}_macroF1"] = {"from_png": got, "metrics_json": exp,
                                            "match": abs(got - exp) < 1e-9}
summed = sum(np.array(CM[("test", l)]) for l in ["de", "en", "nl"])
check["test_lang_sum_equals_overall_window"] = bool((summed == np.array(CM_ALL[("test", "window")])).all())
for (split, lev), cm in CM_ALL.items():
    key = "test" if split == "test" else "validation"
    lvl = "window_level" if lev == "window" else "document_level"
    s = summary(cm)
    check[f"{split}_{lev}_acc"] = {"from_png": s["accuracy"], "metrics_json": metrics[lvl][key]["accuracy"],
                                   "match": abs(s["accuracy"] - metrics[lvl][key]["accuracy"]) < 1e-9}
    check[f"{split}_{lev}_qwk"] = {"from_png": s["qwk"], "metrics_json": metrics[lvl][key]["qwk"],
                                   "match": abs(s["qwk"] - metrics[lvl][key]["qwk"]) < 1e-6}
for k, v in check.items():
    print(k, v)
assert all(v["match"] if isinstance(v, dict) else v for v in check.values()), "transcription mismatch"

# ── 1. Dataset sizes per language ───────────────────────────────────────────
DS_LANG = {"merlin_de": "German", "elg_cefr_de": "German", "elg_cefr_en": "English",
           "icle500_en": "English", "cambridge_exams_en": "English", "elg_cefr_nl": "Dutch"}
rows, recs = [], []
for ds, s in dedup["per_dataset"].items():
    recs.append({"language": DS_LANG[ds], "dataset": ds, **s})
    rows.append([DS_LANG[ds], ds.replace("_", r"\_"), fmt_int(s["n_raw"]), fmt_int(s["n_plus"]),
                 fmt_int(s["n_removed_na"]), fmt_int(s["n_final"])])
ds_df = pd.DataFrame(recs)
ds_df.to_csv(os.path.join(TAB_DIR, "b6_dataset_sizes.csv"), index=False)
by_lang = ds_df.groupby("language")[["n_raw", "n_final"]].sum()
split_lang = {sp: {lg: sum(labels["by_split"][sp]["by_lang_label"][lg].values())
                   for lg in ["de", "en", "nl"]} for sp in ["train", "validation", "test"]}
fin = {lg: sum(split_lang[sp][lg] for sp in split_lang) for lg in ["de", "en", "nl"]}
rows.append(["Total", "", fmt_int(ds_df.n_raw.sum()), fmt_int(ds_df.n_plus.sum()),
             fmt_int(ds_df.n_removed_na.sum()), fmt_int(ds_df.n_final.sum())])
xdup = dedup["en_cross_dataset_dups"]
tex_table(
    os.path.join(TAB_DIR, "b6_dataset_sizes.tex"),
    caption="UniversalCEFR training data before and after cleaning (German, English, Dutch)",
    label="tab:clf-data",
    header=["Language", "Dataset", "Raw rows", "Plus labels", "Removed (NA)", "After cleaning"],
    rows=rows,
    midrules=(len(rows) - 1,),
    notes=[
        "Raw rows: rows downloaded. Plus labels: rows with a plus level (for example B1+), mapped to the "
        "base level. After cleaning: unique texts after removing invalid rows, within-dataset duplicates "
        "and, for elg\\_cefr\\_nl, aggregating 3,596 rater rows to 1,195 texts.",
        f"A further {xdup} English texts occur in both elg\\_cefr\\_en and cambridge\\_exams\\_en and are kept "
        f"once, giving {fmt_int(labels['overall']['total'])} texts: German {fmt_int(fin['de'])}, English "
        f"{fmt_int(fin['en'])}, Dutch {fmt_int(fin['nl'])}. Portuguese (cople2\\_pt, peapl2\\_pt) is not used.",
    ],
    colspec="llrrrr",
    size=r"\small",
)

# ── 2. Class distribution per language and split ───────────────────────────
rows, recs = [], []
for lg, name in [("de", "German"), ("en", "English"), ("nl", "Dutch")]:
    for sp in ["train", "validation", "test"]:
        c = labels["by_split"][sp]["by_lang_label"][lg]
        tot = sum(c.values())
        recs.append({"language": name, "split": sp, **c, "total": tot})
        rows.append([name if sp == "train" else "", sp.capitalize()] +
                    [f"{c[l]} ({100 * c[l] / tot:.0f})" for l in CEFR] + [fmt_int(tot)])
for sp in ["train", "validation", "test"]:
    c = labels["by_split"][sp]["by_label"]
    tot = labels["by_split"][sp]["total"]
    recs.append({"language": "All", "split": sp, **c, "total": tot})
    rows.append(["All" if sp == "train" else "", sp.capitalize()] +
                [f"{c[l]} ({100 * c[l] / tot:.0f})" for l in CEFR] + [fmt_int(tot)])
pd.DataFrame(recs).to_csv(os.path.join(TAB_DIR, "b6_class_distribution.csv"), index=False)
tex_table(
    os.path.join(TAB_DIR, "b6_class_distribution.tex"),
    caption="CEFR class distribution by language and split",
    label="tab:clf-classes",
    header=["Language", "Split", "A", "B1", "B2", "C", "Total"],
    rows=rows, midrules=(3, 6, 9),
    notes=["Documents per class with row percentages in parentheses. A combines A1 and A2; "
           "C combines C1 and C2. Splits are stratified, seed 42."],
    colspec="llrrrrr", size=r"\small",
)

# ── 3. Class weights ────────────────────────────────────────────────────────
cw_used = manifest["training_config"]["class_weights"]
cw = pd.DataFrame({"level": CEFR, "used_in_loss_window_level": [cw_used[l] for l in CEFR],
                   "class_weights_json_document_level": [cw_json[l] for l in CEFR]})
cw.to_csv(os.path.join(TAB_DIR, "b6_class_weights.csv"), index=False, float_format="%.4f")
tex_table(
    os.path.join(TAB_DIR, "b6_class_weights.tex"),
    caption="Class weights in the training loss",
    label="tab:clf-weights",
    header=["Level", "Weight"],
    rows=[[l, f"{cw_used[l]:.2f}"] for l in CEFR],
    notes=["Inverse frequency of training windows (total windows divided by windows of the class), "
           "as recorded in run\\_manifest.json and used in the weighted cross-entropy loss "
           "(label smoothing 0.1) plus EMD term (weight 0.3)."],
    colspec="lr",
)

# ── 4. Overall test metrics ─────────────────────────────────────────────────
rows, recs = [], []
for lvl_key, lvl_name in [("document_level", "Document"), ("window_level", "Window")]:
    for sp_key, sp_name in [("test", "Test"), ("validation", "Validation")]:
        m = metrics[lvl_key][sp_key]
        recs.append({"level": lvl_name, "split": sp_name, "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"], "qwk": m["qwk"],
                     "within_one": m["adjacent_accuracy"]})
        rows.append([lvl_name if sp_key == "test" else "", sp_name, f"{m['accuracy']:.3f}",
                     f"{m['macro_f1']:.3f}", f"{m['qwk']:.3f}", f"{100 * m['adjacent_accuracy']:.1f}"])
pd.DataFrame(recs).to_csv(os.path.join(TAB_DIR, "b6_overall_metrics.csv"), index=False,
                          float_format="%.4f")
tex_table(
    os.path.join(TAB_DIR, "b6_overall_metrics.tex"),
    caption="Classifier performance (run cefr\\_20260817\\_231925)",
    label="tab:clf-metrics",
    header=["Level", "Split", "Accuracy", "Macro-F1", "QWK", r"Within one level (\%)"],
    rows=rows, midrules=(2,),
    notes=["Document level: logits mean-pooled over all 512-token windows of a text, as in corpus "
           "inference (413 test and 413 validation documents). Window level: each window scored separately "
           "(554 test windows). QWK: quadratic weighted kappa. Model selected on validation macro-F1 "
           "(best epoch 3 of 5)."],
    colspec="llrrrr",
)

# ── 5. Per class x language (window level) ─────────────────────────────────
rows, recs = [], []
for split in ["test", "val"]:
    for lg, name in [("de", "German"), ("en", "English"), ("nl", "Dutch")]:
        p, r, f = prf(CM[(split, lg)])
        s = summary(CM[(split, lg)])
        for i, l in enumerate(CEFR):
            recs.append({"split": split, "language": name, "level": l, "support": int(np.array(CM[(split, lg)])[i].sum()),
                         "precision": p[i], "recall": r[i], "f1": f[i]})
        if split == "test":
            rows.append([name] + [f"{p[i]:.2f} / {r[i]:.2f} / {f[i]:.2f}" for i in range(4)] +
                        [f"{f.mean():.2f}", f"{s['accuracy']:.2f}", fmt_int(s["n"])])
prf_df = pd.DataFrame(recs)
prf_df.to_csv(os.path.join(TAB_DIR, "b6_prf_by_language_class.csv"), index=False, float_format="%.4f")
tex_table(
    os.path.join(TAB_DIR, "b6_prf_by_language_class.tex"),
    caption="Precision, recall and F1 by language and class (test set, window level)",
    label="tab:clf-prf",
    header=["Language", "A (P / R / F1)", "B1 (P / R / F1)", "B2 (P / R / F1)", "C (P / R / F1)",
            "Macro-F1", "Accuracy", "Windows"],
    rows=rows,
    notes=["Computed from the per-language confusion matrices saved by 02\\_train\\_final.py for the "
           "selected model, which are window level; document-level predictions per language were not saved. "
           "The values reproduce the per-language macro-F1 in metrics.json exactly.",
           "Validation-set values are in b6\\_prf\\_by\\_language\\_class.csv."],
    colspec="lrrrrrrr", size=r"\scriptsize\setlength{\tabcolsep}{3pt}",
)

# F1 per class x language compact table (test and validation)
rows = []
for split, sname in [("test", "Test"), ("val", "Validation")]:
    for lg, name in [("de", "German"), ("en", "English"), ("nl", "Dutch")]:
        f = prf(CM[(split, lg)])[2]
        rows.append([sname if lg == "de" else "", name] + [f"{x:.2f}" for x in f] + [f"{f.mean():.2f}"])
tex_table(
    os.path.join(TAB_DIR, "b6_f1_class_by_language.tex"),
    caption="F1 by CEFR class and language (window level)",
    label="tab:clf-f1-lang",
    header=["Split", "Language", "A", "B1", "B2", "C", "Macro-F1"],
    rows=rows, midrules=(3,),
    notes=["Window-level F1 from the selected model's per-language confusion matrices."],
    colspec="llrrrrr",
)

# ── 6. Errors per language ─────────────────────────────────────────────────
rows, recs = [], []
for split in ["test", "val"]:
    for lg, name in [("de", "German"), ("en", "English"), ("nl", "Dutch"), ("all", "All")]:
        cm = CM_ALL[(split, "window")] if lg == "all" and split == "test" else \
            (sum(np.array(CM[(split, l)]) for l in ["de", "en", "nl"]) if lg == "all" else CM[(split, lg)])
        s = summary(cm)
        recs.append({"split": split, "language": name, **s})
        if split == "test":
            rows.append([name, fmt_int(s["n"]), fmt_int(s["errors"]),
                         f"{s['errors_1']} ({100 * s['errors_1'] / s['errors']:.0f})",
                         f"{s['errors_2plus']} ({100 * s['errors_2plus'] / s['errors']:.0f})",
                         f"{100 * s['within_one']:.1f}", s["top_confusion"]])
doc = summary(CM_ALL[("test", "doc")])
recs.append({"split": "test", "language": "All (document level)", **doc})
rows.append(["All (document level)", fmt_int(doc["n"]), fmt_int(doc["errors"]),
             f"{doc['errors_1']} ({100 * doc['errors_1'] / doc['errors']:.0f})",
             f"{doc['errors_2plus']} ({100 * doc['errors_2plus'] / doc['errors']:.0f})",
             f"{100 * doc['within_one']:.1f}", doc["top_confusion"]])
pd.DataFrame(recs).to_csv(os.path.join(TAB_DIR, "b6_errors_by_language.csv"), index=False,
                          float_format="%.4f")
tex_table(
    os.path.join(TAB_DIR, "b6_errors_by_language.tex"),
    caption="Classification errors by language (test set)",
    label="tab:clf-errors",
    header=["Language", "n", "Errors", "1 level apart (\\%)", "2+ levels apart (\\%)",
            "Within one (\\%)", "Most frequent confusion"],
    rows=rows, midrules=(3, 4),
    notes=["Window level unless stated. Percentages in the error columns are shares of all errors. "
           "Most frequent confusion is given as true level as predicted level (count)."],
    colspec="lrrrrrl", size=r"\small",
)

# ── 7. Rater agreement context for elg_cefr_nl (dataset level) ─────────────
ra = dedup["rater_agreement"]["elg_cefr_nl"]
apart = {int(k): v for k, v in ra["apart"].items()}
nl_ctx = {"n_texts": ra["n_texts"], "rater_spread_levels": apart,
          "share_any_disagreement": sum(v for k, v in apart.items() if k > 0) / ra["n_texts"]}
print("elg_cefr_nl rater spread:", nl_ctx)

save_json({"transcription_check": check, "dataset_by_language": by_lang.reset_index().to_dict(orient="records"),
           "final_texts_by_language": fin, "class_weights_used": cw_used, "class_weights_json": cw_json,
           "elg_cefr_nl_rater_spread": nl_ctx,
           "errors": recs}, "b6_classifier.json")
print(pd.DataFrame(recs).to_string())
print(prf_df[prf_df.split == "test"].to_string())
