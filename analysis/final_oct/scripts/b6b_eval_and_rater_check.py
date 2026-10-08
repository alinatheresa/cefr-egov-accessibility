# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/b6b_eval_and_rater_check.py
# Purpose: (1) Run classifier/04_eval_per_language.py --run cefr_20260817_231925 (evaluation
#          only: one forward pass over the tokenised validation and test splits; no training).
#          The script is imported and its main() called unchanged; its aggregate_doc_level()
#          is wrapped only to keep the logits and document predictions it already computes,
#          which 04_eval_per_language.py does not write to disk.
#          (2) Rater-supported error analysis for the Dutch test documents (elg_cefr_nl): for
#          each document-level error, check whether any original rater gave the predicted level.
#          Text matching and label normalisation reuse analysis/error_analysis/check_rater_agreement.py.
# Pipeline step: Analysis: evaluation pass of run cefr_20260817_231925 and Dutch rater check (needs
#                the model weights, the tokenised data and the UniversalCEFR download).
# Thesis: Chapter 4: tab:clf-prf-doc, tab:clf-errors-doc, tab:clf-rater-nl; saves the
#         predictions used by b6c_classifier_figures.py and ch5_analysis.py.

import os
import sys
import importlib.util

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import PROJECT_ROOT, RUN_ID, RUN_DIR, TAB_DIR, RES_DIR, CEFR, tex_table, save_json


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = load(os.path.join(PROJECT_ROOT, "classifier", "04_eval_per_language.py"), "eval_per_language")
cra = load(os.path.join(PROJECT_ROOT, "analysis", "error_analysis", "check_rater_agreement.py"),
           "check_rater_agreement")

captured = []
_orig_agg = ev.aggregate_doc_level


def _capturing_agg(logits, y_true, sample_ids, lang_ids):
    out = _orig_agg(logits, y_true, sample_ids, lang_ids)
    captured.append({"logits": np.array(logits), "y_true": np.array(y_true),
                     "sample_ids": np.array(sample_ids), "lang_ids": np.array(lang_ids),
                     "doc_ids": np.array(sorted(set(int(s) for s in sample_ids))),
                     "doc": out})
    return out


ev.aggregate_doc_level = _capturing_agg
sys.argv = ["04_eval_per_language.py", "--run", RUN_ID]
ev.main()
assert len(captured) == 2, "expected validation and test aggregation"

# ── Save predictions ────────────────────────────────────────────────────────
from datasets import load_from_disk  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(os.path.join(RUN_DIR, "mdeberta_cefr_multi"))
metrics = pd.read_json(os.path.join(RUN_DIR, "metrics.json"), typ="series")
ID2LANG = ev.ID2LANG
checks = {}
doc_tables = {}
local_texts = []
for split, cap, mkey in [("validation", captured[0], "validation"), ("test", captured[1], "test")]:
    ds = load_from_disk(os.path.join(ev.PACKED_DIR, split))
    first_window_text = {}
    for row in ds:
        sid = int(row["sample_id"])
        if sid not in first_window_text:
            first_window_text[sid] = tok.decode(row["input_ids"], skip_special_tokens=True)
    n_win = pd.Series(cap["sample_ids"]).value_counts()
    yt, yp, lg = cap["doc"]
    doc = pd.DataFrame({"sample_id": cap["doc_ids"], "lang": [ID2LANG[int(l)] for l in lg],
                        "true_label": [CEFR[i] for i in yt], "pred_label": [CEFR[i] for i in yp]})
    doc["n_windows"] = doc["sample_id"].map(n_win).astype(int)
    doc["distance"] = (doc["true_label"].map(CEFR.index) - doc["pred_label"].map(CEFR.index)).abs()
    doc["text_first_window"] = doc["sample_id"].map(first_window_text)
    # UniversalCEFR texts come from licensed corpora: keep them in a git-ignored local file only
    doc.drop(columns="text_first_window").to_csv(
        os.path.join(RES_DIR, f"b6b_doc_predictions_{split}.csv"), index=False)
    local_texts.append(doc[["sample_id", "lang", "text_first_window"]].assign(split=split))
    win = pd.DataFrame(cap["logits"], columns=[f"logit_{l}" for l in CEFR])
    win.insert(0, "sample_id", cap["sample_ids"])
    win.insert(1, "lang", [ID2LANG[int(l)] for l in cap["lang_ids"]])
    win.insert(2, "true_label", [CEFR[i] for i in cap["y_true"]])
    win["pred_label"] = [CEFR[i] for i in cap["logits"].argmax(1)]
    win.to_csv(os.path.join(RES_DIR, f"b6b_window_logits_{split}.csv"), index=False,
               float_format="%.6f")
    doc_tables[split] = doc
    # checks against metrics.json saved at training time
    m_doc = metrics["document_level"][mkey]
    m_win = metrics["window_level"][mkey]
    acc_doc = (doc.true_label == doc.pred_label).mean()
    acc_win = (win.true_label == win.pred_label).mean()
    checks[split] = {
        "doc_accuracy": acc_doc, "doc_accuracy_metrics_json": m_doc["accuracy"],
        "doc_macro_f1": f1_score(doc.true_label, doc.pred_label, average="macro"),
        "doc_macro_f1_metrics_json": m_doc["macro_f1"],
        "win_accuracy": acc_win, "win_accuracy_metrics_json": m_win["accuracy"],
        "win_per_lang_f1": {l: f1_score(win[win.lang == l].true_label, win[win.lang == l].pred_label,
                                        average="macro") for l in ["de", "en", "nl"]},
        "win_per_lang_f1_metrics_json": m_win["per_lang_f1"],
        "doc_per_lang_f1": {l: f1_score(doc[doc.lang == l].true_label, doc[doc.lang == l].pred_label,
                                        average="macro") for l in ["de", "en", "nl"]},
        "doc_per_lang_f1_metrics_json": m_doc["per_lang_f1"],
    }
for k, v in checks.items():
    print(k, v)

# ── Document-level P/R/F1 per language (test and validation) ───────────────
recs = []
for split, doc in doc_tables.items():
    for lang in ["de", "en", "nl"]:
        d = doc[doc.lang == lang]
        p, r, f, s = [], [], [], []
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, s = precision_recall_fscore_support(d.true_label, d.pred_label, labels=CEFR,
                                                     zero_division=0)
        for i, lvl in enumerate(CEFR):
            recs.append({"split": split, "language": lang, "level": lvl, "support": int(s[i]),
                         "precision": p[i], "recall": r[i], "f1": f[i]})
        recs.append({"split": split, "language": lang, "level": "macro", "support": len(d),
                     "precision": p.mean(), "recall": r.mean(), "f1": f.mean()})
prf_doc = pd.DataFrame(recs)
prf_doc.to_csv(os.path.join(TAB_DIR, "b6_prf_by_language_class_doc.csv"), index=False,
               float_format="%.4f")
LN = {"de": "German", "en": "English", "nl": "Dutch"}
rows = []
for lang in ["de", "en", "nl"]:
    t = prf_doc[(prf_doc.split == "test") & (prf_doc.language == lang)].set_index("level")
    d = doc_tables["test"][doc_tables["test"].lang == lang]
    rows.append([LN[lang]] + [f"{t.loc[l, 'precision']:.2f} / {t.loc[l, 'recall']:.2f} / {t.loc[l, 'f1']:.2f}"
                              for l in CEFR] +
                [f"{t.loc['macro', 'f1']:.2f}", f"{(d.true_label == d.pred_label).mean():.2f}", str(len(d))])
tex_table(
    os.path.join(TAB_DIR, "b6_prf_by_language_class_doc.tex"),
    caption="Precision, recall and F1 by language and class (test set, document level)",
    label="tab:clf-prf-doc",
    header=["Language", "A (P / R / F1)", "B1 (P / R / F1)", "B2 (P / R / F1)", "C (P / R / F1)",
            "Macro-F1", "Accuracy", "Documents"],
    rows=rows,
    notes=["Document-level predictions: logits mean-pooled over all windows of a text, as in corpus "
           "inference. From the evaluation pass of 04\\_eval\\_per\\_language.py; overall accuracy and "
           "per-language macro-F1 reproduce metrics.json."],
    colspec="lrrrrrrr", size=r"\scriptsize\setlength{\tabcolsep}{3pt}",
)

# Document-level error summary per language (test)
err_rows, err_recs = [], []
for lang in ["de", "en", "nl"]:
    d = doc_tables["test"][doc_tables["test"].lang == lang]
    e = d[d.distance > 0]
    top = (e.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False, kind="mergesort"))
    tl = f"{top.index[0][0]} as {top.index[0][1]} ({top.iloc[0]})"
    rec = {"language": LN[lang], "n": len(d), "errors": len(e), "errors_1": int((e.distance == 1).sum()),
           "errors_2plus": int((e.distance >= 2).sum()), "within_one": (d.distance <= 1).mean(),
           "top_confusion": tl}
    err_recs.append(rec)
    err_rows.append([LN[lang], str(rec["n"]), str(rec["errors"]),
                     f"{rec['errors_1']} ({100 * rec['errors_1'] / rec['errors']:.0f})",
                     f"{rec['errors_2plus']} ({100 * rec['errors_2plus'] / rec['errors']:.0f})",
                     f"{100 * rec['within_one']:.1f}", tl])
pd.DataFrame(err_recs).to_csv(os.path.join(TAB_DIR, "b6_errors_by_language_doc.csv"), index=False,
                              float_format="%.4f")
tex_table(
    os.path.join(TAB_DIR, "b6_errors_by_language_doc.tex"),
    caption="Classification errors by language (test set, document level)",
    label="tab:clf-errors-doc",
    header=["Language", "n", "Errors", "1 level apart (\\%)", "2+ levels apart (\\%)",
            "Within one (\\%)", "Most frequent confusion"],
    rows=err_rows,
    notes=["Percentages in the error columns are shares of all errors. Most frequent confusion: "
           "true level as predicted level (count)."],
    colspec="lrrrrrl", size=r"\small",
)

# ── Dutch rater-supported error analysis ───────────────────────────────────
lookup = cra.build_rater_lookup("UniversalCEFR/elg_cefr_nl")
nl = doc_tables["test"][doc_tables["test"].lang == "nl"]
nl_err = nl[nl.distance > 0]
det = []
for r in nl_err.itertuples():
    key, raw = cra.match_text(r.text_first_window, lookup)
    norm = [x for x in (cra.normalize_label(l) for l in raw) if x]
    agree = None if key is None else (r.pred_label in norm)
    det.append({"sample_id": r.sample_id, "true_label": r.true_label, "pred_label": r.pred_label,
                "distance": r.distance, "n_windows": r.n_windows,
                "raw_rater_labels": "|".join(raw) if raw else "NOT_FOUND",
                "norm_rater_labels": "|".join(norm) if norm else "NOT_FOUND",
                "n_raters": len(norm),
                "true_label_in_raters": (r.true_label in norm) if key else None,
                "any_rater_agrees": "UNKNOWN" if agree is None else ("YES" if agree else "NO"),
                "text_excerpt": r.text_first_window[:140].replace("\n", " ")})
det = pd.DataFrame(det)
local_texts.append(det[["sample_id", "text_excerpt"]]
                   .rename(columns={"text_excerpt": "text_first_window"})
                   .assign(split="test_nl_errors", lang="nl"))
pd.concat(local_texts).to_csv(os.path.join(RES_DIR, "local_only_training_texts.csv"), index=False)
det.drop(columns="text_excerpt").to_csv(os.path.join(TAB_DIR, "b6_rater_check_nl_test.csv"),
                                        index=False)
found = det[det.any_rater_agrees != "UNKNOWN"]
yes = (found.any_rater_agrees == "YES")
res = {"n_nl_test_docs": len(nl), "n_errors": len(nl_err), "n_matched": len(found),
       "n_not_found": int((det.any_rater_agrees == "UNKNOWN").sum()),
       "n_rater_agrees": int(yes.sum()), "share_rater_agrees": float(yes.mean()) if len(found) else None,
       "by_distance": {int(k): {"n": int(len(g)), "agree": int((g.any_rater_agrees == "YES").sum())}
                       for k, g in found.groupby("distance")},
       "rater_count_distribution": found.n_raters.value_counts().sort_index().to_dict(),
       "gold_in_rater_labels": int(found.true_label_in_raters.sum())}
# Also: for correct predictions, how often all raters agree (context)
print("Dutch rater check:", res)

rows = [["Dutch test documents", str(res["n_nl_test_docs"])],
        ["Document-level errors", str(res["n_errors"])],
        ["Matched to raw rater labels", str(res["n_matched"])],
        ["At least one rater gave the predicted level",
         f"{res['n_rater_agrees']} ({100 * res['share_rater_agrees']:.0f}\\%)"],
        ["No rater gave the predicted level",
         f"{res['n_matched'] - res['n_rater_agrees']} ({100 * (1 - res['share_rater_agrees']):.0f}\\%)"]]
for k, v in sorted(res["by_distance"].items()):
    rows.append([f"\\quad errors {k} level{'s' if k > 1 else ''} apart: rater-supported",
                 f"{v['agree']} of {v['n']}"])
tex_table(
    os.path.join(TAB_DIR, "b6_rater_check_nl.tex"),
    caption="Rater support for Dutch classification errors (elg\\_cefr\\_nl, test set)",
    label="tab:clf-rater-nl",
    header=["", "n"],
    rows=rows,
    midrules=(3,),
    notes=["Each Dutch text was rated by several raters; the gold label is their aggregate. An error is "
           "rater-supported if at least one original rater label, mapped to A, B1, B2, C, equals the "
           "model's prediction. Document-level predictions of run cefr\\_20260817\\_231925; texts matched "
           "to the raw dataset on their normalised text."],
    colspec="lr",
)
save_json({"checks": checks, "dutch_rater_check": res, "errors_doc": err_recs}, "b6b_eval_rater.json")
