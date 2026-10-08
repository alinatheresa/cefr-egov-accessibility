# Bachelor Thesis, TU Berlin 2026
# Topic: Linguistic Accessibility of E-Government Service Texts in Europe: A Cross-Country Study
# Author: Alina Theresa Huber
# File: analysis/final_oct/scripts/common.py
# Purpose: shared paths, constants, loaders and LaTeX helpers for the final_oct outputs.
#          Reads existing pipeline outputs only; never re-runs inference or categorisation.
# Pipeline step: Shared helpers for all scripts in analysis/final_oct/scripts/.
# Thesis: None directly.

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR    = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(FINAL_DIR))
FIG_DIR      = os.path.join(FINAL_DIR, "figures")
TAB_DIR      = os.path.join(FINAL_DIR, "tables")
RES_DIR      = os.path.join(FINAL_DIR, "results")   # machine-readable intermediate results
for _d in (FIG_DIR, TAB_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

RUN_ID  = "cefr_20260817_231925"
RUN_DIR = os.path.join(PROJECT_ROOT, "classifier", "02_train_output", "runs", RUN_ID)

COUNTRIES = ["germany", "austria", "netherlands", "ireland"]
COUNTRY_LABEL = {"germany": "Germany", "austria": "Austria",
                 "netherlands": "Netherlands", "ireland": "Ireland"}
COUNTRY_COLOUR = {"germany": "#0072B2", "austria": "#D55E00",
                  "netherlands": "#CC79A7", "ireland": "#009E73"}
CORPUS_SIZE = {"germany": 1264, "austria": 3736, "netherlands": 2410, "ireland": 1588}

CEFR = ["A", "B1", "B2", "C"]
CEFR_COLOUR = {"A": "#C0D5F1", "B1": "#6E9BD8", "B2": "#3F6FC4", "C": "#1B3985"}
CEFR_TEXT_COLOUR = {"A": "black", "B1": "black", "B2": "white", "C": "white"}
# Continuous CEFR scale (A to C, index 1 to 4)
CEFR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("cefr", [CEFR_COLOUR[l] for l in CEFR])
# Confusion matrices as in the training script (sklearn ConfusionMatrixDisplay defaults)
CM_CMAP = matplotlib.colormaps["viridis"]


def cm_text_colour(value, cm):
    """Cell text colour as sklearn draws it: darkest map colour on bright cells, brightest on dark."""
    return CM_CMAP(0.0) if value > (cm.max() + cm.min()) / 2 else CM_CMAP(1.0)


SCOPE_COLOUR = {"citizen": "#56B4E9", "business": "#E69F00"}
GRID_COLOUR = "#e0e0e0"
LINE_GREY = "#555555"

CITIZEN_AREAS  = list("ABCDEFGHI")
BUSINESS_AREAS = list("JKLMNOPQ")
AREA_NAMES = {
    "A": "Travel", "B": "Work and retirement", "C": "Vehicles",
    "D": "Residence", "E": "Education or traineeship", "F": "Healthcare",
    "G": "Citizens' and family rights", "H": "Consumer rights",
    "I": "Protection of personal data",
    "J": "Starting, running, closing a business", "K": "Employees (employer side)",
    "L": "Taxes", "M": "Goods", "N": "Services and qualifications",
    "O": "Funding a business", "P": "Public contracts",
    "Q": "Health and safety at work",
}

SDG_CSV = os.path.join(PROJECT_ROOT, "analysis", "sdg", "sdg_categorization_full.csv")
ANALYSIS_DF = os.path.join(PROJECT_ROOT, "analysis", "regression", "analysis_dataframe.csv")
ANNOTATION_CSV = os.path.join(PROJECT_ROOT, "analysis", "sampling",
                              "manual_annotation_sample_ANNOTATED.csv")
ZS_CACHE_CSV = os.path.join(PROJECT_ROOT, "analysis", "sampling", "zeroshot_area_predictions.csv")

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def cefr_csv(country):
    return os.path.join(PROJECT_ROOT, "country_pipelines", country,
                        f"03_cefr_output_{country}", f"{country}_cefr.csv")


def combined_csv(country):
    return os.path.join(PROJECT_ROOT, "country_pipelines", country,
                        f"02_scrape_output_{country}", f"combined_{country}.csv")


def load_cefr(include_unlabelled=False):
    frames = []
    for c in COUNTRIES:
        d = pd.read_csv(cefr_csv(c))
        if len(d) != CORPUS_SIZE[c]:
            raise RuntimeError(f"{c}: {len(d)} CEFR rows, expected {CORPUS_SIZE[c]}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if not include_unlabelled:
        df = df[df["pred_label"].isin(CEFR)].copy()
    df["E"] = df["prob_A"] + 2 * df["prob_B1"] + 3 * df["prob_B2"] + 4 * df["prob_C"]
    return df


def load_text():
    frames = []
    for c in COUNTRIES:
        d = pd.read_csv(combined_csv(c), dtype=str).fillna("")
        d["country"] = c
        frames.append(d[["country", "category", "url", "title", "text_joined"]])
    return pd.concat(frames, ignore_index=True)


def load_sdg():
    sdg = pd.read_csv(SDG_CSV, dtype=str).fillna("")
    sdg = sdg[sdg["country"].isin(COUNTRIES)].copy()
    sdg["rule"] = sdg["assigned_reasoning"].str.startswith("Deterministic")
    return sdg


def scope_of(area):
    if area in CITIZEN_AREAS:
        return "citizen"
    if area in BUSINESS_AREAS:
        return "business"
    return None


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> figures/{name}.pdf/.png")


def save_json(obj, name):
    with open(os.path.join(RES_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


# ── LaTeX helpers (hline borders, threeparttable, no booktabs) ────────────────

def tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def tex_table(path, caption, label, header, rows, notes, colspec=None, midrules=(),
              size=None):
    """header: list of cells (already escaped/LaTeX); rows: list of lists of cells.
    midrules: row indices before which an extra \\hline is drawn."""
    ncol = len(header)
    colspec = colspec or ("l" + "r" * (ncol - 1))
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if size:
        lines.append(size)
    lines += [r"\begin{threeparttable}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\begin{{tabular}}{{{colspec}}}", r"\hline",
              " & ".join(header) + r" \\", r"\hline"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\hline")
        lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\begin{tablenotes}", r"\footnotesize"]
    for n in notes:
        lines.append(rf"\item {n}")
    lines += [r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  table  -> tables/{os.path.basename(path)}")


def fmt_p(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def fmt_int(n):
    return f"{int(n):,}"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centre - half, centre + half)
