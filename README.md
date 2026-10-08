# Linguistic Accessibility of E-Government Service Texts in Europe

Code and result files for the bachelor's thesis *Linguistic Accessibility of E-Government Service
Texts in Europe: A Cross-Country Study* (Alina Theresa Huber, TU Berlin, 2026).

The study compares how hard the texts on four national e-government portals are to read:
Germany (verwaltung.bund.de), Austria (oesterreich.gv.at), the Netherlands (rijksoverheid.nl) and
Ireland (gov.ie). The pages were collected on 25 March 2026. A fine-tuned multilingual classifier
(mDeBERTa-v3-base, trained on German, English and Dutch data from UniversalCEFR) gives every page a
CEFR level (A, B1, B2 or C). A local language model (qwen2.5:7b, zero-shot) assigns each page an
information area from Annex I of the EU Single Digital Gateway Regulation (SDGR). Logistic
regression then relates the chance that a page is harder (B2 or C) to the country and to the type
of content.

## Pipeline

1. **Data collection** (per country): discover URLs from the portal's sitemap (step 00), filter
   and validate them (step 01), extract the page text (step 02).
2. **CEFR classifier**: prepare the UniversalCEFR data, run a small hyperparameter sweep, train the
   final model (run `cefr_20260817_231925`), evaluate it per language.
3. **CEFR inference**: predict a CEFR level for every portal page (512-token windows, stride 128,
   averaged over the whole page).
4. **SDGR categorisation**: assign each page an SDGR area with qwen2.5:7b via Ollama, and validate
   the result against 90 manually annotated pages.
5. **Analysis**: build the analysis data set, fit the regressions and robustness checks, check
   convergent validity against sentence length and Flesch scores, and draw the tables and figures.

## Folder structure

```
cefr-egov-accessibility-public/
|-- country_pipelines/
|   |-- germany/              # 00_discover, 01_clean, 02_scrape scripts and their outputs
|   |-- austria/              #   (same layout for each country)
|   |-- netherlands/
|   `-- ireland/              # 03_cefr_output_<country>/: page-level CEFR predictions
|-- classifier/               # 00_prepare_data ... 04_eval_per_language, run metadata
|-- analysis/
|   |-- data_collection/      # rebuilds and checks the data collection tables
|   |-- sdg/                  # SDGR labels; older categoriser kept only as an imported module
|   |-- regression/           # categoriser used for the labels, build_dataframe, run_models
|   |-- sampling/             # 90-page manual annotation sample and codebook
|   |-- readability/          # sentence length and Flesch scores per page
|   |-- error_analysis/       # classifier error inspection and rater check
|   `-- final_oct/            # final scripts, tables (.tex/.csv), figures (.pdf), results
|-- requirements.txt
|-- LICENSE
`-- README.md
```

`analysis/final_oct/` holds the final analysis run used in the thesis (October 2026); every table,
figure and quoted number in the thesis that comes from code was produced there or by the scripts
it calls.

Every script resolves its paths from its own location, so the repository works wherever it is
cloned. Each `.py` file starts with a header that names its pipeline step and the thesis section
or LaTeX label it produces.

## Setup

Python 3.11 (tested with 3.11.15).

```bash
pip install -r requirements.txt
```

The SDGR categorisation needs [Ollama](https://ollama.com) running locally with the model
`qwen2.5:7b`:

```bash
ollama pull qwen2.5:7b
```

## How to run the steps

Run the commands from the repository root. Seeds are fixed at 42 throughout.

**1. Data collection** (network access; `<c>` = germany, austria, netherlands, ireland)

```bash
python country_pipelines/<c>/00_discover_urls_<c>.py
python country_pipelines/<c>/01_clean_urls_<c>.py
python country_pipelines/<c>/02_scrape_urls_<c>.py
python analysis/data_collection/make_collection_tables.py   # checks the Chapter 3 and Annex tables
```

The portals change all the time. Running the collection scripts again will **not** reproduce the
corpus of 25 March 2026. The repository therefore ships the URL lists that step 01 kept
(`01_clean_output_<c>/`), the counts and rejection reasons, and a copy of `combined_<c>.csv`
without the page text (see "What is not included").

All page text was extracted from the static HTML. The German scraper contains a Playwright
fallback for pages rendered by JavaScript; it did not run when the corpus was collected (thesis
Section 3.4.3), so Playwright is not needed and is not in `requirements.txt`.

`make_collection_tables.py` rebuilds the corpus construction funnel, the rejected URLs by reason,
the page-length table and the Annex URL path tables from the shipped summary files, reject lists
and `combined_<c>.csv`, and checks every row and every count in the table notes against the
shipped `.tex` files. It only reads files.

**2. Classifier** (downloads UniversalCEFR from Hugging Face)

```bash
python classifier/00_prepare_data.py           # German, English, Dutch; seed 42
python classifier/01_sweep_params.py           # learning rate x effective batch size
python classifier/02_train_final.py            # final model, seed 42
python classifier/04_eval_per_language.py --run cefr_20260817_231925
```

Portuguese was part of an earlier design and was removed from the final model (thesis
Section 4.2.1). The hyperparameter sweep ran once, on the four-language data including Portuguese,
and was not repeated. `02_train_final.py` reads its result from
`classifier/01_sweep_output/configs/best_params.json` (learning rate 2e-5, gradient accumulation
8, effective batch 16), so the final model uses those settings (thesis Section 4.2.2).

A new training run gets a new time-stamped run ID; the thesis uses run `cefr_20260817_231925`.
Its metrics, manifest, selection summary and training log are in
`classifier/02_train_output/runs/cefr_20260817_231925/` (machine-specific path prefixes in the
JSON files are replaced by `<repo>/`). Exact reproduction is not guaranteed on Apple MPS
hardware, which the script reports when it sets the seed.

**3. CEFR inference** (needs the trained model and the page texts)

```bash
python classifier/03_infer_cefr.py --country <c>     # or --all for the four countries
```

This writes `country_pipelines/<c>/03_cefr_output_<c>/<c>_cefr.csv`.

**4. SDGR categorisation and validation** (needs Ollama and the page texts)

```bash
python analysis/regression/sdg_categorize_full.py         # labels -> analysis/sdg/sdg_categorization_full.csv
python analysis/sampling/sample_manual_annotation.py      # draws the 90-page sample (seed 42)
```

`analysis/regression/sdg_categorize_full.py` produced the labels on 21 June 2026 (reasoning
prompt, deterministic rule for Austrian "Leben in der Gemeinde/Stadt" pages). That run also
covered Portugal; Portugal has since been removed from the script's country list and from the
label file.

The older categoriser in `analysis/sdg/sdg_categorize_full.py` is not used for any thesis result.
It is kept only as a module: `sample_manual_annotation.py` imports it to build the codebook, and
`b2_unread_text.py` reads its prompt. Do not run it on its own: a full run would overwrite the
label file.

The validation (thesis Section 3.6.3) is computed by `analysis/final_oct/scripts/b3_zeroshot_validation.py`.
That script also reads `analysis/sampling/zeroshot_area_predictions.csv`, a 26 August re-run of
the 90 pages with the older prompt, and writes it as extra rows into
`tables/b3_zeroshot_validation.tex`. The re-run is not part of the thesis: the thesis reports only
the rows for the corpus labels and the majority and random baselines. The file is shipped only
because the script cannot run without it.

**5. Analysis**

```bash
python analysis/regression/build_dataframe.py     # -> analysis/regression/analysis_dataframe.csv
python analysis/regression/run_models.py          # model grid (cross-check)
python analysis/readability/convergent_validity_flesch.py   # needs the page texts

cd analysis/final_oct/scripts
python b1_descriptives.py
python b3_zeroshot_validation.py
python b4_regression.py
python b5_figures.py
python sens_regression.py
python ch5_analysis.py
python ch5b_analysis.py            # needs the page texts
python ch5b_cefr_by_scope.py
python ch5b_e_by_area.py
python b6_classifier_tables.py
python b6b_eval_and_rater_check.py # needs the model and the tokenised data
python b6c_classifier_figures.py
python b5_readability_and_copies.py   # needs the page texts and the model
python b2_unread_text.py --qwen-tokenizer <dir>   # needs the page texts and the qwen2.5 tokenizer
python b7_examples.py              # needs the page texts
python ch6_convergent.py
```

Outputs go to `analysis/final_oct/tables/`, `figures/` and `results/`. Scripts that run later in
the list read CSV files written by earlier ones (for example `b4_regression.py` before
`b5_figures.py`, `sens_regression.py` and `ch5_analysis.py`). Some scripts also draw figures or
write tables that are not in the thesis; those outputs are not shipped and are listed in
`.gitignore`.

## What is not included, and why

| Not included | Why | Steps that need it |
|---|---|---|
| Page texts (`title`, `text_joined` in `combined_<c>.csv`), sentence files, the text of the annotation sample, text snippets in the SDGR labels, the example texts written by `b7_examples.py` | The texts belong to the portal operators and are not republished | CEFR inference, SDGR categorisation, annotation sampling, readability measures, `ch5b_analysis.py`, `b2_unread_text.py`, `b5_readability_and_copies.py`, `b7_examples.py` |
| Trained model weights and checkpoints | Size (about 1 GB); can be retrained | CEFR inference, `04_eval_per_language.py`, `inspect_classifier_errors.py`, `b6b_eval_and_rater_check.py`, the model check in `b5_readability_and_copies.py` |
| Tokenised UniversalCEFR data and raw UniversalCEFR texts | Third-party corpora with their own licences; download from https://huggingface.co/UniversalCEFR | Classifier training and evaluation |
| Raw URL lists from step 00 | Size (Ireland alone over 16 MB); the per-path counts are in the shipped step 00 summary files | Re-running step 01 |

Where a shipped file used to contain page text, it is kept under its original name with the text
columns removed, so the scripts still find it: `combined_<c>.csv` (URL, category, sentence and
word counts), `analysis/sdg/sdg_categorization_full.csv` (snippet removed; the Portuguese rows of
the original run removed), and `analysis/sampling/manual_annotation_sample_ANNOTATED.csv` (URL,
country, manual label). Rerunning the scrapers overwrites `combined_<c>.csv` with a version that
contains text again.

The following run from the shipped files alone: `make_collection_tables.py`,
`build_dataframe.py`, `run_models.py`, `b1_descriptives.py`, `b3_zeroshot_validation.py`,
`b4_regression.py`, `b5_figures.py` (it uses the saved token counts in
`tables/b5_fig4_tokens_E_de_at.csv` when the model is missing), `sens_regression.py`,
`ch5_analysis.py`, `ch5b_cefr_by_scope.py`, `ch5b_e_by_area.py`, `b6_classifier_tables.py`,
`b6c_classifier_figures.py` and `ch6_convergent.py`.

## Licence

The code is released under the MIT licence (see `LICENSE`). The licence covers the code only. It
does not cover the portal content, the UniversalCEFR data or the thesis text.
