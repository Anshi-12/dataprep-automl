# CMPE 255 · Data Preparation, EDA & AutoML (AutoViML) + Agentic EDA
**Anshita Shah · MSSE, San José State University · Fall 2026**

End-to-end data preparation on two messy Kaggle datasets: a **tabular** one (NYC Airbnb) and a **time series** (household power, 2M rows).
Each goes through detailed EDA → cleaning → imputation → anomaly removal & clustering → feature engineering & selection → **AutoViML** models.
In Part 2, AI **data agents** do the same EDA, and I audit their work against my manual EDA.

## 🎥 Video walkthroughs (code + user journey)
| Video | Covers | Link |
|---|---|---|
| 1 | Part 1 (i): Tabular, NYC Airbnb: user journey + cell-by-cell code walkthrough + results | ▶️ **[YouTube: ADD LINK]**(https://youtu.be/ADD_LINK) |
| 2 | Part 1 (ii): Time series, Household Power: user journey + code walkthrough + forecasting results | ▶️ **[YouTube: ADD LINK]**(https://youtu.be/ADD_LINK) |
| 3 | Part 2: Agentic EDA. Colab Data Science Agent live + my Gemini agent + audit vs manual EDA | ▶️ **[YouTube: ADD LINK]**(https://youtu.be/ADD_LINK) |

## 📁 Repository
```
├── part1_tabular_airbnb/
│   ├── 01_airbnb_eda_prep_autoviml.ipynb        ← executed notebook (all outputs visible)
│   ├── 01_airbnb_eda_prep_autoviml.py           ← same notebook as plain Python (jupytext), easy to diff
│   └── outputs_airbnb/  figures/ · reports/ (ydata-profiling, Sweetviz HTML) · model_results.csv · AutoViML logs
├── part1_timeseries_power/
│   ├── 02_power_timeseries_eda_prep_autoviml.ipynb
│   └── outputs_power/   figures/ · reports/ · model_results.csv · AutoViML logs
├── part2_agentic_eda/
│   ├── 03_agentic_eda_gemini_agent.ipynb        ← Tool B: my Gemini code-execution agent + audit
│   ├── PROMPTS_colab_data_science_agent.md      ← Tool A: exact prompts for Colab's Data Science Agent
│   ├── 03b_colab_ds_agent_airbnb.ipynb          ← Tool A's generated notebook (added after the run)
│   └── outputs_agent/   agent_trace.md · agent_report.md · audit CSVs · figures
├── video_scripts/   VIDEO_1 · VIDEO_2 · VIDEO_3 (timestamped scripts)
├── docs/CELL_GUIDE.md   cell-by-cell map of every notebook
├── HOW_TO_RUN.md · requirements.txt
```
**Run it:** see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**. In Colab: open notebook → *Run all* → it installs the AutoViML pins and restarts once → *Run all* again.

---

## Part 1 (i) — Tabular: NYC Airbnb 2019
**Data:** [Kaggle: dgomonov/new-york-city-airbnb-open-data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), 48,895 listings × 16 columns (numeric, low/high-cardinality categorical, free text, date, lat/long).
**Task:** predict nightly price (regression on `log1p(price)`).

| Step | What I did | Key finding |
|---|---|---|
| Audit | types, missingness, duplicates, impossible values | `reviews_per_month` & `last_review` missing in the **same 10,052 rows, all with 0 reviews** → structural, not random. 11 listings priced **$0**; `minimum_nights` up to **1,250** |
| Auto-EDA | **ydata-profiling**, **Sweetviz** (vs. price), **pandas-dq** | table of tool findings vs. my interpretation (where tools would mislead) |
| Manual EDA | 14 figures: missingness matrix, log-price, borough × room type, geo maps, Spearman, text keywords | location × room type dominate; "luxury/penthouse" titles ≈2× median; the 30-night spike is NYC's 30-day rule, not noise |
| Cleaning | logged rule list | removed 11 $0 prices + 14 `minimum_nights` > 365; text normalized; types fixed |
| Imputation | benchmark **mean / median / KNN / MICE** on masked values | KNN best (MAE 0.71) > MICE 0.86 > median 1.13 > mean 1.24, **but** the real gaps are structural zeros, so zero + `never_reviewed` flag; KNN reused for 218 suspect `minimum_nights` |
| Anomalies | IQR, robust z, **Isolation Forest**, **LOF**, **DBSCAN** (haversine) | consensus policy (≥3 detectors) removed only **41** rows (single detectors flagged 489–2,101 each) |
| Clustering | **K-Means** (elbow + silhouette) | 7 market segments → used as a feature |
| Features | haversine distances, equi-depth bins, keyword flags, **TF-IDF→SVD (LSA)**, OOV bucket + frequency + out-of-fold target encoding | |
| Selection | **filter** (mutual info) + **wrapper** (forward SFS) + **embedded** (Lasso, RF) voting | 73 candidates → **45** features (≥2 of 4 votes, redundancy removed), on training rows only |
| Models | baselines + **AutoViML** (Random Forest, XGBoost, CatBoost) | table below |

**Hold-out results (20% test, 9,766 listings):**

| Model | RMSE (log) | R² (log) | MAE ($) | Median abs. error ($) |
|---|---|---|---|---|
| Baseline: training median | 0.703 | −0.01 | 86.7 | 46.0 |
| **OLS linear regression (selected features)** | **0.448** | **0.590** | 59.8 | 24.8 |
| AutoViML D: CatBoost | 0.454 | 0.579 | 60.2 | 24.6 |
| AutoViML C: XGBoost | 0.456 | 0.575 | 60.1 | 24.7 |
| AutoViML B: Random Forest | 0.471 | 0.547 | 61.6 | 25.5 |

**Honest finding:** after removing a target-leakage bug (price-based anomaly flags used as features had inflated R² to ~0.67), linear regression on the engineered features narrowly beats AutoViML's boosted models. The feature engineering carried the signal. AutoViML's linear mode (LinearSVR grid) did not converge on 38k rows, so OLS represents the linear family.

## Part 1 (ii) — Time series: Household Electric Power Consumption
**Data:** [Kaggle: uciml/electric-power-consumption-data-set](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set), 2,075,259 one-minute readings (Dec 2006 – Nov 2010), 7 signals.
**Task:** forecast next-hour average household power (kW). Chronological split: train 2007–2009, test = all of 2010.

| Step | What I did | Key finding |
|---|---|---|
| Parse & audit | `'?'` → NaN, date+time → index, gap-run analysis | 1.25% missing in **71 gaps**; all 7 sensors drop together; 6 gaps > 1 day (max ≈ 5 days) |
| Auto-EDA | ydata-profiling **TS mode**, **Sweetviz train-vs-test compare**, pandas-dq | non-stationary + seasonal target; kitchen circuit zero 84.5% of hours |
| Manual EDA | overview, hour×weekday heatmap, monthly boxplots, **STL**, **ACF/PACF**, sub-meter shares | daily/weekly/annual seasonality; lags 1–3, 24, 168 matter; August vacations |
| Cleaning | **physics rules**: I ≈ √(P²+Q²)/V, sub-meters ≤ total, voltage within ±10% | **524** inconsistent minutes → NaN (434 current mismatches > 2 A, 33 sub-meters > total, 57 voltage out of range) |
| Imputation | benchmark **6 methods × 6 gap lengths** on hidden real data | linear interpolation wins ≤ 1 h (MAE 0.11 kW at 1 min); time-of-week median profile wins ≥ 6 h → **hybrid imputer**, switching at 360 min |
| Reduction | resample minute → hourly (+ within-hour max/std/imputed share) | 2.07M → ~34.6k rows |
| Anomalies | **Hampel**, **STL residual**, **Isolation Forest** (points); **K-Means + DBSCAN on 24-h profiles** (days) | 866 hours flagged by ≥2 detectors → 732 training hours smoothed; **k = 2** day types (winter-high vs. summer/vacation-low); 607 training hours on DBSCAN-anomalous days removed; 2010 left raw |
| Features | lags, rolling stats, calendar + French holidays (sin/cos), Fourier, **Haar wavelets**, **SAX** words, yesterday's day type | |
| Selection | same filter/wrapper/embedded voting, train period only | 62 candidates → **35** features; top: intensity, within-hour max, unmetered load, lag 167 h, lag 23 h |
| Models | naive baselines (persistence, same hour yesterday/last week), OLS, **AutoViML** (RF, XGBoost, CatBoost) | table below |

**Test results (all of 2010, 7,916 hours, never seen during cleaning/selection/training):**

| Model | RMSE (kW) | MAE (kW) | R² | vs. persistence |
|---|---|---|---|---|
| Naive: same hour last week | 0.823 | 0.570 | −0.13 | |
| Naive: same hour yesterday | 0.811 | 0.547 | −0.10 | |
| Naive: persistence (next hour = this hour) | 0.614 | 0.402 | 0.37 | baseline |
| OLS linear regression | 0.507 | 0.360 | 0.57 | −17% |
| AutoViML B: Random Forest | 0.489 | 0.344 | 0.60 | −20% |
| AutoViML C: XGBoost | 0.475 | 0.332 | 0.62 | −23% |
| **AutoViML D: CatBoost** | **0.467** | **0.321** | **0.64** | **−24%** |

## Part 2 — Agentic EDA (AI agents instead of manual EDA)
| Tool | What it is | Autonomy (lecture's L0–L5) |
|---|---|---|
| **A. Colab Data Science Agent** (Gemini in Colab) | goal in plain English → plan → agent writes, runs, and debugs cells | L2 → Proto-L3 |
| **B. My Gemini code-execution agent** | plan → `run_python` tool → observe → repeat; step budget, sandbox rules, full trace, **self-verification turn** | Proto-L3 |

Both agents are scored against **14 ground-truth findings** from my manual EDA (Part 1), e.g. did the agent explain *why* `reviews_per_month` is missing,
catch the $0 prices, and recognize the 30-night rule? Results and reflection: Section 5–6 of `03_agentic_eda_gemini_agent.ipynb`.

| Metric | Colab DS Agent (A) | Gemini agent (B) |
|---|---|---|
| Ground-truth findings covered (of 14) | _fill after run_ | _fill after run_ |
| Wrong / hallucinated claims caught in verification | | |
| Time | | |

---

## 🤖 How a coding agent was used (disclosure)
As the assignment allows, I used an AI coding assistant (Claude) to help write and debug the notebook code, and to fix library-compatibility issues
(AutoViML's numpy/scipy/IPython pins and its XGBoost `updater` bug). **My decisions:** choosing the datasets and the target, the cleaning rules,
interpreting every chart, the imputation and anomaly policies (e.g. structural zero-fill instead of KNN, the consensus rule, keeping the 30-night spike),
and the Part 2 evaluation design. Every number in this README comes from the executed notebooks.

## References
- Course slides: *CMPE 255 — Data Preparation / DataPrepOps* (feature portability, imputation, MICE/KNN, wavelets, SAX, feature selection, AutoViML, data agents L0–L5)
- AutoViML: https://github.com/AutoViML/Auto_ViML · pandas-dq: https://github.com/AutoViML/pandas_dq
- ydata-profiling: https://github.com/ydataai/ydata-profiling · Sweetviz: https://github.com/fbdesignpro/sweetviz
- Colab Data Science Agent: https://docs.cloud.google.com/colab/docs/use-data-science-agent
- Hebrail & Berard (2006), *Individual Household Electric Power Consumption*, UCI ML Repository, doi:10.24432/C58K54
