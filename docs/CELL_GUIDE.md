# Cell guide

Cell numbers count every cell (markdown + code) from the top, matching the video scripts.

## `part1_tabular_airbnb/01_airbnb_eda_prep_autoviml.ipynb`

| Cell | Type | Section | What it does |
|---|---|---|---|
| 1 | 📝 md | Part 1 (i) — Tabular: NYC Airbnb 2019 | Part 1 (i) — Tabular: NYC Airbnb 2019 |
| 2 | 📝 md | 0. Setup | 0. Setup |
| 3 | 🐍 code | 0. Setup | import importlib.metadata as md, os, sys |
| 4 | 🐍 code | 0. Setup | Output folders. Everything this notebook produces is saved next to it. |
| 5 | 📝 md | 1. Load the data | 1. Load the data |
| 6 | 🐍 code | 1. Load the data | DATA = Path("data"); DATA.mkdir(exist_ok=True) |
| 7 | 📝 md | 2. First look & data audit | 2. First look & data audit |
| 8 | 🐍 code | 2. First look & data audit | def audit(df): |
| 9 | 🐍 code | 2. First look & data audit | num_cols = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_cou |
| 10 | 🐍 code | 2. First look & data audit | checks = { |
| 11 | 📝 md | 2. First look & data audit | Audit findings (these drive the cleaning plan): |
| 12 | 📝 md | 3. Automated EDA ("auto EDA" + "auto DS") | 3. Automated EDA ("auto EDA" + "auto DS") |
| 13 | 🐍 code | 3. Automated EDA ("auto EDA" + "auto DS") | from ydata_profiling import ProfileReport |
| 14 | 🐍 code | 3. Automated EDA ("auto EDA" + "auto DS") | import sweetviz as sv |
| 15 | 🐍 code | 3. Automated EDA ("auto EDA" + "auto DS") | from pandas_dq import dq_report |
| 16 | 📝 md | What the automated tools found, and where my manual reading  | What the automated tools found, and where my manual reading disagrees |
| 17 | 📝 md | 4. Manual EDA & visualization | 4. Manual EDA & visualization |
| 18 | 🐍 code | 4. Manual EDA & visualization | fig, ax = plt.subplots(figsize=(9, 3.2)) |
| 19 | 📝 md | 4.2 Target: price is extremely right-skewed. Log space revea | 4.2 Target: price is extremely right-skewed. Log space reveals the structure |
| 20 | 🐍 code | 4.2 Target: price is extremely right-skewed. Log space revea | p = raw.loc[raw.price > 0, "price"] |
| 21 | 📝 md | 4.3 Price by borough × room type | 4.3 Price by borough × room type |
| 22 | 🐍 code | 4.3 Price by borough × room type | fig, ax = plt.subplots(figsize=(10, 4)) |
| 23 | 📝 md | 4.4 Spatial view: where the expensive listings are | 4.4 Spatial view: where the expensive listings are |
| 24 | 🐍 code | 4.4 Spatial view: where the expensive listings are | fig, axes = plt.subplots(1, 2, figsize=(13, 5.2)) |
| 25 | 📝 md | 4.5 Numeric relationships | 4.5 Numeric relationships |
| 26 | 🐍 code | 4.5 Numeric relationships | corr_cols = num_cols + ["latitude", "longitude"] |
| 27 | 🐍 code | 4.5 Numeric relationships | fig, axes = plt.subplots(1, 3, figsize=(13, 3.4)) |
| 28 | 📝 md | 4.6 Time: when were listings last reviewed? | 4.6 Time: when were listings last reviewed? |
| 29 | 🐍 code | 4.6 Time: when were listings last reviewed? | lr = pd.to_datetime(raw.last_review) |
| 30 | 📝 md | 4.7 Text: do listing titles carry price signal? | 4.7 Text: do listing titles carry price signal? |
| 31 | 🐍 code | 4.7 Text: do listing titles carry price signal? | kw = ["luxury", "penthouse", "loft", "studio", "cozy", "private", "shared", "spacious", "view", "room", "bed", |
| 32 | 📝 md | 4.7 Text: do listing titles carry price signal? | Manual EDA takeaways |
| 33 | 📝 md | 5. Cleaning | 5. Cleaning |
| 34 | 🐍 code | 5. Cleaning | 5.1 Types |
| 35 | 📝 md | 6. Imputation: compare strategies, then choose | 6. Imputation: compare strategies, then choose |
| 36 | 🐍 code | 6. Imputation: compare strategies, then choose | from sklearn.experimental import enable_iterative_imputer  # noqa: F401 |
| 37 | 🐍 code | 6. Imputation: compare strategies, then choose | fig, ax = plt.subplots(figsize=(7, 2.8)) |
| 38 | 📝 md | 6. Imputation: compare strategies, then choose | Decision: the model-based imputers (KNN/MICE) beat mean/median by a wide margin when values are missing at ran |
| 39 | 🐍 code | 6. Imputation: compare strategies, then choose | df["never_reviewed"] = (df.number_of_reviews == 0).astype(int) |
| 40 | 📝 md | 7. Anomaly detection & clustering | 7. Anomaly detection & clustering |
| 41 | 🐍 code | 7. Anomaly detection & clustering | LOF on (location, room type, log price): flags listings priced very differently from nearby, similar listings |
| 42 | 🐍 code | 7. Anomaly detection & clustering | pd.crosstab(df.n_flags, df.room_type, margins=True) |
| 43 | 🐍 code | 7. Anomaly detection & clustering | fig, axes = plt.subplots(1, 3, figsize=(14, 4.2)) |
| 44 | 📝 md | Anomaly treatment policy | Anomaly treatment policy |
| 45 | 🐍 code | Anomaly treatment policy | n = len(df) |
| 46 | 📝 md | 7.2 Clustering: market segments (K-Means) | 7.2 Clustering: market segments (K-Means) |
| 47 | 🐍 code | 7.2 Clustering: market segments (K-Means) | seg_cols = ["latitude", "longitude", "room_entire", "room_shared", "log_min_nights", "log_reviews", "availabil |
| 48 | 🐍 code | 7.2 Clustering: market segments (K-Means) | km = KMeans(n_clusters=best_k, n_init=20, random_state=RNG).fit(Zs) |
| 49 | 🐍 code | 7.2 Clustering: market segments (K-Means) | fig, ax = plt.subplots(figsize=(6.5, 6.5)) |
| 50 | 📝 md | 8. Feature engineering | 8. Feature engineering |
| 51 | 🐍 code | 8. Feature engineering | spatial → numeric |
| 52 | 📝 md | 9. Feature selection: filter, wrapper, embedded | 9. Feature selection: filter, wrapper, embedded |
| 53 | 🐍 code | 9. Feature selection: filter, wrapper, embedded | Leakage guard: every column computed FROM price is excluded (the target, and the price-based anomaly flags/sco |
| 54 | 🐍 code | 9. Feature selection: filter, wrapper, embedded | redundancy filter among the kept features: drop the weaker of any pair with \|ρ\| > 0.9 |
| 55 | 📝 md | 10. Models with AutoViML | 10. Models with AutoViML |
| 56 | 🐍 code | 10. Models with AutoViML | from sklearn.model_selection import train_test_split |
| 57 | 🐍 code | 10. Models with AutoViML | AutoViML 0.2.2 bug with its own pinned xgboost 1.7.6: it forces updater='grow_colmaker' together with tree_met |
| 58 | 📝 md | 11. Results & takeaways | 11. Results & takeaways |
| 59 | 🐍 code | 11. Results & takeaways | res = pd.DataFrame(results).set_index("model").round(4) |
| 60 | 🐍 code | 11. Results & takeaways | r = res.sort_values("RMSE_log", ascending=False) |
| 61 | 🐍 code | 11. Results & takeaways | best = res.RMSE_log.idxmin() |
| 62 | 📝 md | Takeaways (from the executed run) | Takeaways (from the executed run) |

## `part1_timeseries_power/02_power_timeseries_eda_prep_autoviml.ipynb`

| Cell | Type | Section | What it does |
|---|---|---|---|
| 1 | 📝 md | Part 1 (ii) — Time Series: Household Electric Power Consumpt | Part 1 (ii) — Time Series: Household Electric Power Consumption |
| 2 | 📝 md | 0. Setup | 0. Setup |
| 3 | 🐍 code | 0. Setup | import importlib.metadata as md, os, sys |
| 4 | 🐍 code | 0. Setup | import warnings, shutil, urllib.request, zipfile, time, re, io, contextlib |
| 5 | 📝 md | 1. Load & parse | 1. Load & parse |
| 6 | 🐍 code | 1. Load & parse | DATA = Path("data"); DATA.mkdir(exist_ok=True) |
| 7 | 🐍 code | 1. Load & parse | raw.dtypes.to_frame("dtype after read") |
| 8 | 🐍 code | 1. Load & parse | Parse the two text columns into one timestamp index (explicit format = 10× faster and no day/month ambiguity) |
| 9 | 📝 md | 2. Data audit | 2. Data audit |
| 10 | 🐍 code | 2. Data audit | full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq="min") |
| 11 | 🐍 code | 2. Data audit | Gap structure: how long is each run of consecutive missing minutes? |
| 12 | 🐍 code | 2. Data audit | ts.describe(percentiles=[.001, .01, .5, .99, .999]).T.round(3) |
| 13 | 📝 md | 2. Data audit | Audit findings: |
| 14 | 📝 md | 3. Automated EDA | 3. Automated EDA |
| 15 | 🐍 code | 3. Automated EDA | hourly_raw = ts.resample("h").mean() |
| 16 | 🐍 code | 3. Automated EDA | import sweetviz as sv |
| 17 | 🐍 code | 3. Automated EDA | from pandas_dq import dq_report |
| 18 | 📝 md | Reading the automated reports | Reading the automated reports |
| 19 | 📝 md | 4. Manual EDA & visualization | 4. Manual EDA & visualization |
| 20 | 🐍 code | 4. Manual EDA & visualization | daily = ts.GAP.resample("D").mean() |
| 21 | 📝 md | 4.2 Seasonality at three scales: hour × weekday, month | 4.2 Seasonality at three scales: hour × weekday, month |
| 22 | 🐍 code | 4.2 Seasonality at three scales: hour × weekday, month | fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]}) |
| 23 | 📝 md | 4.3 Decomposition (STL) and autocorrelation | 4.3 Decomposition (STL) and autocorrelation |
| 24 | 🐍 code | 4.3 Decomposition (STL) and autocorrelation | from statsmodels.tsa.seasonal import STL |
| 25 | 📝 md | 4.4 What is the energy used for? Sub-metering share | 4.4 What is the energy used for? Sub-metering share |
| 26 | 🐍 code | 4.4 What is the energy used for? Sub-metering share | wh = ts.copy() |
| 27 | 📝 md | 4.4 What is the energy used for? Sub-metering share | Manual EDA takeaways |
| 28 | 📝 md | 5. Cleaning & physical-consistency rules | 5. Cleaning & physical-consistency rules |
| 29 | 🐍 code | 5. Cleaning & physical-consistency rules | clean = ts.copy() |
| 30 | 🐍 code | 5. Cleaning & physical-consistency rules | bad = rules.any(axis=1) |
| 31 | 📝 md | 6. Gap imputation: benchmark, then choose by gap length | 6. Gap imputation: benchmark, then choose by gap length |
| 32 | 🐍 code | 6. Gap imputation: benchmark, then choose by gap length | the hidden stretch itself must be fully observed so we know the truth; ≥ 2 weeks of history for seasonal metho |
| 33 | 🐍 code | 6. Gap imputation: benchmark, then choose by gap length | fig, ax = plt.subplots(figsize=(9, 3.6)) |
| 34 | 📝 md | 6. Gap imputation: benchmark, then choose by gap length | Decision: a hybrid imputer. For gaps up to the crossover length the benchmark shows, use linear interpolation. |
| 35 | 🐍 code | 6. Gap imputation: benchmark, then choose by gap length | show one long gap before/after |
| 36 | 📝 md | 7. Data reduction: resample to hourly | 7. Data reduction: resample to hourly |
| 37 | 🐍 code | 7. Data reduction: resample to hourly | imputed["Sub4_other"] = imputed.GAP * 1000 / 60 - imputed[["Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac"]]. |
| 38 | 📝 md | 8. Anomaly detection & clustering | 8. Anomaly detection & clustering |
| 39 | 🐍 code | 8. Anomaly detection & clustering | from sklearn.ensemble import IsolationForest |
| 40 | 🐍 code | 8. Anomaly detection & clustering | Only the TRAINING period (before 2010) is cleaned: the Hampel median is centered (it looks ±12 h ahead), |
| 41 | 📝 md | 8.2 Clustering daily load profiles (day types) | 8.2 Clustering daily load profiles (day types) |
| 42 | 🐍 code | 8.2 Clustering daily load profiles (day types) | D = H.GAP.groupby(H.index.date).apply(lambda s: s.values if len(s) == 24 else None).dropna() |
| 43 | 📝 md | 8.2 Clustering daily load profiles (day types) | Reading the clusters: silhouette picks k = 2: a high-consumption winter day type and a low-load type that domi |
| 44 | 📝 md | 9. Feature engineering | 9. Feature engineering |
| 45 | 🐍 code | 9. Feature engineering | Wavelets: Haar DWT (level 3) of each trailing 24-h window |
| 46 | 🐍 code | 9. Feature engineering | Wavelet example straight from the lecture: {141, 15, 100, 182} → averages {78, 141}, deltas {−63, 41} |
| 47 | 📝 md | Chronological split + anomaly elimination on training rows o | Chronological split + anomaly elimination on training rows only |
| 48 | 🐍 code | Chronological split + anomaly elimination on training rows o | TEST_START = "2010-01-01" |
| 49 | 📝 md | 10. Feature selection: filter, wrapper, embedded | 10. Feature selection: filter, wrapper, embedded |
| 50 | 🐍 code | 10. Feature selection: filter, wrapper, embedded | from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector |
| 51 | 🐍 code | 10. Feature selection: filter, wrapper, embedded | fig, ax = plt.subplots(figsize=(8, 5.5)) |
| 52 | 📝 md | 11. Models: naive baselines vs AutoViML | 11. Models: naive baselines vs AutoViML |
| 53 | 🐍 code | 11. Models: naive baselines vs AutoViML | from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score |
| 54 | 🐍 code | 11. Models: naive baselines vs AutoViML | AutoViML 0.2.2 bug with its own pinned xgboost 1.7.6: it forces updater='grow_colmaker' together with tree_met |
| 55 | 📝 md | 12. Results | 12. Results |
| 56 | 🐍 code | 12. Results | res = pd.DataFrame(results).set_index("model").round(4) |
| 57 | 🐍 code | 12. Results | r_ = res.sort_values("RMSE_kW", ascending=False) |
| 58 | 🐍 code | 12. Results | err = pd.DataFrame({"abs_err": np.abs(yte.values - (preds[best] if best in preds else ols.predict(Xte[selected |
| 59 | 📝 md | Takeaways (from the executed run) | Takeaways (from the executed run) |

## `part2_agentic_eda/03_agentic_eda_gemini_agent.ipynb`

| Cell | Type | Section | What it does |
|---|---|---|---|
| 1 | 📝 md | Part 2 — Exploratory Data Analysis with AI agents instead of | Part 2 — Exploratory Data Analysis with AI agents instead of by hand |
| 2 | 📝 md | 0. Setup | 0. Setup |
| 3 | 🐍 code | 0. Setup | import importlib.util, subprocess, sys |
| 4 | 📝 md | 1. Load the same dataset as Part 1 | 1. Load the same dataset as Part 1 |
| 5 | 🐍 code | 1. Load the same dataset as Part 1 | DATA = Path("data"); DATA.mkdir(exist_ok=True) |
| 6 | 📝 md | 2. Connect to Gemini | 2. Connect to Gemini |
| 7 | 🐍 code | 2. Connect to Gemini | from google import genai |
| 8 | 📝 md | 3. The agent: one tool, a plan–act–observe loop, and guardra | 3. The agent: one tool, a plan–act–observe loop, and guardrails |
| 9 | 🐍 code | 3. The agent: one tool, a plan–act–observe loop, and guardra | Dataset overview |
| 10 | 📝 md | 4. Run the agent | 4. Run the agent |
| 11 | 🐍 code | 4. Run the agent | GOAL = ("Perform a thorough exploratory data analysis of this NYC Airbnb 2019 dataset to prepare it for a mode |
| 12 | 📝 md | 4.2 Verification turn (human-in-the-loop) | 4.2 Verification turn (human-in-the-loop) |
| 13 | 🐍 code | 4.2 Verification turn (human-in-the-loop) | n_before = len(TRACE) |
| 14 | 🐍 code | 4.2 Verification turn (human-in-the-loop) | Save the full, auditable trace |
| 15 | 📝 md | 5. Audit: agent vs. my manual EDA (ground truth from Part 1) | 5. Audit: agent vs. my manual EDA (ground truth from Part 1) |
| 16 | 🐍 code | 5. Audit: agent vs. my manual EDA (ground truth from Part 1) | raw = df |
| 17 | 📝 md | 5.1 Score the Colab Data Science Agent (Tool A) the same way | 5.1 Score the Colab Data Science Agent (Tool A) the same way |
| 18 | 🐍 code | 5.1 Score the Colab Data Science Agent (Tool A) the same way | a_path = OUT / "colab_agent_output.txt" |
| 19 | 📝 md | 5.2 Process metrics | 5.2 Process metrics |
| 20 | 🐍 code | 5.2 Process metrics | metrics = { |
| 21 | 📝 md | 6. Reflection: agentic EDA vs. manual EDA | 6. Reflection: agentic EDA vs. manual EDA |
