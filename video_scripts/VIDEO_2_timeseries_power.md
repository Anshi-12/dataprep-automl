# Video 2 — Part 1 (ii): Time-series data prep + AutoViML forecasting (Household Power)
**Target length:** 12–15 min · Same recording setup as Video 1 · Upload to YouTube as *Unlisted* → link in README.md

**Before recording:** open the executed `02_power_timeseries_eda_prep_autoviml.ipynb` in Colab. Also open `outputs_power/reports/sweetviz_power_train_vs_test.html` in a second tab.

---

## Segment 1 — User journey (0:00 – 2:00)

| Time | Show | Say |
|---|---|---|
| 0:00 | Repo → `part1_timeseries_power/` | "This is the time-series half of Part 1: one household's electricity use, recorded every minute for four years, about 2 million rows." |
| 0:25 | Colab, cells 3 & 6 | "Same user flow as the tabular notebook: open it from GitHub, Run all, and it restarts once for the pinned AutoViML versions. The loader gets the file from Kaggle, or the mirror if Kaggle isn't available." |
| 0:50 | Cell 1 column table | "The goal is to forecast the next hour's average power, a typical utility task. The data is messy in time-series-specific ways: missing values coded as question marks, gaps from one minute to five days, and sensor glitches." |
| 1:30 | Scroll to cell 56 | "Where we end up: AutoViML's CatBoost beats the naive 'next hour equals this hour' forecast by 24% on a full unseen year. Here's how." |

## Segment 2 — Code walkthrough (2:00 – 12:30)

| Time | Cell(s) | Show | Say |
|---|---|---|---|
| 2:00 | 6–8 | read_csv with `na_values=['?']`, datetime parsing | "Two cleaning steps happen at load time: question marks become NaN, and the separate date and time text columns are merged into one timestamp with an explicit format." |
| 2:40 | 10–11 | Audit + gap-length table | "The timestamp grid is complete, so all the missingness is the question-mark marker, and when one sensor drops, all seven drop. It's 1.25% missing, but gap length matters more: most gaps are a few minutes, and 6 are longer than a day." |
| 3:30 | 15 + ydata tab | Alerts list | "ydata-profiling in time-series mode flags the target as non-stationary and seasonal, the kitchen circuit as zero 84% of the time, and 421 fully missing hours." |
| 4:00 | 16 + Sweetviz tab | Train vs test compare | "I used Sweetviz to compare my future training years against the 2010 test year, before modeling, to check for distribution shift." |
| 4:30 | 20, 22 | Overview + hour×weekday heatmap + month boxplot | "Three seasonalities: an evening peak every day, different weekend behavior, and winter at roughly double summer. And this summer dip is a vacation." |
| 5:10 | 24 | STL + ACF/PACF | "STL separates trend, annual season, and residual. The ACF peaks at 24 and 168 hours, so those are the lags to engineer." |
| 5:40 | 26 | Sub-metering stacked area | "About half the energy isn't on any sub-meter, so I derive that 'unmetered' remainder as a feature." |
| 6:00 | 29–30 | Physics rules table + residual histogram | "Cleaning uses physics: current should equal √(P² + Q²) / V, sub-meters can't exceed the total, and French grid voltage stays within ±10%. Violations are sensor errors, so I set them to missing and impute them." |
| 6:50 | 32–33 | Benchmark table + line chart | "Imputation is benchmarked by gap length: I hide real data in gaps of 1 minute up to 3 days and compare six methods. Interpolation wins short gaps, and seasonal methods win long ones." |
| 7:40 | 35 | Before/after imputation example | "So the final imputer is a hybrid. Here a multi-hour outage gets a realistic daily shape instead of a flat line." |
| 8:10 | 37 | Hourly table | "Then data reduction: I resample to hourly. Minute-level detail survives as within-hour max, std, and the share of imputed minutes." |
| 8:40 | 39–40 | Detector table + flagged anomalies plot | "Point anomalies: Hampel filter, STL residuals, and Isolation Forest. I can't delete rows from the middle of a time series, so hours flagged by two detectors are **replaced** with the rolling median in the feature series only." |
| 9:20 | 42 | Day-type profiles + calendar | "Clustering 24-hour profiles gives 2 day types: a high-load winter type and a low-load type that peaks in July and August. DBSCAN flags unusual days, which removes 607 training hours, and those days are **removed from training only**. The test year stays raw." |
| 10:00 | 45–46 | Feature cell; wavelet check | "Features use only information up to now: lags at 1, 2, 3, 24, and 168 hours; rolling stats; calendar plus French holidays with sin/cos encoding; Haar wavelet coefficients of the last 24 hours. This cell reproduces the lecture's wavelet example. Plus a SAX word for the last day's shape, and yesterday's cluster." |
| 11:00 | 48 | Split output | "The split is chronological: 2007–2009 train, all of 2010 test." |
| 11:20 | 50–51 | Votes table + importance | "Selection uses the same filter/wrapper/embedded voting, on training data only. The top features are current intensity, the within-hour maximum, and the unmetered load, followed by the 1-week and 1-day lags." |
| 11:50 | 53–54 | Baselines + AutoViML loop | "Every forecast must beat naive baselines: persistence, same hour yesterday, and same hour last week. Then OLS, and AutoViML with Random Forest, XGBoost, and CatBoost." |

## Segment 3 — Results (12:30 – 14:00)

| Time | Cell | Say |
|---|---|---|
| 12:30 | 56–57 | "Persistence gets an RMSE of 0.614 kW. OLS gets 0.507, Random Forest 0.489, XGBoost 0.475, and CatBoost is best at 0.467: 24% better, with an R² of 0.64." |
| 13:00 | 57 (week plot), 58 | "Here's a February week: the forecast tracks the evening peaks. The error-by-hour chart shows most error sits in the evening peak, when people switch appliances on unpredictably." |
| 13:30 | 59 | "Takeaways: imputation depends on gap length, physics rules catch errors statistics miss, and in time series you replace anomalies instead of deleting rows. Weather data would be my next improvement." |

**Coding-agent disclosure (one sentence):** same as Video 1.
