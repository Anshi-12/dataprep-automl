# Video 1 — Part 1 (i): Tabular data prep + AutoViML on NYC Airbnb
**Target length:** 12–15 min · **Record with:** OBS / Loom / QuickTime (screen + mic) · **Upload:** YouTube → *Unlisted* → paste the link in README.md

**Before recording:** open the *executed* notebook in Colab (all outputs visible). Have the GitHub repo open in a second tab, plus `outputs_airbnb/reports/ydata_profile_airbnb.html` in a third tab.
Zoom the browser to 110–125% so the code is readable.

> Cell numbers = position in the notebook counting markdown and code cells from the top (the same numbering as `docs/CELL_GUIDE.md`).
> **Say** lines are a guide. Say them in your own words, and don't read them word for word.

---

## Segment 1 — User journey (0:00 – 2:30)

| Time | Show on screen | Say |
|---|---|---|
| 0:00 | Your face/cam or the repo README | "Hi, I'm Anshita. This is Part 1 of the data-preparation assignment for CMPE 255: a full EDA, cleaning, and AutoML pipeline on a messy tabular dataset, the NYC Airbnb 2019 data from Kaggle." |
| 0:20 | GitHub repo root → scroll the folder tree | "The repo has one folder per dataset plus Part 2. Each notebook is committed **executed**, so every chart and table is visible without running anything." |
| 0:40 | Colab: File → Open notebook → GitHub → paste repo URL | "As a user, you open it straight from GitHub in Colab." |
| 1:00 | Cell 3 (install cell) | "AutoViML needs numpy 1.x, but Colab ships numpy 2. This cell detects that, installs the pinned versions, and restarts the runtime on purpose. You just click Run all a second time." |
| 1:30 | Cell 6 (loader) | "The loader tries a local file first, then Kaggle through kagglehub, then a GitHub mirror of the same file, so it never blocks on credentials." |
| 1:50 | Scroll quickly to cell 59 (results table) | "Spoiler: the best model explains about 59% of the variance in log price, with a median error of about $25 a night. And the surprise is that plain linear regression on my engineered features edged out AutoML. Now let's see how we got there." |

## Segment 2 — Code walkthrough (2:30 – 12:30)

| Time | Cell(s) | Show | Say (key points) |
|---|---|---|---|
| 2:30 | 1 | Intro table of data types | "I picked this dataset because it has every data type from the lecture: numeric, low- and high-cardinality categorical, free text, a date, and coordinates." |
| 3:00 | 8–10 | Audit table + checks | "Before plotting anything, I audit structure. Two findings drive everything: reviews_per_month and last_review are missing in exactly the same 10,052 rows, and **every one has zero reviews**. And 11 listings have a price of $0, which Airbnb doesn't allow." |
| 3:45 | 13 + switch to the ydata HTML tab | Scroll the alerts and the missing-values heatmap | "For auto-EDA I used ydata-profiling. It raised 10 alerts: 20.6% missing in two columns, skewed minimum nights, 36% zeros in availability, and correlated review columns." |
| 4:15 | 14, 15 | Sweetviz + pandas-dq output | "Sweetviz looks at every feature against price, and pandas-dq is the auto-data-science report from AutoViML's author. It suggests a fix per column." |
| 4:40 | 16 | The comparison table | "This table is the important part: what the tools said versus what I concluded. For example, pandas-dq suggests imputing reviews_per_month, but the tools can't see *why* it's missing." |
| 5:10 | 18–20 | Missingness matrix, price histograms | "The matrix shows the two columns missing together. Price is extremely skewed, up to $10,000, so I model log price." |
| 5:40 | 22, 24 | Borough × room type, maps | "Location and room type dominate: Manhattan entire homes median $191, versus about $50–60 for private rooms in the outer boroughs. The map shows the price gradient around Midtown." |
| 6:10 | 31 | Keyword table | "Even the listing title carries signal: 'luxury' and 'penthouse' listings are almost twice the median, and 'shared' and 'hostel' are far below." |
| 6:30 | 34 | Cleaning log | "Cleaning is a logged list of rules: normalize text, drop the impossible $0 prices, and drop minimum nights over a year." |
| 7:00 | 36–39 | Imputation benchmark chart | "This is the imputation experiment. I hid 15% of known values and compared mean, median, KNN, and MICE. KNN and MICE win by a lot, but only when data is missing at random. Here it isn't, so the honest fill is zero plus a 'never reviewed' flag. I reuse KNN later where values really are unknown." |
| 8:00 | 41–43 | Detector counts + three anomaly maps | "For anomalies I ran five detectors: IQR, a robust z-score, Isolation Forest, Local Outlier Factor, and DBSCAN on coordinates. Each one alone flags a lot of legitimate listings." |
| 8:40 | 45 | Removed examples + policy | "So I only remove rows that **three detectors agree on**, or extreme prices that Isolation Forest also flags. That removed only 41 rows, while single detectors flagged between 489 and 2,101. Suspicious minimum-nights values get KNN-imputed, but I keep the 30-night spike, because that's NYC's 30-day rental rule, not noise." |
| 9:20 | 47–49 | Elbow/silhouette, segment table, segment map | "K-Means with k chosen by silhouette gives 7 market segments. The cluster id becomes a feature." |
| 9:50 | 51 | Feature-engineering cell; scroll slowly | "Feature engineering follows the lecture's portability ideas: haversine distance to Times Square and other landmarks, equi-depth binning, keyword flags, LSA (TF-IDF plus SVD) on titles, and for 221 neighbourhoods an out-of-fold target encoding with an 'Other' bucket so there's no leakage." |
| 10:40 | 53–54 | Votes table + importance chart | "Feature selection uses all three families: mutual information as a filter, forward selection as a wrapper, and Lasso plus Random Forest as embedded methods. A feature survives with at least two votes. 73 candidates went down to 45." |
| 11:20 | 56–57 | Baselines + AutoViML loop | "Finally, AutoViML. I run it with Random Forest, XGBoost, and CatBoost. AutoViML does its own feature reduction and a randomized hyperparameter search. Its linear mode didn't converge on 38k rows, so plain linear regression covers the linear family." |

## Segment 3 — Results & takeaways (12:30 – 14:00)

| Time | Cell | Say |
|---|---|---|
| 12:30 | 59–60 | "On the 20% hold-out, the median baseline has an RMSE of 0.70 in log space. Linear regression gets 0.448, CatBoost 0.454, XGBoost 0.456, and Random Forest 0.471. I also want to be transparent: my first run accidentally used price-based anomaly flags as features. That's target leakage, and it inflated R² to 0.67. I fixed it, and these are the honest numbers." |
| 13:00 | 61 | "Linear regression winning tells me the engineered features (room type, the neighbourhood encoding, distance to Times Square, and the text features) already capture the structure. Feature engineering mattered more than the model." |
| 13:20 | 62 | "Three lessons: impute based on *why* data is missing; use consensus before deleting outliers; and engineered features mattered more than which model I picked." |
| 13:45 | Repo README | "Part 2 does the same EDA with AI agents. That's Video 3. Thanks!" |

**Coding-agent disclosure (say once, around 2:20):** "I used an AI coding agent to help write and debug this code, as the assignment allows. I chose the datasets and the methods, checked every output, and the interpretations are mine."
