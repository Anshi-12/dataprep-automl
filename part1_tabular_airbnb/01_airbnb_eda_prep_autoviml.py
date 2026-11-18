# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 1 (i) — Tabular: NYC Airbnb 2019
# ## EDA → Cleaning → Anomalies & Clustering → Imputation → Feature Engineering & Selection → AutoViML
#
# **CMPE 255 Data Mining · Fall 2026 · Anshita Shah**
#
# **Dataset:** [New York City Airbnb Open Data (Kaggle, dgomonov)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — 48,895 listings, 16 columns.
#
# **Why this dataset:** it mixes every data type covered in the Data Preparation lecture in one table:
#
# | Type | Columns |
# |---|---|
# | Numeric (continuous / count) | `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365` |
# | Categorical (low / high cardinality) | `neighbourhood_group` (5), `room_type` (3), `neighbourhood` (221) |
# | Free text | `name`, `host_name` |
# | Datetime | `last_review` |
# | Spatial | `latitude`, `longitude` |
# | Identifiers | `id`, `host_id` |
#
# **ML task:** predict the nightly price of a listing (regression on `log1p(price)`).
#
# **Notebook map**
# 0. Setup · 1. Load · 2. First look & data audit · 3. Automated EDA (ydata-profiling, Sweetviz, pandas-dq) · 4. Manual EDA & visualization ·
# 5. Cleaning · 6. Imputation (compare simple / KNN / MICE) · 7. Anomaly detection & clustering · 8. Feature engineering ·
# 9. Feature selection (filter / wrapper / embedded) · 10. AutoViML models · 11. Results & takeaways

# %% [markdown]
# ## 0. Setup
# AutoViML 0.2.2 pins `numpy<2`, `scipy==1.11.4`, `scikit-learn<=1.5.2`, and `xgboost<=1.7.6`. Colab ships numpy 2.x, so the
# cell below installs the pinned stack **once** and restarts the runtime automatically.
# After the restart, use **Runtime → Run all** again. The cell sees numpy 1.x and skips the install.

# %%
import importlib.metadata as md, os, sys
def _v(pkg):
    try: return md.version(pkg)
    except md.PackageNotFoundError: return None

NEEDS_INSTALL = (_v("numpy") or "2").startswith("2") or _v("autoviml") is None
if NEEDS_INSTALL:
    print("Installing pinned AutoViML stack (≈2 min). The runtime will restart. Then click Runtime → Run all.")
    # !pip install -q "autoviml==0.2.2" "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.5.2" "xgboost==1.7.6" "setuptools<81" "ydata-profiling==4.12.2" "sweetviz==2.3.3" "pandas-dq==1.29" PyWavelets kagglehub
    os.kill(os.getpid(), 9)   # restart the Colab runtime so the new numpy is loaded
else:
    print("Environment ready:", {p: _v(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "autoviml", "xgboost", "catboost"]})

# %%
import warnings, shutil, urllib.request, time, re
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
RNG = 42
np.random.seed(RNG)

# Output folders. Everything this notebook produces is saved next to it.
OUT = Path("outputs_airbnb"); FIG = OUT / "figures"; REP = OUT / "reports"
for p in (FIG, REP): p.mkdir(parents=True, exist_ok=True)

# One consistent, colorblind-validated chart style for every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ANOMALY = "#d03b3b"; MUTED = "#9a9993"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ad",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.prop_cycle": plt.cycler(color=PALETTE), "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8.5,
})
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 1. Load the data
# The loader tries, in order: (1) a local copy in `data/`, (2) **kagglehub** (official Kaggle source), (3) a public GitHub mirror of the same
# Kaggle file (so the notebook still runs without Kaggle credentials).

# %%
DATA = Path("data"); DATA.mkdir(exist_ok=True)
CSV = DATA / "AB_NYC_2019.csv"
MIRROR = "https://raw.githubusercontent.com/rajtulluri/Airbnb-Data-Exploratory-Analysis/master/AB_NYC_2019.csv"

def load_airbnb():
    if not CSV.exists():
        try:
            import kagglehub
            path = kagglehub.dataset_download("dgomonov/new-york-city-airbnb-open-data")
            shutil.copy(next(Path(path).rglob("AB_NYC_2019.csv")), CSV)
            print("Loaded from Kaggle via kagglehub")
        except Exception as e:
            print(f"kagglehub unavailable ({type(e).__name__}); using GitHub mirror of the Kaggle file")
            urllib.request.urlretrieve(MIRROR, CSV)
    return pd.read_csv(CSV)

raw = load_airbnb()
print(raw.shape)
raw.head()

# %% [markdown]
# ## 2. First look & data audit
# Before any plotting I audit **structure**: types, missingness, duplicates, cardinality, and impossible values.
# This gives a checklist that the cleaning section works through.

# %%
def audit(df):
    a = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
        "example": df.iloc[0],
    })
    return a
audit(raw)

# %%
num_cols = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]
desc = raw[num_cols].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
desc["skew"] = raw[num_cols].skew()
desc.round(2)

# %%
checks = {
    "exact duplicate rows": raw.duplicated().sum(),
    "duplicate listing ids": raw["id"].duplicated().sum(),
    "price == 0": (raw.price == 0).sum(),
    "price > $1,000/night": (raw.price > 1000).sum(),
    "minimum_nights > 365": (raw.minimum_nights > 365).sum(),
    "reviews_per_month missing": raw.reviews_per_month.isna().sum(),
    "…of which number_of_reviews == 0": (raw.reviews_per_month.isna() & (raw.number_of_reviews == 0)).sum(),
    "last_review missing": raw.last_review.isna().sum(),
    "availability_365 == 0": (raw.availability_365 == 0).sum(),
    "name missing": raw.name.isna().sum(),
    "host_name missing": raw.host_name.isna().sum(),
    "lat/long outside NYC bbox": (~raw.latitude.between(40.49, 40.92) | ~raw.longitude.between(-74.27, -73.68)).sum(),
}
pd.Series(checks, name="count").to_frame()

# %% [markdown]
# **Audit findings (these drive the cleaning plan):**
# - **Missingness is structural, not random.** `reviews_per_month` and `last_review` are missing for exactly the same 10,052 rows, and every one of them has `number_of_reviews == 0`. *No review → no review date → no review rate.* This is *missing-by-design*, so the right fix is `0` plus a "never reviewed" flag, not a statistical guess.
# - **Invalid target values:** 11 listings have `price == 0`. Airbnb doesn't allow free listings, so these are data-entry errors and are removed from the training target.
# - **Extreme values:** a few hundred listings charge > $1,000/night, and some have `minimum_nights` > 365 (up to 1,250 nights). These are either long-term rentals posted on a short-term platform or typos. Section 7 decides which are legitimate.
# - **Heavy right skew** in `price`, `minimum_nights`, `number_of_reviews`, and `calculated_host_listings_count` points to log transforms.
# - No duplicate rows or IDs, and every coordinate falls inside NYC.

# %% [markdown]
# ## 3. Automated EDA ("auto EDA" + "auto DS")
# Three automated tools, each answering a different question:
# - **ydata-profiling**: a full univariate and bivariate profile with alerts (correlations, missing patterns, skew, high cardinality).
# - **Sweetviz**: compares the target `price` against every feature.
# - **pandas-dq** (`dq_report`): an "auto data-science" quality report with a suggested fix for each column. pandas-dq is by the author of AutoViML.
#
# The HTML reports are saved under `outputs_airbnb/reports/` (open them in a browser). My manual reading of them follows the cells.

# %%
from ydata_profiling import ProfileReport
t = time.time()
profile = ProfileReport(raw.drop(columns=["id", "host_id"]), title="NYC Airbnb 2019 — ydata-profiling",
                        minimal=False, explorative=True, progress_bar=False,
                        correlations={"auto": {"calculate": True}, "pearson": {"calculate": True}, "spearman": {"calculate": True}},
                        interactions={"continuous": False}, samples={"head": 5, "tail": 0})
profile.to_file(REP / "ydata_profile_airbnb.html")
print(f"ydata-profiling report saved in {time.time()-t:.0f}s →", REP / "ydata_profile_airbnb.html")
alerts = [str(a) for a in profile.get_description().alerts]
print(f"{len(alerts)} alerts raised:"); print("\n".join("  • " + a for a in alerts))

# %%
import sweetviz as sv
sv_df = raw.drop(columns=["id", "host_id", "name", "host_name"]).copy()
sv_report = sv.analyze([sv_df, "Airbnb NYC"], target_feat="price", pairwise_analysis="off")
sv_report.show_html(str(REP / "sweetviz_airbnb.html"), open_browser=False, layout="vertical")

# %%
from pandas_dq import dq_report
dq = dq_report(raw, target="price", verbose=0, html=False, csv_engine="pandas")
dq

# %% [markdown]
# ### What the automated tools found, and where my manual reading disagrees
# | Automated finding | My interpretation / action |
# |---|---|
# | ydata flags `reviews_per_month` and `last_review` as ~20.6% **missing** | Not random: the profile's missingness heatmap shows the two columns are missing *together*. In Section 2, every missing row has 0 reviews. → structural fill, not KNN/MICE. |
# | ydata flags **high skew** for `price`, `minimum_nights`, `calculated_host_listings_count` | Model `log1p(price)` and log-transform the count features. |
# | ydata flags `availability_365` with many **zeros** (~36%) | A zero can mean "fully booked" *or* "inactive listing". It can't be resolved from the data, so I add an `is_inactive` flag (0 availability **and** never reviewed). |
# | ydata flags **high correlation** `number_of_reviews` ↔ `reviews_per_month` | Expected (one is a rate of the other). Keep both, but the selection step will drop one if it is redundant. |
# | Sweetviz: `room_type` and `neighbourhood_group` have the strongest association with price | Confirms the domain intuition: *what* you rent and *where* matter most. |
# | pandas-dq suggests capping outliers in `price` / `minimum_nights` | Tools can't tell a typo from a legitimate luxury penthouse. Section 7 uses multiple detectors plus domain rules instead of blind capping. |
# | pandas-dq: `neighbourhood` is high-cardinality (221) | One-hot would add 221 sparse columns, so I use frequency + out-of-fold target encoding plus an OOV ("Other") bucket for rare neighbourhoods (per lecture). |

# %% [markdown]
# ## 4. Manual EDA & visualization
# ### 4.1 Missingness pattern

# %%
fig, ax = plt.subplots(figsize=(9, 3.2))
miss = raw.isna()
cols_with_na = miss.columns[miss.any()]
sample = miss[cols_with_na].sample(2500, random_state=RNG).sort_values(list(cols_with_na))
ax.imshow(sample.T.values, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#e6e5e0", "#2a78d6"]), interpolation="nearest")
ax.set_yticks(range(len(cols_with_na))); ax.set_yticklabels(cols_with_na); ax.set_xticks([]); ax.grid(False)
ax.set_title("Missing values (blue) across 2,500 sampled listings: last_review and reviews_per_month are missing together")
save(fig, "01_missingness_matrix")

# %% [markdown]
# ### 4.2 Target: price is extremely right-skewed. Log space reveals the structure

# %%
p = raw.loc[raw.price > 0, "price"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].hist(p, bins=120, color=PALETTE[0]); axes[0].set_title("Raw price (long tail to $10,000)"); axes[0].set_xlabel("$ / night")
axes[1].hist(p[p <= 500], bins=60, color=PALETTE[0]); axes[1].set_title("Price ≤ $500 (≈97% of listings)"); axes[1].set_xlabel("$ / night")
axes[2].hist(np.log1p(p), bins=60, color=PALETTE[2]); axes[2].set_title("log1p(price): close to normal"); axes[2].set_xlabel("log1p($)")
save(fig, "02_price_distribution")
print(f"skew raw={p.skew():.2f} → log1p={np.log1p(p).skew():.2f}")

# %% [markdown]
# ### 4.3 Price by borough × room type

# %%
fig, ax = plt.subplots(figsize=(10, 4))
order = raw.groupby("neighbourhood_group").price.median().sort_values(ascending=False).index
sns.boxplot(data=raw[raw.price.between(1, 1000)], x="neighbourhood_group", y="price", hue="room_type", order=order,
            showfliers=False, linewidth=0.8, palette=PALETTE[:3], ax=ax)
ax.set_title("Price by borough and room type (outliers hidden; price ≤ $1,000)"); ax.set_xlabel(""); ax.set_ylabel("$ / night")
ax.legend(title="", ncols=3, loc="upper right")
save(fig, "03_price_borough_roomtype")
pd.crosstab(raw.neighbourhood_group, raw.room_type, values=raw.price, aggfunc="median").round(0)

# %% [markdown]
# ### 4.4 Spatial view: where the expensive listings are

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
d = raw[raw.price.between(1, 1000)]
sc = axes[0].scatter(d.longitude, d.latitude, c=np.log1p(d.price), s=1.2, cmap="Blues", alpha=.7)
axes[0].set_title("Listings colored by log price (darker = pricier)"); plt.colorbar(sc, ax=axes[0], shrink=.7, label="log1p(price)")
for i, (g, gd) in enumerate(raw.groupby("neighbourhood_group")):
    axes[1].scatter(gd.longitude, gd.latitude, s=1, color=PALETTE[i], label=g, alpha=.6)
axes[1].legend(markerscale=8, loc="upper left"); axes[1].set_title("Boroughs (neighbourhood_group)")
for a in axes: a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.set_aspect("equal")
save(fig, "04_geo_price_map")

# %% [markdown]
# ### 4.5 Numeric relationships

# %%
corr_cols = num_cols + ["latitude", "longitude"]
c = raw.loc[raw.price > 0, corr_cols].assign(price=lambda x: np.log1p(x.price)).rename(columns={"price": "log_price"}).corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(c, annot=True, fmt=".2f", cmap=sns.diverging_palette(250, 25, as_cmap=True), center=0, vmin=-1, vmax=1,
            linewidths=1, linecolor="#fcfcfb", cbar_kws={"shrink": .7}, ax=ax, annot_kws={"size": 7.5})
ax.set_title("Spearman correlations (rank-based, robust to the skew)"); ax.grid(False)
save(fig, "05_spearman_heatmap")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].hist(raw.minimum_nights.clip(upper=60), bins=60, color=PALETTE[1]); axes[0].set_title("minimum_nights (clipped at 60 for view)")
axes[0].axvline(30, color=MUTED, ls="--", lw=1); axes[0].text(31, axes[0].get_ylim()[1]*.8, "30-night rule", fontsize=8, color="#52514e")
axes[1].hist(raw.availability_365, bins=73, color=PALETTE[2]); axes[1].set_title("availability_365: spikes at 0 and 365")
axes[2].hist(np.log1p(raw.calculated_host_listings_count), bins=40, color=PALETTE[6]); axes[2].set_title("log1p(host listing count)")
save(fig, "06_other_distributions")

# %% [markdown]
# ### 4.6 Time: when were listings last reviewed?

# %%
lr = pd.to_datetime(raw.last_review)
fig, ax = plt.subplots(figsize=(10, 3))
lr.dt.to_period("M").value_counts().sort_index().plot(ax=ax, color=PALETTE[0])
ax.set_title("Listings by month of last review (data snapshot ≈ July 2019)"); ax.set_xlabel("")
save(fig, "07_last_review_timeline")
print("last_review range:", lr.min().date(), "→", lr.max().date())

# %% [markdown]
# ### 4.7 Text: do listing titles carry price signal?

# %%
kw = ["luxury", "penthouse", "loft", "studio", "cozy", "private", "shared", "spacious", "view", "room", "bed", "hostel"]
name = raw.name.fillna("").str.lower()
kw_tbl = pd.DataFrame({k: {"listings": name.str.contains(rf"\b{k}\b").sum(),
                           "median_price": raw.loc[name.str.contains(rf"\b{k}\b"), "price"].median()} for k in kw}).T
kw_tbl["vs_overall"] = (kw_tbl.median_price / raw.price.median()).round(2)
kw_tbl.sort_values("median_price", ascending=False)

# %% [markdown]
# **Manual EDA takeaways**
# 1. **Location + room type dominate price.** Manhattan entire homes have a median ~$191 vs ~$50–60 for private rooms in the outer boroughs. The map shows a price gradient centered on lower/mid Manhattan and nearby Brooklyn, so **distance to the center** should be a strong engineered feature.
# 2. **Title words carry real signal.** "luxury" and "penthouse" listings have a median price ~1.8–2.6× the overall median, while "shared", "hostel" and "private" sit well below it. That justifies text features (keyword flags + LSA) instead of dropping `name`.
# 3. **`minimum_nights` has a spike at 30.** NYC law forbids most short rentals under 30 days, so many hosts set 30. This is a *legitimate* mode, not noise, and a `min_nights ≥ 30` flag captures it.
# 4. **`availability_365` is bimodal (0 and 365).** This is evidence of inactive/dormant listings mixed with always-open ones.
# 5. Pairwise numeric correlations with log price are weak (|ρ| < 0.25). Price depends on **categorical/spatial** factors more than numeric counts, which explains why tree-based models should beat linear ones.

# %% [markdown]
# ## 5. Cleaning
# Each rule is logged so the effect of every step is auditable.

# %%
df = raw.copy()
log = []
def step(msg, before, after):
    log.append({"step": msg, "rows_before": before, "rows_after": after, "removed": before - after})

# 5.1 Types
df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
for c in ["neighbourhood_group", "neighbourhood", "room_type"]:
    df[c] = df[c].astype("category")

# 5.2 Text normalisation: trim, collapse whitespace, fill missing with explicit token
for c in ["name", "host_name"]:
    df[c] = df[c].fillna("unknown").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

# 5.3 Duplicates (none expected, but always check after normalisation)
n = len(df); df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"]); step("drop duplicate rows (ignoring id)", n, len(df))

# 5.4 Invalid target: free listings are impossible on Airbnb
n = len(df); df = df[df.price > 0]; step("drop price == 0 (invalid)", n, len(df))

# 5.5 Domain rule: minimum_nights above one year is not a short-term rental listing (typos like 1,250)
n = len(df); df = df[df.minimum_nights <= 365]; step("drop minimum_nights > 365 (domain rule)", n, len(df))

pd.DataFrame(log)

# %% [markdown]
# ## 6. Imputation: compare strategies, then choose
# The lecture lists three options: (a) drop the record, (b) **impute** (mean/median, KNN, MICE), or (c) use a model that tolerates missing values.
# Missingness here is *structural* (no reviews), so the principled fix is `0` plus an indicator. To justify that choice, I **test** the alternatives.
#
# **Experiment:** take listings that *do* have `reviews_per_month`, hide 15% of the values at random, impute with each method, and measure the error.
# This shows how well each imputer would perform *if* the data were missing at random.

# %%
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

feat = ["reviews_per_month", "number_of_reviews", "minimum_nights", "availability_365", "calculated_host_listings_count", "latitude", "longitude"]
known = df.loc[df.reviews_per_month.notna(), feat].copy()
known["days_since_review"] = (df.last_review.max() - df.loc[known.index, "last_review"]).dt.days
known = known.sample(12000, random_state=RNG)
mask = np.random.RandomState(RNG).rand(len(known)) < .15
truth = known.loc[mask, "reviews_per_month"].values
masked = known.copy(); masked.loc[mask, "reviews_per_month"] = np.nan

sc = StandardScaler().fit(masked)
Z = pd.DataFrame(sc.transform(masked), columns=masked.columns, index=masked.index)
def unscale(arr):   # back to original units for the target column only
    j = list(masked.columns).index("reviews_per_month")
    return arr[:, j] * sc.scale_[j] + sc.mean_[j]

imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "Median": SimpleImputer(strategy="median"),
    "KNN (k=10)": KNNImputer(n_neighbors=10, weights="distance"),
    "MICE (IterativeImputer)": IterativeImputer(max_iter=15, random_state=RNG),
}
rows = []
for nm, imp in imputers.items():
    t = time.time(); filled = unscale(imp.fit_transform(Z))[mask]
    rows.append({"method": nm, "MAE": np.mean(np.abs(filled - truth)), "RMSE": np.sqrt(np.mean((filled - truth) ** 2)), "sec": time.time() - t})
imp_res = pd.DataFrame(rows).sort_values("MAE").round(3)
imp_res

# %%
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh(imp_res.method, imp_res.MAE, color=[PALETTE[2] if i == 0 else PALETTE[0] for i in range(len(imp_res))], height=.55)
ax.invert_yaxis(); ax.set_xlabel("MAE on hidden reviews_per_month (lower is better)")
ax.set_title("Imputation benchmark on artificially masked values")
for y, v in enumerate(imp_res.MAE): ax.text(v, y, f"  {v:.3f}", va="center", fontsize=8)
save(fig, "08_imputation_benchmark")

# %% [markdown]
# **Decision:** the model-based imputers (KNN/MICE) beat mean/median by a wide margin *when* values are missing at random, because review rate is
# predictable from review count and recency. **But our real missingness is not random.** All 10k missing rows have zero reviews, so the true rate is
# exactly 0. Applying KNN/MICE would invent activity that never happened. So:
# - `reviews_per_month` → **0**, plus `never_reviewed = 1`
# - `last_review` → kept as NaT and converted to `days_since_last_review`, with never-reviewed listings set to the max+1 sentinel (older than any review) and flagged
#
# KNN imputation **is** used later for the anomalous `minimum_nights` values marked unreliable in Section 7. Those are genuinely unknown,
# and the benchmark above shows KNN/MICE are the strongest choices.

# %%
df["never_reviewed"] = (df.number_of_reviews == 0).astype(int)
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
snapshot = df.last_review.max() + pd.Timedelta(days=1)
df["days_since_last_review"] = (snapshot - df.last_review).dt.days
df["days_since_last_review"] = df["days_since_last_review"].fillna(df["days_since_last_review"].max() + 1)
print("Remaining missing values:\n", df.drop(columns=["last_review"]).isna().sum()[lambda s: s > 0])

# %% [markdown]
# ## 7. Anomaly detection & clustering
# An outlier is either **noise** (a typo, which we remove) or a **legitimate rare event** (a real penthouse, which we keep). The lecture warns not to blindly delete outliers.
# I combine four detectors and act only where they **agree** or a **domain rule** applies.
#
# | Detector | Type | What it catches |
# |---|---|---|
# | IQR (Tukey fences, on log price) | univariate | extreme prices |
# | Robust z-score (median/MAD) | univariate | extreme `minimum_nights` |
# | Isolation Forest | multivariate | unusual *combinations* (e.g. cheap entire home in Midtown with 0 availability) |
# | Local Outlier Factor | multivariate, density | listings priced far from their **geographic neighbours** |
# | DBSCAN on coordinates | spatial clustering | geographically isolated listings |

# %%
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

lp = np.log1p(df.price)
q1, q3 = lp.quantile([.25, .75]); iqr = q3 - q1
df["out_iqr_price"] = ((lp < q1 - 1.5 * iqr) | (lp > q3 + 1.5 * iqr)).astype(int)

mn = np.log1p(df.minimum_nights); mad = (mn - mn.median()).abs().median()
df["out_z_minnights"] = (0.6745 * (mn - mn.median()).abs() / mad > 3.5).astype(int)

X_anom = pd.DataFrame({
    "log_price": lp, "log_min_nights": mn, "log_reviews": np.log1p(df.number_of_reviews),
    "reviews_per_month": df.reviews_per_month, "log_host_listings": np.log1p(df.calculated_host_listings_count),
    "availability_365": df.availability_365, "latitude": df.latitude, "longitude": df.longitude,
    "room_entire": (df.room_type == "Entire home/apt").astype(int), "room_shared": (df.room_type == "Shared room").astype(int),
})
Xs = StandardScaler().fit_transform(X_anom)
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=RNG).fit(Xs)
df["out_iforest"] = (iso.predict(Xs) == -1).astype(int)
df["iforest_score"] = -iso.score_samples(Xs)

# LOF on (location, room type, log price): flags listings priced very differently from nearby, similar listings
X_lof = StandardScaler().fit_transform(X_anom[["latitude", "longitude", "room_entire", "room_shared", "log_price"]])
df["out_lof"] = (LocalOutlierFactor(n_neighbors=35, contamination=0.01).fit_predict(X_lof) == -1).astype(int)

# DBSCAN on coordinates (haversine, eps ≈ 300 m): label -1 = spatially isolated
coords = np.radians(df[["latitude", "longitude"]].values)
db = DBSCAN(eps=0.3 / 6371.0, min_samples=10, metric="haversine", algorithm="ball_tree").fit(coords)
df["geo_isolated"] = (db.labels_ == -1).astype(int)

flags = ["out_iqr_price", "out_z_minnights", "out_iforest", "out_lof", "geo_isolated"]
df["n_flags"] = df[flags].sum(axis=1)
summary = df[flags].sum().to_frame("flagged")
summary["%"] = (summary.flagged / len(df) * 100).round(2)
summary

# %%
pd.crosstab(df.n_flags, df.room_type, margins=True)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
base = dict(s=1, color="#d7d6d0")
axes[0].scatter(df.longitude, df.latitude, **base)
m = df.out_lof == 1; axes[0].scatter(df.longitude[m], df.latitude[m], s=5, color=ANOMALY, label=f"LOF anomaly ({m.sum()})")
axes[0].set_title("LOF: priced unlike its neighbours"); axes[0].legend(loc="upper left")
axes[1].scatter(df.longitude, df.latitude, **base)
m = df.geo_isolated == 1; axes[1].scatter(df.longitude[m], df.latitude[m], s=5, color=ANOMALY, label=f"DBSCAN noise ({m.sum()})")
axes[1].set_title("DBSCAN (~300 m): spatially isolated"); axes[1].legend(loc="upper left")
axes[2].scatter(X_anom.log_min_nights, X_anom.log_price, s=1, color="#d7d6d0")
m = df.out_iforest == 1; axes[2].scatter(X_anom.log_min_nights[m], X_anom.log_price[m], s=5, color=ANOMALY, label=f"Isolation Forest ({m.sum()})")
axes[2].set_xlabel("log1p(minimum_nights)"); axes[2].set_ylabel("log1p(price)"); axes[2].legend(loc="upper left")
axes[2].set_title("Isolation Forest in feature space")
for a in axes[:2]: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
save(fig, "09_anomaly_detectors")

# %% [markdown]
# ### Anomaly treatment policy
# - **Remove** listings flagged by **≥ 3 detectors**. Independent methods agreeing is strong evidence of a data error, not a rare-but-real listing.
# - **Remove** price above the 99.9th percentile *if* also flagged by Isolation Forest (e.g. $10,000 private rooms).
# - **Keep but flag** the rest. The flags themselves are **not** model features: IQR, LOF, and this Isolation Forest all *use price*, so feeding them to a price model would leak the target. Section 8 builds a separate **price-free** anomaly score for modeling.
# - **Re-impute** `minimum_nights` with KNN (best in the Section 6 benchmark) where the robust z-score flags it *and* the value is not a legitimate mode (30, 60, 90, 180, 365 nights are kept).

# %%
n = len(df)
p999 = df.price.quantile(.999)
drop_mask = (df.n_flags >= 3) | ((df.price > p999) & (df.out_iforest == 1))
removed = df[drop_mask]
df = df[~drop_mask].copy(); step("drop consensus anomalies (≥3 detectors, or >p99.9 & iForest)", n, len(df))
print(f"Removed {drop_mask.sum()} rows; examples:")
display(removed.sort_values("n_flags", ascending=False)[["name", "neighbourhood_group", "room_type", "price", "minimum_nights", "n_flags"]].head(8))

legit_modes = {30, 60, 90, 120, 180, 365}
suspect = (df.out_z_minnights == 1) & (~df.minimum_nights.isin(legit_modes))
print(f"minimum_nights values re-imputed with KNN: {suspect.sum()}")
knn_cols = ["minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365", "calculated_host_listings_count", "latitude", "longitude"]
tmp = df[knn_cols].copy(); tmp.loc[suspect, "minimum_nights"] = np.nan
sc2 = StandardScaler().fit(tmp); filled = KNNImputer(n_neighbors=10).fit_transform(sc2.transform(tmp))
df.loc[suspect, "minimum_nights"] = np.round(filled[suspect.values, 0] * sc2.scale_[0] + sc2.mean_[0]).clip(1, 365)
pd.DataFrame(log)

# %% [markdown]
# ### 7.2 Clustering: market segments (K-Means)
# Clustering serves two purposes: it **describes** the market, and the cluster id becomes a **feature**. The lecture notes that clustering
# can act as a feature transformer, and AutoViML's `KMeans_Featurizer` does the same thing automatically.
# I choose *k* with both the elbow (inertia) and the silhouette score.

# %%
seg_cols = ["latitude", "longitude", "room_entire", "room_shared", "log_min_nights", "log_reviews", "availability_365", "log_host_listings"]
X_seg = X_anom.loc[df.index, seg_cols]
seg_scaler = StandardScaler().fit(X_seg); Zs = seg_scaler.transform(X_seg)
ks, inertia, sil = range(2, 11), [], []
samp = np.random.RandomState(RNG).choice(len(Zs), 6000, replace=False)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Zs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Zs[samp], km.labels_[samp]))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(list(ks), inertia, marker="o", markersize=4); axes[0].set_title("Elbow (inertia)"); axes[0].set_xlabel("k")
axes[1].plot(list(ks), sil, marker="o", markersize=4, color=PALETTE[1]); axes[1].set_title("Silhouette (higher = better separated)"); axes[1].set_xlabel("k")
save(fig, "10_kmeans_k_selection")
best_k = list(ks)[int(np.argmax(sil[2:])) + 2]   # prefer k ≥ 4 for an interpretable segmentation
print("chosen k =", best_k)

# %%
km = KMeans(n_clusters=best_k, n_init=20, random_state=RNG).fit(Zs)
df["segment"] = km.labels_
prof = df.groupby("segment").agg(listings=("id", "size"), median_price=("price", "median"),
                                 pct_entire=("room_type", lambda s: (s == "Entire home/apt").mean() * 100),
                                 median_min_nights=("minimum_nights", "median"), median_reviews=("number_of_reviews", "median"),
                                 median_availability=("availability_365", "median"), median_host_listings=("calculated_host_listings_count", "median"),
                                 top_borough=("neighbourhood_group", lambda s: s.mode()[0])).round(1)
prof.sort_values("median_price", ascending=False)

# %%
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for s in sorted(df.segment.unique()):
    d = df[df.segment == s]; ax.scatter(d.longitude, d.latitude, s=1, color=PALETTE[s % 8], label=f"segment {s} (median ${d.price.median():.0f})")
ax.legend(markerscale=8, loc="upper left", fontsize=7.5); ax.set_aspect("equal"); ax.set_title("K-Means market segments"); ax.set_xticks([]); ax.set_yticks([])
save(fig, "11_kmeans_segments_map")

# %% [markdown]
# ## 8. Feature engineering
# Following the lecture's *feature portability* ideas: numeric → categorical (equi-depth binning), categorical → numeric (one-hot, frequency,
# out-of-fold target encoding, OOV bucket), text → numeric (keyword flags + **LSA** with TF-IDF/SVD), spatial → numeric (haversine distances), date → numeric.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

fe = df.copy()
fe["log_price"] = np.log1p(fe.price)                                   # target

# spatial → numeric
def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))
landmarks = {"times_sq": (40.7580, -73.9855), "wall_st": (40.7069, -74.0113), "jfk": (40.6413, -73.7781), "williamsburg": (40.7081, -73.9571)}
for k_, (la, lo) in landmarks.items():
    fe[f"km_to_{k_}"] = haversine_km(fe.latitude, fe.longitude, la, lo)

# skewed counts → log
for c in ["minimum_nights", "number_of_reviews", "calculated_host_listings_count"]:
    fe[f"log_{c}"] = np.log1p(fe[c])

# numeric → categorical (equi-depth binning, from the slides)
fe["host_size_bin"] = pd.qcut(fe.calculated_host_listings_count.rank(method="first"), q=4, labels=["single", "small", "medium", "pro"])
fe["availability_bin"] = pd.cut(fe.availability_365, bins=[-1, 0, 90, 180, 364, 365], labels=["none", "≤3mo", "≤6mo", "<1yr", "always"])

# domain flags
fe["min_nights_30plus"] = (fe.minimum_nights >= 30).astype(int)
fe["is_inactive"] = ((fe.availability_365 == 0) & (fe.never_reviewed == 1)).astype(int)
fe["multi_listing_host"] = (fe.calculated_host_listings_count > 1).astype(int)
fe["review_recency_bucket"] = pd.cut(fe.days_since_last_review, bins=[-1, 30, 180, 365, 1e9], labels=["<1mo", "<6mo", "<1yr", "older/never"])

# text → numeric: keyword flags + length + LSA (TF-IDF → truncated SVD)
nm = fe.name.str.lower()
fe["name_len"] = nm.str.len(); fe["name_words"] = nm.str.split().str.len()
fe["name_upper_ratio"] = fe.name.str.count(r"[A-Z]") / fe.name.str.len().clip(lower=1)
fe["name_exclaim"] = fe.name.str.contains("!").astype(int)
for k_ in ["luxury", "penthouse", "loft", "studio", "cozy", "private", "shared", "spacious", "view", "hostel", "bed"]:
    fe[f"kw_{k_}"] = nm.str.contains(rf"\b{k_}\b").astype(int)
tfidf = TfidfVectorizer(min_df=20, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
T = tfidf.fit_transform(nm)
svd = TruncatedSVD(n_components=8, random_state=RNG); L = svd.fit_transform(T)
for i in range(L.shape[1]): fe[f"lsa_{i}"] = L[:, i]
print(f"TF-IDF vocab = {T.shape[1]} terms → LSA 8 components explain {svd.explained_variance_ratio_.sum():.1%} of variance")
top_terms = pd.DataFrame({f"lsa_{i}": np.array(tfidf.get_feature_names_out())[np.argsort(-svd.components_[i])[:6]] for i in range(4)})
display(top_terms)

# high-cardinality categorical: OOV bucket + frequency encoding + out-of-fold target encoding (each row's encoding never uses its own price)
counts = fe.neighbourhood.value_counts()
fe["neighbourhood_oov"] = fe.neighbourhood.astype(str).where(fe.neighbourhood.map(counts) >= 30, "Other")
fe["neighbourhood_freq"] = fe.neighbourhood.map(counts / len(fe)).astype(float)
fe["neighbourhood_te"] = np.nan
for tr, va in KFold(5, shuffle=True, random_state=RNG).split(fe):
    means = fe.iloc[tr].groupby("neighbourhood_oov").log_price.mean()
    fe.iloc[va, fe.columns.get_loc("neighbourhood_te")] = fe.iloc[va].neighbourhood_oov.map(means).values
fe["neighbourhood_te"] = fe.neighbourhood_te.fillna(fe.log_price.mean())
print("neighbourhoods:", fe.neighbourhood.nunique(), "→ with OOV bucket:", fe.neighbourhood_oov.nunique())

fe["host_name_listing_share"] = fe.groupby("host_id").id.transform("count") / fe.calculated_host_listings_count.clip(lower=1)

# price-free anomaly score (safe to use as a feature: no target information)
X_np = StandardScaler().fit_transform(X_anom.loc[fe.index].drop(columns=["log_price"]))
fe["anomaly_score_noprice"] = -IsolationForest(n_estimators=300, random_state=RNG).fit(X_np).score_samples(X_np)
fe.shape

# %% [markdown]
# ## 9. Feature selection: filter, wrapper, embedded
# The lecture covers three families. I run all three and keep features that **at least two** methods support.
# Selection runs on the **training split only** (the 20% test rows are set aside first).
# - **Filter:** mutual information with the target (model-independent, fast) and a redundancy check (|ρ| > 0.9)
# - **Wrapper:** sequential **forward** selection with a fast linear model (start with 1, keep adding)
# - **Embedded:** L1-regularized **Lasso** coefficients + Random Forest impurity importance

# %%
from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector, VarianceThreshold
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Leakage guard: every column computed FROM price is excluded (the target, and the price-based anomaly flags/scores)
leaky = ["price", "log_price", "out_iqr_price", "out_lof", "out_iforest", "iforest_score", "n_flags"]
drop_cols = ["id", "name", "host_id", "host_name", "last_review", "neighbourhood", "neighbourhood_oov",
             "minimum_nights", "number_of_reviews", "calculated_host_listings_count"] + leaky
cat_cols = ["neighbourhood_group", "room_type", "host_size_bin", "availability_bin", "review_recency_bucket"]
X_all = pd.get_dummies(fe.drop(columns=drop_cols), columns=cat_cols + ["segment"], drop_first=False, dtype=int)
y_all = fe.log_price
X_all = X_all.loc[:, VarianceThreshold(1e-4).fit(X_all).get_support()]
print("candidate features:", X_all.shape[1])

# Hold out the test set NOW: feature selection only ever sees training rows (no selection leakage)
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(X_all.index, test_size=.2, random_state=RNG)
samp_idx = X_all.loc[train_idx].sample(15000, random_state=RNG).index
Xf, yf = X_all.loc[samp_idx], y_all.loc[samp_idx]
Xf_s = pd.DataFrame(StandardScaler().fit_transform(Xf), columns=Xf.columns, index=Xf.index)

mi = pd.Series(mutual_info_regression(Xf, yf, random_state=RNG, n_neighbors=5), index=Xf.columns)
lasso = LassoCV(cv=5, random_state=RNG, n_alphas=40).fit(Xf_s, yf)
lasso_coef = pd.Series(np.abs(lasso.coef_), index=Xf.columns)
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RNG).fit(Xf, yf)
rf_imp = pd.Series(rf.feature_importances_, index=Xf.columns)
t = time.time()
sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=20, direction="forward", cv=3, n_jobs=-1).fit(Xf_s, yf)
print(f"forward selection took {time.time()-t:.0f}s")
sfs_sel = pd.Series(sfs.get_support(), index=Xf.columns)

sel = pd.DataFrame({"MI": mi, "Lasso|coef|": lasso_coef, "RF_importance": rf_imp, "forward_SFS": sfs_sel})
sel["vote_MI"] = sel.MI >= sel.MI.quantile(.5)
sel["vote_Lasso"] = sel["Lasso|coef|"] > 1e-3
sel["vote_RF"] = sel.RF_importance >= sel.RF_importance.quantile(.5)
sel["votes"] = sel[["vote_MI", "vote_Lasso", "vote_RF", "forward_SFS"]].sum(axis=1)
sel = sel.sort_values(["votes", "RF_importance"], ascending=False)
sel.head(25).round(4)

# %%
# redundancy filter among the kept features: drop the weaker of any pair with |ρ| > 0.9
kept = sel.index[sel.votes >= 2].tolist()
cm = X_all.loc[train_idx, kept].corr().abs()
redundant = set()
for i, a in enumerate(kept):
    for b in kept[i + 1:]:
        if cm.loc[a, b] > .9 and b not in redundant and a not in redundant:
            redundant.add(b if sel.loc[a, "RF_importance"] >= sel.loc[b, "RF_importance"] else a)
selected = [f for f in kept if f not in redundant]
print(f"{X_all.shape[1]} candidates → {len(kept)} with ≥2 votes → {len(selected)} after removing redundant {sorted(redundant)}")
dropped = sel.index[sel.votes < 2].tolist()
print("Dropped (fewer than 2 votes):", dropped)

fig, ax = plt.subplots(figsize=(8, 6))
top = sel.loc[selected].sort_values("RF_importance").tail(20)
ax.barh(top.index, top.RF_importance, color=PALETTE[0], height=.6); ax.set_title("Top-20 selected features (Random Forest importance)")
save(fig, "12_feature_importance_selected")

# %% [markdown]
# ## 10. Models with AutoViML
# **AutoViML** (Auto **V**ariant **I**nterpretable **ML**) automatically handles data-type detection, missing values, categorical encoding,
# **feature reduction** (SULOV, which removes correlated features, + recursive XGBoost selection), hyperparameter search, and model training.
#
# I run several configurations to compare model families. Each run gets the same **80/20 hold-out split** and the selected features, and the hold-out is scored in both log space and dollars.
#
# | Run | `Boosting_Flag` | Model family | Extra |
# |---|---|---|---|
# | A | `None` | Linear (AutoViML uses a LinearSVR grid search) | **not run**: liblinear did not converge after 15+ min on 38k rows × 2 CPUs. The OLS baseline below covers the linear family. |
# | B | `False` | Random Forest | |
# | C | `True` | XGBoost | |
# | D | `'CatBoost'` | CatBoost | |
# #
# I also run two plain baselines: predict the training median, and ordinary linear regression on the same features.

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model_df = X_all[selected].copy()
model_df["log_price"] = y_all
model_df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in model_df.columns]   # AutoViML/XGBoost-safe names
train, test = model_df.loc[train_idx].copy(), model_df.loc[test_idx].copy()   # same split used for selection
y_test = test.pop("log_price")
print(train.shape, test.shape)

def score(name, y_true_log, y_pred_log, secs):
    yt, yp = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {"model": name, "RMSE_log": np.sqrt(mean_squared_error(y_true_log, y_pred_log)), "MAE_log": mean_absolute_error(y_true_log, y_pred_log),
            "R2_log": r2_score(y_true_log, y_pred_log), "MAE_$": mean_absolute_error(yt, yp), "MedianAE_$": np.median(np.abs(yt - yp)), "train_sec": secs}

results = [score("Baseline: training median", y_test, np.full(len(y_test), train.log_price.median()), 0)]
t = time.time(); lr_ = LinearRegression().fit(train.drop(columns="log_price"), train.log_price)
results.append(score("Baseline: OLS linear regression", y_test, lr_.predict(test), time.time() - t))
pd.DataFrame(results).round(3)

# %%
# AutoViML 0.2.2 bug with its own pinned xgboost 1.7.6: it forces updater='grow_colmaker' together with tree_method='hist',
# which raises "Only hist can use QuantileDMatrix" and makes Auto_ViML return None. Patch that one line (idempotent).
import importlib.util, pathlib
_f = pathlib.Path(importlib.util.find_spec("autoviml").submodule_search_locations[0]) / "Auto_ViML.py"
_src = _f.read_text()
if "cpu_params['updater'] = 'grow_colmaker'" in _src:
    _f.write_text(_src.replace("cpu_params['updater'] = 'grow_colmaker'", "cpu_params['updater'] = 'grow_quantile_histmaker'")); print("patched AutoViML XGBoost updater")
else:
    print("AutoViML XGBoost patch already applied")
from autoviml.Auto_ViML import Auto_ViML
import contextlib, io

runs = {
    "AutoViML B: Random Forest (Boosting_Flag=False)": dict(Boosting_Flag=False),
    "AutoViML C: XGBoost (Boosting_Flag=True)": dict(Boosting_Flag=True),
    "AutoViML D: CatBoost": dict(Boosting_Flag="CatBoost"),
}
avml = {}
for name, kw in runs.items():
    t = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):      # AutoViML is very chatty; keep its log but print a short summary
        model, feats, trainm, testm = Auto_ViML(train.copy(), "log_price", test.copy(), sample_submission="", hyper_param="RS",
                                               feature_reduction=True, scoring_parameter="", verbose=0, **kw)
    secs = time.time() - t
    pred_col = [c for c in testm.columns if c.startswith("log_price") and c.endswith("predictions")]
    pred_col = "log_price_predictions" if "log_price_predictions" in pred_col else pred_col[0]
    results.append(score(name, y_test, testm[pred_col].values, secs))
    avml[name] = {"model": model, "features": feats, "log": buf.getvalue()}
    (OUT / f"autoviml_log_{name.split(':')[0].replace(' ', '_')}.txt").write_text(buf.getvalue())
    print(f"{name:<55} {len(feats):>3} features kept by AutoViML · RMSE_log={results[-1]['RMSE_log']:.4f} · {secs:.0f}s")

# %% [markdown]
# ## 11. Results & takeaways

# %%
res = pd.DataFrame(results).set_index("model").round(4)
res.to_csv(OUT / "model_results.csv")
res.sort_values("RMSE_log")

# %%
r = res.sort_values("RMSE_log", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
colors = [PALETTE[2] if m == r.RMSE_log.idxmin() else (MUTED if m.startswith("Baseline") else PALETTE[0]) for m in r.index]
axes[0].barh(r.index, r.RMSE_log, color=colors, height=.6); axes[0].set_title("Hold-out RMSE (log price) — lower is better")
axes[1].barh(r.index, r["MAE_$"], color=colors, height=.6); axes[1].set_title("Hold-out MAE in dollars — lower is better")
for a, col in zip(axes, ["RMSE_log", "MAE_$"]):
    for y, v in enumerate(r[col]): a.text(v, y, f" {v:.3f}" if col == "RMSE_log" else f" ${v:.0f}", va="center", fontsize=8)
save(fig, "13_model_comparison")

# %%
best = res.RMSE_log.idxmin()
print("Best model:", best)
print("Features AutoViML kept for it:", avml[best]["features"] if best in avml else "n/a")
bm = avml[best]["model"] if best in avml else None
if bm is not None and hasattr(bm, "feature_importances_"):
    imp = pd.Series(bm.feature_importances_, index=avml[best]["features"]).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.barh(imp.index, imp.values, color=PALETTE[0], height=.6)
    ax.set_title(f"What drives price: top features in the best model"); save(fig, "14_best_model_importance")

# %% [markdown]
# ### Takeaways (from the executed run)
# 1. **Data quality first.** Structural missingness (10,052 never-reviewed listings), 11 $0 prices, and 14 impossible minimum-night values would have silently corrupted any model.
# 2. **Impute by the *reason* values are missing.** KNN (MAE 0.71) and MICE (0.86) clearly beat median (1.13) and mean (1.24) under random masking, but the real gaps are true zeros, so zero + flag was used. KNN re-imputed 218 suspicious `minimum_nights` values.
# 3. **Consensus anomaly detection removed only 41 rows.** A single detector would have flagged 489–2,101 rows, mostly legitimate listings.
# 4. **Leakage check mattered.** My first run accidentally used price-based anomaly flags as features. That inflated R² to ~0.67. With them removed (and selection on training rows only), R² is an honest **≈0.58–0.59**.
# 5. **Linear regression on well-engineered features was the best model** (RMSE 0.448 in log space, median error ≈ $25/night), narrowly ahead of AutoViML CatBoost (0.454) and XGBoost (0.456). Random Forest was slowest and worst (0.471). Once room type, the neighbourhood target encoding, distance to Times Square, and the text features are in place, the remaining signal is close to additive. The feature engineering did more work than the model choice.
# 6. **Limitation:** there is no size, amenity, or photo data, so ~40% of the log-price variance can't be explained from these columns.
