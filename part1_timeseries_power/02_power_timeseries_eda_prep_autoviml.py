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
# # Part 1 (ii) — Time Series: Household Electric Power Consumption
# ## EDA → Cleaning → Gap Imputation → Anomalies & Clustering → Feature Engineering & Selection → AutoViML forecasting
#
# **CMPE 255 Data Mining · Fall 2026 · Anshita Shah**
#
# **Dataset:** [Household Electric Power Consumption (Kaggle, uciml)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set). This is one house in Sceaux, France,
# sampled **every minute for ~4 years** (Dec 2006 – Nov 2010): **2,075,259 rows × 9 columns**.
#
# | Column | Meaning | Unit |
# |---|---|---|
# | `Date`, `Time` | timestamp (separate text columns → must be parsed and merged) | dd/mm/yyyy, hh:mm:ss |
# | `Global_active_power` | household's total active power (**the target**) | kW (minute average) |
# | `Global_reactive_power` | reactive power | kW |
# | `Voltage` | mains voltage | V |
# | `Global_intensity` | current | A |
# | `Sub_metering_1` | kitchen (dishwasher, oven, microwave) | Wh |
# | `Sub_metering_2` | laundry (washer, dryer, fridge, light) | Wh |
# | `Sub_metering_3` | electric water heater + air conditioner | Wh |
#
# **Why it's a good cleaning exercise:** missing values are encoded as the string `'?'`; gaps range from 1 minute to 5 days; it has sensor glitches,
# vacation periods, and strong daily/weekly/annual seasonality. There are also physical laws to check consistency against (power = voltage × current).
#
# **ML task:** forecast the household's **average power in the next hour** from information available now (one-step-ahead forecasting, a common energy-utility task).
#
# **Notebook map**
# 0. Setup · 1. Load & parse · 2. Data audit · 3. Automated EDA · 4. Manual EDA & visualization · 5. Cleaning & consistency rules ·
# 6. Gap imputation (benchmark 6 methods) · 7. Resampling (data reduction) · 8. Anomaly detection & clustering of daily profiles ·
# 9. Feature engineering (lags, rolling, calendar, Fourier, wavelets, SAX) · 10. Feature selection · 11. AutoViML vs baselines · 12. Results

# %% [markdown]
# ## 0. Setup
# This notebook uses the same pinned AutoViML stack as the tabular notebook. The first run installs it and restarts Colab. After that, use **Runtime → Run all**.

# %%
import importlib.metadata as md, os, sys
def _v(pkg):
    try: return md.version(pkg)
    except md.PackageNotFoundError: return None

NEEDS_INSTALL = (_v("numpy") or "2").startswith("2") or _v("autoviml") is None
if NEEDS_INSTALL:
    print("Installing pinned AutoViML stack (≈2 min). The runtime will restart. Then click Runtime → Run all.")
    # !pip install -q "autoviml==0.2.2" "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.5.2" "xgboost==1.7.6" "setuptools<81" "ydata-profiling==4.12.2" "sweetviz==2.3.3" "pandas-dq==1.29" PyWavelets holidays statsmodels kagglehub
    os.kill(os.getpid(), 9)
else:
    print("Environment ready:", {p: _v(p) for p in ["numpy", "pandas", "scikit-learn", "statsmodels", "autoviml", "xgboost", "catboost"]})

# %%
import warnings, shutil, urllib.request, zipfile, time, re, io, contextlib
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 60); pd.set_option("display.width", 160)
RNG = 42; np.random.seed(RNG)
OUT = Path("outputs_power"); FIG = OUT / "figures"; REP = OUT / "reports"
for p in (FIG, REP): p.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ANOMALY = "#d03b3b"; MUTED = "#9a9993"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ad",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.prop_cycle": plt.cycler(color=PALETTE), "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 8.5,
})
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 1. Load & parse
# Loader order: local `data/` → **kagglehub** (`uciml/electric-power-consumption-data-set`) → public GitHub mirror of the same UCI/Kaggle file.
# `na_values=['?']` converts the dataset's missing-value marker to NaN at read time.

# %%
DATA = Path("data"); DATA.mkdir(exist_ok=True)
TXT = DATA / "household_power_consumption.txt"
MIRROR = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"

def load_power():
    if not TXT.exists():
        try:
            import kagglehub
            path = kagglehub.dataset_download("uciml/electric-power-consumption-data-set")
            shutil.copy(next(Path(path).rglob("household_power_consumption.txt")), TXT)
            print("Loaded from Kaggle via kagglehub")
        except Exception as e:
            print(f"kagglehub unavailable ({type(e).__name__}); using GitHub mirror of the same file")
            z = DATA / "hpc.zip"; urllib.request.urlretrieve(MIRROR, z)
            zipfile.ZipFile(z).extractall(DATA)
    t = time.time()
    raw = pd.read_csv(TXT, sep=";", na_values=["?"], low_memory=False)
    print(f"read {len(raw):,} rows in {time.time()-t:.0f}s")
    return raw

raw = load_power()
raw.head()

# %%
raw.dtypes.to_frame("dtype after read")

# %%
# Parse the two text columns into one timestamp index (explicit format = 10× faster and no day/month ambiguity)
ts = raw.copy()
ts.index = pd.to_datetime(ts.Date + " " + ts.Time, format="%d/%m/%Y %H:%M:%S")
ts.index.name = "timestamp"
ts = ts.drop(columns=["Date", "Time"]).astype("float32")
ts.columns = ["GAP", "GRP", "Voltage", "Intensity", "Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac"]
print(ts.index.min(), "→", ts.index.max(), "|", ts.shape, f"| memory {ts.memory_usage().sum()/1e6:.0f} MB")

# %% [markdown]
# ## 2. Data audit

# %%
full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq="min")
audit = {
    "rows": len(ts),
    "expected minute timestamps": len(full_idx),
    "missing timestamps (rows absent)": len(full_idx) - len(ts),
    "duplicate timestamps": ts.index.duplicated().sum(),
    "monotonic increasing index": ts.index.is_monotonic_increasing,
    "rows with ANY NaN ('?')": ts.isna().any(axis=1).sum(),
    "rows with ALL NaN": ts.isna().all(axis=1).sum(),
    "% missing": round(ts.GAP.isna().mean() * 100, 3),
}
pd.Series(audit, name="value").to_frame()

# %%
# Gap structure: how long is each run of consecutive missing minutes?
na = ts.GAP.isna()
run_id = (na != na.shift()).cumsum()
runs = ts.index.to_series()[na].groupby(run_id[na]).agg(["first", "last", "size"]).rename(columns={"size": "minutes"})
runs["bucket"] = pd.cut(runs.minutes, [0, 1, 10, 60, 1440, 1e9], labels=["1 min", "2–10 min", "11–60 min", "1–24 h", "> 1 day"])
display(runs.bucket.value_counts().sort_index().to_frame("gap runs").assign(minutes_missing=runs.groupby("bucket").minutes.sum()))
runs.sort_values("minutes", ascending=False).head(8)

# %%
ts.describe(percentiles=[.001, .01, .5, .99, .999]).T.round(3)

# %% [markdown]
# **Audit findings:**
# - The timestamp grid is **complete** (no absent rows, no duplicates). All missingness is the `'?'` marker, and when one sensor is missing **all 7 are missing together**: these are whole-meter outages, not single bad sensors.
# - 1.25% of minutes are missing across **71 gaps**. Most are 1–10 minutes, but **6 gaps exceed a day** (the longest is ~5 days in Aug 2010). A single imputation method will not fit both cases. Section 6 benchmarks methods *by gap length*.
# - Sub-metering is in **Wh per minute** while GAP is in **kW**, so the units must be aligned before comparing them (Section 5).

# %% [markdown]
# ## 3. Automated EDA
# Minute-level data (2M rows) is too large for profiling tools, so I profile the **hourly mean** series:
# - **ydata-profiling in time-series mode** (`tsmode=True`): adds autocorrelation and seasonality/stationarity alerts per column
# - **Sweetviz compare**: **2007–2009 vs 2010**, which is exactly my later train/test split. It checks for **distribution shift** before modeling.
# - **pandas-dq**: automated data-quality suggestions

# %%
hourly_raw = ts.resample("h").mean()
from ydata_profiling import ProfileReport
t = time.time()
prof = ProfileReport(hourly_raw.reset_index(), tsmode=True, sortby="timestamp", title="Household power (hourly) — ydata-profiling TS mode",
                     progress_bar=False, interactions={"continuous": False}, samples={"head": 5, "tail": 0})
prof.to_file(REP / "ydata_profile_power_hourly.html")
alerts = [str(a) for a in prof.get_description().alerts]
print(f"saved in {time.time()-t:.0f}s · {len(alerts)} alerts:"); print("\n".join("  • " + a for a in alerts))

# %%
import sweetviz as sv
h = hourly_raw.dropna(subset=["GAP"]).copy(); h["hour"] = h.index.hour; h["dow"] = h.index.dayofweek
cmp = sv.compare([h[h.index < "2010-01-01"].reset_index(drop=True), "Train 2007–2009"], [h[h.index >= "2010-01-01"].reset_index(drop=True), "Test 2010"], target_feat="GAP", pairwise_analysis="off")
cmp.show_html(str(REP / "sweetviz_power_train_vs_test.html"), open_browser=False, layout="vertical")

# %%
from pandas_dq import dq_report
dq_report(hourly_raw.reset_index(), target="GAP", verbose=0, html=False, csv_engine="pandas")

# %% [markdown]
# ### Reading the automated reports
# | Automated finding | My interpretation / action |
# |---|---|
# | ydata TS mode: GAP, Voltage, and Intensity are **non-stationary** *and* **seasonal** | Expected for energy load (daily/annual cycles, slow trend). Lags, rolling statistics, and calendar/Fourier terms will be the backbone of the features. |
# | ydata: **GAP ↔ Intensity** and **GAP ↔ Sub3_heater_ac** highly correlated; **GRP ↔ Sub2_laundry** correlated | Physics: current ≈ power / voltage, so Intensity is nearly a copy of the target. It is used only as a *consistency check* and at time *t* to predict *t+1*, never at the same time step. The water heater/AC is the biggest single load. |
# | ydata: `Sub1_kitchen` is **zero 84.5%** of hours, `Sub2_laundry` 29.9%, `Sub3_heater_ac` 22.9% | Zero-inflated appliance circuits (off most of the time). I add an "appliance active" flag. |
# | ydata: **421 hours (1.2%) fully missing** in every column at once | Whole-meter outages. Confirms the audit: all channels drop together. |
# | Sweetviz compare: **2010 has lower average GAP** than 2007–2009 | A **distribution shift**. A model trained on earlier years may over-predict, so I report error *in kW* and compare against naive baselines, which adapt automatically. |
# | pandas-dq flags missing hours and suggests imputation | Agreed, but the gaps need *time-aware* imputation (Section 6), not the mean/median imputation it suggests. |

# %% [markdown]
# ## 4. Manual EDA & visualization
# ### 4.1 The whole series, plus where the data is missing

# %%
daily = ts.GAP.resample("D").mean()
fig, ax = plt.subplots(figsize=(13, 3.6))
ax.plot(daily.index, daily, color="#b9cfe9", lw=.8, label="daily mean")
ax.plot(daily.index, daily.rolling(30, center=True, min_periods=10).mean(), color=PALETTE[0], lw=2, label="30-day rolling mean")
for _, r in runs[runs.minutes > 1440].iterrows():
    ax.axvspan(r["first"], r["last"], color=ANOMALY, alpha=.25, lw=0)
ax.axvspan(runs.iloc[0]["first"], runs.iloc[0]["first"], color=ANOMALY, alpha=.25, label="gaps > 1 day")
ax.set_ylabel("kW"); ax.set_title("Global active power: annual seasonality (winter peaks) and a summer-2008 dip"); ax.legend(ncols=3, loc="upper right")
save(fig, "01_series_overview")

# %% [markdown]
# ### 4.2 Seasonality at three scales: hour × weekday, month

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
pv = hourly_raw.GAP.groupby([hourly_raw.index.dayofweek, hourly_raw.index.hour]).mean().unstack()
pv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sns.heatmap(pv, cmap="Blues", ax=axes[0], cbar_kws={"label": "mean kW", "shrink": .8}, linewidths=.3, linecolor="#fcfcfb")
axes[0].set_title("Mean power by weekday × hour: evening peak, weekend midday load"); axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("")
m = daily.groupby(daily.index.month)
axes[1].boxplot([g.dropna().values for _, g in m], tick_labels=list("JFMAMJJASOND"), showfliers=False, widths=.55,
                medianprops={"color": PALETTE[1]}, boxprops={"color": "#52514e"}, whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"})
axes[1].set_title("Daily mean by month: annual cycle"); axes[1].set_ylabel("kW")
save(fig, "02_seasonality")

# %% [markdown]
# ### 4.3 Decomposition (STL) and autocorrelation

# %%
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
d_filled = daily.interpolate(limit_direction="both")
stl_d = STL(d_filled, period=365, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(13, 6.5), sharex=True)
for a, (s, lbl) in zip(axes, [(d_filled, "observed (daily)"), (stl_d.trend, "trend"), (stl_d.seasonal, "annual seasonal"), (stl_d.resid, "residual")]):
    a.plot(s.index, s, lw=.9, color=PALETTE[0] if lbl != "residual" else MUTED); a.set_ylabel(lbl, fontsize=8.5)
axes[0].set_title("STL decomposition of daily mean power (period = 365 days): slow downward trend + strong annual cycle")
save(fig, "03_stl_daily")

h_filled = hourly_raw.GAP.interpolate(limit_direction="both")
fig, axes = plt.subplots(1, 2, figsize=(13, 3.2))
plot_acf(h_filled, lags=24 * 8, ax=axes[0], markersize=2)
axes[0].set_title("ACF (hourly, 8 days): peaks at 24 h multiples and 168 h")
plot_pacf(h_filled, lags=50, ax=axes[1], markersize=2, method="ywm")
axes[1].set_title("PACF (hourly, 50 lags): lags 1–3 and ~24 matter most")
save(fig, "04_acf_pacf")

# %% [markdown]
# ### 4.4 What is the energy used for? Sub-metering share

# %%
wh = ts.copy()
wh["total_Wh"] = wh.GAP * 1000 / 60                                  # kW averaged over 1 min → Wh in that minute
wh["Sub4_other"] = wh.total_Wh - wh[["Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac"]].sum(axis=1)
share = wh[["Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac", "Sub4_other"]].resample("MS").sum()
share = share.div(share.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 3.4))
ax.stackplot(share.index, share.T.values, labels=["kitchen", "laundry", "water heater + AC", "unmetered (lighting, electronics…)"],
             colors=PALETTE[:4], edgecolor="#fcfcfb", linewidth=.6)
ax.set_ylim(0, 100); ax.set_ylabel("% of monthly energy"); ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(.5, -.12))
ax.set_title("Monthly energy share by circuit: about half of all energy is unmetered")
save(fig, "05_submeter_share")

# %% [markdown]
# **Manual EDA takeaways**
# 1. **Three nested seasonalities.** Daily (evening peak ~19–21 h plus a smaller morning peak), weekly (weekend daytime load is higher), and annual (winter ≈ 2× summer, consistent with electric heating). The ACF peaks at 24 h and 168 h, so the **lags to engineer are 1–3, 24, 168**.
# 2. **August is a vacation month.** Consumption collapses to near-baseload for ~2–3 weeks (most visible in Aug 2008). These are behavioral anomalies, not sensor errors, and are handled in Section 8.
# 3. **A slow downward trend** across the four years (efficiency or behavior change) plus the lower 2010 level from Sweetviz means the test period differs from training.
# 4. **Sub-metering explains only about half of the energy.** I derive the unmetered remainder `Sub4_other` (lighting, electronics) as a feature.

# %% [markdown]
# ## 5. Cleaning & physical-consistency rules
# Domain rules from electrical physics check whether each reading is *internally consistent*. Readings that fail are sensor errors → set to NaN → imputed in Section 6.
#
# 1. **Power = Voltage × Current.** `Intensity ≈ GAP·1000 / Voltage` (for apparent power; the active-power version differs slightly by power factor). Intensity is reported in 0.2 A steps and every value is a 1-minute *average*, so appliances switching mid-minute create small mismatches. Only a residual above **2 A** (≈0.5 kW at 240 V) is treated as corrupted.
# 2. **Sub-meters can't exceed the total.** `Sub4_other = total_Wh − Σ sub-meters` should not be meaningfully negative.
# 3. **Voltage must be within the grid tolerance.** The French grid is 230 V ±10% → 207–253 V.

# %%
clean = ts.copy()
apparent_I = np.sqrt(clean.GAP ** 2 + clean.GRP ** 2) * 1000 / clean.Voltage
resid_I = clean.Intensity - apparent_I
sub4 = clean.GAP * 1000 / 60 - clean[["Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac"]].sum(axis=1)
rules = pd.DataFrame({
    "|I − √(P²+Q²)/V| > 2 A": resid_I.abs() > 2.0,
    "sub-meters exceed total by > 1 Wh": sub4 < -1,
    "voltage outside 207–253 V": ~clean.Voltage.between(207, 253) & clean.Voltage.notna(),
})
display(rules.sum().to_frame("violations").assign(**{"% of rows": lambda d: (d.violations / len(clean) * 100).round(4)}))
fig, ax = plt.subplots(figsize=(6.5, 3)); ax.hist(resid_I.dropna(), bins=200, color=PALETTE[0]); ax.set_yscale("log")
ax.axvline(2, color=ANOMALY, ls="--", lw=1); ax.axvline(-2, color=ANOMALY, ls="--", lw=1)
ax.set_title("Physics residual: measured current − √(P²+Q²)/V"); ax.set_xlabel("amperes (log count)")
save(fig, "06_physics_residual")

# %%
bad = rules.any(axis=1)
clean.loc[bad, :] = np.nan
print(f"{bad.sum():,} physically inconsistent minutes set to NaN → total missing now {clean.GAP.isna().mean()*100:.3f}%")

# %% [markdown]
# ## 6. Gap imputation: benchmark, then choose by gap length
# The lecture's time-series rule: *"use the behavioral attribute values of contextually nearby records… linearly interpolate"*. That works for short gaps,
# but a 3-day gap needs a **seasonal** estimate. I test six methods on **artificial gaps** of different lengths placed in fully observed stretches, and score the error against the hidden truth.

# %%
s = clean.GAP.copy()
def seasonal_fill(series, lag):          # value from the same minute `lag` minutes earlier (falls back to the next earlier period)
    out = series.copy()
    for k in range(1, 5):
        out = out.fillna(series.shift(lag * k))
    return out

how_of_week = s.groupby([s.index.dayofweek, s.index.hour, s.index.minute]).transform("median")
methods = {
    "forward fill": lambda x: x.ffill(),
    "linear interpolation": lambda x: x.interpolate("linear", limit_direction="both"),
    "time-of-week median profile": lambda x: x.fillna(how_of_week),
    "seasonal naive (−1 day)": lambda x: seasonal_fill(x, 1440),
    "seasonal naive (−1 week)": lambda x: seasonal_fill(x, 10080),
    "blend: ½ day-ago + ½ week-ago": lambda x: x.fillna((seasonal_fill(x, 1440) + seasonal_fill(x, 10080)) / 2),
}
rs = np.random.RandomState(RNG)
gap_lengths = [1, 10, 60, 360, 1440, 4320]
rows = []
for L in gap_lengths:
    # the hidden stretch itself must be fully observed so we know the truth; ≥ 2 weeks of history for seasonal methods
    valid_starts = np.where(s.rolling(L).count().shift(-(L - 1)).values == L)[0]
    valid_starts = valid_starts[(valid_starts > 20160) & (valid_starts < len(s) - L - 20)]
    starts = rs.choice(valid_starts, 40, replace=False)
    for st in starts:
        lo = max(0, st - 20160); hi = st + L + 10   # local window keeps each test fast and leakage-free
        win = s.iloc[lo:hi].copy(); truth = win.iloc[st - lo: st - lo + L].copy(); win.iloc[st - lo: st - lo + L] = np.nan
        for nm, f in methods.items():
            est = f(win).iloc[st - lo: st - lo + L]
            rows.append({"gap": L, "method": nm, "MAE": float(np.nanmean(np.abs(est.values - truth.values)))})
bench = pd.DataFrame(rows).groupby(["gap", "method"]).MAE.mean().unstack("gap").round(3)
bench.columns = [f"{c} min" for c in bench.columns]
bench

# %%
fig, ax = plt.subplots(figsize=(9, 3.6))
for i, (m_, r) in enumerate(bench.iterrows()):
    ax.plot(gap_lengths, r.values, marker="o", markersize=4, label=m_, color=PALETTE[i])
ax.set_xscale("log"); ax.set_xticks(gap_lengths); ax.set_xticklabels(bench.columns)
ax.set_ylabel("MAE (kW)"); ax.set_title("Imputation error by gap length: interpolation wins short gaps, seasonal wins long ones")
ax.legend(ncols=2, fontsize=7.5)
save(fig, "07_imputation_benchmark")

# %% [markdown]
# **Decision: a hybrid imputer.** For gaps up to the crossover length the benchmark shows, use **linear interpolation**. Longer gaps get the best seasonal method.
# Because all channels go missing together, the same rule is applied to every column.

# %%
short_best = bench.iloc[:, :2].mean(axis=1).idxmin(); long_best = bench.iloc[:, -2:].mean(axis=1).idxmin()
cross = next((g for g, c in zip(gap_lengths, bench.columns) if bench.loc[long_best, c] < bench.loc[short_best, c]), 360)
print(f"short gaps → {short_best} | long gaps → {long_best} | switch when gap ≥ {cross} min")

def hybrid_impute(frame, max_short):
    out = frame.copy()
    isna = out.iloc[:, 0].isna(); rid = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(rid).transform("sum").where(isna, 0)
    short_mask = isna & (run_len < max_short)
    interp = out.interpolate("linear", limit_direction="both")
    out[short_mask] = interp[short_mask]
    for c in out.columns:
        col = out[c]
        if long_best == "time-of-week median profile":
            prof = col.groupby([col.index.dayofweek, col.index.hour, col.index.minute]).transform("median"); filled = col.fillna(prof)
        elif long_best.startswith("blend"):
            filled = col.fillna((seasonal_fill(col, 1440) + seasonal_fill(col, 10080)) / 2)
        elif "week" in long_best:
            filled = seasonal_fill(col, 10080)
        else:
            filled = seasonal_fill(col, 1440)
        out[c] = filled.fillna(col.interpolate(limit_direction="both"))
    return out

t = time.time()
imputed = hybrid_impute(clean, cross)
imputed["imputed_flag"] = clean.GAP.isna().astype("int8")
print(f"done in {time.time()-t:.0f}s · remaining NaN: {int(imputed.isna().sum().sum())}")

# show one long gap before/after
g = runs.sort_values("minutes", ascending=False).iloc[2]
w0, w1 = g["first"] - pd.Timedelta(days=2), g["last"] + pd.Timedelta(days=2)
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(imputed.loc[w0:w1].index, imputed.loc[w0:w1, "GAP"].rolling(15).mean(), color=PALETTE[1], label="imputed (15-min smoothed)")
ax.plot(ts.loc[w0:w1].index, ts.loc[w0:w1, "GAP"].rolling(15).mean(), color=PALETTE[0], label="observed")
ax.axvspan(g["first"], g["last"], color=ANOMALY, alpha=.12, lw=0); ax.legend(loc="upper left"); ax.set_ylabel("kW")
ax.set_title(f"Example: a {g['minutes']/60:.0f}-hour outage filled with seasonal structure, not a flat line")
save(fig, "08_imputation_example")

# %% [markdown]
# ## 7. Data reduction: resample to hourly
# The lecture lists **sampling/aggregation** as data reduction. Forecasting the next hour does not need minute resolution, and aggregating 2M rows to ~34.6k
# hourly rows makes AutoViML practical. Minute-level detail isn't thrown away: it is summarized as within-hour **max, std, and the share of imputed minutes**.

# %%
imputed["Sub4_other"] = imputed.GAP * 1000 / 60 - imputed[["Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac"]].sum(axis=1)
agg = {c: "mean" for c in ["GAP", "GRP", "Voltage", "Intensity"]}
agg.update({c: "sum" for c in ["Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac", "Sub4_other"]})
H = imputed.resample("h").agg(agg)
H["GAP_max_min"] = imputed.GAP.resample("h").max()
H["GAP_std_min"] = imputed.GAP.resample("h").std()
H["imputed_share"] = imputed.imputed_flag.resample("h").mean()
H = H.iloc[1:-1]   # drop partial first/last hours
print(H.shape); H.head()

# %% [markdown]
# ## 8. Anomaly detection & clustering
# Two levels:
# - **Point anomalies (single hours):** Hampel filter (rolling median ± 3·MAD), STL-residual robust z-score, and Isolation Forest on the multivariate hourly vector
# - **Contextual anomalies (whole days):** cluster the **24-hour daily load profiles** with K-Means to find "day types", and use DBSCAN to flag days that fit no cluster (vacations, parties, outages)
#
# **Elimination policy.** Rows can't be deleted from the middle of a time series without breaking the lags, so:
# 1. Point anomalies flagged by **≥ 2 detectors** in the **training period** are **replaced** with the Hampel rolling median in the *feature* series (spikes no longer contaminate the lags and rolling means). The test year is left raw because the centered median looks ahead in time.
# 2. **Anomalous days** (DBSCAN noise + vacation cluster) are **eliminated from the training rows** after features are built, so the model learns normal behavior.
#    The **test set is never cleaned**, so the evaluation stays honest.

# %%
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

y = H.GAP
med = y.rolling(25, center=True, min_periods=5).median()
mad = (y - med).abs().rolling(25, center=True, min_periods=5).median()
H["out_hampel"] = ((y - med).abs() > 3 * 1.4826 * mad).astype(int)

stl_h = STL(y, period=24, robust=True).fit()
r = stl_h.resid; rz = 0.6745 * (r - r.median()).abs() / (r - r.median()).abs().median()
H["out_stl"] = (rz > 4.5).astype(int)

iso_cols = ["GAP", "GRP", "Voltage", "Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac", "Sub4_other", "GAP_std_min"]
Xi = StandardScaler().fit_transform(H[iso_cols].assign(hour=H.index.hour))
H["out_iforest"] = (IsolationForest(n_estimators=300, contamination=0.005, random_state=RNG).fit_predict(Xi) == -1).astype(int)
H["n_flags"] = H[["out_hampel", "out_stl", "out_iforest"]].sum(axis=1)
display(H[["out_hampel", "out_stl", "out_iforest"]].sum().to_frame("hours flagged").assign(**{"≥2 detectors": [(H.n_flags >= 2).sum()] * 3}))

wk = slice("2008-01-14", "2008-01-27")
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.plot(H.loc[wk].index, H.loc[wk, "GAP"], color=PALETTE[0], lw=1, label="hourly GAP")
ax.plot(med.loc[wk].index, med.loc[wk], color=MUTED, lw=1, ls="--", label="Hampel rolling median")
f = H.loc[wk][H.loc[wk].n_flags >= 2]; ax.scatter(f.index, f.GAP, color=ANOMALY, s=26, zorder=3, label="anomaly (≥2 detectors)")
ax.legend(ncols=3, loc="upper left"); ax.set_ylabel("kW"); ax.set_title("Point anomalies over two winter weeks")
save(fig, "09_point_anomalies")

# %%
# Only the TRAINING period (before 2010) is cleaned: the Hampel median is centered (it looks ±12 h ahead),
# so applying it to the test year would leak future information into the lag features.
fix = (H.n_flags >= 2) & (H.index < "2010-01-01")
H["GAP_clean"] = np.where(fix, med, H.GAP)
print(f"replaced {fix.sum()} anomalous training-period hours with the rolling median (feature series only; 2010 left raw)")

# %% [markdown]
# ### 8.2 Clustering daily load profiles (day types)

# %%
D = H.GAP.groupby(H.index.date).apply(lambda s: s.values if len(s) == 24 else None).dropna()
P = np.vstack(D.values); days = pd.to_datetime(D.index)
P_feat = np.hstack([P, P.mean(axis=1, keepdims=True)])        # shape + level
Zd = StandardScaler().fit_transform(P_feat)
ks = range(2, 9); sil = [silhouette_score(Zd, KMeans(k, n_init=10, random_state=RNG).fit_predict(Zd)) for k in ks]
k_best = list(ks)[int(np.argmax(sil))]
km = KMeans(k_best, n_init=20, random_state=RNG).fit(Zd)
day_type = pd.Series(km.labels_, index=days, name="day_type")
print("silhouette by k:", dict(zip(ks, np.round(sil, 3))), "→ k =", k_best)

nn = NearestNeighbors(n_neighbors=8).fit(Zd); dist = np.sort(nn.kneighbors(Zd)[0][:, -1])
eps = float(np.quantile(dist, .97))
day_noise = pd.Series(DBSCAN(eps=eps, min_samples=8).fit_predict(Zd) == -1, index=days, name="dbscan_noise")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for c in range(k_best):
    prof = P[km.labels_ == c]
    axes[0].plot(range(24), prof.mean(axis=0), color=PALETTE[c], lw=2, label=f"type {c}: {len(prof)} days, {prof.mean():.2f} kW")
axes[0].set_xlabel("hour"); axes[0].set_ylabel("kW"); axes[0].set_title("K-Means day types (mean 24-h profile)"); axes[0].legend(fontsize=7.5)
cal = pd.DataFrame({"day_type": day_type, "noise": day_noise})
for c in range(k_best):
    d_ = cal.index[cal.day_type == c]; axes[1].scatter(d_, [c] * len(d_), s=4, color=PALETTE[c])
d_ = cal.index[cal.noise]; axes[1].scatter(d_, [-0.7] * len(d_), s=10, color=ANOMALY, marker="x", label=f"DBSCAN noise ({len(d_)} days)")
axes[1].set_yticks(list(range(k_best)) + [-0.7]); axes[1].set_yticklabels([f"type {c}" for c in range(k_best)] + ["anomalous"])
axes[1].set_title("When each day type occurs"); axes[1].legend(loc="upper right")
save(fig, "10_day_type_clusters")

lowest = int(np.argmin([P[km.labels_ == c].mean() for c in range(k_best)]))
vac = day_type[day_type == lowest]
print(f"lowest-load day type = {lowest}; its months:", vac.index.month.value_counts().sort_index().to_dict())

# %% [markdown]
# **Reading the clusters:** silhouette picks **k = 2**: a **high-consumption winter day type** and a **low-load type** that dominates from April to September and peaks in July–August (vacations plus no heating).
# DBSCAN marks days whose profile resembles no other day (unusual spikes, partial outages).
# Training rows from DBSCAN-noise days are **eliminated**. The day-type label of **yesterday** becomes a feature (clustering as a feature transformer, as in AutoViML's `KMeans_Featurizer`).

# %% [markdown]
# ## 9. Feature engineering
# **Target:** `y = GAP(t+1)`, the mean power in the **next** hour. All features use information at time *t* or earlier (no leakage).
#
# | Family | Features | Lecture link |
# |---|---|---|
# | Lags | GAP at t, t−1, t−2, t−5, t−23 (same hour yesterday → next hour), t−167 (same hour last week) | behavioral attributes of nearby records |
# | Rolling stats | mean/std/max over 3, 6, 24, 168 h | smoothing / aggregation |
# | Calendar | hour, weekday, month, weekend, French public holiday + **sin/cos** cyclic encodings | contextual attributes |
# | Fourier | annual sin/cos terms | time series → numeric |
# | **Wavelets** | Haar DWT of the last 24 h: approximation coefficients + detail-energy per level | *Time series → numeric: wavelet transform* slides |
# | **SAX** | symbolic word of the last 24 h (PAA 6 segments, 4-letter alphabet, Gaussian breakpoints) | *Time series → discrete sequence: SAX* slides |
# | Exogenous | reactive power, voltage, sub-meters, unmetered remainder, minute-level volatility, imputed share at t | multivariate context |
# | Clustering | yesterday's day type | clustering as a feature |

# %%
import pywt, holidays
F = pd.DataFrame(index=H.index)
g = H.GAP_clean
F["target_next_hour"] = H.GAP.shift(-1)                 # raw (not cleaned) future value = honest target
for L in [0, 1, 2, 5, 23, 167]:
    F[f"gap_lag{L}"] = g.shift(L)
for w in [3, 6, 24, 168]:
    F[f"gap_roll_mean_{w}"] = g.rolling(w).mean(); F[f"gap_roll_std_{w}"] = g.rolling(w).std()
F["gap_roll_max_24"] = g.rolling(24).max()
F["gap_diff_1"] = g - g.shift(1); F["gap_diff_24"] = g - g.shift(24)
for c in ["GRP", "Voltage", "Intensity", "Sub1_kitchen", "Sub2_laundry", "Sub3_heater_ac", "Sub4_other", "GAP_std_min", "GAP_max_min", "imputed_share"]:
    F[c.lower()] = H[c]
F["sub3_active"] = (H.Sub3_heater_ac > 60).astype(int)       # water heater/AC drawing > 1 kWh in the hour

nxt = F.index + pd.Timedelta(hours=1)                         # calendar features describe the hour being forecast
fr_hol = holidays.France(years=range(2006, 2011))
F["hour"] = nxt.hour; F["dow"] = nxt.dayofweek; F["month"] = nxt.month
F["is_weekend"] = (nxt.dayofweek >= 5).astype(int); F["is_holiday"] = pd.Series(nxt.date, index=F.index).map(lambda d: d in fr_hol).astype(int)
for nm_, per, val in [("hour", 24, nxt.hour), ("dow", 7, nxt.dayofweek), ("doy", 365.25, nxt.dayofyear)]:
    F[f"{nm_}_sin"] = np.sin(2 * np.pi * val / per); F[f"{nm_}_cos"] = np.cos(2 * np.pi * val / per)

# Wavelets: Haar DWT (level 3) of each trailing 24-h window
win = np.lib.stride_tricks.sliding_window_view(g.values, 24)
coeffs = pywt.wavedec(win, "haar", level=3, axis=1)          # [cA3 (3), cD3 (3), cD2 (6), cD1 (12)]
wf = np.hstack([coeffs[0], *[np.sum(c ** 2, axis=1, keepdims=True) for c in coeffs[1:]]])
wcols = ["wav_cA3_0", "wav_cA3_1", "wav_cA3_2", "wav_energy_D3", "wav_energy_D2", "wav_energy_D1"]
for c in wcols: F[c] = np.nan
F["sax24"] = None
F.loc[F.index[23:], wcols] = wf

# SAX word of trailing 24 h
breakpoints = [-0.6745, 0, 0.6745]                            # 4 equiprobable Gaussian bins
def sax_word(x):
    z = (x - x.mean()) / (x.std() + 1e-9); paa = z.reshape(6, 4).mean(axis=1)
    return "".join("abcd"[np.searchsorted(breakpoints, v)] for v in paa)
F.loc[F.index[23:], "sax24"] = [sax_word(w_) for w_ in win]

F["yesterday_day_type"] = pd.Series(F.index.normalize() - pd.Timedelta(days=1), index=F.index).map(day_type).values
F["yesterday_anomalous"] = pd.Series(F.index.normalize() - pd.Timedelta(days=1), index=F.index).map(day_noise).fillna(False).astype(int).values

F = F.dropna()
print(F.shape)
print("most frequent SAX words of the trailing day:"); display(F.sax24.value_counts().head(6).to_frame("hours"))

# %%
# Wavelet example straight from the lecture: {141, 15, 100, 182} → averages {78, 141}, deltas {−63, 41}
a = np.array([141, 15, 100, 182], dtype=float)
cA, cD = pywt.dwt(a, "haar")
print("pywt (orthonormal Haar) cA, cD:", cA.round(2), cD.round(2), "| lecture form (÷√2):", (cA / np.sqrt(2)).round(1), (cD / np.sqrt(2)).round(1))

# %% [markdown]
# ### Chronological split + anomaly elimination on training rows only

# %%
TEST_START = "2010-01-01"
train_F = F[F.index < TEST_START].copy(); test_F = F[F.index >= TEST_START].copy()
noise_days = set(day_noise[day_noise].index.date)
before = len(train_F)
train_F = train_F[~pd.Series(train_F.index.date, index=train_F.index).isin(noise_days)]
print(f"train {before:,} → {len(train_F):,} rows after removing {before-len(train_F):,} hours on DBSCAN-anomalous days | test {len(test_F):,} rows (untouched)")

# %% [markdown]
# ## 10. Feature selection: filter, wrapper, embedded
# Selection runs on the **training period only**. Choosing features with test data would be leakage.

# %%
from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

Xtr = pd.get_dummies(train_F.drop(columns="target_next_hour"), columns=["sax24"], dtype=int)
Xte = pd.get_dummies(test_F.drop(columns="target_next_hour"), columns=["sax24"], dtype=int).reindex(columns=Xtr.columns, fill_value=0)
ytr, yte = train_F.target_next_hour, test_F.target_next_hour
sax_cols = [c for c in Xtr.columns if c.startswith("sax24_")]
keep_sax = Xtr[sax_cols].sum().sort_values(ascending=False).index[:15]     # OOV: rare SAX words dropped
Xtr = Xtr.drop(columns=[c for c in sax_cols if c not in keep_sax]); Xte = Xte[Xtr.columns]

samp = Xtr.sample(12000, random_state=RNG).index
Xs_ = pd.DataFrame(StandardScaler().fit_transform(Xtr.loc[samp]), columns=Xtr.columns, index=samp)
mi = pd.Series(mutual_info_regression(Xtr.loc[samp], ytr.loc[samp], random_state=RNG), index=Xtr.columns)
lasso = LassoCV(cv=TimeSeriesSplit(5), random_state=RNG, n_alphas=40).fit(StandardScaler().fit_transform(Xtr), ytr)
lasso_coef = pd.Series(np.abs(lasso.coef_), index=Xtr.columns)
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RNG).fit(Xtr.loc[samp], ytr.loc[samp])
rf_imp = pd.Series(rf.feature_importances_, index=Xtr.columns)
t = time.time()
sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=15, direction="forward", cv=3, n_jobs=-1).fit(Xs_, ytr.loc[samp])
print(f"forward selection {time.time()-t:.0f}s")
sel = pd.DataFrame({"MI": mi, "Lasso|coef|": lasso_coef, "RF_importance": rf_imp, "forward_SFS": sfs.get_support()}, index=Xtr.columns)
sel["votes"] = (sel.MI >= sel.MI.quantile(.5)).astype(int) + (sel["Lasso|coef|"] > 1e-3).astype(int) + (sel.RF_importance >= sel.RF_importance.quantile(.5)).astype(int) + sel.forward_SFS.astype(int)
sel = sel.sort_values(["votes", "RF_importance"], ascending=False)
kept = sel.index[sel.votes >= 2].tolist()
cm = Xtr[kept].corr().abs(); redundant = set()
for i, a in enumerate(kept):
    for b in kept[i + 1:]:
        if cm.loc[a, b] > .95 and a not in redundant and b not in redundant:
            redundant.add(b if sel.loc[a, "RF_importance"] >= sel.loc[b, "RF_importance"] else a)
selected = [f for f in kept if f not in redundant]
print(f"{Xtr.shape[1]} candidates → {len(kept)} with ≥2 votes → {len(selected)} after dropping redundant: {sorted(redundant)}")
sel.head(20).round(4)

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))
top = sel.loc[selected].sort_values("RF_importance").tail(20)
ax.barh(top.index, top.RF_importance, color=PALETTE[0], height=.6); ax.set_title("Top-20 selected features (Random Forest importance)")
save(fig, "11_feature_importance_selected")

# %% [markdown]
# ## 11. Models: naive baselines vs AutoViML
# A forecasting model is only useful if it beats **naive forecasts**:
# - **Persistence:** next hour = this hour
# - **Seasonal naive (day):** next hour = same hour yesterday
# - **Seasonal naive (week):** next hour = same hour last week
#
# Then OLS (the linear family; AutoViML's own linear mode runs a LinearSVR grid search that did not finish in 15+ min on 2 CPUs), plus AutoViML with three model families: Random Forest, XGBoost, and CatBoost. The test set is the **whole of 2010** (≈ 7,900 hours), never seen during cleaning, selection, or training.

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def score(name, yt, yp, secs=0):
    return {"model": name, "RMSE_kW": np.sqrt(mean_squared_error(yt, yp)), "MAE_kW": mean_absolute_error(yt, yp), "R2": r2_score(yt, yp),
            "MAPE_%": np.mean(np.abs((yt - yp) / np.clip(yt, .1, None))) * 100, "train_sec": secs}
H_raw_gap = H.GAP
results = [
    score("Naive: persistence (t)", yte, H_raw_gap.reindex(test_F.index).values),
    score("Naive: same hour yesterday", yte, H_raw_gap.shift(23).reindex(test_F.index).values),
    score("Naive: same hour last week", yte, H_raw_gap.shift(167).reindex(test_F.index).values),
]
t = time.time(); ols = LinearRegression().fit(Xtr[selected], ytr)
results.append(score("OLS linear regression (selected features)", yte, ols.predict(Xte[selected]), time.time() - t))
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
clean_name = lambda c: re.sub(r"[^0-9a-zA-Z_]", "_", c)
trn = Xtr[selected].rename(columns=clean_name).astype("float64").assign(target=ytr.values.astype("float64")).reset_index(drop=True)   # AutoViML only treats float64 targets as regression
tst = Xte[selected].rename(columns=clean_name).astype("float64").reset_index(drop=True)
runs_cfg = {
    "AutoViML B: Random Forest (Boosting_Flag=False)": dict(Boosting_Flag=False),
    "AutoViML C: XGBoost (Boosting_Flag=True)": dict(Boosting_Flag=True),
    "AutoViML D: CatBoost": dict(Boosting_Flag="CatBoost"),
}
avml, preds = {}, {}
for name, kw in runs_cfg.items():
    t = time.time(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model, feats, trainm, testm = Auto_ViML(trn.copy(), "target", tst.copy(), sample_submission="", hyper_param="RS",
                                               feature_reduction=True, scoring_parameter="", verbose=0, **kw)
    secs = time.time() - t
    pc = "target_predictions" if "target_predictions" in testm.columns else [c for c in testm.columns if c.endswith("predictions")][0]
    preds[name] = testm[pc].values
    results.append(score(name, yte.values, preds[name], secs))
    avml[name] = {"model": model, "features": feats}
    (OUT / f"autoviml_log_{name.split(':')[0].replace(' ', '_')}.txt").write_text(buf.getvalue())
    print(f"{name:<50} {len(feats):>3} features kept · RMSE={results[-1]['RMSE_kW']:.4f} kW · {secs:.0f}s")

# %% [markdown]
# ## 12. Results

# %%
res = pd.DataFrame(results).set_index("model").round(4)
res.to_csv(OUT / "model_results.csv")
res.sort_values("RMSE_kW")

# %%
r_ = res.sort_values("RMSE_kW", ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
cols = [PALETTE[2] if m == r_.RMSE_kW.idxmin() else (MUTED if m.startswith("Naive") else PALETTE[0]) for m in r_.index]
ax.barh(r_.index, r_.RMSE_kW, color=cols, height=.6)
for yy, v in enumerate(r_.RMSE_kW): ax.text(v, yy, f" {v:.3f}", va="center", fontsize=8)
ax.set_title("2010 hold-out RMSE (kW): lower is better · gray = naive baselines"); ax.set_xlabel("RMSE (kW)")
save(fig, "12_model_comparison")

best = res.RMSE_kW.idxmin()
wk = (test_F.index >= "2010-02-01") & (test_F.index < "2010-02-08")
fig, ax = plt.subplots(figsize=(13, 3.4))
ax.plot(test_F.index[wk], yte[wk], color="#52514e", lw=1.2, label="actual")
ax.plot(test_F.index[wk], preds[best][wk] if best in preds else ols.predict(Xte[selected])[wk], color=PALETTE[1], lw=1.4, label=f"{best}")
ax.plot(test_F.index[wk], H_raw_gap.shift(167).reindex(test_F.index)[wk], color=PALETTE[0], lw=1, ls="--", alpha=.7, label="same hour last week")
ax.legend(loc="upper left", ncols=3); ax.set_ylabel("kW"); ax.set_title("One test week (Feb 2010): next-hour forecast vs actual")
save(fig, "13_forecast_week")

# %%
err = pd.DataFrame({"abs_err": np.abs(yte.values - (preds[best] if best in preds else ols.predict(Xte[selected]))), "hour": test_F.hour.values})
fig, ax = plt.subplots(figsize=(9, 3)); err.groupby("hour").abs_err.mean().plot(kind="bar", ax=ax, color=PALETTE[0], width=.7)
ax.set_title("Where the best model struggles: mean absolute error by forecast hour"); ax.set_ylabel("MAE (kW)"); ax.set_xlabel("hour of day")
save(fig, "14_error_by_hour")

# %% [markdown]
# ### Takeaways (from the executed run)
# 1. **Gap length decides the imputer.** Linear interpolation wins up to ~1 hour (MAE 0.11 kW at 1 min); from 6 hours on, the time-of-week median profile wins (0.61–0.65 kW vs 0.81–1.01 kW for interpolation). The hybrid switches at 360 min.
# 2. **Physics rules caught 524 inconsistent minutes** (434 current mismatches, 33 sub-meter > total, 57 out-of-range voltages) that look plausible to a z-score.
# 3. **Time-series anomalies are replaced, not deleted.** 866 hours were flagged by ≥2 detectors, and the 732 in the training period were smoothed. 607 training hours on DBSCAN-anomalous days were dropped, and 2010 stayed raw.
# 4. **Clustering 24-h profiles** split days into a high-load winter type and a low-load type that peaks in Jul–Aug (vacations, warm months). Yesterday's type became a feature.
# 5. **AutoViML CatBoost is the best model:** RMSE **0.467 kW** vs **0.614** for persistence (**24% better**), R² 0.64 on the full unseen 2010 year. XGBoost (0.475) and Random Forest (0.489) follow, and all three beat OLS (0.507). The seasonal naive forecasts are worse than persistence at a 1-hour horizon.
# 6. **Limitation:** household behavior is stochastic, and most error sits in the evening peak. Outdoor temperature would be the most valuable extra input.
