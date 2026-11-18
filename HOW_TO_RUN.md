# How to run everything

Three notebooks, run in this order. Each one downloads its own data, so no manual downloads are needed.

| # | Notebook | Runtime | Approx. time (free Colab CPU) |
|---|---|---|---|
| 1 | `part1_tabular_airbnb/01_airbnb_eda_prep_autoviml.ipynb` | Colab CPU | ~25–35 min (most of it in AutoViML) |
| 2 | `part1_timeseries_power/02_power_timeseries_eda_prep_autoviml.ipynb` | Colab CPU | ~25–35 min |
| 3 | `part2_agentic_eda/03_agentic_eda_gemini_agent.ipynb` | fresh Colab CPU runtime + Gemini API key | ~5–10 min |
| 3b | Colab Data Science Agent (no notebook to open; follow `part2_agentic_eda/PROMPTS_colab_data_science_agent.md`) | blank Colab notebook | ~20 min |

---

## Option A — Google Colab (recommended)

### Notebooks 1 and 2 (AutoViML)
1. Open the notebook in Colab using either method:
   - **From GitHub:** `File → Open notebook → GitHub tab` → paste the repo URL → pick the notebook, **or**
   - **Upload:** `File → Upload notebook` → choose the `.ipynb` from this zip.
2. **Runtime → Run all.**
3. The **first code cell** sees Colab's numpy 2.x, installs the pinned AutoViML stack (~2 min), and then **restarts the runtime on purpose**.
   You'll see "Your session crashed / restarted". **That is expected.**
4. Click **Runtime → Run all** again. The first cell now prints `Environment ready: {...}` and the rest runs top to bottom.
5. Outputs appear inline and are also saved to the Colab file browser:
   - `outputs_airbnb/figures/*.png`, `outputs_airbnb/reports/*.html` (ydata-profiling + Sweetviz), `outputs_airbnb/model_results.csv`
   - `outputs_power/...` for the time-series notebook
6. Download the executed notebook: **File → Download → .ipynb**. Commit it so the outputs are visible on GitHub.

> **Kaggle data:** the loader tries `kagglehub` first. If Colab asks for Kaggle credentials or it fails, the notebook automatically falls back to a
> public GitHub mirror of the *same* Kaggle file, so it never blocks.

### Notebook 3 (Gemini agent)
1. Get a free API key: https://aistudio.google.com/apikey
2. In Colab, open the notebook, click **🔑 Secrets** (left sidebar) → add `GOOGLE_API_KEY` → toggle **Notebook access** on.
3. Use a **fresh runtime** (`Runtime → Disconnect and delete runtime` if you just ran notebook 1 or 2), then **Run all**.
4. After it finishes, fill in the `manual_check` column (Section 5) and the reflection table (Section 6) **in your own words**.
5. Run Tool A (Colab Data Science Agent) using `PROMPTS_colab_data_science_agent.md`. Save its notebook as `03b_colab_ds_agent_airbnb.ipynb`,
   paste its summary into `outputs_agent/colab_agent_output.txt`, and re-run Section 5.1 to score it.

---

## Option B — Locally (Python 3.12)

```bash
cd cmpe255-dataprep-automl
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Open each notebook and **Run All**. The install cell skips itself because numpy 1.26 is already present.
For notebook 3, `export GOOGLE_API_KEY=...` first (or paste the key when prompted).

To execute headless (this is how the committed outputs were produced):
```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3000 \
  part1_tabular_airbnb/01_airbnb_eda_prep_autoviml.ipynb --output 01_airbnb_eda_prep_autoviml.ipynb
```

---

## Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `ImportError: numpy.core.multiarray failed to import` / `numpy.dtype size changed` | The runtime wasn't restarted after the install. **Runtime → Restart session**, then Run all. |
| `ModuleNotFoundError: No module named 'distutils'` | Python 3.12 removed distutils. `pip install "setuptools<81"` (already in the install cell). |
| `No module named 'pkg_resources'` (ydata-profiling) | setuptools ≥ 81 removed it. `pip install "setuptools<81"`. |
| `cannot import name 'display' from 'IPython.core.display'` | IPython 9 removed it. `pip install "ipython<9"` (Colab already ships IPython 7, so local only). |
| AutoViML seems frozen on `Boosting_Flag=None` | Its linear mode runs a LinearSVR grid that does not converge on 38k rows. That is why the notebooks use OLS for the linear family instead. |
| AutoViML prints hundreds of lines | Expected. The notebooks capture its log to `outputs_*/autoviml_log_*.txt` and print a one-line summary per model. |
| Gemini `404 model not found` | The notebook auto-picks the newest Flash model your key can use. To force one, set `MODEL_OVERRIDE = "gemini-2.5-flash"` (or any name from `client.models.list()`). |
| Gemini `429 / quota` | The free tier is rate-limited. Wait a minute and re-run the agent cell, or lower `MAX_STEPS`. |
| AutoViz | Not used: `autoviz` pulls a bokeh/holoviews version that conflicts with AutoViML's pins. ydata-profiling + Sweetviz + pandas-dq cover auto-EDA instead. |
