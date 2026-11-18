# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 2 — Exploratory Data Analysis with AI agents instead of by hand
# **CMPE 255 Data Mining · Fall 2026 · Anshita Shah**
#
# **Goal:** hand the *same* NYC Airbnb dataset from Part 1 to AI data agents, let them do the EDA, then **audit** their work against my manual EDA.
#
# **Agentic tools used**
# | # | Tool | How it works | Autonomy level (lecture's L0–L5 ladder) |
# |---|---|---|---|
# | A | **Colab Data Science Agent** (Gemini in Colab) | Upload the CSV, describe the goal in the Gemini side panel. It writes a plan, generates cells, runs them, and fixes its own errors. | **L2 → Proto-L3**: the human sets the goal and approves the plan; the agent executes multi-step workflows |
# | B | **Gemini code-execution agent (this notebook)** | A transparent agent loop I built with the Gemini API. The model plans, calls a `run_python` tool, reads the output, and decides the next step until it writes a report. Every step is logged. | **Proto-L3**: the human sets the goal and guardrails (step budget, sandboxed namespace), and the agent orchestrates the analysis itself |
#
# Tool A runs entirely in the Colab UI. Its prompts and recording steps are in `PROMPTS_colab_data_science_agent.md`, and its generated notebook is saved as
# `03b_colab_ds_agent_airbnb.ipynb`. **Tool B is this notebook.** Section 5 scores **both** agents against the ground truth established manually in Part 1.
#
# **Why build Tool B as well?** Agents in a UI are a black box. Building the loop exposes *how* an agent works: plan → act (code) → observe (output) → reflect.
# It also lets me measure the reliability challenges the lecture lists: **hallucination, cascading errors, and verification**.

# %% [markdown]
# ## 0. Setup
# Run this notebook in a **fresh** Colab runtime (it does not need the AutoViML pins).
#
# **API key (free):** get one at https://aistudio.google.com/apikey, then add it in Colab under **🔑 Secrets** (left sidebar) as `GOOGLE_API_KEY` and enable notebook access.

# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("google.genai") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai"], check=True)

import os, io, re, json, time, contextlib, traceback, textwrap, shutil, urllib.request
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Markdown, Image

OUT = Path("outputs_agent"); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load the same dataset as Part 1

# %%
DATA = Path("data"); DATA.mkdir(exist_ok=True)
CSV = DATA / "AB_NYC_2019.csv"
MIRROR = "https://raw.githubusercontent.com/rajtulluri/Airbnb-Data-Exploratory-Analysis/master/AB_NYC_2019.csv"
if not CSV.exists():
    try:
        import kagglehub
        p = kagglehub.dataset_download("dgomonov/new-york-city-airbnb-open-data")
        shutil.copy(next(Path(p).rglob("AB_NYC_2019.csv")), CSV)
    except Exception as e:
        print(f"kagglehub unavailable ({type(e).__name__}) → GitHub mirror"); urllib.request.urlretrieve(MIRROR, CSV)
df = pd.read_csv(CSV)
print(df.shape)

# %% [markdown]
# ## 2. Connect to Gemini
# The model is picked automatically: the newest stable **Flash** model your key can access (fast, free tier, strong at tool use). Set `MODEL_OVERRIDE` to force a specific one.

# %%
from google import genai
from google.genai import types

def get_key():
    try:
        from google.colab import userdata
        k = userdata.get("GOOGLE_API_KEY")
        if k: return k
    except Exception:
        pass
    if os.environ.get("GOOGLE_API_KEY"): return os.environ["GOOGLE_API_KEY"]
    from getpass import getpass
    return getpass("Paste your Gemini API key: ")

client = genai.Client(api_key=get_key())

MODEL_OVERRIDE = ""   # e.g. "gemini-2.5-flash"
def pick_model():
    if MODEL_OVERRIDE: return MODEL_OVERRIDE
    names = []
    for m in client.models.list():
        nm = m.name.split("/")[-1]; acts = getattr(m, "supported_actions", None) or []
        if "generateContent" in acts and "flash" in nm and not any(x in nm for x in ["lite", "tts", "image", "live", "audio", "exp", "preview", "8b", "thinking"]):
            names.append(nm)
    def ver(n):
        v = re.findall(r"gemini-(\d+(?:\.\d+)?)", n); return float(v[0]) if v else 0
    names = sorted(set(names), key=lambda n: (ver(n), -len(n)))
    if not names: raise RuntimeError("No Flash model available. Set MODEL_OVERRIDE.")
    return names[-1]
MODEL = pick_model()
print("Using model:", MODEL)

# %% [markdown]
# ## 3. The agent: one tool, a plan–act–observe loop, and guardrails
# **Design choices (and why):**
# - **One tool, `run_python`.** The agent can write and run any pandas/matplotlib code. This is the most general "action space", the same idea as Colab's agent and OpenAI's Code Interpreter.
# - **A persistent namespace.** Variables survive between steps (`df` is preloaded), just like notebook cells.
# - **Guardrails:** a step budget (`MAX_STEPS`), output truncation (so a huge `print(df)` can't flood the context), the raw `df` protected by passing a copy, no file deletion, and every call logged to a trace.
# - **Observability:** every step's code, output, and figures are saved to `outputs_agent/agent_trace.md`. That trace is the evidence for my evaluation.

# %%
MAX_STEPS = 25
NS = {"pd": pd, "np": np, "plt": plt, "sns": sns, "df": df.copy()}
TRACE = []

def run_python(code: str) -> str:
    """Execute Python code in a persistent Jupyter-like namespace and return what it printed.

    The namespace already contains: df (pandas DataFrame of NYC Airbnb 2019 listings), pd, np, plt, sns.
    Use print() to see results; the last expression is NOT echoed. Matplotlib figures are saved and shown to the human automatically.
    Keep each call focused (one analysis step). Output longer than 4,000 characters is truncated.

    Args:
        code: Python source code to execute.
    Returns:
        Captured stdout, or the error traceback if the code failed.
    """
    step = len(TRACE) + 1
    if step > MAX_STEPS:
        return "STEP BUDGET EXHAUSTED. Stop calling tools and write your final report now."
    if re.search(r"\b(os\.remove|shutil\.rmtree|rmdir|unlink|subprocess|os\.system)\b", code):
        return "BLOCKED: file deletion and shell commands are not allowed in this sandbox."
    buf = io.StringIO(); t = time.time(); ok = True
    plt.close("all")
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, f"<agent-step-{step}>", "exec"), NS)
    except Exception:
        ok = False; buf.write("ERROR:\n" + "".join(traceback.format_exc().splitlines(keepends=True)[-6:]))
    figs = []
    for i, num in enumerate(plt.get_fignums()):
        f = FIG / f"step{step:02d}_{i}.png"; plt.figure(num).savefig(f, bbox_inches="tight", dpi=110); figs.append(str(f))
    plt.close("all")
    out = buf.getvalue()
    if len(out) > 4000: out = out[:4000] + f"\n…[truncated {len(out)-4000} chars]"
    TRACE.append({"step": step, "code": code, "output": out, "ok": ok, "figures": figs, "sec": round(time.time() - t, 2)})
    display(Markdown(f"**🤖 Agent step {step}** ({'ok' if ok else 'error'}, {time.time()-t:.1f}s)"))
    display(Markdown(f"```python\n{code}\n```")); print(out[:1500] + ("…" if len(out) > 1500 else ""))
    for f in figs: display(Image(f, width=650))
    return out if out.strip() else "(no output: use print() to see results)"

SYSTEM = """You are an autonomous senior data analyst performing exploratory data analysis (EDA).
You have ONE tool, run_python, which executes code in a persistent namespace where `df` is already loaded.

Work like a careful human analyst:
1. Start with a short numbered PLAN (as text) before your first tool call.
2. Then execute it step by step. After each observation, briefly state what you learned and what you will do next.
3. Investigate at least: schema/dtypes, missing values AND *why* they are missing (check co-occurrence with other columns),
   duplicates, invalid/impossible values, outliers in every numeric column, distribution shape/skew, the target `price`
   versus categorical, numeric, spatial (latitude/longitude) and text (name) features, and data-quality issues a model would trip on.
4. Make at least 4 informative plots (matplotlib/seaborn). Give each a title.
5. Every number you put in the final report MUST come from a tool output. Never guess.
6. You have a budget of {max_steps} tool calls. Be efficient: combine related checks into one call.

Finish with a FINAL REPORT in markdown with exactly these sections:
## Dataset overview
## Data quality issues (each with the exact count and the evidence)
## Key insights about price
## Recommended cleaning & feature-engineering steps (ordered, with justification)
## Limitations / what I could not determine
"""

# %% [markdown]
# ## 4. Run the agent
# ### 4.1 The autonomous EDA run (goal only, no step-by-step instructions)

# %%
GOAL = ("Perform a thorough exploratory data analysis of this NYC Airbnb 2019 dataset to prepare it for a model that predicts nightly price. "
        "Find every data-quality problem a careful analyst would fix before modeling.")

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM.format(max_steps=MAX_STEPS),
        tools=[run_python],
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=MAX_STEPS + 2),
    ),
)
t0 = time.time()
resp = chat.send_message(GOAL)
report = resp.text or ""
elapsed = time.time() - t0
display(Markdown("---\n# 🤖 Agent's final report\n" + report))
print(f"\n⏱ {elapsed:.0f}s · {len(TRACE)} tool calls · {sum(not s['ok'] for s in TRACE)} errors (self-corrected)")

# %% [markdown]
# ### 4.2 Verification turn (human-in-the-loop)
# The lecture's reliability challenge is **hallucinated or inconsistent insights**. As the human supervisor, I ask the agent to *re-verify* every count in its report with fresh code.
# This mirrors the "verifiable evaluation harness" idea on the "How to prepare for L5 agents" slide.

# %%
n_before = len(TRACE)
verify = chat.send_message(
    "Verification pass: for EVERY number in your 'Data quality issues' section, write one run_python call that recomputes all of them "
    "from the raw df and prints a table: issue | reported | recomputed | match?. Then list any claim you got wrong and correct it.")
display(Markdown("# 🔎 Verification result\n" + (verify.text or "")))
print(f"verification used {len(TRACE) - n_before} extra tool calls")

# %%
# Save the full, auditable trace
lines = [f"# Agent trace — {MODEL} — {datetime.now():%Y-%m-%d %H:%M}\n", f"**Goal:** {GOAL}\n"]
for s in TRACE:
    lines += [f"\n## Step {s['step']} ({'ok' if s['ok'] else 'ERROR'}, {s['sec']}s)\n", "```python", s["code"], "```", "```text", s["output"][:3000], "```"]
    lines += [f"![]({Path(f).relative_to(OUT)})" for f in s["figures"]]
lines += ["\n# Final report\n", report, "\n# Verification\n", verify.text or ""]
(OUT / "agent_trace.md").write_text("\n".join(lines))
(OUT / "agent_report.md").write_text(report)
print("saved:", OUT / "agent_trace.md", "|", OUT / "agent_report.md")

# %% [markdown]
# ## 5. Audit: agent vs. my manual EDA (ground truth from Part 1)
# Each item below was established *by hand* in Part 1, with exact counts. The checker searches the agent's report for evidence of each one,
# and **I then confirm every row manually** by reading the report (the `manual_check` column). A keyword match only suggests a finding; it does not prove the agent found it.

# %%
raw = df
GROUND_TRUTH = [
    ("Missing reviews_per_month / last_review (10,052 rows)", rf"10,?052|20\.6", "Is missingness quantified?"),
    ("Missingness is structural: all missing rows have 0 reviews", r"(number_of_reviews|reviews?)\D{0,40}(==?|equal|are|is|have)\s*0|never (been )?reviewed|no reviews", "Did it explain WHY values are missing (MNAR)?"),
    ("price == 0 is invalid (11 rows)", r"\b11\b[^.\n]{0,60}(price|zero|\$0|free)|(price|zero|\$0)[^.\n]{0,60}\b11\b", "Invalid target values"),
    ("Extreme prices up to $10,000 / long right tail", r"10,?000|right[- ]skew|long tail|log[- ]?transform", "Outliers in target"),
    ("minimum_nights > 365 (up to 1,250)", r"1,?250|minimum_nights[^.\n]{0,80}(365|outlier|extreme)", "Domain-invalid values"),
    ("30-night spike is legitimate (NYC 30-day rule)", r"\b30\b[^.\n]{0,60}(night|day)", "Context-aware outlier reasoning"),
    ("availability_365 == 0 for ~36% (inactive vs booked)", r"availability(_365)?[^.\n]{0,80}(\b0\b|zero|3[56](\.\d)?\s?%)", "Ambiguous zeros"),
    ("Missing name (16) / host_name (21)", r"(name|host)[^.\n]{0,60}\b(16|21)\b|\b(16|21)\b[^.\n]{0,60}(name|host)", "Text column gaps"),
    ("room_type & borough drive price (Manhattan entire home highest)", r"manhattan[^.\n]{0,120}(entire|highest|expensive)|(entire|room_type)[^.\n]{0,120}manhattan", "Main drivers of price"),
    ("Spatial / geographic price pattern", r"latitude|longitude|geograph|spatial|map", "Uses the spatial columns"),
    ("Text in listing name carries signal (luxury, shared…)", r"luxury|penthouse|keyword|name length|tf-?idf|text feature", "Uses the text column"),
    ("High-cardinality neighbourhood (221) needs encoding strategy", r"221|high[- ]cardinality|target encod|frequency encod", "Encoding strategy"),
    ("Recommends log(price) as target", r"log\(?1?p?\(?\s*price|log[- ]transform(ed)? (the )?(target|price)|log of price", "Target transform"),
    ("No duplicate rows/ids", r"no duplicate|0 duplicate|duplicates?\W{0,5}0", "Duplicate check"),
]

def audit_report(text, name):
    rows = []
    for item, pattern, why in GROUND_TRUTH:
        hit = re.search(pattern, text, flags=re.I)
        rows.append({"ground-truth finding (Part 1)": item, "what it tests": why, f"{name}: auto-match": "✅" if hit else "—",
                     f"{name}: evidence": (text[max(0, hit.start() - 60): hit.end() + 60].replace("\n", " ") if hit else ""),
                     f"{name}: manual_check": "TODO ✅/❌/partial"})
    return pd.DataFrame(rows)

full_text_B = report + "\n" + (verify.text or "")   # score what the agent REPORTED, not raw tool output
audit_B = audit_report(full_text_B, "Gemini agent (B)")
display(audit_B)
print(f"Auto-matched coverage, Tool B: {(audit_B.iloc[:, 2] == '✅').mean():.0%} of {len(GROUND_TRUTH)} ground-truth findings")

# %% [markdown]
# ### 5.1 Score the Colab Data Science Agent (Tool A) the same way
# After running Tool A (see `PROMPTS_colab_data_science_agent.md`), copy its **final summary text** plus any printed outputs into
# `outputs_agent/colab_agent_output.txt`, then run this cell.

# %%
a_path = OUT / "colab_agent_output.txt"
if a_path.exists():
    audit_A = audit_report(a_path.read_text(), "Colab DS Agent (A)")
    both = audit_B.iloc[:, :3].merge(audit_A.iloc[:, [0, 2]], on="ground-truth finding (Part 1)")
    display(both)
    print(f"Coverage: A = {(audit_A.iloc[:, 2] == '✅').mean():.0%} · B = {(audit_B.iloc[:, 2] == '✅').mean():.0%}")
    audit_A.to_csv(OUT / "audit_colab_agent.csv", index=False)
else:
    print("Paste Tool A's output into", a_path, "and re-run this cell.")
audit_B.to_csv(OUT / "audit_gemini_agent.csv", index=False)

# %% [markdown]
# ### 5.2 Process metrics
# These measure *how* the agent worked, not just what it concluded: speed, self-correction, and hallucination after verification.

# %%
metrics = {
    "wall-clock time (s)": round(elapsed, 0),
    "tool calls (EDA run)": n_before,
    "tool calls (verification)": len(TRACE) - n_before,
    "code errors hit": sum(not s["ok"] for s in TRACE),
    "errors self-corrected (error followed by a successful step)": sum(1 for a, b in zip(TRACE, TRACE[1:]) if not a["ok"] and b["ok"]),
    "plots produced": sum(len(s["figures"]) for s in TRACE),
    "claims corrected during verification": "fill in manually after reading 4.2",
}
pd.Series(metrics, name="value").to_frame()

# %% [markdown]
# ## 6. Reflection: agentic EDA vs. manual EDA
# *Fill this in after running both agents. Keep it honest and specific; this section is what makes the assignment yours.*
#
# | Dimension | Manual (Part 1) | Colab DS Agent (A) | Gemini code agent (B) |
# |---|---|---|---|
# | Time to first useful insight | ~hours | | |
# | Coverage of the 14 ground-truth findings | 14 / 14 (by definition) | | |
# | Found something I missed? | — | | |
# | Wrong / hallucinated claims | — | | |
# | Explained *why* data was missing (MNAR) | yes | | |
# | Domain reasoning (e.g. NYC 30-day rule) | yes | | |
# | Reproducible / auditable | notebook | generated notebook | full trace in `agent_trace.md` |
# | Autonomy level (L0–L5) | L0 | L2 → Proto-L3 | Proto-L3 |
#
# **Where agents shine:** fast breadth, boilerplate profiling, plots, and self-correcting code errors.
#
# **Where the human is still needed:** *why*-questions (the missingness mechanism), domain rules, judging whether an outlier is noise or a real penthouse,
# and verifying numbers. These are exactly the L3 → L4 gaps the lecture describes (verification and governance).
#
# **My conclusion:** *(2–4 sentences in your own words)*
