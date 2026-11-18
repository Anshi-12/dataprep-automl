"""Generate docs/CELL_GUIDE.md: a cell-by-cell map of every notebook (used by the video scripts)."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NBS = [
    "part1_tabular_airbnb/01_airbnb_eda_prep_autoviml.ipynb",
    "part1_timeseries_power/02_power_timeseries_eda_prep_autoviml.ipynb",
    "part2_agentic_eda/03_agentic_eda_gemini_agent.ipynb",
]

def summary(cell):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    if not lines: return ""
    if cell["cell_type"] == "markdown":
        return re.sub(r"[#*`|]", "", lines[0]).strip()[:110]
    comment = next((l.lstrip("# ").strip() for l in lines if l.startswith("#") and len(l) > 3), None)
    return (comment or lines[0])[:110].replace("|", "\\|")

out = ["# Cell guide", "", "Cell numbers count every cell (markdown + code) from the top, matching the video scripts.", ""]
for nb_path in NBS:
    nb = json.loads((ROOT / nb_path).read_text())
    out += [f"## `{nb_path}`", "", "| Cell | Type | Section | What it does |", "|---|---|---|---|"]
    section = ""
    for i, c in enumerate(nb["cells"], 1):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        m = re.match(r"\s*#{1,3}\s+(.*)", src)
        if c["cell_type"] == "markdown" and m: section = m.group(1).strip()[:60]
        out.append(f"| {i} | {'📝 md' if c['cell_type'] == 'markdown' else '🐍 code'} | {section} | {summary(c)} |")
    out.append("")
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "CELL_GUIDE.md").write_text("\n".join(out))
print("wrote docs/CELL_GUIDE.md")
