# Video 3 — Part 2: EDA with AI agents instead of by hand
**Target length:** 10–12 min · Upload to YouTube as *Unlisted* → link in README.md

This video is mostly **live**: you record the Colab Data Science Agent while it works, then walk through the Gemini agent notebook.
Follow `part2_agentic_eda/PROMPTS_colab_data_science_agent.md` while recording Segment 2.

---

## Segment 1 — Framing (0:00 – 1:30)

| Time | Show | Say |
|---|---|---|
| 0:00 | Lecture slide "Data Agent Autonomy Levels L0–L5" | "Part 2 asks: can an AI agent do the EDA I did by hand in Part 1? The lecture frames data agents on an L0 to L5 autonomy ladder, and today's tools sit around L2 to proto-L3." |
| 0:40 | Notebook 03, cell 1 table | "I tested two agents on the same Airbnb data: Google's Colab Data Science Agent, which is a polished product, and a small agent I built with the Gemini API so I could see every step it takes. Then I scored both against the ground truth from Part 1." |

## Segment 2 — User journey: Colab Data Science Agent, live (1:30 – 5:30)

| Time | Show | Say |
|---|---|---|
| 1:30 | Blank Colab → upload CSV → Gemini panel | "As a user, I open a blank notebook, upload the CSV, and open the Gemini panel." |
| 2:00 | Paste Prompt 1 → plan appears | "I give it only the goal. It writes a plan first. [Read 2–3 plan steps aloud.] I approve it. That approval is the human-in-the-loop part." |
| 2:40 | Cells being generated and run (speed up in editing) | "It writes and runs cells by itself. When it hits an error, it reads the error and fixes its own code. [Point at one if it happens.]" |
| 3:40 | Prompt 2 (missingness why) | "Now I test reasoning, not just plotting: *why* is reviews_per_month missing? [Say whether it found the zero-reviews link.]" |
| 4:20 | Prompt 3 (outlier judgment) | "Does it know a $10,000 private room is an error but a $5,000 penthouse might be real? Does it know NYC's 30-day rule? [Say what it did.]" |
| 5:00 | Prompt 5 (self-verification table) | "Last, I make it verify its own numbers. [Say whether any were wrong.]" |

## Segment 3 — Code walkthrough: the Gemini agent I built (5:30 – 9:30)

| Time | Cell | Say |
|---|---|---|
| 5:30 | 7 | "The notebook picks the newest Flash model my API key can access. The key lives in Colab Secrets, never in the code." |
| 6:00 | 9 (scroll run_python + SYSTEM) | "The agent has one tool, run_python. It executes code in a persistent namespace, like notebook cells. The guardrails: a 25-step budget, truncated output, shell and delete commands blocked, and every step logged. The system prompt requires a plan first, a minimum set of checks, and 'every number must come from a tool output'." |
| 7:00 | 11 (outputs: steps scrolling) | "This is the plan-act-observe loop in action: [N] tool calls, [E] errors, all self-corrected. [Show 2 plots it made.] It finished in [T] seconds." |
| 7:50 | 11 (final report) | "Here's its final report. [Highlight one good finding and one miss.]" |
| 8:20 | 13 | "Then a verification turn. As the supervisor, I ask it to recompute every number it reported. This is the 'verifiable evaluation harness' idea from the lecture. [Say whether any claim was corrected.]" |
| 9:00 | 14 | "The full trace (code, outputs, figures) is saved to agent_trace.md, so the analysis can be audited." |

## Segment 4 — Audit & reflection (9:30 – 11:30)

| Time | Cell | Say |
|---|---|---|
| 9:30 | 16 (audit table) | "Now the scoring. These 14 findings are ground truth from my manual EDA. The checker searches each agent's report, and I confirmed every row by hand. The Gemini agent covered [x] of 14 and the Colab agent [y] of 14." |
| 10:20 | 20 (metrics) | "Process metrics: time, tool calls, errors, self-corrections." |
| 10:40 | 21 (reflection table) | "My conclusion: agents are excellent at breadth and speed, minutes instead of hours. But they missed [the *why* behind missingness / the 30-day rule / ...], which needs domain reasoning. That's exactly the gap between L3 and L4 in the lecture: verification and judgment are still the human's job." |
| 11:15 | — | "Thanks for watching. All three notebooks and the agent trace are in the repo." |
