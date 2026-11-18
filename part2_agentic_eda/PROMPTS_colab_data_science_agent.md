# Tool A — Colab Data Science Agent: exact steps & prompts

The Colab Data Science Agent is Gemini built into Google Colab. It turns a plain-English goal into a plan, then writes, runs, and debugs the notebook cells itself.
It's free with a personal Google account.

> ⏱ About 20–25 minutes, including screen recording for Video 3.

## 1. Prepare
1. Download `AB_NYC_2019.csv` from Kaggle: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
   (or reuse `data/AB_NYC_2019.csv` from the Part 1 run).
2. Open a **new blank notebook**: https://colab.research.google.com → *New notebook*.
3. Rename it `03b_colab_ds_agent_airbnb.ipynb`.
4. **Start screen recording now.** This run is the "user journey" segment of Video 3.

## 2. Hand the data to the agent
1. Left sidebar → 📁 **Files** → upload `AB_NYC_2019.csv`.
2. Open the **Gemini** panel (✦ button, bottom of the notebook or top-right toolbar).
3. Make sure the CSV is attached to the chat. Use the file's ⋮ menu → *Add to Gemini*, or the **+** in the prompt box.

## 3. Prompt sequence (paste one at a time, and let each finish)

**Prompt 1: the goal (autonomous EDA)**
```
I want to predict the nightly price of NYC Airbnb listings using AB_NYC_2019.csv.
Do a thorough exploratory data analysis to prepare this data for modeling:
schema and dtypes, missing values and WHY they are missing, duplicates,
invalid or impossible values, outliers in every numeric column, distributions,
and how price relates to borough, room type, location (latitude/longitude),
and the listing name text. Make clear plots. Report exact counts for every
data-quality problem you find.
```
→ The agent shows a **plan**. Read it out loud on the recording, then click **Accept & run** (or *Execute plan*).
If it asks for permission to run each step, click **Accept** each time. That approval click *is* the human-in-the-loop part of an L2/L3 agent, so mention it.

**Prompt 2: push on reasoning (tests the "why")**
```
For reviews_per_month and last_review: are the missing values random?
Check whether they co-occur with number_of_reviews == 0 and tell me the
correct imputation strategy, with evidence.
```

**Prompt 3: outlier judgment (noise vs. legitimate)**
```
Examine listings with price == 0, price > 1000, and minimum_nights > 365.
For each group, decide whether it is a data error or a legitimate listing,
and justify it. Also explain the spike at minimum_nights = 30.
```

**Prompt 4: cleaning + features**
```
Based on your analysis, clean the dataset and engineer features for a price
model (target encoding or frequency encoding for neighbourhood, distance to
Manhattan, text features from name, log-transformed price). Show the final
feature table.
```

**Prompt 5: self-verification (reliability check)**
```
Verify your own work: recompute every count you reported earlier from the raw
CSV and show a table of reported vs recomputed values. Correct anything
that was wrong.
```

## 4. Save the evidence
1. **File → Download → .ipynb**, and save it into this folder as `03b_colab_ds_agent_airbnb.ipynb`.
2. Copy the agent's final summary text (and any printed counts) into `outputs_agent/colab_agent_output.txt`.
   Section 5.1 of `03_agentic_eda_gemini_agent.ipynb` scores it against the Part 1 ground truth.
3. Take 3 screenshots for the README: the plan, one generated plot, and the verification table → `outputs_agent/screenshots/`.

## 5. What to watch for (note these down for the reflection)
- Did the plan cover data quality *before* jumping to charts?
- Did it explain **why** `reviews_per_month` is missing (0 reviews), or just fill it with the mean?
- Did it call the $10,000 listings errors or keep them? Did it know about NYC's 30-day rule?
- How many code errors did it hit, and did it fix them on its own?
- Any number in its summary that is **wrong** when you check it? (Ground truth: 10,052 missing reviews_per_month, 11 zero prices, max minimum_nights 1,250, 221 neighbourhoods, 16 missing names, 21 missing host names, no duplicate rows.)
