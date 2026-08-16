# ADHD Data Explorer

A Streamlit app with four tabs exploring public ADHD-related data, plus an
AI chat tab that answers questions about whatever data is currently loaded.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501

## Tabs

### 1. Prevalence & Demographics -- works immediately, no download needed
Uses `data/prevalence_national.csv` and `data/treatment_state_range.csv`, compiled
from CDC / National Survey of Children's Health (NSCH) published statistics
(2022-2023, U.S. children ages 3-17):
- https://www.cdc.gov/adhd/data/index.html
- https://nschdata.org/browse/survey/results?q=10167

Optional upgrade: upload a CSV with `state` and `rate` columns at the bottom of
this tab to render a state-by-state choropleth map. Get state-level data from:
- https://www.childhealthdata.org (free interactive query tool)
- https://developers.americashealthrankings.org (free API, signup required)

### 2. Adult Health & Activity -- requires a Kaggle download
Dataset: "ADHD Diagnosis Data", also known as HYPERAKTIV -- clinical info,
wrist activity features, and CPT-II attention-test scores for adults with
ADHD and matched controls.
- https://www.kaggle.com/datasets/arashnic/adhd-diagnosis-data

To get it:
1. Create a free Kaggle account if you don't have one.
2. Open the link above and use the **Data Explorer** to download individual
   files -- you don't need the whole 551MB dataset. Grab just:
   - `patient_info.csv` (required -- has the ADHD/control label + clinical
     scale scores)
   - `features.csv` (optional -- wrist accelerometer summary features)
   - `CPT_II_ConnersContinuousPerformanceTest.csv` (optional -- attention
     test scores)
3. Skip the `activity_data/`, `hrv_data/`, and `hyperaktiv_with_controls/`
   folders -- those hold raw per-second sensor data and account for almost
   all of the 551MB; this tab doesn't use them.
4. Upload the file(s) through the three uploaders on this tab. All three
   files are semicolon-delimited (`;`), which the app handles automatically.

The tab joins whichever files you upload on the shared `ID` column, uses the
real `ADHD` (1/0) column as the group label, and lets you compare clinical
scale scores, wrist-activity summary stats, and CPT attention-test scores
between the ADHD and control groups via box plots.

### 3. EEG Signals -- requires a Kaggle download
Dataset: "EEG Dataset for ADHD" (raw EEG channel data, children with and
without ADHD).
- https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd

Same process as tab 2: download, unzip, upload the CSV. The tab plots a raw
waveform for a chosen channel and, if there's a label/group column, compares
channel distributions between groups.

### 4. Chat with the Data -- requires a free Anthropic API key
A chat interface (built with `st.chat_message` / `st.chat_input`) that answers
questions using only the data currently loaded elsewhere in the app -- the
national prevalence tables (always available), and the Adult Health /
EEG data if you've uploaded it in those tabs. It's told explicitly not to use
outside knowledge or make clinical claims, and to say so when the loaded data
doesn't answer the question.

To enable it:
1. Get a free API key at [console.anthropic.com](https://console.anthropic.com)
   (pay-as-you-go after that -- this app uses the cheap Haiku model, so casual
   use is a few cents at most).
2. **On Streamlit Community Cloud:** open your app's dashboard -> Settings ->
   Secrets, and add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
3. **Running locally:** create a file at `.streamlit/secrets.toml` in the
   project with the same line. Don't commit this file -- add
   `.streamlit/secrets.toml` to a `.gitignore` so the key never ends up in
   GitHub.

Never paste your API key directly into `app.py` or upload it to GitHub --
secrets belong in Streamlit's secrets manager, not in code.

## Notes

- This app is for exploratory / educational purposes only -- it is not a
  diagnostic tool and shouldn't be used to draw clinical conclusions.
- Charts use a colorblind-safe categorical palette (validated per the
  project's dataviz guidelines).
- Tabs 2 and 3 are schema-flexible: they inspect whatever columns are in the
  uploaded CSV rather than hardcoding exact Kaggle column names, since dataset
  versions can change slightly.
