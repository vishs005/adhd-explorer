# ADHD Data Explorer

A Streamlit app with three tabs exploring public ADHD-related data.

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
Dataset: "ADHD Diagnosis Data" (heart rate / activity data from adults with and
without ADHD).
- https://www.kaggle.com/datasets/arashnic/adhd-diagnosis-data

To get it:
1. Create a free Kaggle account if you don't have one.
2. Open the link above and click "Download" (no API key needed for a manual
   download through the browser).
3. Unzip and upload the CSV through the app's file uploader on this tab.

The tab auto-detects numeric columns, lets you pick a metric + a group column
(e.g. ADHD vs. control) and shows box plots / distributions, plus a trend line
if it finds a time or date column.

### 3. EEG Signals -- requires a Kaggle download
Dataset: "EEG Dataset for ADHD" (raw EEG channel data, children with and
without ADHD).
- https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd

Same process as tab 2: download, unzip, upload the CSV. The tab plots a raw
waveform for a chosen channel and, if there's a label/group column, compares
channel distributions between groups.

## Notes

- This app is for exploratory / educational purposes only -- it is not a
  diagnostic tool and shouldn't be used to draw clinical conclusions.
- Charts use a colorblind-safe categorical palette (validated per the
  project's dataviz guidelines).
- Tabs 2 and 3 are schema-flexible: they inspect whatever columns are in the
  uploaded CSV rather than hardcoding exact Kaggle column names, since dataset
  versions can change slightly.
