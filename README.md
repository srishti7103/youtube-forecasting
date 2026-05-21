# Closed-Loop YouTube Performance Forecasting Pipeline

This repository implements a scientific, closed-loop machine learning forecasting pipeline designed to help new or unsuccessful creators identify high-potential video topics, optimize their title/duration structure, predict video performance (virality multiplier relative to channel baseline) *before* publishing, and close the loop by tracking post-publish prediction error.

## Pipeline Architecture & Workflow

The project is structured around 6 progressive Jupyter notebooks that map to the data engineering, machine learning, and revalidation loop:

```
youtube-forecasting/
├── config.py                 # Niche channels and API parameters
├── src/
│   ├── collector.py          # Data collection via YouTube Data API v3
│   ├── features.py           # Feature extraction & baseline logic
│   └── model.joblib          # Serialized Random Forest model & extractor
├── notebooks/
│   ├── 01_niche_analysis.ipynb      # Niche definition & seed channel mappings
│   ├── 02_data_collection.ipynb     # Fetching raw historical video statistics
│   ├── 03_eda_and_features.ipynb     # Extracting features & profiling viral outliers
│   ├── 04_model_building.ipynb      # Training models via GroupKFold cross-validation
│   ├── 05_prediction_logging.ipynb  # Pre-publish performance logging
│   └── 06_model_revalidation.ipynb  # Post-publish error tracking & revalidation
├── data/
│   ├── raw/                  # Raw JSON/CSV statistics fetched from API
│   └── processed/            # Cleaned training feature matrices
├── predictions/
│   └── predictions_log.csv   # Historical predictions vs. actual views
└── reports/
    ├── findings.md           # Niche insights & viral outlier profiles
    └── feature_correlations.png
```

---

## Getting Started

### 1. Installation & Environment Setup

Clone the repository and install the dependencies in a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows Powershell
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration & API Key

Copy `.env.example` to `.env` and fill in your YouTube Data API v3 key:

```ini
YOUTUBE_API_KEY=your_google_cloud_api_key_here
```

---

## Step-by-Step Execution Guide

### Phase 1: Niche Profiling & Data Collection
1. **`01_niche_analysis.ipynb`**: Set up the two niches:
   - **Popular Niche**: Personal Finance India (highly saturated)
   - **Underserved Niche**: India Economy & Data Stories (high demand, lower supply)
2. **`02_data_collection.ipynb`**: Connects to the YouTube Data API and collects the last ~50 videos from the 10 target channels defined in `config.py` (totaling ~500+ videos for training).

### Phase 2: Exploratory Data Analysis & Modeling
3. **`03_eda_and_features.ipynb`**: Parses raw data using our modular `YouTubeFeatureExtractor` in `src/features.py`. It extracts text hook counts, emotional trigger tags, duration variables, and upload schedules, while profiling the **top 5% viral outliers** (videos that outperform the channel's median views by >3x).
4. **`04_model_building.ipynb`**: Trains predictive models (Ridge Regression, Random Forests, and HistGradientBoosting) targeting the **Virality Ratio** ($\log_{10}(\text{views}/\text{median\_views} + 0.01)$). Models are evaluated using **GroupKFold** (grouped by `channel_id`) to ensure they generalize to brand-new channels. Saves the final trained predictor to `src/model.joblib`.

### Phase 3: Closed-Loop Prediction & Revalidation
5. **`05_prediction_logging.ipynb`**: Use this **before publishing** a new video. Input your proposed title, video duration, and planned upload hour. The script extracts features, runs the model, estimates your expected views/virality, and appends a record to `predictions/predictions_log.csv` with `is_revalidated = 0`.
6. **`06_model_revalidation.ipynb`**: Use this **7 to 14 days after publishing**. Paste the actual YouTube video ID next to your logged prediction in `predictions/predictions_log.csv` and run this script. It automatically queries the YouTube API for the actual view count, updates the log, and generates a validation accuracy report (MAE, MAPE) along with diagnostic plots.

---

## Key Predictive Features

Our feature extractor (`src/features.py`) calculates variables engineered specifically to capture click-trigger and viewer retention indicators:
* **Virality Ratio (Target)**: $\log_{10}(\text{views} / \text{channel\_median\_views} + 0.01)$, which normalizes performance across large and small channels.
* **Duration Minutes**: Modeled to capture the sweet spot (e.g. 10-15 min) for watch-time optimization.
* **Title Hook Intensity**: Detects questions (`?`), numerical metrics (numbers/rupee symbols), all-caps word triggers, and emotional power words (e.g. *crash*, *scam*, *warning*, *secrets*).
* **Upload Schedule**: Captures day-of-week and peak viewing hours in Indian Standard Time (IST).
