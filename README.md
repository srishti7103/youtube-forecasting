# Closed-Loop YouTube Performance Forecasting Pipeline & Creator Playbook

This repository implements a production-grade, closed-loop machine learning forecasting pipeline and Creator Playbook CLI. Trained on **26,860 real records** harvested across **8 distinct YouTube niches**, the system predicts video performance trajectories ($\log_{10}(\text{views} + 1)$) at 7, 30, and 90-day horizons *before* publishing, and tracks post-publish prediction accuracy.

---

## 🏗️ Repository Architecture

The project is structured around 5 sequential Jupyter notebooks, core modular source modules, and a CLI utility that map to the data engineering, machine learning ensemble, and revalidation loop:

```
youtube-forecasting/
├── config.py                 # Niche settings, subscriber thresholds, and search keywords
├── predict_video.py          # Interactive Creator Playbook CLI utility
├── requirements.txt          # Python package dependencies
├── src/
│   ├── __init__.py           # Package initialization
│   ├── collector.py          # Data harvesting via YouTube Data API v3 (Mock mode fallback)
│   ├── niche_analyzer.py     # Niche scoring across 5 dimensions & feature profile comparisons
│   ├── scaled_harvest.py     # High-scale crawler balancing successful/small channels across 8 niches
│   ├── features.py           # Pre-publish feature engineering (45 engineered attributes)
│   ├── recommender.py        # Ensemble prediction engine, confidence intervals, & playbook generator
│   └── model.joblib          # Serialized ensemble model bundle (PyTorch state + LightGBM + scalar)
├── notebooks/
│   ├── 01_niche_analysis.ipynb      # Pilot 2-niche definition, profiling, and comparative analysis
│   ├── 03_eda_and_features.ipynb     # Feature extraction, profiles, and baseline calculations at scale
│   ├── 04_model_building.ipynb      # Ensemble training with chronological split & decay weights
│   ├── 05_prediction_logging.ipynb  # Pre-publish forecasting and database logging
│   └── 06_model_revalidation.ipynb  # Post-publish API revalidation & calibration checks
├── data/
│   ├── raw/
│   │   ├── popular_niche/
│   │   │   └── videos.csv    # Pilot popular niche harvested videos
│   │   ├── underserved_niche/
│   │   │   └── videos.csv    # Pilot underserved niche harvested videos
│   │   └── scaled_niche/
│   │       ├── videos.csv    # Balanced scale harvested dataset (26,860 videos across 8 niches)
│   │       └── harvest_summary.csv  # Harvesting crawl summary report
│   └── processed/
│       ├── niche_comparison.csv # Pilot 2-niche statistical comparison
│       ├── features_popular.csv # Pilot popular niche engineered features
│       ├── features_underserved.csv # Pilot underserved niche engineered features
│       └── scaled_niche/
│           └── features.csv  # Core engineered feature matrix for training
├── predictions/
│   └── predictions_log.csv   # Pre-publish forecast log database (tracks live tests)
└── reports/                  # Diagnostic plots, heatmaps, and analytical summaries
    ├── findings.md           # Niche insights & viral outlier profiles
    ├── ensemble_actual_vs_predicted.png  # Model accuracy regression scatter plot
    ├── niche_view_distribution.png       # Long-tail view frequency histogram
    └── revalidation_dashboard.png        # Calibration & accuracy tracking dashboard
```

---

## ⚡ Getting Started

### 1. Installation & Environment Setup

Clone the repository and install the dependencies inside a Python virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows Powershell
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration & API Credentials

Create a `.env` file in the root directory and add your Google Cloud YouTube API Key:

```ini
YOUTUBE_API_KEY=your_google_cloud_api_key_here
```

---

## 🔄 Step-by-Step Pipeline Guide

### Phase 1: Niche Profiling & Dataset Harvesting
1. **`01_niche_analysis.ipynb`**: Perform pilot niche profiling (comparing saturated *Personal Finance India* vs. underserved *India Economy & Data Stories*) using `src/niche_analyzer.py` to identify demand, opportunity, and competition baselines.
2. **`src/scaled_harvest.py`**: Crawls the YouTube API to discover balanced channels (successful vs. small/stagnant) and harvests up to 500 uploads per channel across all 8 niches, saving the results to `data/raw/scaled_niche/videos.csv`.
3. **`03_eda_and_features.ipynb`**: Runs exploratory data analysis on the scale-harvested dataset, engineers 45 features using `src/features.py` (text hooks, schedules, baseline history), and stores the feature matrix in `data/processed/scaled_niche/features.csv`.

### Phase 2: Modeling & Ensemble Training
4. **`04_model_building.ipynb`**: Trains a **Hybrid PyTorch MLP + LightGBM Regressor Ensemble** targeting $\log_{10}(\text{views} + 1)$.
   * **Temporal Splitting**: Sorted by upload date; training on the oldest 80% and validating on the newest 20% to prevent lookahead leakage.
   * **Sample Weighting**: Applies a time-decay loss weight (newest videos weight = 1.0, decaying linearly to 0.3 for videos older than 1 year).
   * **Performance**: Achieves a **Validation $R^2 \approx 0.6744$** and a **Validation MAE $\approx 0.495$** in log space (predictions are on average within $\approx 3.12\text{x}$ of actual views across a 6-order-of-magnitude range).

### Phase 3: Pre-Publish Logging & API Revalidation
5. **`05_prediction_logging.ipynb`**: Logs predictions in `predictions/predictions_log.csv` before publishing.
6. **`06_model_revalidation.ipynb`**: Fetches actual views for logged videos using the YouTube API after 7–14 days to audit and calibrate prediction confidence bands.

---

## 🛠️ Running the Creator Playbook CLI

To predict views and generate a **4-part optimization playbook** for a video concept, execute:

```powershell
python predict_video.py
```

### The Playbook Includes:
* **[1] Views & Velocity Horizons:** Expected views at 7, 30, and 90 days with 80% confidence bands and Virality Ratios.
* **[2] Title Analysis & Reframing:** Length check, sentiment metrics, emotional hook analyzer, and three suggested click-inducing reframes.
* **[3] Format & Style Insights:** Optimization guides tailored to the selected niche style.
* **[4] Discoverability SEO Checklist:** Metadata tags and description mapping instructions.

