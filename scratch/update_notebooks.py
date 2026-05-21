import json
import os

def create_eda_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "cell_1",
                "metadata": {},
                "source": [
                    "# 03. Exploratory Data Analysis & Feature Extraction\n",
                    "\n",
                    "This notebook loads the raw scaled YouTube dataset of 8 niches, runs the upgraded `YouTubeFeatureExtractor` to calculate rolling baselines, chronological sequences, and format styles, profiles outliers, and generates insights."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_2",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "sns.set_theme(style='whitegrid', palette='muted')\n",
                    "plt.rcParams.update({'figure.figsize': [11, 6], 'font.size': 11})\n",
                    "\n",
                    "sys.path.append('..')\n",
                    "from src.features import YouTubeFeatureExtractor"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_3",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the scaled dataset\n",
                    "raw_path = '../data/raw/scaled_niche/videos.csv'\n",
                    "if not os.path.exists(raw_path):\n",
                    "    raise FileNotFoundError(f\"Dataset not found at {raw_path}. Run harvester first.\")\n",
                    "\n",
                    "df_raw = pd.read_csv(raw_path)\n",
                    "print(f\"Loaded {len(df_raw)} videos from {df_raw['channel_id'].nunique()} channels.\")\n",
                    "print(f\"Columns: {list(df_raw.columns)}\")\n",
                    "print(f\"Niches represented: {df_raw['niche'].value_counts() if 'niche' in df_raw.columns else 'None'}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_4",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run Feature Extraction\n",
                    "extractor = YouTubeFeatureExtractor()\n",
                    "extractor.fit(df_raw)\n",
                    "df_feat = extractor.transform(df_raw)\n",
                    "\n",
                    "print(\"✅ Features extracted successfully!\")\n",
                    "print(f\"Features shape: {df_feat.shape}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_5",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save processed features\n",
                    "out_dir = '../data/processed/scaled_niche'\n",
                    "os.makedirs(out_dir, exist_ok=True)\n",
                    "features_path = os.path.join(out_dir, 'features.csv')\n",
                    "df_feat.to_csv(features_path, index=False)\n",
                    "print(f\"✅ Processed features saved to: {features_path}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_6",
                "metadata": {},
                "source": [
                    "## 1. Niche Analysis: Median Views & Subscribers"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_7",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Group by niche and show median views and subscribers\n",
                    "niche_stats = df_raw.groupby('niche').agg(\n",
                    "    video_count=('video_id', 'count'),\n",
                    "    median_views=('view_count', 'median'),\n",
                    "    median_likes=('like_count', 'median')\n",
                    ").reset_index()\n",
                    "\n",
                    "print(\"=== Niche Performance Summary ===\")\n",
                    "print(niche_stats.to_string(index=False))\n",
                    "\n",
                    "# Plot Median Views per Niche\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "sns.barplot(data=niche_stats.sort_values('median_views', ascending=False), x='median_views', y='niche', hue='niche', legend=False)\n",
                    "plt.title('Median Views per Video across 8 Niches', fontsize=14, fontweight='bold')\n",
                    "plt.xlabel('Median Views')\n",
                    "plt.ylabel('Niche')\n",
                    "plt.tight_layout()\n",
                    "os.makedirs('../reports', exist_ok=True)\n",
                    "plt.savefig('../reports/niche_median_views.png', dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_8",
                "metadata": {},
                "source": [
                    "## 2. Long-Form vs Shorts Duration Profile"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_9",
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 5))\n",
                    "df_feat['duration_minutes'].plot(kind='hist', bins=50, color='#4C72B0', edgecolor='white')\n",
                    "plt.axvline(1.0, color='red', linestyle='--', label='Shorts Boundary (60s)')\n",
                    "plt.title('Video Duration Distribution (Minutes)', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Duration (minutes)')\n",
                    "plt.ylabel('Video Count')\n",
                    "plt.legend()\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../reports/duration_distribution.png', dpi=150)\n",
                    "plt.show()\n",
                    "\n",
                    "shorts_pct = (df_feat['is_shorts'].mean() * 100)\n",
                    "print(f\"Shorts (<=60s) representation: {shorts_pct:.2f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_10",
                "metadata": {},
                "source": [
                    "## 3. Engagement Rate by Channel Category (Successful vs Small)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_11",
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.boxplot(data=df_feat, x='channel_category', y='engagement_rate', showfliers=False, hue='channel_category', legend=False)\n",
                    "plt.title('Engagement Rate (Likes+Comments / Views) by Channel Tier', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Channel Tier')\n",
                    "plt.ylabel('Engagement Rate')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../reports/engagement_rate_by_tier.png', dpi=150)\n",
                    "plt.show()\n",
                    "\n",
                    "print(df_feat.groupby('channel_category')['engagement_rate'].median())"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_12",
                "metadata": {},
                "source": [
                    "## 4. Feature Correlations with Log Views"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_13",
                "metadata": {},
                "outputs": [],
                "source": [
                    "corr_features = [\n",
                    "    'duration_minutes', 'title_char_count', 'title_word_count',\n",
                    "    'title_has_number', 'title_has_question', 'title_emotional_trigger_count',\n",
                    "    'title_is_hindi_or_hinglish', 'tags_count', 'hashtag_count',\n",
                    "    'log_channel_subscribers', 'log_rolling_channel_baseline',\n",
                    "    'log_video_sequence_number', 'log_channel_age_days_at_upload',\n",
                    "    'is_highly_edited', 'log_views'\n",
                    "]\n",
                    "corr_features = [f for f in corr_features if f in df_feat.columns]\n",
                    "correlations = df_feat[corr_features].corr()['log_views'].drop('log_views').sort_values()\n",
                    "\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "colors = ['#DD8452' if x > 0 else '#4C72B0' for x in correlations.values]\n",
                    "sns.barplot(x=correlations.values, y=correlations.index, palette=colors, hue=correlations.index, legend=False)\n",
                    "plt.axvline(0, color='black', linewidth=0.8)\n",
                    "plt.title('Feature Correlations with Log Views Target', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Pearson Correlation')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../reports/feature_correlations.png', dpi=150)\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open('notebooks/03_eda_and_features.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Rewrote notebooks/03_eda_and_features.ipynb successfully.")

def create_model_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "m1",
                "metadata": {},
                "source": [
                    "# 04. Model Building — Hybrid PyTorch + LightGBM Ensemble\n",
                    "\n",
                    "This notebook trains a hybrid PyTorch MLP + LightGBM Regressor ensemble model to predict $\\log_{10}(\\text{views} + 1)$ across 8 niches, applying decay weights for video age, and saves the final model bundle to `src/model.joblib`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c1",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import joblib\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.optim as optim\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                    "import lightgbm as lgb\n",
                    "\n",
                    "sys.path.append('..')\n",
                    "from src.features import YouTubeFeatureExtractor\n",
                    "from src.recommender import PyTorchMLP, YouTubeRecommender"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c2",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load processed feature matrix\n",
                    "features_path = '../data/processed/scaled_niche/features.csv'\n",
                    "if not os.path.exists(features_path):\n",
                    "    raise FileNotFoundError(f\"Processed features not found at {features_path}. Run 03_eda_and_features.ipynb first.\")\n",
                    "\n",
                    "df_feat = pd.read_csv(features_path)\n",
                    "print(f\"Loaded feature matrix of shape: {df_feat.shape}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c3",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Define the complete pre-upload feature list\n",
                    "FEATURES_LIST = [\n",
                    "    'duration_minutes',\n",
                    "    'is_optimal_duration',\n",
                    "    'is_shorts',\n",
                    "    'title_char_count',\n",
                    "    'title_word_count',\n",
                    "    'title_has_number',\n",
                    "    'title_has_question',\n",
                    "    'title_has_exclamation',\n",
                    "    'title_has_colon',\n",
                    "    'title_has_money_ref',\n",
                    "    'title_has_all_caps',\n",
                    "    'title_emotional_trigger_count',\n",
                    "    'title_is_hindi_or_hinglish',\n",
                    "    'title_mentions_india',\n",
                    "    'title_sentiment_polarity',\n",
                    "    'title_sentiment_subjectivity',\n",
                    "    'tags_count',\n",
                    "    'hashtag_count',\n",
                    "    'has_hashtags',\n",
                    "    'description_char_count',\n",
                    "    'publish_hour_ist',\n",
                    "    'publish_day_of_week',\n",
                    "    'is_weekend',\n",
                    "    'is_prime_time',\n",
                    "    'log_channel_subscribers',\n",
                    "    'log_rolling_channel_baseline',\n",
                    "    'log_video_sequence_number',\n",
                    "    'log_channel_age_days_at_upload',\n",
                    "    'days_since_published',\n",
                    "    'log_days_since_published',\n",
                    "    'is_highly_edited',\n",
                    "    'niche_personal_finance',\n",
                    "    'niche_geopolitics_economy',\n",
                    "    'niche_tech_gadgets',\n",
                    "    'niche_gaming',\n",
                    "    'niche_comedy_entertainment',\n",
                    "    'niche_education_science',\n",
                    "    'niche_makeup_beauty',\n",
                    "    'niche_vlogging_lifestyle',\n",
                    "    'voice_type_human_own',\n",
                    "    'voice_type_ai_generated',\n",
                    "    'voice_type_music_only',\n",
                    "    'presentation_style_face_presenter',\n",
                    "    'presentation_style_faceless_broll',\n",
                    "    'presentation_style_animated_visuals'\n",
                    "]\n",
                    "\n",
                    "# Ensure all features exist in DataFrame\n",
                    "FEATURES_LIST = [f for f in FEATURES_LIST if f in df_feat.columns]\n",
                    "print(f\"Using {len(FEATURES_LIST)} features for model training:\")\n",
                    "print(FEATURES_LIST)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c4",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Chronological split to evaluate forecasting generalization\n",
                    "# Sort descending by days_since_published (oldest first, newest last)\n",
                    "df_sorted = df_feat.sort_values('days_since_published', ascending=False).reset_index(drop=True)\n",
                    "\n",
                    "X = df_sorted[FEATURES_LIST]\n",
                    "y = df_sorted['log_views']\n",
                    "days = df_sorted['days_since_published']\n",
                    "\n",
                    "split_idx = int(len(X) * 0.8)\n",
                    "X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]\n",
                    "y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]\n",
                    "days_train, days_val = days.iloc[:split_idx], days.iloc[split_idx:]\n",
                    "\n",
                    "print(f\"Training set  : {len(X_train)} samples\")\n",
                    "print(f\"Validation set: {len(X_val)} samples\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c5",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compute linear decay sample weights\n",
                    "# Videos <= 1 year old get 1.0; older videos decay linearly to 0.3 up to 3 years\n",
                    "w_train = np.ones(len(X_train))\n",
                    "mask_tr = days_train > 365\n",
                    "w_train[mask_tr] = np.maximum(0.3, 1.0 - (days_train[mask_tr] - 365) / 730.0 * 0.7)\n",
                    "\n",
                    "w_val = np.ones(len(X_val))\n",
                    "mask_val = days_val > 365\n",
                    "w_val[mask_val] = np.maximum(0.3, 1.0 - (days_val[mask_val] - 365) / 730.0 * 0.7)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c6",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Train LightGBM model\n",
                    "print(\"Training LightGBM Regressor...\")\n",
                    "lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)\n",
                    "lgb_val = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=lgb_train)\n",
                    "\n",
                    "params = {\n",
                    "    'objective': 'regression',\n",
                    "    'metric': 'rmse',\n",
                    "    'learning_rate': 0.05,\n",
                    "    'num_leaves': 31,\n",
                    "    'max_depth': 6,\n",
                    "    'feature_fraction': 0.8,\n",
                    "    'verbose': -1,\n",
                    "    'seed': 42\n",
                    "}\n",
                    "\n",
                    "lgb_model = lgb.train(\n",
                    "    params,\n",
                    "    lgb_train,\n",
                    "    num_boost_round=300,\n",
                    "    valid_sets=[lgb_train, lgb_val],\n",
                    "    callbacks=[lgb.log_evaluation(period=50)]\n",
                    ")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c7",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Train PyTorch MLP model\n",
                    "print(\"Training PyTorch MLP...\")\n",
                    "X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)\n",
                    "y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)\n",
                    "w_train_tensor = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)\n",
                    "\n",
                    "X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)\n",
                    "y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)\n",
                    "w_val_tensor = torch.tensor(w_val, dtype=torch.float32).unsqueeze(1)\n",
                    "\n",
                    "mlp = PyTorchMLP(X_train.shape[1])\n",
                    "optimizer = optim.Adam(mlp.parameters(), lr=0.001)\n",
                    "\n",
                    "epochs = 100\n",
                    "batch_size = 128\n",
                    "\n",
                    "for epoch in range(1, epochs + 1):\n",
                    "    mlp.train()\n",
                    "    permutation = torch.randperm(X_train_tensor.size(0))\n",
                    "    epoch_loss = 0\n",
                    "    \n",
                    "    for i in range(0, X_train_tensor.size(0), batch_size):\n",
                    "        indices = permutation[i:i+batch_size]\n",
                    "        batch_x = X_train_tensor[indices]\n",
                    "        batch_y = y_train_tensor[indices]\n",
                    "        batch_w = w_train_tensor[indices]\n",
                    "        \n",
                    "        optimizer.zero_grad()\n",
                    "        outputs = mlp(batch_x)\n",
                    "        loss = ( (outputs - batch_y) ** 2 * batch_w ).mean()\n",
                    "        loss.backward()\n",
                    "        optimizer.step()\n",
                    "        epoch_loss += loss.item() * len(indices)\n",
                    "        \n",
                    "    epoch_loss /= X_train_tensor.size(0)\n",
                    "    if epoch % 20 == 0 or epoch == 1:\n",
                    "        mlp.eval()\n",
                    "        with torch.no_grad():\n",
                    "            val_out = mlp(X_val_tensor)\n",
                    "            val_loss = ( (val_out - y_val_tensor) ** 2 * w_val_tensor ).mean().item()\n",
                    "        print(f\"Epoch {epoch:3d}/{epochs} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c8",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Evaluate Ensemble on Validation Set\n",
                    "lgb_pred = lgb_model.predict(X_val)\n",
                    "mlp.eval()\n",
                    "with torch.no_grad():\n",
                    "    mlp_pred = mlp(X_val_tensor).numpy().flatten()\n",
                    "\n",
                    "ens_pred = 0.5 * lgb_pred + 0.5 * mlp_pred\n",
                    "\n",
                    "ens_rmse = np.sqrt(mean_squared_error(y_val, ens_pred))\n",
                    "ens_mae = mean_absolute_error(y_val, ens_pred)\n",
                    "ens_r2 = r2_score(y_val, ens_pred)\n",
                    "\n",
                    "print(f\"Validation Ensemble RMSE: {ens_rmse:.4f}\")\n",
                    "print(f\"Validation Ensemble MAE : {ens_mae:.4f} (log10 views)\")\n",
                    "print(f\"Validation Ensemble R2  : {ens_r2:.4f}\")\n",
                    "\n",
                    "# Plot predictions vs actuals\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "plt.scatter(y_val, ens_pred, alpha=0.3, color='#4C72B0')\n",
                    "plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)\n",
                    "plt.title('Actual vs Predicted Log Views (Ensemble Validation)', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Actual Log Views')\n",
                    "plt.ylabel('Predicted Log Views')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../reports/ensemble_actual_vs_predicted.png', dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c9",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Fit the global extractor on ALL data and save the model bundle\n",
                    "raw_path = '../data/raw/scaled_niche/videos.csv'\n",
                    "df_raw = pd.read_csv(raw_path)\n",
                    "extractor_global = YouTubeFeatureExtractor()\n",
                    "extractor_global.fit(df_raw)\n",
                    "\n",
                    "bundle = {\n",
                    "    'lightgbm_model': lgb_model,\n",
                    "    'pytorch_model_state': mlp.state_dict(),\n",
                    "    'features_list': FEATURES_LIST,\n",
                    "    'extractor': extractor_global,\n",
                    "    'rmse': float(ens_rmse),\n",
                    "    'niche_medians': {n: float(df_raw[df_raw['niche'] == n]['view_count'].median())\n",
                    "                      for n in df_raw['niche'].unique()}\n",
                    "}\n",
                    "\n",
                    "bundle_path = '../src/model.joblib'\n",
                    "joblib.dump(bundle, bundle_path)\n",
                    "print(f\"✅ Model bundle successfully saved to: {bundle_path}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c10",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Demo prediction using YouTubeRecommender\n",
                    "recommender = YouTubeRecommender.load(bundle_path)\n",
                    "res = recommender.predict(\n",
                    "    title=\"How I Built a Profitable Faceless YouTube Channel in 30 Days\",\n",
                    "    duration_minutes=14.0,\n",
                    "    publish_hour_ist=19,\n",
                    "    publish_day_of_week=3,\n",
                    "    niche='personal_finance',\n",
                    "    subscribers=25000,\n",
                    "    voice_type='human_own',\n",
                    "    presentation_style='faceless_broll',\n",
                    "    is_highly_edited=1\n",
                    ")\n",
                    "\n",
                    "recommender.print_report(res)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('notebooks/04_model_building.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Rewrote notebooks/04_model_building.ipynb successfully.")

if __name__ == '__main__':
    # Adjust paths relative to script location
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(base)) # change directory to workspace root
    create_eda_notebook()
    create_model_notebook()
