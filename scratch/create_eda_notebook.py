import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# 03. Exploratory Data Analysis & Feature Extraction\n\n"
        "This notebook loads the raw YouTube metrics collected in Step 1, extracts engineered features using our modular `YouTubeFeatureExtractor` class, profiles the **viral outliers** in each niche, and saves the final processed features for model training."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import os\n"
        "import sys\n\n"
        "# Set up plotting styles\n"
        "sns.set_theme(style='whitegrid')\n"
        "plt.rcParams['figure.figsize'] = [10, 6]\n"
        "plt.rcParams['font.size'] = 11\n\n"
        "# Import feature extractor from src\n"
        "sys.path.append('..')\n"
        "from src.features import YouTubeFeatureExtractor"
    ),
    nbf.v4.new_code_cell(
        "# Load the raw datasets collected from the YouTube API\n"
        "try:\n"
        "    popular_raw = pd.read_csv('../data/raw/popular_niche/videos.csv')\n"
        "    underserved_raw = pd.read_csv('../data/raw/underserved_niche/videos.csv')\n"
        "    print(f'Popular niche raw data: {len(popular_raw)} videos')\n"
        "    print(f'Underserved niche raw data: {len(underserved_raw)} videos')\n"
        "except FileNotFoundError as e:\n"
        "    print(f'Error loading raw files: {e}. Did you run 01_niche_analysis.ipynb first?')\n"
    ),
    nbf.v4.new_code_cell(
        "# Extract features\n"
        "extractor_popular = YouTubeFeatureExtractor()\n"
        "extractor_popular.fit(popular_raw)\n"
        "popular_features = extractor_popular.transform(popular_raw)\n\n"
        "extractor_underserved = YouTubeFeatureExtractor()\n"
        "extractor_underserved.fit(underserved_raw)\n"
        "underserved_features = extractor_underserved.transform(underserved_raw)\n\n"
        "print('Features extracted successfully!')\n"
        "print('Popular features columns:', list(popular_features.columns))"
    ),
    nbf.v4.new_code_cell(
        "# Save processed feature matrices\n"
        "os.makedirs('../data/processed', exist_ok=True)\n"
        "popular_features.to_csv('../data/processed/features_popular.csv', index=False)\n"
        "underserved_features.to_csv('../data/processed/features_underserved.csv', index=False)\n"
        "print('Features saved to data/processed/!')"
    ),
    nbf.v4.new_markdown_cell(
        "## Profiling Viral Outliers\n\n"
        "A **viral outlier** is defined as a video that gets more than **3x the channel's median views**. "
        "These are videos that broke out of the channel's baseline subscriber bubble. Let's inspect the top performing viral outliers in each niche."
    ),
    nbf.v4.new_code_cell(
        "# Isolate and print viral outliers for Popular Niche\n"
        "pop_viral = popular_raw[popular_features['is_viral_outlier'] == 1].copy()\n"
        "views_col = 'view_count' if 'view_count' in pop_viral.columns else 'views'\n"
        "pop_viral['outperform_ratio'] = pop_viral[views_col] / popular_features['channel_median_views']\n"
        "pop_viral = pop_viral.sort_values(by='outperform_ratio', ascending=False)\n\n"
        "print(f'=== Personal Finance India: {len(pop_viral)} Viral Outliers Found (out of {len(popular_raw)}) ===\\n')\n"
        "for idx, row in pop_viral.head(10).iterrows():\n"
        "    print(f\"- [{row['channel_title']}] {row['title']}\")\n"
        "    print(f\"  Views: {int(row[views_col]):,} | {row['outperform_ratio']:.1f}x channel median views\\n\")"
    ),
    nbf.v4.new_code_cell(
        "# Isolate and print viral outliers for Underserved Niche\n"
        "und_viral = underserved_raw[underserved_features['is_viral_outlier'] == 1].copy()\n"
        "views_col = 'view_count' if 'view_count' in und_viral.columns else 'views'\n"
        "und_viral['outperform_ratio'] = und_viral[views_col] / underserved_features['channel_median_views']\n"
        "und_viral = und_viral.sort_values(by='outperform_ratio', ascending=False)\n\n"
        "print(f'=== India Economy & Data Stories: {len(und_viral)} Viral Outliers Found (out of {len(underserved_raw)}) ===\\n')\n"
        "for idx, row in und_viral.head(10).iterrows():\n"
        "    print(f\"- [{row['channel_title']}] {row['title']}\")\n"
        "    print(f\"  Views: {int(row[views_col]):,} | {row['outperform_ratio']:.1f}x channel median views\\n\")"
    ),
    nbf.v4.new_markdown_cell(
        "## Feature Comparisons: Viral vs. Regular Videos\n\n"
        "Let's compare the average feature values of viral outliers vs. regular videos to identify the performance drivers."
    ),
    nbf.v4.new_code_cell(
        "features_to_check = [\n"
        "    'duration_minutes', \n"
        "    'title_has_number', \n"
        "    'title_has_question', \n"
        "    'title_has_all_caps', \n"
        "    'title_emotional_trigger_count', \n"
        "    'tags_count'\n"
        "]\n\n"
        "print('=== Personal Finance India: Feature Means by Virality ===')\n"
        "pop_comparison = popular_features.groupby('is_viral_outlier')[features_to_check].mean().T\n"
        "pop_comparison.columns = ['Regular Videos', 'Viral Outliers']\n"
        "print(pop_comparison.round(3))\n\n"
        "print('\\n=== India Economy & Data Stories: Feature Means by Virality ===')\n"
        "und_comparison = underserved_features.groupby('is_viral_outlier')[features_to_check].mean().T\n"
        "und_comparison.columns = ['Regular Videos', 'Viral Outliers']\n"
        "print(und_comparison.round(3))"
    ),
    nbf.v4.new_markdown_cell(
        "## Correlation Matrix Heatmaps\n\n"
        "Let's visualize the correlation between our engineered features and the **Virality Ratio** (`virality_ratio_median` = $\\log_{10}(\\text{views}/\\text{channel median} + 0.01)$)."
    ),
    nbf.v4.new_code_cell(
        "plt.figure(figsize=(15, 6))\n\n"
        "plt.subplot(1, 2, 1)\n"
        "pop_corr = popular_features[['virality_ratio_median'] + features_to_check].corr()\n"
        "sns.heatmap(pop_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', cbar=False)\n"
        "plt.title('Personal Finance India: Feature Correlations with Virality')\n\n"
        "plt.subplot(1, 2, 2)\n"
        "und_corr = underserved_features[['virality_ratio_median'] + features_to_check].corr()\n"
        "sns.heatmap(und_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')\n"
        "plt.title('India Economy & Data Stories: Feature Correlations with Virality')\n\n"
        "plt.tight_layout()\n"
        "os.makedirs('reports', exist_ok=True)\n"
        "plt.savefig('reports/feature_correlations.png', dpi=150)\n"
        "plt.show()"
    )
]

nb['cells'] = cells

with open('notebooks/03_eda_and_features.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created notebooks/03_eda_and_features.ipynb successfully!")
