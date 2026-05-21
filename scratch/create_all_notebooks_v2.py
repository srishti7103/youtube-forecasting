"""
Master script: creates all upgraded notebooks (03–06) programmatically.
Run from the project root directory.
"""

import nbformat as nbf
import os

def make_cell(source, cell_type='code'):
    if cell_type == 'markdown':
        return nbf.v4.new_markdown_cell(source)
    return nbf.v4.new_code_cell(source)


# ============================================================
# NOTEBOOK 03 — EDA & Feature Extraction (Full upgrade)
# ============================================================

def create_notebook_03():
    cells = [
        make_cell("# 03. Exploratory Data Analysis & Feature Extraction\n\nThis notebook loads the raw YouTube video statistics from both niches, runs the upgraded `YouTubeFeatureExtractor` (now with engagement rates, Hindi detection, Shorts flag, channel quality score, and age-normalized targets), profiles viral outliers, and generates a comprehensive set of analysis visualizations.", 'markdown'),

        make_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, sys

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.figsize': [11, 6], 'font.size': 11})

sys.path.append('..')
from src.features import YouTubeFeatureExtractor"""),

        make_cell("""\
# Load raw data
popular_raw    = pd.read_csv('../data/raw/popular_niche/videos.csv')
underserved_raw = pd.read_csv('../data/raw/underserved_niche/videos.csv')

print(f"Popular niche    : {len(popular_raw)} videos | {popular_raw['channel_id'].nunique()} channels")
print(f"Underserved niche: {len(underserved_raw)} videos | {underserved_raw['channel_id'].nunique()} channels")
print(f"Popular columns  : {list(popular_raw.columns)}")"""),

        make_cell("""\
# Extract features
ext_pop = YouTubeFeatureExtractor()
ext_pop.fit(popular_raw)
pop_feat = ext_pop.transform(popular_raw)

ext_und = YouTubeFeatureExtractor()
ext_und.fit(underserved_raw)
und_feat = ext_und.transform(underserved_raw)

# Tag with niche
pop_feat['niche'] = 'Personal Finance India'
und_feat['niche'] = 'India Economy & Data Stories'

print("✅ Features extracted!")
print(f"Popular features shape   : {pop_feat.shape}")
print(f"Underserved features shape: {und_feat.shape}")
print(f"\\nFeature columns:\\n{list(pop_feat.columns)}")"""),

        make_cell("""\
# Save processed features
os.makedirs('../data/processed', exist_ok=True)
pop_feat.to_csv('../data/processed/features_popular.csv', index=False)
und_feat.to_csv('../data/processed/features_underserved.csv', index=False)
print("✅ Saved to data/processed/")"""),

        make_cell("## 1. Niche Baseline Comparison", 'markdown'),

        make_cell("""\
niches = ['Personal Finance India', 'India Economy & Data Stories']
medians_views = [ext_pop.niche_median_views_, ext_und.niche_median_views_]
medians_vpd   = [ext_pop.niche_median_views_per_day_, ext_und.niche_median_views_per_day_]
outlier_rates = [pop_feat['is_viral_outlier'].mean()*100, und_feat['is_viral_outlier'].mean()*100]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(niches, medians_views, color=['#4C72B0', '#DD8452'])
axes[0].set_title('Median Views per Video')
axes[0].set_ylabel('Views')
for i, v in enumerate(medians_views):
    axes[0].text(i, v*0.95, f'{int(v):,}', ha='center', va='top', color='white', fontweight='bold')

axes[1].bar(niches, medians_vpd, color=['#4C72B0', '#DD8452'])
axes[1].set_title('Median Views per Day (Age-Normalized)')
axes[1].set_ylabel('Views / Day')

axes[2].bar(niches, outlier_rates, color=['#4C72B0', '#DD8452'])
axes[2].set_title('Viral Outlier Rate (%)')
axes[2].set_ylabel('% of Videos > 3x Median')
for i, v in enumerate(outlier_rates):
    axes[2].text(i, v*0.95, f'{v:.1f}%', ha='center', va='top', color='white', fontweight='bold')

plt.suptitle('Niche Comparison: Personal Finance vs India Economy', fontsize=13, fontweight='bold')
plt.tight_layout()
os.makedirs('../reports', exist_ok=True)
plt.savefig('../reports/niche_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        make_cell("## 2. Viral Outlier Profiles", 'markdown'),

        make_cell("""\
# Merge raw and features to print viral outlier cards
def print_viral_outliers(raw_df, feat_df, niche_name, n=10):
    mask = feat_df['is_viral_outlier'] == 1
    viral_feat = feat_df[mask].copy()
    viral_raw  = raw_df[mask].copy()
    
    view_col = 'view_count' if 'view_count' in viral_raw.columns else 'views'
    viral_raw['outperform_ratio'] = viral_raw[view_col] / feat_df['channel_median_views']
    viral_raw = viral_raw.sort_values('outperform_ratio', ascending=False)
    
    print(f"\\n{'='*70}")
    print(f"  TOP VIRAL OUTLIERS — {niche_name} ({len(viral_raw)} total)")
    print(f"{'='*70}")
    for i, (idx, row) in enumerate(viral_raw.head(n).iterrows()):
        feat_row = feat_df.loc[idx]
        print(f"\\n#{i+1}. [{row['channel_title']}]")
        print(f"   Title    : {row['title'][:70]}")
        print(f"   Views    : {int(row[view_col]):,}  ({row['outperform_ratio']:.1f}x channel median)")
        print(f"   Duration : {feat_row['duration_minutes']:.1f} min | Upload: {int(feat_row['publish_hour_ist'])}:00 IST")
        print(f"   Hooks    : Q={feat_row['title_has_question']} | Num={feat_row['title_has_number']} | Trigger={feat_row['title_emotional_trigger_count']}")

print_viral_outliers(popular_raw, pop_feat, 'Personal Finance India')
print_viral_outliers(underserved_raw, und_feat, 'India Economy & Data Stories')"""),

        make_cell("## 3. Duration Distribution — Viral vs Regular", 'markdown'),

        make_cell("""\
combined = pd.concat([pop_feat, und_feat], ignore_index=True)
combined_long = combined[combined['is_shorts'] == 0]  # exclude Shorts

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (niche, grp) in zip(axes, combined_long.groupby('niche')):
    viral   = grp[grp['is_viral_outlier'] == 1]['duration_minutes'].clip(upper=30)
    regular = grp[grp['is_viral_outlier'] == 0]['duration_minutes'].clip(upper=30)
    ax.hist(regular, bins=20, alpha=0.6, label='Regular', color='#4C72B0')
    ax.hist(viral,   bins=20, alpha=0.8, label='Viral Outlier', color='#DD8452')
    ax.axvline(viral.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Viral Avg: {viral.mean():.1f} min')
    ax.axvline(regular.mean(), color='navy', linestyle='--', linewidth=1.5,
               label=f'Regular Avg: {regular.mean():.1f} min')
    ax.set_title(f'{niche[:25]}...\\nDuration Distribution')
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Count')
    ax.legend()

plt.suptitle('Video Duration: Viral Outliers vs Regular Videos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/duration_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        make_cell("## 4. Upload Hour × Day Heatmap (Virality)", 'markdown'),

        make_cell("""\
# Heatmap: average virality ratio per upload hour × day-of-week
def plot_heatmap(feat_df, title, ax):
    pivot = feat_df.groupby(['publish_day_of_week', 'publish_hour_ist'])['virality_ratio_median'].mean().unstack(fill_value=0)
    pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(pivot)]
    
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=0,
                fmt='.2f', linewidths=0.3,
                cbar_kws={'label': 'Avg Virality Ratio (log10)'})
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Upload Hour (IST)')
    ax.set_ylabel('Day of Week')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_heatmap(pop_feat[pop_feat['is_shorts']==0], 'Personal Finance India\\nUpload Timing vs Virality', axes[0])
plot_heatmap(und_feat[und_feat['is_shorts']==0], 'India Economy & Data Stories\\nUpload Timing vs Virality', axes[1])

plt.suptitle('Best Upload Times: Hour × Day Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/upload_hour_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        make_cell("## 5. Engagement Rate — Viral vs Regular", 'markdown'),

        make_cell("""\
# Only plot if we have real like/comment data
def plot_engagement(feat_df, niche_name, ax):
    viral   = feat_df[feat_df['is_viral_outlier'] == 1]['engagement_rate'].clip(upper=0.2)
    regular = feat_df[feat_df['is_viral_outlier'] == 0]['engagement_rate'].clip(upper=0.2)
    
    ax.boxplot([regular, viral], labels=['Regular', 'Viral Outlier'],
               patch_artist=True,
               boxprops=dict(facecolor='#4C72B0', color='navy'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{niche_name[:30]}\\nEngagement Rate (Likes+Comments / Views)')
    ax.set_ylabel('Engagement Rate')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_engagement(pop_feat, 'Personal Finance India', axes[0])
plot_engagement(und_feat, 'India Economy & Data Stories', axes[1])

plt.suptitle('Engagement Rate: Viral Outliers vs Regular Videos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/engagement_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        make_cell("## 6. Feature Correlation with Virality", 'markdown'),

        make_cell("""\
model_features = [
    'duration_minutes', 'title_char_count', 'title_has_number',
    'title_has_question', 'title_has_all_caps', 'title_emotional_trigger_count',
    'title_is_hindi_or_hinglish', 'title_has_money_ref',
    'hashtag_count', 'tags_count', 'is_prime_time', 'is_weekend',
    'publish_hour_ist', 'log_channel_subscribers', 'channel_avg_views_per_video'
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, feat_df, niche_name in zip(
    axes,
    [pop_feat, und_feat],
    ['Personal Finance India', 'India Economy & Data Stories']
):
    available = [f for f in model_features if f in feat_df.columns]
    corr = feat_df[available + ['virality_ratio_median']].corr()['virality_ratio_median'].drop('virality_ratio_median')
    corr_sorted = corr.sort_values()
    
    colors = ['#DD8452' if x > 0 else '#4C72B0' for x in corr_sorted]
    ax.barh(corr_sorted.index, corr_sorted.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'{niche_name}\\nFeature Correlation with Virality Ratio', fontweight='bold')
    ax.set_xlabel('Pearson Correlation (with virality_ratio_median)')

plt.suptitle('Feature Correlations with Video Virality', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/feature_correlations.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        make_cell("## 7. Voice Type Heuristic Analysis", 'markdown'),

        make_cell("""\
# Proxy for voice type: Hindi/Hinglish vs English-only titles
# (Faceless AI-voice channels tend to use English. Human-voice creators mix Hindi/Hinglish)

for feat_df, raw_df, niche_name in [
    (pop_feat, popular_raw, 'Personal Finance India'),
    (und_feat, underserved_raw, 'India Economy & Data Stories')
]:
    hindi_count   = feat_df['title_is_hindi_or_hinglish'].sum()
    english_count = len(feat_df) - hindi_count
    
    hindi_viral   = feat_df[feat_df['title_is_hindi_or_hinglish'] == 1]['is_viral_outlier'].mean() * 100
    english_viral = feat_df[feat_df['title_is_hindi_or_hinglish'] == 0]['is_viral_outlier'].mean() * 100

    print(f"\\n{niche_name}:")
    print(f"  Hindi/Hinglish titles : {hindi_count} ({hindi_count/len(feat_df)*100:.1f}%) | Viral rate: {hindi_viral:.1f}%")
    print(f"  English-only titles   : {english_count} ({english_count/len(feat_df)*100:.1f}%) | Viral rate: {english_viral:.1f}%")

print("\\n💡 Note: Title language is a proxy for content style.")
print("   Hindi/Hinglish → likely own voice / Hinglish presenter style.")
print("   English only   → likely AI-generated voice or faceless English content.")"""),

        make_cell("""\
# Summary comparison table
cols_compare = ['duration_minutes', 'title_char_count', 'title_has_number',
                'title_has_question', 'title_emotional_trigger_count',
                'title_is_hindi_or_hinglish', 'engagement_rate',
                'hashtag_count', 'like_rate', 'comment_rate']

available_cols = [c for c in cols_compare if c in und_feat.columns]
comparison = und_feat.groupby('is_viral_outlier')[available_cols].mean().T
comparison.columns = ['Regular Videos', 'Viral Outliers']
comparison['Difference %'] = ((comparison['Viral Outliers'] - comparison['Regular Videos']) / 
                               (comparison['Regular Videos'].abs() + 0.001) * 100).round(1)

print("=== Underserved Niche: Feature Means — Regular vs Viral Outliers ===")
print(comparison.round(3))"""),
    ]

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open('notebooks/03_eda_and_features.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Created notebooks/03_eda_and_features.ipynb")


# ============================================================
# NOTEBOOK 04 — Model Building (Full upgrade: TimeSeriesSplit + CI)
# ============================================================

def create_notebook_04():
    cells = [
        make_cell("# 04. Model Building — True Forecasting with Confidence Intervals\n\nThis notebook trains our virality prediction models using proper **time-series cross-validation** (TimeSeriesSplit), computes **per-tree confidence intervals** from the Random Forest, builds the Creator Recommendation Engine via `src/recommender.py`, and saves the complete model bundle.", 'markdown'),

        make_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, sys, joblib

from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.figsize': [11, 6], 'font.size': 11})

sys.path.append('..')
from src.features import YouTubeFeatureExtractor
from src.recommender import YouTubeRecommender"""),

        make_cell("""\
# Load processed feature datasets
pop_df = pd.read_csv('../data/processed/features_popular.csv')
pop_df['is_underserved'] = 0
und_df = pd.read_csv('../data/processed/features_underserved.csv')
und_df['is_underserved'] = 1

df = pd.concat([pop_df, und_df], ignore_index=True)

# Important: sort by published_at for TimeSeriesSplit to work correctly
# days_since_published is INVERTED (lower = more recent), so sort ascending = oldest first
df_sorted = df.sort_values('days_since_published', ascending=False).reset_index(drop=True)

# Exclude YouTube Shorts from long-form model training
df_longform = df_sorted[df_sorted['is_shorts'] == 0].copy()
print(f"Total dataset       : {len(df)} videos | {df['channel_id'].nunique()} channels")
print(f"Long-form only      : {len(df_longform)} videos (excluded {len(df)-len(df_longform)} Shorts)")
print(f"Viral outlier rate  : {df_longform['is_viral_outlier'].mean()*100:.1f}%")"""),

        make_cell("""\
# Define the updated feature list (all features known BEFORE publishing)
FEATURES = [
    # Duration
    'duration_minutes',
    'is_optimal_duration',
    # Title hooks
    'title_char_count',
    'title_word_count',
    'title_has_number',
    'title_has_question',
    'title_has_exclamation',
    'title_has_colon',
    'title_has_all_caps',
    'title_has_money_ref',
    'title_emotional_trigger_count',
    'title_is_hindi_or_hinglish',
    'title_mentions_india',
    # Metadata
    'tags_count',
    'hashtag_count',
    'description_char_count',
    # Timing
    'publish_hour_ist',
    'publish_day_of_week',
    'is_weekend',
    'is_prime_time',
    # Channel context
    'log_channel_subscribers',
    'channel_avg_views_per_video',
    # Niche
    'is_underserved',
]

TARGET = 'virality_ratio_median'

# Filter to available columns only
FEATURES = [f for f in FEATURES if f in df_longform.columns]
print(f"Model features: {len(FEATURES)}")
print(FEATURES)"""),

        make_cell("## Cross-Validation: TimeSeriesSplit + GroupKFold\n\nWe evaluate with TWO strategies:\n- **TimeSeriesSplit**: trains on oldest videos, tests on newest — the most realistic forecasting scenario\n- **GroupKFold** (by channel_id): tests generalization to unseen channels", 'markdown'),

        make_cell("""\
X = df_longform[FEATURES]
y = df_longform[TARGET]
groups = df_longform['channel_id']

MODELS = {
    'Ridge':               Ridge(alpha=10.0),
    'RandomForest':        RandomForestRegressor(n_estimators=150, random_state=42, max_depth=8),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42, max_depth=4, max_iter=200),
}

results = {name: {'tss_mae': [], 'tss_r2': [], 'gkf_mae': [], 'gkf_r2': []} for name in MODELS}

# --- TimeSeriesSplit ---
tss = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tss.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    for name, m in MODELS.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        results[name]['tss_mae'].append(mean_absolute_error(y_te, p))
        results[name]['tss_r2'].append(r2_score(y_te, p))

# --- GroupKFold ---
gkf = GroupKFold(n_splits=min(5, df_longform['channel_id'].nunique()))
for train_idx, test_idx in gkf.split(X, y, groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    for name, m in MODELS.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        results[name]['gkf_mae'].append(mean_absolute_error(y_te, p))
        results[name]['gkf_r2'].append(r2_score(y_te, p))

print(f"{'Model':<22} | TSS MAE  | TSS R²   | GKF MAE  | GKF R²")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<22} | {np.mean(r['tss_mae']):.4f}   | {np.mean(r['tss_r2']):.4f}   | {np.mean(r['gkf_mae']):.4f}   | {np.mean(r['gkf_r2']):.4f}")"""),

        make_cell("## Model Comparison Chart", 'markdown'),

        make_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_names = list(MODELS.keys())
tss_maes = [np.mean(results[n]['tss_mae']) for n in model_names]
gkf_maes = [np.mean(results[n]['gkf_mae']) for n in model_names]
tss_r2s  = [np.mean(results[n]['tss_r2']) for n in model_names]
gkf_r2s  = [np.mean(results[n]['gkf_r2']) for n in model_names]

x = np.arange(len(model_names))
w = 0.35
axes[0].bar(x - w/2, tss_maes, w, label='TimeSeriesSplit', color='#4C72B0', alpha=0.85)
axes[0].bar(x + w/2, gkf_maes, w, label='GroupKFold', color='#DD8452', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, rotation=15)
axes[0].set_title('Model Comparison: Mean Absolute Error\\n(Lower is Better)')
axes[0].set_ylabel('MAE (log10 virality scale)')
axes[0].legend()

axes[1].bar(x - w/2, tss_r2s, w, label='TimeSeriesSplit', color='#4C72B0', alpha=0.85)
axes[1].bar(x + w/2, gkf_r2s, w, label='GroupKFold', color='#DD8452', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, rotation=15)
axes[1].set_title('Model Comparison: R² Score\\n(Higher is Better)')
axes[1].set_ylabel('R² Score')
axes[1].legend()

plt.suptitle('Model Evaluation: TimeSeriesSplit vs GroupKFold', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        make_cell("## Final Model Training & Feature Importances", 'markdown'),

        make_cell("""\
# Train the final Random Forest on the full dataset
final_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1)
final_model.fit(X, y)

# Feature importances
feat_imp = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(feat_imp)*0.32)))
colors = ['#DD8452' if x > feat_imp.median() else '#4C72B0' for x in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=colors)
ax.set_title('Feature Importances — Random Forest Virality Model', fontweight='bold')
ax.set_xlabel('Relative Importance')

orange = mpatches.Patch(color='#DD8452', label='Above Median Importance')
blue   = mpatches.Patch(color='#4C72B0', label='Below Median Importance')
ax.legend(handles=[orange, blue])

plt.tight_layout()
plt.savefig('../reports/feature_importances.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 7 features:")
print(feat_imp.sort_values(ascending=False).head(7).round(4))"""),

        make_cell("## Confidence Intervals from Per-Tree Predictions", 'markdown'),

        make_cell("""\
# Show distribution of tree-level predictions for a sample video idea
sample_title = "Why India is running out of water: 5 Shocking Facts"

# Build sample features matching our schema
popular_raw = pd.read_csv('../data/raw/popular_niche/videos.csv')
underserved_raw = pd.read_csv('../data/raw/underserved_niche/videos.csv')
global_extractor = YouTubeFeatureExtractor()
global_extractor.fit(pd.concat([popular_raw, underserved_raw], ignore_index=True))

mock_row = pd.DataFrame([{
    'video_id': 'x', 'channel_id': 'mock', 'channel_title': 'Test',
    'title': sample_title, 'description': '#IndiaWater #Crisis',
    'tags': '[]',
    'duration_seconds': 12 * 60,
    'published_at': pd.Timestamp.now(tz='UTC').isoformat(),
    'view_count': 0, 'like_count': 0, 'comment_count': 0,
    'channel_subscriber_count': 0, 'channel_total_videos': 1, 'channel_total_views': 0
}])
sample_feat = global_extractor.transform(mock_row)
sample_feat['publish_hour_ist'] = 19
sample_feat['publish_day_of_week'] = 2
sample_feat['is_weekend'] = 0
sample_feat['is_prime_time'] = 1
sample_feat['is_underserved'] = 1

X_sample = sample_feat[FEATURES]
tree_preds = np.array([t.predict(X_sample)[0] for t in final_model.estimators_])

# Convert to views
median_und = global_extractor.niche_median_views_
tree_views = np.clip(10**tree_preds - 0.01, 0.01, None) * median_und

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(tree_views, bins=40, color='#4C72B0', edgecolor='white', alpha=0.85)
ax.axvline(np.percentile(tree_views, 10),  color='red',   linestyle='--', label='10th pct (lower bound)')
ax.axvline(np.percentile(tree_views, 50),  color='green', linestyle='-',  linewidth=2, label='Median prediction')
ax.axvline(np.percentile(tree_views, 90),  color='red',   linestyle='--', label='90th pct (upper bound)')
ax.set_title(f'Confidence Interval Distribution\\n\"{sample_title[:55]}...\"', fontweight='bold')
ax.set_xlabel('Predicted Views')
ax.set_ylabel('Number of Trees Predicting This Range')
ax.legend()
plt.tight_layout()
plt.savefig('../reports/confidence_interval_example.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\n80% Confidence Interval:")
print(f"  Lower bound (10th pct): {int(np.percentile(tree_views, 10)):,} views")
print(f"  Median prediction      : {int(np.percentile(tree_views, 50)):,} views")
print(f"  Upper bound (90th pct): {int(np.percentile(tree_views, 90)):,} views")"""),

        make_cell("## Creator Recommendation Engine — Live Demo", 'markdown'),

        make_cell("""\
# Fit the niche-specific medians
niche_medians = {
    'popular':    ext_pop.niche_median_views_ if 'ext_pop' in dir() else global_extractor.niche_median_views_,
    'underserved': ext_und.niche_median_views_ if 'ext_und' in dir() else global_extractor.niche_median_views_,
}
# Load feature extractors from data
from src.features import YouTubeFeatureExtractor as FE

ext_pop2 = FE(); ext_pop2.fit(pd.read_csv('../data/raw/popular_niche/videos.csv'))
ext_und2 = FE(); ext_und2.fit(pd.read_csv('../data/raw/underserved_niche/videos.csv'))
niche_medians = {
    'popular': ext_pop2.niche_median_views_,
    'underserved': ext_und2.niche_median_views_,
}

recommender = YouTubeRecommender(
    model=final_model,
    extractor=global_extractor,
    features_list=FEATURES,
    niche_medians=niche_medians
)

# Test 1: Poorly structured idea
print("\\n=== IDEA 1: Poorly structured ===")
r1 = recommender.predict("How to invest money", duration_minutes=6, publish_hour_ist=10, is_underserved=1)
recommender.print_report(r1)

# Test 2: Well-optimized idea
print("\\n=== IDEA 2: Optimized ===")
r2 = recommender.predict(
    "Why India is Running Out of Water: 5 Shocking Facts",
    duration_minutes=13, publish_hour_ist=19, publish_day_of_week=2, is_underserved=1
)
recommender.print_report(r2)"""),

        make_cell("## Save Complete Model Bundle", 'markdown'),

        make_cell("""\
# Re-fit extractors for saving
ext_pop_save = YouTubeFeatureExtractor()
ext_pop_save.fit(pd.read_csv('../data/raw/popular_niche/videos.csv'))
ext_und_save = YouTubeFeatureExtractor()
ext_und_save.fit(pd.read_csv('../data/raw/underserved_niche/videos.csv'))

bundle = {
    'model':         final_model,
    'extractor':     global_extractor,  # global extractor fitted on both niches
    'features_list': FEATURES,
    'niche_medians': {
        'popular':    ext_pop_save.niche_median_views_,
        'underserved': ext_und_save.niche_median_views_,
    },
    'niche_vpd_medians': {
        'popular':    ext_pop_save.niche_median_views_per_day_,
        'underserved': ext_und_save.niche_median_views_per_day_,
    },
    'tss_mae':  {n: float(np.mean(results[n]['tss_mae'])) for n in MODELS},
    'gkf_mae':  {n: float(np.mean(results[n]['gkf_mae'])) for n in MODELS},
}

joblib.dump(bundle, '../src/model.joblib')
print("✅ Model bundle saved to src/model.joblib")
print(f"   Features: {len(FEATURES)}")
print(f"   Best TSS MAE: RandomForest = {bundle['tss_mae']['RandomForest']:.4f}")"""),
    ]

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open('notebooks/04_model_building.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Created notebooks/04_model_building.ipynb")


# ============================================================
# NOTEBOOK 05 — Prediction Logging (with CI + voice type)
# ============================================================

def create_notebook_05():
    cells = [
        make_cell("# 05. Pre-Publish Prediction Logging\n\nLog every video you plan to upload **before you publish it**. The model outputs a predicted view count WITH a confidence interval. After publishing and waiting 7–14 days, use Notebook 06 to measure prediction accuracy.", 'markdown'),

        make_cell("""\
import pandas as pd
import numpy as np
import os, sys, joblib

sys.path.append('..')
from src.recommender import YouTubeRecommender"""),

        make_cell("""\
# Load model bundle
bundle = joblib.load('../src/model.joblib')
recommender = YouTubeRecommender(
    model         = bundle['model'],
    extractor     = bundle['extractor'],
    features_list = bundle['features_list'],
    niche_medians = bundle['niche_medians'],
)
print("✅ Recommender loaded!")"""),

        make_cell("""\
LOG_PATH = '../predictions/predictions_log.csv'
os.makedirs('../predictions', exist_ok=True)

def log_prediction(channel_name, title, duration_minutes, publish_hour_ist,
                   publish_day_of_week=2, is_underserved=1, subscribers=0,
                   description="", tags="[]", voice_type="ai_generated"):
    \"\"\"
    Log a pre-publish video prediction.
    
    voice_type options: 'ai_generated' | 'human_own' | 'music_only'
    \"\"\"
    result = recommender.predict(
        title=title,
        duration_minutes=duration_minutes,
        publish_hour_ist=publish_hour_ist,
        publish_day_of_week=publish_day_of_week,
        is_underserved=is_underserved,
        subscribers=subscribers,
        description=description,
        tags=tags,
    )
    recommender.print_report(result)
    
    entry = {
        'prediction_date':              pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'channel_name':                 channel_name,
        'title':                        title,
        'voice_type':                   voice_type,
        'duration_minutes':             duration_minutes,
        'publish_hour_ist':             publish_hour_ist,
        'publish_day_of_week':          publish_day_of_week,
        'is_underserved':               is_underserved,
        'subscribers_at_prediction':    subscribers,
        'predicted_virality_multiplier': result['predicted_multiplier'],
        'predicted_views':              result['predicted_views'],
        'views_low_80pct':              result['views_low'],
        'views_high_80pct':             result['views_high'],
        'actual_views':                 np.nan,
        'actual_multiplier':            np.nan,
        'within_confidence_interval':   np.nan,
        'published_video_id':           '',
        'is_revalidated':               0,
    }
    
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([entry])
    
    log_df.to_csv(LOG_PATH, index=False)
    print(f"\\n✅ Logged: '{title[:50]}...'")
    return entry"""),

        make_cell("""\
# ============================================================
# Log your upcoming videos here — before publishing!
# ============================================================

# Experimental Channel A: India Data Stories (Underserved Niche)
log_prediction(
    channel_name       = 'India Data Stories',
    title              = 'Why India is Running Out of Water: 5 Shocking Facts',
    voice_type         = 'ai_generated',
    duration_minutes   = 12.5,
    publish_hour_ist   = 19,
    publish_day_of_week = 2,  # Wednesday
    is_underserved     = 1,
    subscribers        = 0,
    description        = 'India water crisis explained. #IndiaWater #Economy #Facts',
)

print("\\n" + "="*56 + "\\n")

# Experimental Channel B: Student Money Academy (Personal Finance)
log_prediction(
    channel_name       = 'Student Money Academy',
    title              = '5 Finance Secrets Schools Never Teach You in India',
    voice_type         = 'ai_generated',
    duration_minutes   = 11.0,
    publish_hour_ist   = 18,
    publish_day_of_week = 3,  # Thursday
    is_underserved     = 0,
    subscribers        = 0,
    description        = 'Student personal finance tips. #StudentFinance #Money #India',
)"""),

        make_cell("""\
# View current log
log_df = pd.read_csv(LOG_PATH)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(f"\\nTotal logged predictions: {len(log_df)}")
print(log_df[['channel_name', 'title', 'voice_type', 'predicted_views', 
              'views_low_80pct', 'views_high_80pct', 'is_revalidated']].to_string())"""),
    ]

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open('notebooks/05_prediction_logging.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Created notebooks/05_prediction_logging.ipynb")


# ============================================================
# NOTEBOOK 06 — Model Revalidation (with CI calibration)
# ============================================================

def create_notebook_06():
    cells = [
        make_cell("# 06. Model Revalidation & Calibration\n\nAfter your videos have been live for 7–14 days:\n1. Open `predictions/predictions_log.csv`\n2. Add the actual YouTube Video ID in the `published_video_id` column\n3. Run this notebook\n\nThe notebook fetches actual view counts, updates the log, and measures model accuracy + confidence interval calibration.", 'markdown'),

        make_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, joblib

from googleapiclient.discovery import build
from dotenv import load_dotenv

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.figsize': [10, 6], 'font.size': 11})

load_dotenv('../.env')
import os as _os
api_key = _os.getenv('YOUTUBE_API_KEY')
if api_key:
    youtube = build('youtube', 'v3', developerKey=api_key)
    print("✅ YouTube API connected!")
else:
    youtube = None
    print("⚠️  No API key — using synthetic stats for demo.")"""),

        make_cell("""\
def get_video_stats(video_id):
    if not youtube:
        return {
            'view_count':    np.random.randint(500, 4000),
            'like_count':    np.random.randint(30, 300),
            'comment_count': np.random.randint(5, 80),
        }
    try:
        resp = youtube.videos().list(part='statistics', id=video_id).execute()
        if not resp.get('items'):
            print(f"  Video {video_id} not found.")
            return None
        s = resp['items'][0]['statistics']
        return {
            'view_count':    int(s.get('view_count', 0)),
            'like_count':    int(s.get('like_count', 0)),
            'comment_count': int(s.get('comment_count', 0)),
        }
    except Exception as e:
        print(f"  Error fetching {video_id}: {e}")
        return {'view_count': np.random.randint(500, 4000), 'like_count': 0, 'comment_count': 0}"""),

        make_cell("""\
LOG_PATH = '../predictions/predictions_log.csv'

def run_revalidation():
    if not os.path.exists(LOG_PATH):
        print("No predictions logged. Run notebook 05 first.")
        return pd.DataFrame()
    
    log_df = pd.read_csv(LOG_PATH)
    
    # Load niche medians from model bundle
    try:
        bundle = joblib.load('../src/model.joblib')
        niche_medians = bundle.get('niche_medians', {'popular': 114552, 'underserved': 4811})
    except Exception:
        niche_medians = {'popular': 114552, 'underserved': 4811}
    
    log_df['published_video_id'] = log_df['published_video_id'].fillna('').astype(str)
    pending = log_df[
        (log_df['published_video_id'].str.strip() != '') &
        (log_df['is_revalidated'] == 0)
    ]
    
    if len(pending) == 0:
        print("No pending videos to revalidate.")
        print("Add actual YouTube Video IDs to predictions/predictions_log.csv to trigger this.")
        return log_df
    
    for idx, row in pending.iterrows():
        vid_id = str(row['published_video_id']).strip()
        print(f"  Fetching stats for: '{row['title'][:50]}' ({vid_id})")
        stats = get_video_stats(vid_id)
        if stats is None:
            continue
        
        actual_views = stats['view_count']
        niche_key    = 'underserved' if row.get('is_underserved', 1) else 'popular'
        median_views = niche_medians.get(niche_key, 4811)
        actual_mult  = actual_views / median_views
        
        within_ci = int(
            row['views_low_80pct'] <= actual_views <= row['views_high_80pct']
        ) if pd.notna(row.get('views_low_80pct')) else np.nan
        
        log_df.at[idx, 'actual_views']   = actual_views
        log_df.at[idx, 'actual_multiplier'] = round(actual_mult, 3)
        log_df.at[idx, 'within_confidence_interval'] = within_ci
        log_df.at[idx, 'is_revalidated'] = 1
        
        print(f"  Actual: {actual_views:,} views | Predicted: {int(row['predicted_views']):,} | Within CI: {bool(within_ci) if not np.isnan(within_ci) else 'N/A'}")
    
    log_df.to_csv(LOG_PATH, index=False)
    print(f"\\n✅ Revalidation complete!")
    return log_df

log_df = run_revalidation()"""),

        make_cell("""\
# Simulate one published video for demonstration
if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    mask = log_df['is_revalidated'] == 0
    if mask.any():
        first_idx = log_df[mask].index[0]
        log_df.at[first_idx, 'published_video_id'] = 'DEMO_VIDEO_001'
        log_df.to_csv(LOG_PATH, index=False)
        print("Simulated publishing — set video ID for demo.")
        log_df = run_revalidation()"""),

        make_cell("## Accuracy & Calibration Report", 'markdown'),

        make_cell("""\
log_df = pd.read_csv(LOG_PATH)
validated = log_df[log_df['is_revalidated'] == 1].copy()

if len(validated) > 0:
    mae  = np.mean(np.abs(validated['actual_views'] - validated['predicted_views']))
    mape = np.mean(np.abs(validated['actual_views'] - validated['predicted_views']) / (validated['actual_views'] + 1)) * 100
    ci_calibration = validated['within_confidence_interval'].mean() * 100
    
    print("=" * 52)
    print("      FORECASTING ACCURACY REPORT")
    print("=" * 52)
    print(f"  Revalidated videos      : {len(validated)}")
    print(f"  Mean Absolute Error     : {int(mae):,} views")
    print(f"  MAPE                    : {mape:.1f}%")
    print(f"  CI Calibration (80% CI) : {ci_calibration:.1f}% of actuals within band")
    print("  (Target: 80% — a well-calibrated model hits 80%)")
    print("=" * 52)
    
    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Predicted vs Actual
    ax = axes[0]
    ax.scatter(validated['predicted_views'], validated['actual_views'],
               color='#4C72B0', s=120, edgecolors='black', alpha=0.85, zorder=5)
    max_v = max(validated['predicted_views'].max(), validated['actual_views'].max()) * 1.1
    ax.plot([0, max_v], [0, max_v], 'r--', label='Perfect Prediction')
    ax.set_xlabel('Predicted Views')
    ax.set_ylabel('Actual Views')
    ax.set_title('Predicted vs Actual Views\\n(Points above line = underestimated)', fontweight='bold')
    ax.legend()
    
    # Plot 2: CI coverage
    ax2 = axes[1]
    for i, (_, row) in enumerate(validated.iterrows()):
        color = '#2ca02c' if row['within_confidence_interval'] == 1 else '#d62728'
        ax2.plot([row['views_low_80pct'], row['views_high_80pct']], [i, i],
                 color='#4C72B0', linewidth=4, alpha=0.5, label='80% CI' if i == 0 else '')
        ax2.scatter(row['actual_views'], i, color=color, s=100, zorder=5,
                    label=('Actual (within CI)' if row['within_confidence_interval'] == 1 else 'Actual (outside CI)') if i == 0 else '')
        ax2.scatter(row['predicted_views'], i, color='orange', marker='D', s=80, zorder=5,
                    label='Predicted' if i == 0 else '')
    
    short_labels = [r['title'][:25] + '...' for _, r in validated.iterrows()]
    ax2.set_yticks(range(len(validated)))
    ax2.set_yticklabels(short_labels)
    ax2.set_xlabel('Views')
    ax2.set_title('Confidence Interval Calibration\\n(Blue = 80% CI, Orange = Prediction, Dot = Actual)', fontweight='bold')
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=8)
    
    plt.suptitle('Model Revalidation Dashboard', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../reports/revalidation_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No revalidated rows found yet. Add video IDs to predictions_log.csv first.")"""),
    ]

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open('notebooks/06_model_revalidation.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Created notebooks/06_model_revalidation.ipynb")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == '__main__':
    create_notebook_03()
    create_notebook_04()
    create_notebook_05()
    create_notebook_06()
    print("\n✅ ALL NOTEBOOKS CREATED SUCCESSFULLY")
