import sys, os
sys.path.insert(0, os.path.abspath('.'))

import joblib, pandas as pd, numpy as np

# Check model bundle
bundle = joblib.load('src/model.joblib')
print('=== MODEL BUNDLE ===')
print(f"Features ({len(bundle['features_list'])}): {bundle['features_list']}")
print(f"TSS MAE  RF  : {bundle['tss_mae']['RandomForest']:.4f}")
print(f"TSS MAE  Ridge: {bundle['tss_mae']['Ridge']:.4f}")
print(f"TSS MAE  HGB  : {bundle['tss_mae']['HistGradientBoosting']:.4f}")
print(f"Niche medians: popular={bundle['niche_medians']['popular']:,.0f} | underserved={bundle['niche_medians']['underserved']:,.0f}")

# Check report files
print()
print('=== REPORT FILES ===')
for f in sorted(os.listdir('reports')):
    size = os.path.getsize(f'reports/{f}')
    print(f'  {f:45s} ({size/1024:.1f} KB)')

# Check predictions log
print()
print('=== PREDICTIONS LOG ===')
log = pd.read_csv('predictions/predictions_log.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
cols = ['channel_name', 'voice_type', 'predicted_views', 'views_low_80pct', 'views_high_80pct']
print(log[cols].to_string())

# Live recommender demo
print()
print('=== RECOMMENDER DEMO ===')
from src.recommender import YouTubeRecommender
engine = YouTubeRecommender.load('src/model.joblib')
r = engine.predict(
    title="Why India is Running Out of Water: 5 Shocking Facts",
    duration_minutes=13,
    publish_hour_ist=19,
    publish_day_of_week=2,
    is_underserved=1,
)
engine.print_report(r)
