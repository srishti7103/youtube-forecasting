import os
import sys

# Ensure output streams use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Ensure src/ is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import YouTubeFeatureExtractor
from src.recommender import PyTorchMLP, YouTubeRecommender

def main():
    print("=" * 60)
    # Use ascii-only characters for safety
    print("               MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load harvested dataset
    raw_path = os.path.join("data", "raw", "scaled_niche", "videos.csv")
    if not os.path.exists(raw_path):
        print(f"[ERROR] Harvested dataset not found at: {raw_path}")
        print("Please wait for the harvester to complete.")
        return

    print(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded {len(df_raw)} records from {df_raw['channel_id'].nunique()} channels.")

    # 2. Extract features
    print("Extracting features using YouTubeFeatureExtractor...")
    extractor = YouTubeFeatureExtractor()
    extractor.fit(df_raw)
    df_feat = extractor.transform(df_raw)
    print(f"Feature matrix shape: {df_feat.shape}")

    # Define features and target
    FEATURES_LIST = [
        'duration_minutes',
        'is_optimal_duration',
        'is_shorts',
        'title_char_count',
        'title_word_count',
        'title_has_number',
        'title_has_question',
        'title_has_exclamation',
        'title_has_colon',
        'title_has_money_ref',
        'title_has_all_caps',
        'title_emotional_trigger_count',
        'title_is_hindi_or_hinglish',
        'title_mentions_india',
        'title_sentiment_polarity',
        'title_sentiment_subjectivity',
        'tags_count',
        'hashtag_count',
        'has_hashtags',
        'description_char_count',
        'publish_hour_ist',
        'publish_day_of_week',
        'is_weekend',
        'is_prime_time',
        'log_channel_subscribers',
        'log_rolling_channel_baseline',
        'log_video_sequence_number',
        'log_channel_age_days_at_upload',
        'days_since_published',
        'log_days_since_published',
        'is_highly_edited',
        'niche_personal_finance',
        'niche_geopolitics_economy',
        'niche_tech_gadgets',
        'niche_gaming',
        'niche_comedy_entertainment',
        'niche_education_science',
        'niche_makeup_beauty',
        'niche_vlogging_lifestyle',
        'voice_type_human_own',
        'voice_type_ai_generated',
        'voice_type_music_only',
        'presentation_style_face_presenter',
        'presentation_style_faceless_broll',
        'presentation_style_animated_visuals'
    ]

    # Check for missing features
    missing = [f for f in FEATURES_LIST if f not in df_feat.columns]
    if missing:
        print(f"[WARN] Missing features from extractor: {missing}")
        # Dynamically filter features
        FEATURES_LIST = [f for f in FEATURES_LIST if f in df_feat.columns]

    X = df_feat[FEATURES_LIST]
    y = df_feat['log_views']  # target is log10(views + 1)
    print(f"Target variable statistics:\n{y.describe()}")

    # Sort chronologically to prevent lookahead bias in validation (smaller days_since_published = newer)
    # Sort descending by days_since_published so oldest is first, newest is last.
    sort_idx = df_feat['days_since_published'].argsort()[::-1]
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    days_sorted = df_feat['days_since_published'].iloc[sort_idx].reset_index(drop=True)

    # Chronological Split: 80% train, 20% validation
    split_idx = int(len(X_sorted) * 0.8)
    X_train, X_val = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_val = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
    days_train, days_val = days_sorted.iloc[:split_idx], days_sorted.iloc[split_idx:]

    print(f"Train set: {len(X_train)} samples (older videos).")
    print(f"Val set:   {len(X_val)} samples (newer videos).")

    # 3. Calculate sample weights based on upload age (recent videos = 1.0, older decay to 0.3)
    # Linear decay from 1.0 to 0.3 between 365 days and 1095 days (3 years)
    w_train = np.ones(len(X_train))
    mask_tr = days_train > 365
    w_train[mask_tr] = np.maximum(0.3, 1.0 - (days_train[mask_tr] - 365) / 730.0 * 0.7)

    w_val = np.ones(len(X_val))
    mask_val = days_val > 365
    w_val[mask_val] = np.maximum(0.3, 1.0 - (days_val[mask_val] - 365) / 730.0 * 0.7)

    print(f"Train sample weights - Median: {np.median(w_train):.2f}, Mean: {w_train.mean():.2f}")

    # 4. Train LightGBM Regressor
    print("\nTraining LightGBM model...")
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # Custom logging callback to print training progress safely
    callbacks = [lgb.log_evaluation(period=20)]
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_val],
        callbacks=callbacks
    )
    print("LightGBM training completed.")

    # 5. Train PyTorch MLP Model
    print("\nTraining PyTorch MLP model...")
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    w_train_tensor = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    w_val_tensor = torch.tensor(w_val, dtype=torch.float32).unsqueeze(1)

    input_dim = X_train.shape[1]
    mlp = PyTorchMLP(input_dim)
    
    optimizer = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train loops
    epochs = 100
    batch_size = 128
    
    for epoch in range(1, epochs + 1):
        mlp.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]
            batch_w = w_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = mlp(batch_x)
            
            # Weighted MSE loss
            loss_raw = (outputs - batch_y) ** 2
            loss = (loss_raw * batch_w).mean()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
            
        epoch_loss /= X_train_tensor.size(0)
        
        if epoch % 20 == 0 or epoch == 1:
            # Eval on validation
            mlp.eval()
            with torch.no_grad():
                val_outputs = mlp(X_val_tensor)
                val_loss_raw = (val_outputs - y_val_tensor) ** 2
                val_loss = (val_loss_raw * w_val_tensor).mean().item()
            print(f"  Epoch {epoch:3d}/{epochs} | Train Weighted Loss: {epoch_loss:.5f} | Val Weighted Loss: {val_loss:.5f}")

    print("PyTorch MLP training completed.")

    # 6. Evaluate Ensemble (Average) on Validation Set
    print("\nEvaluating models on validation set...")
    lgb_pred = lgb_model.predict(X_val)
    
    mlp.eval()
    with torch.no_grad():
        mlp_pred = mlp(X_val_tensor).numpy().flatten()
        
    ens_pred = 0.5 * lgb_pred + 0.5 * mlp_pred
    
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
    mlp_rmse = np.sqrt(mean_squared_error(y_val, mlp_pred))
    ens_rmse = np.sqrt(mean_squared_error(y_val, ens_pred))
    
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    ens_mae = mean_absolute_error(y_val, ens_pred)
    
    ens_r2 = r2_score(y_val, ens_pred)

    print("-" * 60)
    print(f"LightGBM RMSE : {lgb_rmse:.4f}")
    print(f"PyTorch MLP RMSE: {mlp_rmse:.4f}")
    print(f"Ensemble RMSE   : {ens_rmse:.4f}")
    print(f"Ensemble MAE    : {ens_mae:.4f} (log10 scale)")
    print(f"Ensemble R2     : {ens_r2:.4f}")
    print("-" * 60)

    # 7. Save model.joblib bundle
    model_dir = "src"
    os.makedirs(model_dir, exist_ok=True)
    bundle_path = os.path.join(model_dir, "model.joblib")
    
    print(f"Saving model bundle to {bundle_path}...")
    bundle = {
        'lightgbm_model': lgb_model,
        'pytorch_model_state': mlp.state_dict(),
        'features_list': FEATURES_LIST,
        'extractor': extractor,
        'rmse': float(ens_rmse),
        'niche_medians': {n: float(df_raw[df_raw['niche'] == n]['view_count'].median()) 
                          for n in df_raw['niche'].unique() if 'niche' in df_raw.columns}
    }
    joblib.dump(bundle, bundle_path)
    print("✅ Model bundle saved successfully!")

    # 8. Sanity check: load using YouTubeRecommender
    print("\nVerifying YouTubeRecommender load and prediction...")
    recommender = YouTubeRecommender.load(bundle_path)
    res = recommender.predict(
        title="Why Indian Startups are Collapsing: Case Study",
        duration_minutes=15.0,
        publish_hour_ist=18,
        publish_day_of_week=4,
        niche='geopolitics_economy',
        subscribers=50000,
        voice_type='human_own',
        presentation_style='faceless_broll',
        is_highly_edited=1
    )
    print("Verification report:")
    recommender.print_report(res)
    print("All checks completed successfully!")

if __name__ == "__main__":
    main()
