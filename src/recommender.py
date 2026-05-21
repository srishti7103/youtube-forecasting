"""
YouTube Creator Recommendation Engine
======================================
Loads the PyTorch + LightGBM ensemble and features to predict video view velocity
and generate a comprehensive Creator Playbook.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from textblob import TextBlob

from src.features import EMOTIONAL_TRIGGERS

# ---------------------------------------------------------------------------
# PyTorch MLP Architecture Definition
# ---------------------------------------------------------------------------
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Recommender Engine
# ---------------------------------------------------------------------------
class YouTubeRecommender:
    """
    Loads a trained PyTorch + LightGBM model bundle and provides 
    multi-horizon predictions and Creator Playbook.
    """

    def __init__(self, lgb_model, pytorch_state, features_list, extractor, rmse, niche_medians):
        self.lgb_model = lgb_model
        self.features_list = features_list
        self.extractor = extractor
        self.rmse = rmse if rmse > 0 else 0.25
        self.niche_medians = niche_medians

        # Reconstruct PyTorch Model
        input_dim = len(features_list)
        self.pytorch_model = PyTorchMLP(input_dim)
        self.pytorch_model.load_state_dict(pytorch_state)
        self.pytorch_model.eval()

    @classmethod
    def load(cls, model_path=None):
        """Load the recommender from a saved model.joblib bundle."""
        if model_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base, 'model.joblib')

        bundle = joblib.load(model_path)
        return cls(
            lgb_model     = bundle['lightgbm_model'],
            pytorch_state = bundle['pytorch_model_state'],
            features_list = bundle['features_list'],
            extractor     = bundle['extractor'],
            rmse          = bundle.get('rmse', 0.25),
            niche_medians = bundle.get('niche_medians', {})
        )

    def _build_mock_row(self, title, duration_minutes, publish_hour_ist,
                        publish_day_of_week, niche, subscribers,
                        voice_type, presentation_style, is_highly_edited,
                        description, tags, rolling_channel_baseline, video_seq, channel_age_days):
        """Build a single-row DataFrame for feature extraction."""
        return pd.DataFrame([{
            'video_id':                 'mock_video',
            'channel_id':               'mock_channel',
            'channel_title':            'Mock Channel',
            'title':                    title,
            'description':              description,
            'tags':                     tags,
            'duration_seconds':         duration_minutes * 60,
            'published_at':             pd.Timestamp.now(tz='UTC').isoformat(),
            'channel_subscriber_count': subscribers,
            'channel_total_videos':     100,
            'channel_total_views':      subscribers * 50,
            'channel_created':          (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=channel_age_days)).isoformat(),
            'niche':                    niche,
            'voice_type':               voice_type,
            'presentation_style':       presentation_style,
            'is_highly_edited':         is_highly_edited,
            'rolling_channel_baseline': rolling_channel_baseline,
            'video_sequence_number':    video_seq,
            'channel_age_days_at_upload': channel_age_days,
            'view_count':               0,
            'like_count':               0,
            'comment_count':            0
        }])

    def predict(self, title, duration_minutes=12.0, publish_hour_ist=19,
                publish_day_of_week=2, niche='personal_finance', subscribers=10000,
                voice_type='human_own', presentation_style='face_presenter', is_highly_edited=1,
                description="", tags="[]", rolling_channel_baseline=None, video_seq=100, channel_age_days=365):
        """
        Predict views for 7, 30, and 90 days.
        """
        # Default rolling baseline to 10% of subscribers if not provided
        if rolling_channel_baseline is None or rolling_channel_baseline <= 0:
            rolling_channel_baseline = max(subscribers * 0.1, 100.0)

        horizons = [7, 30, 90]
        predictions = {}

        for h in horizons:
            # Build row
            mock_row = self._build_mock_row(
                title, duration_minutes, publish_hour_ist,
                publish_day_of_week, niche, subscribers,
                voice_type, presentation_style, is_highly_edited,
                description, tags, rolling_channel_baseline, video_seq, channel_age_days
            )

            # Apply Feature Extractor
            features = self.extractor.transform(mock_row)

            # Override temporal features explicitly
            features['publish_hour_ist']    = publish_hour_ist
            features['publish_day_of_week'] = publish_day_of_week
            features['is_weekend']          = int(publish_day_of_week >= 5)
            features['is_prime_time']       = int(17 <= publish_hour_ist <= 21)
            
            # Explicitly override days_since_published to model the milestone
            features['days_since_published'] = h
            features['log_days_since_published'] = np.log10(h + 1)

            # Select model features
            X = features[self.features_list]

            # 1. LightGBM prediction
            lgb_pred = self.lgb_model.predict(X)[0]

            # 2. PyTorch prediction
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            with torch.no_grad():
                pytorch_pred = self.pytorch_model(X_tensor).item()

            # 3. Ensemble (average logs)
            ens_pred_log = 0.5 * lgb_pred + 0.5 * pytorch_pred

            # Convert back to views
            predicted_views = max(0, int(10 ** ens_pred_log - 1))

            # 80% Confidence Interval (z=1.28)
            log_low = ens_pred_log - 1.28 * self.rmse
            log_high = ens_pred_log + 1.28 * self.rmse
            
            views_low = max(0, int(10 ** log_low - 1))
            views_high = max(0, int(10 ** log_high - 1))

            # Calculate relative performance (V-Ratio)
            v_ratio = predicted_views / (rolling_channel_baseline + 1)

            predictions[h] = {
                'predicted_views': predicted_views,
                'views_low': views_low,
                'views_high': views_high,
                'v_ratio': round(v_ratio, 2)
            }

        # Determine virality classification based on 30-day forecast
        v_30 = predictions[30]['v_ratio']
        if v_30 < 0.5:
            virality_tier = "Stagnant (Underperforming baseline)"
        elif v_30 < 1.5:
            virality_tier = "Baseline (Normal channel performance)"
        elif v_30 < 3.0:
            virality_tier = "Growth (Outperforming baseline)"
        else:
            virality_tier = "Viral Outlier (Massive breakout potential!)"

        # Niche name mapping
        niche_names = {
            'personal_finance': 'Personal Finance India',
            'geopolitics_economy': 'Geopolitics & Economy',
            'tech_gadgets': 'Tech & Gadgets',
            'gaming': 'Gaming',
            'comedy_entertainment': 'Comedy & Entertainment',
            'education_science': 'Education & Science',
            'makeup_beauty': 'Makeup & Beauty',
            'vlogging_lifestyle': 'Vlogging & Lifestyle'
        }

        # Style insights
        style_insights = self._get_style_insights(niche, voice_type, presentation_style, is_highly_edited)

        # Title analysis
        title_analysis = self._analyze_title(title)

        return {
            'title': title,
            'niche': niche_names.get(niche, niche),
            'duration_minutes': duration_minutes,
            'publish_hour_ist': publish_hour_ist,
            'publish_day_of_week': publish_day_of_week,
            'voice_type': voice_type,
            'presentation_style': presentation_style,
            'is_highly_edited': is_highly_edited,
            'rolling_baseline': rolling_channel_baseline,
            'virality_tier': virality_tier,
            'predictions': predictions,
            'title_analysis': title_analysis,
            'style_insights': style_insights
        }

    def _analyze_title(self, title):
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        recs = []
        n_chars = len(title)
        if n_chars > 65:
            recs.append(f"Trim title to under 65 characters (currently {n_chars} chars) to prevent truncation on mobile devices.")
        elif n_chars < 45:
            recs.append(f"Expand title to 45-65 characters (currently {n_chars} chars) to include rich keywords and a curiosity hook.")

        if '?' not in title:
            recs.append("Add a curiosity question (e.g. 'Is this...?', 'Why...?') to spike click-through rates (CTR).")

        if not any(c.isdigit() for c in title):
            recs.append("Integrate a specific number or percentage (e.g. '₹5 Lakhs', '3 Mistakes') to make the title more concrete.")

        # Money or India anchor
        if not re.search(r'[₹]|rupee|crore|lakh|india|bharat', title.lower()):
            recs.append("Add a local anchor if relevant (e.g., ₹ rupees, India, Bharat) to increase search match rates.")

        triggers = [t for t in EMOTIONAL_TRIGGERS if t in title.lower()]

        # Generate Alternative Title Suggestions
        alternatives = []
        clean_title = re.sub(r'[?!.]', '', title).strip()
        alternatives.append(f"Why {clean_title}? (The Hidden Truth)")
        alternatives.append(f"Is {clean_title} Actually Worth It? (₹5 Lakh Scam)")
        alternatives.append(f"3 Big Mistakes with {clean_title} (And How to Save)")

        return {
            'char_count': n_chars,
            'emotional_triggers': triggers,
            'sentiment_polarity': round(polarity, 2),
            'sentiment_subjectivity': round(subjectivity, 2),
            'suggestions': recs,
            'alternatives': alternatives
        }

    def _get_style_insights(self, niche, voice_type, presentation_style, is_highly_edited):
        warnings = []
        success_tips = []

        # Voice Type Analysis
        if voice_type == 'ai_generated':
            warnings.append("AI-generated voice carries a heavy audience retention penalty. Highly authoritative channels in this niche use natural human narration.")
        else:
            success_tips.append("Using human narration builds subscriber loyalty and long-term search trust.")

        # Presentation Style Analysis
        if niche in ['geopolitics_economy', 'education_science']:
            if presentation_style == 'face_presenter':
                warnings.append("Case study / Education content performs better when supported heavily by faceless B-roll or animated visuals rather than a talking-head shot.")
            else:
                success_tips.append("Faceless B-roll/animations keep viewer retention high for data-heavy narratives.")
        elif niche in ['makeup_beauty', 'vlogging_lifestyle', 'comedy_entertainment']:
            if presentation_style != 'face_presenter':
                warnings.append("Beauty, lifestyle, and sketch comedy require high emotional connection. A presenter's face is crucial for these genres.")
            else:
                success_tips.append("Showing your face builds strong visual trust and higher click-through on thumbnails.")

        # Editing quality
        if is_highly_edited == 0:
            warnings.append("Fast-paced niches require high-quality editing (motion graphics, sound effects) to maintain retention above 45%.")
        else:
            success_tips.append("High editing frequency aligns with successful channel baselines in this niche.")

        return {
            'warnings': warnings,
            'success_tips': success_tips
        }

    def print_report(self, result):
        """Print a structured, encoding-safe Creator Playbook."""
        def safe_print(text):
            try:
                print(text)
            except UnicodeEncodeError:
                clean = text
                replacements = {
                    "★": "*",
                    "•": "-",
                    "✔": "[OK]",
                    "✘": "[FAIL]",
                    "🏆": "[VIRAL]",
                    "💡": "[TIP]",
                    "⚠️": "[WARN]",
                    "📈": "[GROWTH]",
                    "📉": "[STAGNANT]",
                    "₹": "Rs.",
                    "—": "-"
                }
                for k, v in replacements.items():
                    clean = clean.replace(k, v)
                clean = clean.encode('ascii', errors='replace').decode('ascii')
                print(clean)

        safe_print("\n" + "=" * 65)
        safe_print("                   CREATOR FORECASTING PLAYBOOK")
        safe_print("=" * 65)
        safe_print(f" Proposed Title : {result['title']}")
        safe_print(f" Niche          : {result['niche']}")
        safe_print(f" Format/Style   : {result['presentation_style'].upper()} | {result['voice_type'].upper()} | {'HIGHLY EDITED' if result['is_highly_edited'] else 'LIGHTLY EDITED'}")
        safe_print(f" Sub Baseline   : {result['rolling_baseline']:,} views (preceding uploads median)")
        safe_print("-" * 65)
        
        # 1. Virality Forecast Table
        safe_print(" [1] VIEWS & VELOCITY HORIZONS")
        safe_print(f"  Virality Tier : {result['virality_tier']}")
        safe_print(f"  {'Horizon':<10} | {'Expected Views':<16} | {'80% Confidence Band':<22} | {'V-Ratio':<8}")
        safe_print(f"  {'-'*10} | {'-'*16} | {'-'*22} | {'-'*8}")
        for h in [7, 30, 90]:
            p = result['predictions'][h]
            safe_print(f"  {h:<2} Days     | {p['predicted_views']:<16,} | {p['views_low']:,} - {p['views_high']:,} | {p['v_ratio']:.2f}x")
        
        safe_print("-" * 65)
        
        # 2. Title Reframing
        safe_print(" [2] TITLE ANALYSIS & REFRAMING")
        safe_print(f"  Character Count  : {result['title_analysis']['char_count']} (Ideal: 45-65)")
        safe_print(f"  Sentiment Tone   : Polarity {result['title_analysis']['sentiment_polarity']:.2f} | Subjectivity {result['title_analysis']['sentiment_subjectivity']:.2f}")
        if result['title_analysis']['emotional_triggers']:
            safe_print(f"  Emotional Hooks  : {', '.join(result['title_analysis']['emotional_triggers'])}")
        
        safe_print("\n  Advisory Warnings:")
        for sugg in result['title_analysis']['suggestions']:
            safe_print(f"    ✔ {sugg}")
            
        safe_print("\n  Suggested Title Reframes:")
        for idx, alt in enumerate(result['title_analysis']['alternatives']):
            safe_print(f"    Alt {idx+1}: {alt}")

        safe_print("-" * 65)

        # 3. Format & Style Strategy
        safe_print(" [3] FORMAT & STYLE INSIGHTS")
        if result['style_insights']['warnings']:
            safe_print("  Warnings:")
            for w in result['style_insights']['warnings']:
                safe_print(f"    ⚠️  {w}")
        if result['style_insights']['success_tips']:
            safe_print("  Success Drivers:")
            for t in result['style_insights']['success_tips']:
                safe_print(f"    💡 {t}")

        safe_print("-" * 65)

        # 4. Discoverability SEO Guide
        safe_print(" [4] DISCOVERABILITY SEO CHECKLIST")
        safe_print("  - Description: Include 3 relevant hashtags at the bottom of description.")
        safe_print("  - Keyword Anchoring: Ensure your main keyword is in the first 2 lines.")
        safe_print("  - Tag Injection: Use 5-10 specific tags mapped to your niche.")
        safe_print("=" * 65 + "\n")
