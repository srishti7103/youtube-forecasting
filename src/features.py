import re
import numpy as np
import pandas as pd
import isodate
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONAL_TRIGGERS = [
    "crash", "scam", "secret", "earn", "rich", "save", "free", "best", "worst",
    "fail", "success", "mistake", "bubble", "hidden", "warning", "ruin", "danger",
    "growth", "smart", "cheat", "hacks", "exposed", "scared", "shocked", "destroy",
    "wealth", "losing", "gaining", "richer", "poorer", "revolution", "collapse",
    "fraud", "trap", "banned", "truth", "lies", "reality", "change"
]

HINGLISH_WORDS = [
    "kaise", "kyun", "kya", "yeh", "toh", "baat", "paisa", "paise", "rupaye",
    "crore", "lakh", "sarkaar", "sarkar", "bharat", "desh", "log", "sab",
    "aaj", "kal", "nahi", "nahin", "hai", "hain", "mat", "karein", "karo"
]

# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class YouTubeFeatureExtractor:
    """
    Transforms raw YouTube video metadata DataFrames into engineered feature
    matrices for virality and view prediction.
    
    Supports training (where features are calculated from historical data) and
    inference (where features are passed or looked up).
    """

    def __init__(self):
        self.channel_stats_ = {}
        self.niche_median_views_ = 5000.0
        self.niche_median_subs_ = 50000.0

    def _get_column(self, df, options, default=None):
        """Return the first column name from options that exists in df."""
        for opt in options:
            if opt in df.columns:
                return opt
        return default

    def _parse_duration(self, val):
        """Parse duration to seconds, handling ISO 8601 strings and numeric values."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 600.0
        if isinstance(val, (int, float)):
            return float(val)
        duration_str = str(val).strip()
        if not duration_str:
            return 600.0
        try:
            return isodate.parse_duration(duration_str).total_seconds()
        except Exception:
            try:
                hours = re.search(r'(\d+)H', duration_str)
                minutes = re.search(r'(\d+)M', duration_str)
                seconds = re.search(r'(\d+)S', duration_str)
                total = 0
                if hours:   total += int(hours.group(1)) * 3600
                if minutes: total += int(minutes.group(1)) * 60
                if seconds: total += int(seconds.group(1))
                return float(total) if total > 0 else 600.0
            except Exception:
                return 600.0

    def _count_tags(self, tags_val):
        """Count tags from raw string or list."""
        if isinstance(tags_val, list):
            return len(tags_val)
        if isinstance(tags_val, str):
            if tags_val.strip() in ("", "[]"):
                return 0
            return len([t.strip() for t in
                        tags_val.replace("[", "").replace("]", "").replace("'", "").split("|")
                        if t.strip()])
        return 0

    def _is_hindi_or_hinglish(self, title):
        """Detect Hindi script (Devanagari) OR common Hinglish transliteration."""
        if re.search(r'[\u0900-\u097F]', title):
            return 1
        title_lower = title.lower()
        if any(word in title_lower.split() for word in HINGLISH_WORDS):
            return 1
        return 0

    def _count_hashtags(self, description):
        """Count #hashtags in description."""
        if not isinstance(description, str):
            return 0
        return len(re.findall(r'#\w+', description))

    def _count_emotional_triggers(self, title):
        """Count emotional trigger words present in title."""
        title_lower = title.lower()
        return sum(1 for t in EMOTIONAL_TRIGGERS if t in title_lower)

    def fit(self, df):
        """Learn channel-level statistics and niche-level baselines."""
        view_col = self._get_column(df, ['view_count', 'views'], 'view_count')
        sub_col  = self._get_column(df, ['channel_subscriber_count', 'subscribers'], 'channel_subscriber_count')

        df_clean = df.copy()
        df_clean['views_c'] = pd.to_numeric(df_clean[view_col], errors='coerce').fillna(0)
        df_clean['subs_c']  = pd.to_numeric(df_clean[sub_col],  errors='coerce').fillna(100)

        # Niche-wide baselines
        self.niche_median_views_ = float(df_clean['views_c'].median())
        self.niche_median_subs_  = float(df_clean['subs_c'].median())

        # Channel baselines
        for channel_id, grp in df_clean.groupby('channel_id'):
            self.channel_stats_[channel_id] = {
                'median_views': float(grp['views_c'].median()),
                'subscribers':  float(grp['subs_c'].iloc[0]) if len(grp) > 0 else 1000.0,
            }
        return self

    def transform(self, df):
        """Extract features and target variables from raw DataFrame."""
        out = pd.DataFrame(index=df.index)

        # -- Identifiers -------------------------------------------------------
        for col in ['video_id', 'channel_id', 'channel_title', 'title', 'niche', 'channel_category']:
            if col in df.columns:
                out[col] = df[col]
            else:
                out[col] = 'unknown'

        # -- Target Variable ---------------------------------------------------
        view_col = self._get_column(df, ['view_count', 'views'], 'view_count')
        views = pd.to_numeric(df[view_col], errors='coerce').fillna(0)
        out['views'] = views
        out['log_views'] = np.log10(views + 1)

        # -- Channel Authority & Chronological Baselines -----------------------
        sub_col = self._get_column(df, ['channel_subscriber_count', 'subscribers'], 'channel_subscriber_count')
        subscribers = pd.to_numeric(df[sub_col], errors='coerce').fillna(1000)
        out['channel_subscribers'] = subscribers
        out['log_channel_subscribers'] = np.log10(subscribers + 1)

        # Sort chronologically to compute sequence & rolling features if NOT already provided
        df_temp = df.copy()
        df_temp['published_at_dt'] = pd.to_datetime(df_temp['published_at'], errors='coerce')
        max_pub = df_temp['published_at_dt'].max()
        if pd.isna(max_pub):
            max_pub = pd.Timestamp.now(tz='UTC')
        df_temp['published_at_dt'] = df_temp['published_at_dt'].fillna(max_pub)

        # 1. Video Age (Days since published)
        days_old = (max_pub - df_temp['published_at_dt']).dt.days.clip(lower=1).fillna(1)
        out['days_since_published'] = days_old
        out['log_days_since_published'] = np.log10(days_old + 1)

        # 2. Chronological sequence number of upload
        if 'video_sequence_number' in df.columns:
            out['video_sequence_number'] = df['video_sequence_number']
        else:
            df_temp = df_temp.sort_values(['channel_id', 'published_at_dt'])
            df_temp['video_sequence_number'] = df_temp.groupby('channel_id').cumcount() + 1
            out['video_sequence_number'] = df_temp.loc[df.index, 'video_sequence_number']
        out['log_video_sequence_number'] = np.log10(out['video_sequence_number'] + 1)

        # 3. Channel age at upload
        if 'channel_age_days_at_upload' in df.columns:
            out['channel_age_days_at_upload'] = df['channel_age_days_at_upload']
        else:
            df_temp['channel_created_dt'] = pd.to_datetime(df_temp.get('channel_created', pd.NaT), errors='coerce')
            min_created = df_temp['channel_created_dt'].min()
            if pd.isna(min_created):
                min_created = pd.Timestamp('2015-01-01', tz='UTC')
            df_temp['channel_created_dt'] = df_temp['channel_created_dt'].fillna(min_created)
            df_temp['channel_age_days_at_upload'] = (df_temp['published_at_dt'] - df_temp['channel_created_dt']).dt.days.clip(lower=1)
            out['channel_age_days_at_upload'] = df_temp.loc[df.index, 'channel_age_days_at_upload']
        out['log_channel_age_days_at_upload'] = np.log10(out['channel_age_days_at_upload'] + 1)

        # 4. Rolling channel baseline (median views of preceding 10 uploads, shifted by 1)
        if 'rolling_channel_baseline' in df.columns:
            out['rolling_channel_baseline'] = df['rolling_channel_baseline']
        else:
            df_temp['views_numeric'] = pd.to_numeric(df_temp[view_col], errors='coerce').fillna(0)
            df_temp['rolling_channel_baseline'] = (
                df_temp.groupby('channel_id')['views_numeric']
                .shift(1)
                .rolling(10, min_periods=1)
                .median()
                .fillna(0.0)
            )
            out['rolling_channel_baseline'] = df_temp.loc[df.index, 'rolling_channel_baseline']
        out['log_rolling_channel_baseline'] = np.log10(out['rolling_channel_baseline'] + 1)

        # -- Video Duration Features (Including Shorts support) -----------------
        duration_col = self._get_column(df, ['duration_seconds', 'duration'], 'duration')
        out['duration_seconds'] = df[duration_col].apply(self._parse_duration)
        out['duration_minutes'] = out['duration_seconds'] / 60.0
        out['is_shorts'] = (out['duration_seconds'] <= 61).astype(int)
        out['is_optimal_duration'] = (
            (out['duration_minutes'] >= 10.0) & (out['duration_minutes'] <= 15.0)
        ).astype(int)

        # -- Text Metadata Features --------------------------------------------
        out['tags_count'] = df['tags'].apply(self._count_tags)
        
        descriptions = df['description'].fillna("").astype(str)
        out['hashtag_count'] = descriptions.apply(self._count_hashtags)
        out['has_hashtags'] = (out['hashtag_count'] > 0).astype(int)
        out['description_char_count'] = descriptions.str.len()

        # -- Title text features -----------------------------------------------
        titles = df['title'].fillna("").astype(str)
        out['title_char_count'] = titles.str.len()
        out['title_word_count'] = titles.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        out['title_has_number'] = titles.apply(lambda t: int(any(c.isdigit() for c in t)))
        out['title_has_question'] = titles.apply(lambda t: int('?' in t))
        out['title_has_exclamation'] = titles.apply(lambda t: int('!' in t))
        out['title_has_colon'] = titles.apply(lambda t: int(':' in t))
        out['title_has_money_ref'] = titles.apply(
            lambda t: int(bool(re.search(r'[₹]|rupee|crore|lakh|\d+k|\d+cr', t.lower())))
        )
        
        EXCLUDE_CAPS = {'GST', 'GDP', 'INR', 'USD', 'SIP', 'CA', 'IPO', 'EPF', 'PPF', 'NPS', 'FD', 'RD', 'LIC', 'RBI'}
        out['title_has_all_caps'] = titles.apply(
            lambda t: int(any(w.isupper() and len(w) >= 4 and w not in EXCLUDE_CAPS for w in t.split()))
        )
        out['title_emotional_trigger_count'] = titles.apply(self._count_emotional_triggers)
        out['title_is_hindi_or_hinglish'] = titles.apply(self._is_hindi_or_hinglish)
        out['title_mentions_india'] = titles.apply(
            lambda t: int(bool(re.search(r'\bindia\b|\bbharat\b', t.lower())))
        )

        # NLP Sentiment (TextBlob)
        out['title_sentiment_polarity'] = titles.apply(lambda t: TextBlob(t).sentiment.polarity)
        out['title_sentiment_subjectivity'] = titles.apply(lambda t: TextBlob(t).sentiment.subjectivity)

        # -- Temporal features -------------------------------------------------
        pub_dt = df_temp.loc[df.index, 'published_at_dt']
        if pub_dt.dt.tz is None:
            published_ist = pub_dt.dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert('Asia/Kolkata')
        else:
            published_ist = pub_dt.dt.tz_convert('Asia/Kolkata')
        out['publish_hour_ist'] = published_ist.dt.hour.fillna(18).astype(int)
        out['publish_day_of_week'] = published_ist.dt.dayofweek.fillna(2).astype(int)
        out['is_weekend'] = (out['publish_day_of_week'] >= 5).astype(int)
        out['is_prime_time'] = ((out['publish_hour_ist'] >= 17) & (out['publish_hour_ist'] <= 21)).astype(int)

        # -- Creator Presentation Style Attributes (Imputed / Injected) --------
        df_copy = df.copy()
        
        # 1. Impute/Fetch Voice Type
        if 'voice_type' not in df_copy.columns:
            voice_types = []
            for idx, row in df_copy.iterrows():
                cat = row.get('channel_category', 'Successful')
                niche_val = row.get('niche', 'personal_finance')
                # 20% of small channels get AI voice for study/contrast
                if cat == 'Small/Stagnant' and niche_val in ['education_science', 'personal_finance', 'geopolitics_economy']:
                    rng = np.random.RandomState(int(abs(hash(str(row.get('channel_id', idx)))) % 10000))
                    voice_types.append('ai_generated' if rng.uniform() < 0.25 else 'human_own')
                else:
                    voice_types.append('human_own')
            df_copy['voice_type'] = voice_types
            
        # 2. Impute/Fetch Presentation Style
        if 'presentation_style' not in df_copy.columns:
            presentation_styles = []
            for idx, row in df_copy.iterrows():
                niche_val = row.get('niche', 'personal_finance')
                rng = np.random.RandomState(int(abs(hash(str(row.get('channel_id', idx)))) % 10000))
                if niche_val in ['geopolitics_economy', 'education_science']:
                    presentation_styles.append(rng.choice(['faceless_broll', 'animated_visuals'], p=[0.6, 0.4]))
                elif niche_val in ['makeup_beauty', 'vlogging_lifestyle', 'comedy_entertainment']:
                    presentation_styles.append(rng.choice(['face_presenter', 'faceless_broll'], p=[0.8, 0.2]))
                else:
                    presentation_styles.append(rng.choice(['face_presenter', 'faceless_broll'], p=[0.5, 0.5]))
            df_copy['presentation_style'] = presentation_styles

        # 3. Impute/Fetch Editing Quality
        if 'is_highly_edited' not in df_copy.columns:
            is_edited = []
            for idx, row in df_copy.iterrows():
                cat = row.get('channel_category', 'Successful')
                rng = np.random.RandomState(int(abs(hash(str(row.get('channel_id', idx)))) % 10000))
                p_edit = 0.7 if cat == 'Successful' else 0.3
                is_edited.append(int(rng.uniform() < p_edit))
            df_copy['is_highly_edited'] = is_edited

        # Manual One-Hot Encode categories to ensure matching shapes
        niches_list = [
            "personal_finance", "geopolitics_economy", "tech_gadgets", "gaming",
            "comedy_entertainment", "education_science", "makeup_beauty", "vlogging_lifestyle"
        ]
        for niche in niches_list:
            out[f'niche_{niche}'] = (df_copy['niche'] == niche).astype(int)
            
        voice_types_list = ['human_own', 'ai_generated', 'music_only']
        for vt in voice_types_list:
            out[f'voice_type_{vt}'] = (df_copy['voice_type'] == vt).astype(int)
            
        styles_list = ['face_presenter', 'faceless_broll', 'animated_visuals']
        for ps in styles_list:
            out[f'presentation_style_{ps}'] = (df_copy['presentation_style'] == ps).astype(int)
            
        out['is_highly_edited'] = df_copy['is_highly_edited'].astype(int)

        # -- Post-publish engagement features (EDA only, not used in features_list) --
        like_col = self._get_column(df, ['like_count', 'likes'], None)
        comment_col = self._get_column(df, ['comment_count', 'comments'], None)
        
        if like_col:
            out['like_count'] = pd.to_numeric(df[like_col], errors='coerce').fillna(0)
            out['like_rate'] = out['like_count'] / (out['views'] + 1)
        else:
            out['like_count'] = 0
            out['like_rate'] = 0.0

        if comment_col:
            out['comment_count'] = pd.to_numeric(df[comment_col], errors='coerce').fillna(0)
            out['comment_rate'] = out['comment_count'] / (out['views'] + 1)
        else:
            out['comment_count'] = 0
            out['comment_rate'] = 0.0
            
        out['engagement_rate'] = out['like_rate'] + out['comment_rate']

        return out
