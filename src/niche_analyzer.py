import pandas as pd
import numpy as np

class NicheAnalyzer:
    """
    Takes collected video data and scores a niche across 5 dimensions.
    Compares popular vs underserved to guide content strategy.
    """

    def __init__(self, df: pd.DataFrame, niche_name: str):
        self.df = df.copy()
        self.niche_name = niche_name
        self._clean()

    def _clean(self):
        # Remove videos with 0 views (private/deleted but still in playlist)
        self.df = self.df[self.df["view_count"] > 0]

        # Remove Shorts (under 60 seconds) — different algorithm entirely
        self.df = self.df[self.df["duration_seconds"] >= 60]

        # Only keep videos at least 14 days old (views still stabilising before that)
        self.df["published_at"] = pd.to_datetime(self.df["published_at"], utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14)
        self.df = self.df[self.df["published_at"] < cutoff]

    def score_niche(self) -> dict:
        """Calculate all 5 niche scores."""
        small = self.df[self.df["channel_subscriber_count"] < 50_000]
        large = self.df[self.df["channel_subscriber_count"] >= 50_000]

        # 1. Audience demand score — median views across ALL channels
        demand_score = self.df["view_count"].median()

        # 2. New creator opportunity — median views for channels under 50k subs
        # This is the most important number for you
        new_creator_score = small["view_count"].median() if len(small) > 0 else 0

        # 3. Competition intensity — how many videos uploaded per month
        months_span = (
            self.df["published_at"].max() - self.df["published_at"].min()
        ).days / 30
        competition_score = len(self.df) / max(months_span, 1)

        # 4. Viral potential — % of videos that got 10x the median views
        median_views = self.df["view_count"].median()
        viral_threshold = median_views * 10
        viral_pct = len(self.df[self.df["view_count"] > viral_threshold]) / len(self.df) * 100

        # 5. Engagement quality — avg likes / views ratio
        self.df["engagement_rate"] = (
            self.df["like_count"] / self.df["view_count"].replace(0, np.nan)
        )
        engagement_score = self.df["engagement_rate"].median() * 100

        return {
            "niche": self.niche_name,
            "videos_analysed": len(self.df),
            "demand_score_median_views": round(demand_score),
            "new_creator_opportunity_median_views": round(new_creator_score),
            "competition_videos_per_month": round(competition_score, 1),
            "viral_potential_pct": round(viral_pct, 2),
            "engagement_rate_pct": round(engagement_score, 3),
            "opportunity_ratio": round(demand_score / max(competition_score, 1)),
        }

    def best_performing_features(self) -> dict:
        """Find what features top-performing videos share in this niche."""
        top_25 = self.df.nlargest(int(len(self.df) * 0.25), "view_count")
        bottom_75 = self.df[~self.df["video_id"].isin(top_25["video_id"])]

        findings = {
            "top_videos_avg_duration_mins": round(top_25["duration_seconds"].mean() / 60, 1),
            "bottom_videos_avg_duration_mins": round(bottom_75["duration_seconds"].mean() / 60, 1),
            "top_videos_pct_with_number_in_title": round(top_25["title_has_number"].mean() * 100, 1),
            "bottom_videos_pct_with_number_in_title": round(bottom_75["title_has_number"].mean() * 100, 1),
            "top_videos_avg_title_length": round(top_25["title_length"].mean(), 1),
            "bottom_videos_avg_title_length": round(bottom_75["title_length"].mean(), 1),
            "best_upload_hour_ist": top_25["publish_hour_ist"].mode()[0] if len(top_25) > 0 else None,
            "best_upload_day": top_25["publish_day_of_week"].mode()[0] if len(top_25) > 0 else None,
            "top_videos_avg_tags_count": round(top_25["tags_count"].mean(), 1),
        }

        return findings
