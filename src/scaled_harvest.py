import os
import sys
import time

# Ensure output streams use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from googleapiclient.discovery import build

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class YouTubeScaledHarvester:
    """
    Automated high-scale YouTube dataset harvester.
    Searches channels, balances successful vs small creators,
    and harvests up to 500 videos per channel across 8 niches.
    """
    def __init__(self):
        if not config.YOUTUBE_API_KEY:
            raise ValueError("[ERROR] YOUTUBE_API_KEY is not defined in the environment or config.py")
        
        self.youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)
        self.quota_used = 0

    def log_quota(self, units):
        self.quota_used += units
        print(f"   [Quota Info] Used {units} units. Session total: {self.quota_used} units.")

    def search_channels(self, niche_name, keywords):
        """Search for channel IDs using niche keywords."""
        print(f"\n[Search] Searching channels for Niche: '{niche_name}'...")
        channel_ids = set()
        
        for kw in keywords[:2]:  # Use top 2 keywords to save quota
            try:
                request = self.youtube.search().list(
                    q=kw,
                    type="channel",
                    part="snippet",
                    maxResults=25
                )
                response = request.execute()
                self.log_quota(100)  # Search costs 100 units
                
                for item in response.get("items", []):
                    cid = item["snippet"]["channelId"]
                    channel_ids.add(cid)
            except Exception as e:
                print(f"  [ERROR] Channel search failed for keyword '{kw}': {e}")
                
            time.sleep(0.2)
            
        print(f"  Discovered {len(channel_ids)} unique channel candidates.")
        return list(channel_ids)

    def get_channels_details(self, channel_ids):
        """Fetch details (subs, views, creation, uploads playlist) for channels."""
        print(f"[Info] Fetching details for {len(channel_ids)} channels...")
        details = []
        
        # Batch requests in chunks of 50
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i+50]
            try:
                request = self.youtube.channels().list(
                    id=",".join(batch),
                    part="snippet,statistics,contentDetails"
                )
                response = request.execute()
                self.log_quota(1)
                
                for item in response.get("items", []):
                    stats = item.get("statistics", {})
                    snippet = item.get("snippet", {})
                    content = item.get("contentDetails", {})
                    
                    details.append({
                        "channel_id": item["id"],
                        "channel_name": snippet.get("title"),
                        "channel_created": snippet.get("publishedAt"),
                        "subscriber_count": int(stats.get("subscriberCount", 0)),
                        "total_videos": int(stats.get("videoCount", 0)),
                        "total_views": int(stats.get("viewCount", 0)),
                        "uploads_playlist_id": content.get("relatedPlaylists", {}).get("uploads")
                    })
            except Exception as e:
                print(f"  [ERROR] Failed to fetch channel details: {e}")
                
        return details

    def select_balanced_channels(self, channels_details):
        """Sort and select 5 successful and 5 small channels."""
        # Filter out empty channels or those without uploads playlist
        valid_channels = [c for c in channels_details if c["uploads_playlist_id"] and c["total_videos"] >= 10]
        
        if not valid_channels:
            return [], []
            
        # Sort by subscribers descending
        valid_channels.sort(key=lambda x: x["subscriber_count"], reverse=True)
        
        # If we have less than 10 channels, split 50/50
        if len(valid_channels) < 10:
            mid = len(valid_channels) // 2
            successful = valid_channels[:mid]
            small = valid_channels[mid:]
        else:
            successful = valid_channels[:5]
            # Small channels: select the bottom 5 that have at least 1,000 subscribers
            potential_small = [c for c in valid_channels[5:] if c["subscriber_count"] >= 1000]
            if len(potential_small) >= 5:
                small = potential_small[-5:]
            else:
                small = valid_channels[-5:]
                
        return successful, small

    def harvest_uploads_playlist(self, playlist_id, max_videos):
        """Retrieve video IDs from a channel's uploads playlist."""
        video_ids = []
        next_page_token = None
        
        while len(video_ids) < max_videos:
            try:
                request = self.youtube.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                self.log_quota(1)
                
                items = response.get("items", [])
                if not items:
                    break
                    
                for item in items:
                    video_ids.append(item["contentDetails"]["videoId"])
                    
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
                time.sleep(0.1)
            except Exception as e:
                print(f"  [ERROR] Playlist items fetch failed: {e}")
                break
                
        return video_ids[:max_videos]

    def fetch_videos_details(self, video_ids):
        """Batch fetch statistics and snippet metadata for videos."""
        records = []
        
        # Batch in chunks of 50
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            try:
                request = self.youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(batch)
                )
                response = request.execute()
                self.log_quota(1)
                
                for item in response.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    content = item.get("contentDetails", {})
                    
                    title = snippet.get("title", "")
                    description = snippet.get("description", "")
                    tags = snippet.get("tags", [])
                    published_at = snippet.get("publishedAt", "")
                    duration = content.get("duration", "PT0S")
                    
                    records.append({
                        "video_id": item["id"],
                        "channel_id": snippet.get("channelId"),
                        "channel_title": snippet.get("channelTitle"),
                        "title": title,
                        "description": description,
                        "tags": "|".join(tags) if isinstance(tags, list) else "",
                        "published_at": published_at,
                        "duration": duration,
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                    })
            except Exception as e:
                print(f"  [ERROR] Failed to fetch video details: {e}")
                
            time.sleep(0.05)
            
        return pd.DataFrame(records)

    def harvest_all(self):
        """Main pipeline to crawl and harvest data across all 8 niches."""
        all_dfs = []
        niche_summary = []
        
        for niche_id, niche_info in config.NICHES.items():
            print("\n" + "="*80)
            print(f" PROCESSING NICHE: {niche_info['name'].upper()}")
            print("="*80)
            
            # Discover channels
            raw_channel_ids = self.search_channels(niche_info['name'], niche_info['keywords'])
            if not raw_channel_ids:
                print(f"No channels found for niche: {niche_id}")
                continue
                
            # Get statistics and uploads playlists
            channels_details = self.get_channels_details(raw_channel_ids)
            
            # Select balanced set
            succ_channels, small_channels = self.select_balanced_channels(channels_details)
            selected_channels = succ_channels + small_channels
            
            print(f"\nSelected {len(selected_channels)} balanced channels:")
            for ch in selected_channels:
                category = "Successful" if ch in succ_channels else "Small/Stagnant"
                print(f"  - [{category}] {ch['channel_name']} ({ch['subscriber_count']:,} subs, {ch['total_videos']} vids)")
            
            # Harvest uploads for each channel
            for ch in selected_channels:
                category = "Successful" if ch in succ_channels else "Small/Stagnant"
                print(f"\n[Harvest] Harvesting {ch['channel_name']} ({category})...")
                
                # Fetch uploads playlist video IDs
                video_ids = self.harvest_uploads_playlist(ch["uploads_playlist_id"], config.MAX_VIDEOS_PER_CHANNEL)
                print(f"  Retrieved {len(video_ids)} video IDs from uploads playlist.")
                
                if not video_ids:
                    continue
                    
                # Fetch details for those videos
                df_vids = self.fetch_videos_details(video_ids)
                if df_vids.empty:
                    continue
                    
                # Add channel details
                df_vids["channel_subscriber_count"] = ch["subscriber_count"]
                df_vids["channel_total_videos"] = ch["total_videos"]
                df_vids["channel_total_views"] = ch["total_views"]
                df_vids["channel_created"] = ch["channel_created"]
                df_vids["niche"] = niche_id
                df_vids["channel_category"] = category
                
                all_dfs.append(df_vids)
                niche_summary.append({
                    "niche": niche_id,
                    "channel_name": ch["channel_name"],
                    "category": category,
                    "subscribers": ch["subscriber_count"],
                    "videos_collected": len(df_vids)
                })
                
                # Sleep to prevent hitting rate limits
                time.sleep(0.5)

        if not all_dfs:
            print("\n[CRITICAL] No data was collected.")
            return
            
        final_df = pd.concat(all_dfs, ignore_index=True)
        print("\n" + "="*80)
        print(" HARVEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Total videos collected: {len(final_df):,}")
        print(f"Total unique channels:  {final_df['channel_id'].nunique()}")
        print(f"Total session quota:    {self.quota_used} units")
        
        # Save output
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "scaled_niche")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "videos.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved dataset to: {out_path}")
        
        # Save summary report
        summary_path = os.path.join(out_dir, "harvest_summary.csv")
        pd.DataFrame(niche_summary).to_csv(summary_path, index=False)
        print(f"Saved harvest summary to: {summary_path}\n")

if __name__ == "__main__":
    try:
        harvester = YouTubeScaledHarvester()
        harvester.harvest_all()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
