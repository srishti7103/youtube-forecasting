import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=api_key)

channel_ids = [
    # Popular
    "UCwAdQUuPT6laN-AQR17fe1g",  # Pranjal Kamra
    "UCe3qdG0A_gr-sEdat5y2twQ",  # CA Rachana Ranade
    "UCVOTBwF0vnSxMRIbfSE_K_g",  # Labour Law Advisor
    "UCwVEhEzsjLym_u1he4XWFkg",  # Sharan Hegde
    "UCqW8jxh4tH1Z1sWPbkGWL4g",  # Akshat Shrivastava
    # Underserved
    "UCIfANh2BltdF-XixRwKFRqQ",  # 1Finance
    "UC_gUM8rL-Lrg6O3adPW9K1g",  # Wion
    "UCKZozRVHRYsYHGEyNKuhhdA",  # Think School
    "UCUaB-S9K3D2r4-5pP7x6f3A",  # Finology
    "UCR-foyF-C6VuAlwy3KZMkgA"   # Vivek Bindra
]

response = youtube.channels().list(
    part="snippet,statistics,contentDetails",
    id=",".join(channel_ids)
).execute()

items = response.get("items", [])
print(f"Retrieved {len(items)} channels out of {len(channel_ids)}")
for idx, item in enumerate(items):
    name = item["snippet"]["title"]
    subs = item["statistics"].get("subscriberCount", "0")
    uploads = item["contentDetails"]["relatedPlaylists"].get("uploads", "")
    print(f"[{idx+1}] Name: '{name}' | Subs: {subs} | Uploads Playlist: '{uploads}'")
