import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=api_key)

channel_queries = {
    "Pranjal Kamra": "Pranjal Kamra",
    "CA Rachana Ranade": "CA Rachana Ranade",
    "Labour Law Advisor": "Labour Law Advisor",
    "Sharan Hegde": "Sharan Hegde",
    "Akshat Shrivastava": "Akshat Shrivastava",
    "1Finance": "1Finance",
    "Wion": "WION",
    "Think School": "Think School",
    "Finology": "Finology",
    "Vivek Bindra": "Dr. Vivek Bindra"
}

resolved = {}

for name, query in channel_queries.items():
    try:
        response = youtube.search().list(
            q=query,
            type="channel",
            part="id,snippet",
            maxResults=1
        ).execute()
        
        items = response.get("items", [])
        if items:
            channel_id = items[0]["id"]["channelId"]
            title = items[0]["snippet"]["title"]
            resolved[name] = (channel_id, title)
            print(f"Resolved '{name}' -> '{title}' (ID: {channel_id})")
        else:
            print(f"Could not resolve '{name}'")
    except Exception as e:
        print(f"Error resolving '{name}': {e}")

print("\n--- Copy and paste this into config.py ---")
print(resolved)
