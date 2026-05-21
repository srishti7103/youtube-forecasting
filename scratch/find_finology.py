import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=api_key)

response = youtube.search().list(
    q="Finology Ticker",
    type="channel",
    part="id,snippet",
    maxResults=3
).execute()

for item in response.get("items", []):
    print("Channel Title:", item["snippet"]["title"])
    print("Channel ID:", item["id"]["channelId"])
