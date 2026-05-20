import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Niches you are testing
NICHES = {
    "popular": {
        "name": "Personal Finance India",
        "keywords": ["mutual funds India", "save tax India", "SIP investment"],
        "channels": [
            "UCy3d3hmt2-UKNkPEZqAXahg",  # Pranjal Kamra
            "UCmA5OxhY5PVpGMvkBUFkMEQ",  # CA Rachana Ranade
            "UCvr6pNHmkIVe-EBb1Dz-aBw",  # Labour Law Advisor
            "UC3pzSBV16R1GKBK9V6OXoGw",  # Sharan Hegde
            "UCXBMblREgIsFUZrKWCqnIaQ",  # Akshat Shrivastava
        ]
    },
    "underserved": {
        "name": "India Economy and Data Stories",
        "keywords": ["India economy explained", "India statistics facts", "Indian data stories"],
        "channels": [
            "UCHqSxMJDLf5CXBbkqzuiGcQ",  # 1Finance
            "UCBwbY5hGhbMEJRtB9U5JKJA",  # Wion
            "UC-BaQBDRWqqFTJWz27Xkz5Q",  # Think School
            "UCVOPQIv7MBiONdO8qTFwVVQ",  # Finology
            "UCVEFg5XPVbUVfF_8bkZfkNQ",  # Vivek Bindra (for contrast)
        ]
    }
}

# API quota per day: 10,000 units
# search.list = 100 units each — use sparingly
# videos.list = 1 unit per 50 videos — use freely
# channels.list = 1 unit per 50 channels — use freely

MAX_VIDEOS_PER_CHANNEL = 100
MIN_CHANNEL_SUBS = 5_000
MAX_CHANNEL_SUBS = 800_000
