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
            "UCwAdQUuPT6laN-AQR17fe1g",  # Pranjal Kamra
            "UCe3qdG0A_gr-sEdat5y2twQ",  # CA Rachana Ranade
            "UCVOTBwF0vnSxMRIbfSE_K_g",  # Labour Law Advisor
            "UCwVEhEzsjLym_u1he4XWFkg",  # Sharan Hegde
            "UCqW8jxh4tH1Z1sWPbkGWL4g",  # Akshat Shrivastava
        ]
    },
    "underserved": {
        "name": "India Economy and Data Stories",
        "keywords": ["India economy explained", "India statistics facts", "Indian data stories"],
        "channels": [
            "UCIfANh2BltdF-XixRwKFRqQ",  # 1Finance
            "UC_gUM8rL-Lrg6O3adPW9K1g",  # Wion
            "UCKZozRVHRYsYHGEyNKuhhdA",  # Think School
            "UCNXapAc8mXTwW82MTncdfzQ",  # Finology
            "UCR-foyF-C6VuAlwy3KZMkgA",  # Vivek Bindra (for contrast)
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
