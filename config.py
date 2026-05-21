import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# 8 Niches and their query keywords for automated channel discovery
NICHES = {
    "personal_finance": {
        "name": "Personal Finance India",
        "keywords": ["personal finance India", "mutual funds tax India", "SIP investment India", "save money India"]
    },
    "geopolitics_economy": {
        "name": "Geopolitics & Economy",
        "keywords": ["India geopolitics", "Indian economy analysis", "business case studies India", "geopolitics case study"]
    },
    "tech_gadgets": {
        "name": "Tech & Gadgets",
        "keywords": ["smartphone review India", "unboxing tech Hindi", "gadget reviews India", "tech news India"]
    },
    "gaming": {
        "name": "Gaming",
        "keywords": ["pubg mobile India gameplay", "free fire stream India", "gaming channel Hindi", "PC games India"]
    },
    "comedy_entertainment": {
        "name": "Comedy & Entertainment",
        "keywords": ["funny comedy sketch Hindi", "standup comedy India", "roasting channel India", "memes comedy Hindi"]
    },
    "education_science": {
        "name": "Education & Science",
        "keywords": ["science explainers Hindi", "space facts India", "education animation India", "knowledge facts Hindi"]
    },
    "makeup_beauty": {
        "name": "Makeup & Beauty",
        "keywords": ["makeup tutorial India", "beauty product review India", "skincare routine Hindi", "makeup tips India"]
    },
    "vlogging_lifestyle": {
        "name": "Vlogging & Lifestyle",
        "keywords": ["daily vlog India", "travel vlogger Hindi", "lifestyle routine India", "minivlogs India"]
    }
}

# Harvesting configuration
MAX_VIDEOS_PER_CHANNEL = 500

# Channel mix criteria
SUCC_MIN_SUBS = 500_000      # Target > 500k subscribers for successful channels
SMALL_MIN_SUBS = 5_000       # Target > 5k subscribers for small channels
SMALL_MAX_SUBS = 50_000      # Target < 50k subscribers for small/stagnant channels
