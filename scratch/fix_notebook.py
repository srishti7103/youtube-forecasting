import json
import os

def fix_notebook():
    notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "01_niche_analysis.ipynb"))
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    
    # 1. Update Cell 1 (index 0) - Imports and Niche structures
    if len(cells) > 0:
        cells[0]['source'] = [
            "# Cell 1 — Imports\n",
            "import sys\n",
            "sys.path.append(\"..\")\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from src.collector import YouTubeCollector\n",
            "from src.niche_analyzer import NicheAnalyzer\n",
            "from config import MAX_VIDEOS_PER_CHANNEL\n",
            "\n",
            "POPULAR_NICHE = {\n",
            "    \"name\": \"Personal Finance India\",\n",
            "    \"channels\": [\n",
            "        \"UCwAdQUuPT6laN-AQR17fe1g\",\n",
            "        \"UCe3qdG0A_gr-sEdat5y2twQ\",\n",
            "        \"UCVOTBwF0vnSxMRIbfSE_K_g\",\n",
            "        \"UCwVEhEzsjLym_u1he4XWFkg\",\n",
            "        \"UCqW8jxh4tH1Z1sWPbkGWL4g\"\n",
            "    ]\n",
            "}\n",
            "\n",
            "UNDERSERVED_NICHE = {\n",
            "    \"name\": \"India Economy and Data Stories\",\n",
            "    \"channels\": [\n",
            "        \"UCIfANh2BltdF-XixRwKFRqQ\",\n",
            "        \"UC_gUM8rL-Lrg6O3adPW9K1g\",\n",
            "        \"UCKZozRVHRYsYHGEyNKuhhdA\",\n",
            "        \"UCNXapAc8mXTwW82MTncdfzQ\",\n",
            "        \"UCR-foyF-C6VuAlwy3KZMkgA\"\n",
            "    ]\n",
            "}\n"
        ]
        print("Cell 0 (Imports) updated.")

    # 2. Update Cell 2 (index 1) - Popular niche collection
    if len(cells) > 1:
        cells[1]['source'] = [
            "# Cell 2 — Collect data for both niches\n",
            "collector = YouTubeCollector()\n",
            "\n",
            "popular_df = collector.collect_niche(\n",
            "    POPULAR_NICHE,\n",
            "    max_videos_per_channel=MAX_VIDEOS_PER_CHANNEL\n",
            ")\n",
            "popular_df.to_csv(\"../data/raw/popular_niche/videos.csv\", index=False)\n",
            "print(f\"Popular niche: {len(popular_df)} videos saved\")"
        ]
        print("Cell 1 (Popular) updated.")

    # 3. Update Cell 3 (index 2) - Underserved niche collection
    if len(cells) > 2:
        cells[2]['source'] = [
            "# Cell 3\n",
            "underserved_df = collector.collect_niche(\n",
            "    UNDERSERVED_NICHE,\n",
            "    max_videos_per_channel=MAX_VIDEOS_PER_CHANNEL\n",
            ")\n",
            "underserved_df.to_csv(\"../data/raw/underserved_niche/videos.csv\", index=False)\n",
            "print(f\"Underserved niche: {len(underserved_df)} videos saved\")"
        ]
        print("Cell 2 (Underserved) updated.")
        
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("Notebook updated successfully.")

if __name__ == "__main__":
    fix_notebook()
