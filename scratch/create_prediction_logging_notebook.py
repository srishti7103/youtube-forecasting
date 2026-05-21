import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# 05. Pre-Publish Prediction Logging\n\n"
        "This notebook is used **before publishing** any new video on your experimental channels. "
        "By entering the details of your video idea, the model estimates its virality multiplier and view performance. "
        "The prediction is logged to `predictions/predictions_log.csv` to establish the baseline for scientific re-validation."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import os\n"
        "import sys\n"
        "import joblib\n\n"
        "# Add parent directory to path so we can import src\n"
        "sys.path.append('..')\n"
        "from src.features import YouTubeFeatureExtractor"
    ),
    nbf.v4.new_code_cell(
        "# Load trained model artifacts\n"
        "try:\n"
        "    model_data = joblib.load('../src/model.joblib')\n"
        "    model = model_data['model']\n"
        "    features_list = model_data['features_list']\n"
        "    extractor = model_data['extractor']\n"
        "    print('Model and Feature Extractor loaded successfully!')\n"
        "except FileNotFoundError:\n"
        "    print('Error: model.joblib not found. Run 04_model_building.ipynb first.')"
    ),
    nbf.v4.new_markdown_cell(
        "## Prediction Logging Function\n\n"
        "Run the following cell to define the logging function."
    ),
    nbf.v4.new_code_cell(
        "def log_video_prediction(channel_name, title, duration_minutes, publish_hour_ist, publish_day_of_week=2, is_underserved=1, subscribers=0):\n"
        "    # Create mock dataframe row for feature extractor\n"
        "    mock_row = pd.DataFrame([{\n"
        "        'title': title,\n"
        "        'description': '',\n"
        "        'tags': '[]',\n"
        "        'duration_seconds': duration_minutes * 60,\n"
        "        'published_at': pd.Timestamp.now(tz='UTC').isoformat(),\n"
        "        'view_count': 0,\n"
        "        'channel_subscriber_count': subscribers,\n"
        "        'channel_id': 'mock_channel_id'\n"
        "    }])\n"
        "    \n"
        "    # Extract features using globally fitted extractor\n"
        "    features = extractor.transform(mock_row)\n"
        "    \n"
        "    # Override temporal features explicitly to match parameters\n"
        "    features['publish_hour_ist'] = publish_hour_ist\n"
        "    features['publish_day_of_week'] = publish_day_of_week\n"
        "    features['is_weekend'] = int(publish_day_of_week >= 5)\n"
        "    features['is_underserved'] = is_underserved\n"
        "    \n"
        "    X_pred = features[features_list]\n"
        "    pred_virality = model.predict(X_pred)[0]\n"
        "    \n"
        "    # Convert log10 target back to views and multiplier\n"
        "    multiplier = 10**pred_virality - 0.01\n"
        "    median_baseline = extractor.niche_median_views_\n"
        "    estimated_views = max(0, multiplier * median_baseline)\n"
        "    \n"
        "    log_path = '../predictions/predictions_log.csv'\n"
        "    os.makedirs('../predictions', exist_ok=True)\n"
        "    \n"
        "    new_entry = {\n"
        "        'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),\n"
        "        'channel_name': channel_name,\n"
        "        'title': title,\n"
        "        'duration_minutes': duration_minutes,\n"
        "        'publish_hour_ist': publish_hour_ist,\n"
        "        'publish_day_of_week': publish_day_of_week,\n"
        "        'is_underserved': is_underserved,\n"
        "        'subscribers_at_pred': subscribers,\n"
        "        'predicted_virality_multiplier': round(multiplier, 3),\n"
        "        'predicted_views': int(estimated_views),\n"
        "        'actual_views': np.nan,\n"
        "        'actual_virality_multiplier': np.nan,\n"
        "        'is_revalidated': 0,\n"
        "        'published_video_id': ''\n"
        "    }\n"
        "    \n"
        "    if os.path.exists(log_path):\n"
        "        log_df = pd.read_csv(log_path)\n"
        "        log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)\n"
        "    else:\n"
        "        log_df = pd.DataFrame([new_entry])\n"
        "        \n"
        "    log_df.to_csv(log_path, index=False)\n"
        "    print(f'Successfully logged prediction for video Idea: \"{title}\"')\n"
        "    print(f'Predicted views: {int(estimated_views):,} | Multiplier: {multiplier:.2f}x')\n"
        "    return new_entry"
    ),
    nbf.v4.new_markdown_cell(
        "## Log Predictions for Your Next Videos\n\n"
        "Let's pre-populate the log file with predictions for your first 3 scheduled videos on the two new experimental channels."
    ),
    nbf.v4.new_code_cell(
        "# Experimental Channel A: India Data Stories (Underserved Niche)\n"
        "log_video_prediction(\n"
        "    channel_name='India Data Stories',\n"
        "    title='Why India is running out of water: 5 Shocking Facts',\n"
        "    duration_minutes=12.5,\n"
        "    publish_hour_ist=19,\n"
        "    publish_day_of_week=2, # Wednesday\n"
        "    is_underserved=1,\n"
        "    subscribers=0\n"
        ")\n\n"
        "# Experimental Channel B: Student Money Academy (Personal Finance Niche)\n"
        "log_video_prediction(\n"
        "    channel_name='Student Money Academy',\n"
        "    title='5 Student Finance Secrets they don\\'t teach you in school',\n"
        "    duration_minutes=10.0,\n"
        "    publish_hour_ist=18,\n"
        "    publish_day_of_week=4, # Friday\n"
        "    is_underserved=0,\n"
        "    subscribers=0\n"
        ")\n\n"
        "# Let's check the created logs\n"
        "log_df = pd.read_csv('../predictions/predictions_log.csv')\n"
        "display(log_df)"
    )
]

nb['cells'] = cells

with open('notebooks/05_prediction_logging.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created notebooks/05_prediction_logging.ipynb successfully!")
