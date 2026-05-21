import sys
import os

# Ensure src/ is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.recommender import YouTubeRecommender

def main():
    print("=" * 60)
    print("      YOUTUBE VIDEO PERFORMANCE PREDICTION TOOL")
    print("=" * 60)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "model.joblib")
    try:
        recommender = YouTubeRecommender.load(model_path)
    except Exception as e:
        print(f"Error loading model from src/model.joblib: {e}")
        print("Please build the model first using the notebooks or ensure src/model.joblib is present.")
        sys.exit(1)
        
    title = input("Enter proposed video title: ").strip()
    if not title:
        print("Error: Video title cannot be empty.")
        sys.exit(1)
        
    print("\nSelect Niche:")
    print("  [1] Personal Finance India")
    print("  [2] Geopolitics & Economy")
    print("  [3] Tech & Gadgets")
    print("  [4] Gaming")
    print("  [5] Comedy & Entertainment")
    print("  [6] Education & Science")
    print("  [7] Makeup & Beauty")
    print("  [8] Vlogging & Lifestyle")
    niche_choice = input("Choice [1-8, default 1]: ").strip()
    
    niche_map = {
        '1': 'personal_finance',
        '2': 'geopolitics_economy',
        '3': 'tech_gadgets',
        '4': 'gaming',
        '5': 'comedy_entertainment',
        '6': 'education_science',
        '7': 'makeup_beauty',
        '8': 'vlogging_lifestyle'
    }
    niche = niche_map.get(niche_choice, 'personal_finance')

    duration_input = input("\nEnter video duration in minutes [default 12.0]: ").strip()
    try:
        duration_minutes = float(duration_input) if duration_input else 12.0
    except ValueError:
        print("Invalid duration, using default 12.0 minutes.")
        duration_minutes = 12.0

    print("\nSelect Voice Type:")
    print("  [1] Natural Human narration (Own Voice) [default]")
    print("  [2] AI-generated text-to-speech")
    print("  [3] Background Music only (no voice)")
    voice_choice = input("Choice [1-3, default 1]: ").strip()
    voice_map = {'1': 'human_own', '2': 'ai_generated', '3': 'music_only'}
    voice_type = voice_map.get(voice_choice, 'human_own')

    print("\nSelect Presentation Style:")
    print("  [1] Face Presenter (Talking Head / On-camera) [default]")
    print("  [2] Faceless B-Roll (footage / slide deck)")
    print("  [3] Animated Visuals (motion graphics / custom assets)")
    style_choice = input("Choice [1-3, default 1]: ").strip()
    style_map = {'1': 'face_presenter', '2': 'faceless_broll', '3': 'animated_visuals'}
    presentation_style = style_map.get(style_choice, 'face_presenter')

    print("\nIs the video highly edited?")
    print("  [1] Yes (motion graphics, sound design, fast pacing) [default]")
    print("  [2] No (simple cuts, raw presentation)")
    edit_choice = input("Choice [1-2, default 1]: ").strip()
    is_highly_edited = 0 if edit_choice == '2' else 1

    print("\nPlanned upload hour (in Indian Standard Time - IST):")
    hour_input = input("Enter hour [0-23, default 19 (7 PM)]: ").strip()
    try:
        publish_hour_ist = int(hour_input) if hour_input else 19
    except ValueError:
        print("Invalid hour, using default 19 IST.")
        publish_hour_ist = 19

    print("\nPlanned upload day of the week:")
    print("  0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday")
    day_input = input("Enter day [0-6, default 2 (Wednesday)]: ").strip()
    try:
        publish_day_of_week = int(day_input) if day_input else 2
    except ValueError:
        print("Invalid day, using default 2 (Wednesday).")
        publish_day_of_week = 2

    subs_input = input("\nEnter current channel subscribers [default 10000]: ").strip()
    try:
        subscribers = int(subs_input) if subs_input else 10000
    except ValueError:
        print("Invalid subscriber count, using default 10000.")
        subscribers = 10000

    baseline_input = input("\nEnter rolling channel baseline (median views of preceding 10 uploads, optional): ").strip()
    try:
        rolling_channel_baseline = float(baseline_input) if baseline_input else None
    except ValueError:
        print("Invalid baseline, using default auto-calculation.")
        rolling_channel_baseline = None

    seq_input = input("\nEnter video sequence number (upload experience index, default 100): ").strip()
    try:
        video_seq = int(seq_input) if seq_input else 100
    except ValueError:
        print("Invalid sequence, using default 100.")
        video_seq = 100

    age_input = input("\nEnter channel age in days (default 365): ").strip()
    try:
        channel_age_days = int(age_input) if age_input else 365
    except ValueError:
        print("Invalid age, using default 365.")
        channel_age_days = 365

    description = input("\nEnter brief draft description (optional): ").strip()

    print("\nGenerating prediction report...\n")

    result = recommender.predict(
        title=title,
        duration_minutes=duration_minutes,
        publish_hour_ist=publish_hour_ist,
        publish_day_of_week=publish_day_of_week,
        niche=niche,
        subscribers=subscribers,
        voice_type=voice_type,
        presentation_style=presentation_style,
        is_highly_edited=is_highly_edited,
        description=description,
        rolling_channel_baseline=rolling_channel_baseline,
        video_seq=video_seq,
        channel_age_days=channel_age_days
    )

    recommender.print_report(result)
    
if __name__ == "__main__":
    main()
