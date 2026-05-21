import pandas as pd
import numpy as np

def main():
    pop = pd.read_csv('data/processed/features_popular.csv')
    und = pd.read_csv('data/processed/features_underserved.csv')
    
    print("=" * 65)
    print("           SHORTS VS. LONG-FORM COMPARATIVE ANALYSIS")
    print("=" * 65)
    
    for df, name in [(pop, "Personal Finance India (Popular)"), (und, "India Economy & Data Stories (Underserved)")]:
        print(f"\nNICHE: {name}")
        print("-" * 50)
        
        # Split into Shorts and Long-form
        shorts = df[df['is_shorts'] == 1]
        long_form = df[df['is_shorts'] == 0]
        
        print(f"Total Videos: {len(df)}")
        print(f"  * Shorts (<= 60s): {len(shorts)} videos ({len(shorts)/len(df)*100:.1f}%)")
        print(f"  * Long-form (> 60s): {len(long_form)} videos ({len(long_form)/len(df)*100:.1f}%)")
        
        print("\nMedian View Counts:")
        print(f"  * Shorts Median Views: {shorts['views'].median():,.0f}" if len(shorts) > 0 else "  * Shorts Median Views: N/A")
        print(f"  * Long-form Median Views: {long_form['views'].median():,.0f}" if len(long_form) > 0 else "  * Long-form Median Views: N/A")
        
        print("\nAverage View Counts:")
        print(f"  * Shorts Avg Views: {shorts['views'].mean():,.0f}" if len(shorts) > 0 else "  * Shorts Avg Views: N/A")
        print(f"  * Long-form Avg Views: {long_form['views'].mean():,.0f}" if len(long_form) > 0 else "  * Long-form Avg Views: N/A")
        
        # Calculate Virality Outliers (videos with views > 3x channel median views)
        # Using virality_ratio_median > 3.0
        viral_shorts = shorts[shorts['views'] > shorts['channel_median_views'] * 3.0]
        viral_long = long_form[long_form['views'] > long_form['channel_median_views'] * 3.0]
        
        print("\nVirality Success Rate (Outperforming Channel Median by >3x):")
        pct_viral_shorts = (len(viral_shorts) / len(shorts) * 100) if len(shorts) > 0 else 0
        pct_viral_long = (len(viral_long) / len(long_form) * 100) if len(long_form) > 0 else 0
        print(f"  * Shorts Virality Rate: {pct_viral_shorts:.2f}% ({len(viral_shorts)}/{len(shorts)})")
        print(f"  * Long-form Virality Rate: {pct_viral_long:.2f}% ({len(viral_long)}/{len(long_form)})")
        
    print("=" * 65)

if __name__ == "__main__":
    main()
