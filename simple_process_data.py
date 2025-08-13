#!/usr/bin/env python3
"""
Simple processing of existing data with basic misogyny detection.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging

from text_processing import TextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_process_data():
    """Process all existing data files with basic misogyny detection."""
    print("🔬 Simple MisogynyWatch Data Processing")
    print("=" * 40)
    
    # Data directories
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Find all data files
    reddit_files = list(raw_dir.glob("reddit_*.csv"))
    twitter_files = list(raw_dir.glob("twitter_*.csv"))
    
    print(f"📊 Found {len(reddit_files)} Reddit files")
    print(f"📊 Found {len(twitter_files)} Twitter files")
    
    all_reddit_data = []
    all_twitter_data = []
    
    # Process Reddit files
    print(f"\n🔴 Loading Reddit Data...")
    for file in reddit_files:
        try:
            print(f"📄 Loading {file.name}...")
            df = pd.read_csv(file)
            print(f"   Loaded {len(df)} records")
            all_reddit_data.append(df)
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    
    # Process Twitter files  
    print(f"\n🐦 Loading Twitter Data...")
    for file in twitter_files:
        try:
            print(f"📄 Loading {file.name}...")
            df = pd.read_csv(file)
            print(f"   Loaded {len(df)} records")
            all_twitter_data.append(df)
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    
    # Combine and process Reddit data
    if all_reddit_data:
        print(f"\n🔴 Processing Reddit Data...")
        reddit_combined = pd.concat(all_reddit_data, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(reddit_combined)
        if 'comment_id' in reddit_combined.columns:
            reddit_combined = reddit_combined.drop_duplicates(subset=['comment_id'])
        else:
            reddit_combined = reddit_combined.drop_duplicates()
        final_count = len(reddit_combined)
        
        print(f"📊 Reddit Data Summary:")
        print(f"   Total records: {initial_count}")
        print(f"   Unique records: {final_count}")
        print(f"   Duplicates removed: {initial_count - final_count}")
        
        # Add basic misogyny detection
        reddit_processed = add_misogyny_analysis(reddit_combined, text_processor, 'body')
        
        # Save processed Reddit data
        reddit_output = processed_dir / "reddit_processed.csv"
        reddit_processed.to_csv(reddit_output, index=False)
        print(f"💾 Saved processed Reddit data: {reddit_output}")
        
    else:
        print("⚠️ No Reddit data found")
        reddit_processed = pd.DataFrame()
    
    # Combine and process Twitter data
    if all_twitter_data:
        print(f"\n🐦 Processing Twitter Data...")
        twitter_combined = pd.concat(all_twitter_data, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(twitter_combined)
        if 'tweet_id' in twitter_combined.columns:
            twitter_combined = twitter_combined.drop_duplicates(subset=['tweet_id'])
        else:
            twitter_combined = twitter_combined.drop_duplicates()
        final_count = len(twitter_combined)
        
        print(f"📊 Twitter Data Summary:")
        print(f"   Total records: {initial_count}")
        print(f"   Unique records: {final_count}")
        print(f"   Duplicates removed: {initial_count - final_count}")
        
        # Add basic misogyny detection
        twitter_processed = add_misogyny_analysis(twitter_combined, text_processor, 'text')
        
        # Save processed Twitter data
        twitter_output = processed_dir / "twitter_processed.csv"
        twitter_processed.to_csv(twitter_output, index=False)
        print(f"💾 Saved processed Twitter data: {twitter_output}")
        
    else:
        print("⚠️ No Twitter data found")
        twitter_processed = pd.DataFrame()
    
    # Analysis summary
    analyze_results(reddit_processed, twitter_processed)
    
    return reddit_processed, twitter_processed

def add_misogyny_analysis(df, text_processor, text_column):
    """Add misogyny analysis to dataframe."""
    
    # Check if text column exists
    if text_column not in df.columns:
        print(f"❌ Column '{text_column}' not found. Available columns: {list(df.columns)}")
        return df
    
    print(f"🔬 Analyzing {len(df)} posts for misogyny...")
    
    # Add new columns for analysis results
    misogyny_scores = []
    is_misogynistic = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0 and idx > 0:
            print(f"   Processed {idx}/{len(df)} posts...")
        
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        
        if text and len(text.strip()) > 0:
            # Simple misogyny detection (returns float)
            score = text_processor.detect_misogyny(text)
            misogyny_scores.append(score)
            is_misogynistic.append(score > 0.3)  # Threshold for misogynistic content
        else:
            misogyny_scores.append(0.0)
            is_misogynistic.append(False)
    
    # Add analysis columns
    df_copy = df.copy()
    df_copy['misogyny_score'] = misogyny_scores
    df_copy['is_misogynistic'] = is_misogynistic
    df_copy['processed_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Completed analysis of {len(df)} posts")
    
    return df_copy

def analyze_results(reddit_df, twitter_df):
    """Analyze and display misogyny detection results."""
    print(f"\n🔍 Misogyny Detection Results")
    print("=" * 32)
    
    total_posts = 0
    total_misogyny = 0
    
    if not reddit_df.empty:
        reddit_misogyny = reddit_df['is_misogynistic'].sum()
        reddit_total = len(reddit_df)
        reddit_percent = (reddit_misogyny / reddit_total) * 100 if reddit_total > 0 else 0
        
        print(f"🔴 Reddit Results:")
        print(f"   Misogynistic posts: {reddit_misogyny:,} / {reddit_total:,} ({reddit_percent:.1f}%)")
        
        # Score distribution
        high_scores = (reddit_df['misogyny_score'] >= 0.7).sum()
        medium_scores = ((reddit_df['misogyny_score'] >= 0.3) & (reddit_df['misogyny_score'] < 0.7)).sum()
        low_scores = ((reddit_df['misogyny_score'] > 0) & (reddit_df['misogyny_score'] < 0.3)).sum()
        
        print(f"   Score distribution:")
        print(f"     High (0.7+): {high_scores:,} ({(high_scores/reddit_total)*100:.1f}%)")
        print(f"     Medium (0.3-0.7): {medium_scores:,} ({(medium_scores/reddit_total)*100:.1f}%)")
        print(f"     Low (0-0.3): {low_scores:,} ({(low_scores/reddit_total)*100:.1f}%)")
        
        total_posts += reddit_total
        total_misogyny += reddit_misogyny
    
    if not twitter_df.empty:
        twitter_misogyny = twitter_df['is_misogynistic'].sum()
        twitter_total = len(twitter_df)
        twitter_percent = (twitter_misogyny / twitter_total) * 100 if twitter_total > 0 else 0
        
        print(f"\n🐦 Twitter Results:")
        print(f"   Misogynistic posts: {twitter_misogyny:,} / {twitter_total:,} ({twitter_percent:.1f}%)")
        
        # Score distribution
        high_scores = (twitter_df['misogyny_score'] >= 0.7).sum()
        medium_scores = ((twitter_df['misogyny_score'] >= 0.3) & (twitter_df['misogyny_score'] < 0.7)).sum()
        low_scores = ((twitter_df['misogyny_score'] > 0) & (twitter_df['misogyny_score'] < 0.3)).sum()
        
        print(f"   Score distribution:")
        print(f"     High (0.7+): {high_scores:,} ({(high_scores/twitter_total)*100:.1f}%)")
        print(f"     Medium (0.3-0.7): {medium_scores:,} ({(medium_scores/twitter_total)*100:.1f}%)")
        print(f"     Low (0-0.3): {low_scores:,} ({(low_scores/twitter_total)*100:.1f}%)")
        
        total_posts += twitter_total
        total_misogyny += twitter_misogyny
    
    # Combined results
    if total_posts > 0:
        combined_percent = (total_misogyny / total_posts) * 100
        print(f"\n📊 Combined Results:")
        print(f"   Total misogynistic posts: {total_misogyny:,} / {total_posts:,} ({combined_percent:.1f}%)")
        print(f"   Dataset size: {total_posts:,} social media posts")

def main():
    """Main processing function."""
    print("🚀 MisogynyWatch Simple Data Processing")
    print("=" * 40)
    
    reddit_df, twitter_df = simple_process_data()
    
    print(f"\n✅ Processing Complete!")
    print(f"📁 Processed files saved in: data/processed/")
    print(f"🔬 Ready for advanced analysis and visualization!")
    
    # Show file locations
    if not reddit_df.empty:
        print(f"🔴 Reddit data: data/processed/reddit_processed.csv")
    if not twitter_df.empty:
        print(f"🐦 Twitter data: data/processed/twitter_processed.csv")

if __name__ == "__main__":
    main()
