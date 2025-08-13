#!/usr/bin/env python3
"""
Custom Reddit data collection with working subreddits and no user profile issues.
"""

from reddit_scraper import RedditScraper
import pandas as pd
import time
from datetime import datetime

def collect_reddit_data():
    """Collect Reddit data from working subreddits with safe limits."""
    scraper = RedditScraper()
    
    if not scraper.reddit:
        print("❌ Reddit API not available")
        return
    
    print("🔴 Reddit Data Collection (Fixed Version)")
    print("=" * 45)
    
    # Use subreddits that typically work and have active content
    working_subreddits = [
        'MensRights',           # Usually accessible
        'relationship_advice',  # Very active, good content  
        'unpopularopinion',    # Often has controversial posts
        'AmItheAsshole',       # Good for analyzing attitudes
        'dating_advice',       # Dating-related discussions
        'AskMen',              # Men's perspectives
        'AskWomen',            # Women's perspectives  
        'TwoXChromosomes'      # Women's community
    ]
    
    all_data = []
    
    for subreddit in working_subreddits:
        print(f"\n📊 Collecting from r/{subreddit}...")
        
        try:
            # Collect with conservative limits to avoid issues
            df = scraper.collect_subreddit_comments(
                subreddit_name=subreddit,
                limit=100,  # Conservative limit per subreddit
                time_filter='week',  # Recent data only
                sort_by='hot'  # Most engaging content
            )
            
            if not df.empty:
                print(f"✅ Collected {len(df)} comments from r/{subreddit}")
                all_data.append(df)
                
                # Save individual subreddit data as backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scraper.save_data(df, f'reddit_{subreddit}_{timestamp}')
                
            else:
                print(f"⚠️ No data collected from r/{subreddit}")
                
        except Exception as e:
            print(f"❌ Error with r/{subreddit}: {e}")
            continue
        
        # Rate limiting between subreddits
        print("⏱️ Waiting 5 seconds...")
        time.sleep(5)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['comment_id'])
        final_count = len(combined_df)
        
        print(f"\n📈 Collection Summary:")
        print(f"Total comments collected: {initial_count}")
        print(f"Unique comments after deduplication: {final_count}")
        print(f"Removed {initial_count - final_count} duplicates")
        
        # Save combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'reddit_combined_collection_{timestamp}'
        scraper.save_data(combined_df, filename)
        
        print(f"💾 Saved combined dataset: {filename}.csv")
        
        # Show subreddit breakdown
        if 'subreddit' in combined_df.columns:
            print(f"\n📊 Data by subreddit:")
            subreddit_counts = combined_df['subreddit'].value_counts()
            for subreddit, count in subreddit_counts.items():
                print(f"   r/{subreddit}: {count} comments")
        
        return combined_df
    else:
        print("❌ No Reddit data collected from any subreddit")
        return pd.DataFrame()

def collect_specific_searches():
    """Collect Reddit data using search for specific terms."""
    scraper = RedditScraper()
    
    print(f"\n🔍 Collecting Reddit search results...")
    
    # Search terms related to misogyny research
    search_terms = [
        'misogyny',
        'red pill',
        'Andrew Tate',
        'alpha male',
        'female nature',
        'traditional gender roles'
    ]
    
    all_search_data = []
    
    for term in search_terms:
        print(f"🔍 Searching for: '{term}'...")
        try:
            search_df = scraper.search_comments(
                query=term,
                limit=50,  # Conservative limit per search term
                sort='relevance'
            )
            
            if not search_df.empty:
                print(f"✅ Found {len(search_df)} comments for '{term}'")
                all_search_data.append(search_df)
            else:
                print(f"⚠️ No results for '{term}'")
                
        except Exception as e:
            print(f"❌ Error searching for '{term}': {e}")
            continue
        
        time.sleep(3)  # Rate limiting
    
    if all_search_data:
        search_combined = pd.concat(all_search_data, ignore_index=True)
        search_combined = search_combined.drop_duplicates(subset=['comment_id'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'reddit_search_results_{timestamp}'
        scraper.save_data(search_combined, filename)
        
        print(f"💾 Saved search results: {filename}.csv ({len(search_combined)} comments)")
        return search_combined
    
    return pd.DataFrame()

def main():
    """Run comprehensive Reddit data collection."""
    print("🚀 Comprehensive Reddit Data Collection")
    print("=" * 42)
    
    # Method 1: Collect from working subreddits
    subreddit_data = collect_reddit_data()
    
    # Small delay between collection methods
    time.sleep(10)
    
    # Method 2: Search for specific terms
    search_data = collect_specific_searches()
    
    # Final summary
    total_comments = len(subreddit_data) + len(search_data)
    print(f"\n🎉 Final Collection Summary:")
    print(f"Subreddit comments: {len(subreddit_data)}")
    print(f"Search results: {len(search_data)}")
    print(f"Total Reddit comments: {total_comments}")
    
    print(f"\n💾 Data files created in: data/raw/")
    print(f"📊 Ready for text processing and misogyny analysis!")

if __name__ == "__main__":
    main()
