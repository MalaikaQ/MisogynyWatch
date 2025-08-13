"""
Quick real data collection test for MisogynyWatch
"""
import pandas as pd
from reddit_scraper import RedditScraper
from twitter_scraper import TwitterScraper
import time
from datetime import datetime

def collect_real_sample_data():
    """Collect a small sample of real data for testing."""
    
    print("🚀 REAL DATA COLLECTION TEST")
    print("=" * 35)
    
    # Test accessible Reddit communities
    accessible_communities = [
        'MensRights',      # Usually accessible
        'dating_advice',   # Public community
        'relationship_advice',  # Public community
        'AskMen'          # Public community
    ]
    
    reddit = RedditScraper()
    all_reddit_data = []
    
    print("📱 Testing Reddit communities...")
    for community in accessible_communities:
        print(f"   Trying r/{community}...")
        try:
            # Quick collection without heavy user profiling
            df = reddit.collect_subreddit_comments(community, limit=5, time_filter='week')
            if not df.empty:
                print(f"   ✅ r/{community}: {len(df)} posts")
                df['community'] = community
                all_reddit_data.append(df)
                break  # Just get one working community for test
            else:
                print(f"   ❌ r/{community}: No accessible posts")
        except Exception as e:
            print(f"   ❌ r/{community}: {str(e)[:50]}...")
    
    # Combine Reddit data
    reddit_df = pd.concat(all_reddit_data, ignore_index=True) if all_reddit_data else pd.DataFrame()
    
    # Test Twitter with accessible keywords
    print()
    print("🐦 Testing Twitter...")
    twitter = TwitterScraper()
    twitter_df = pd.DataFrame()
    
    try:
        # Use less controversial terms to avoid rate limits
        print("   Searching for 'dating advice'...")
        twitter_df = twitter.collect_tweets_by_keywords(['dating advice'], max_results=10)
        if not twitter_df.empty:
            print(f"   ✅ Twitter: {len(twitter_df)} tweets")
        else:
            print("   ❌ Twitter: No tweets collected")
    except Exception as e:
        print(f"   ❌ Twitter: {str(e)[:50]}...")
    
    # Results summary
    print()
    print("📊 REAL DATA COLLECTION RESULTS:")
    print(f"   Reddit posts: {len(reddit_df)}")
    print(f"   Twitter tweets: {len(twitter_df)}")
    
    if not reddit_df.empty:
        print()
        print("🔍 REDDIT REAL DATA PROOF:")
        print(f"   Sample IDs: {reddit_df['id'].head(3).tolist()}")
        print(f"   Communities: {reddit_df['community'].unique()}")
        
        # Save the real Reddit data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reddit_filename = f"data/raw/real_reddit_sample_{timestamp}.csv"
        reddit_df.to_csv(reddit_filename, index=False)
        print(f"   ✅ Saved to: {reddit_filename}")
    
    if not twitter_df.empty:
        print()
        print("🔍 TWITTER REAL DATA PROOF:")
        print(f"   Sample IDs: {twitter_df['tweet_id'].head(3).tolist()}")
        
        # Save the real Twitter data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        twitter_filename = f"data/raw/real_twitter_sample_{timestamp}.csv"
        twitter_df.to_csv(twitter_filename, index=False)
        print(f"   ✅ Saved to: {twitter_filename}")
    
    return reddit_df, twitter_df

if __name__ == "__main__":
    reddit_data, twitter_data = collect_real_sample_data()
    
    print()
    print("🎯 CONCLUSION:")
    if not reddit_data.empty or not twitter_data.empty:
        print("✅ REAL DATA COLLECTION: WORKING!")
        print("✅ APIs are functional for authentic social media data")
        print("✅ Ready to scale up for full research collection")
    else:
        print("❌ No real data collected - need to investigate further")
