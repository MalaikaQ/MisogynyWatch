#!/usr/bin/env python3
"""
Strategic Twitter collection that works around rate limits.
"""

from twitter_scraper import TwitterScraper
import pandas as pd
import time
from datetime import datetime

def strategic_twitter_collection():
    """Collect Twitter data using rate-limit friendly methods."""
    scraper = TwitterScraper()
    
    if not scraper.client:
        print("X Twitter API not available")
        return
    
    print("Twitter Strategic Collection (Rate Limit Optimized)")
    print("=" * 55)
    
    all_data = []
    
    # Strategy 1: Different keyword sets to diversify content
    print("\nPhase 1: Diverse keyword searches...")
    
    keyword_groups = [
        # Group 1: Direct misogyny terms
        ['misogyny', 'misogynistic'],
        
        # Group 2: Red-pill terms  
        ['red pill', 'redpill'],
        
        # Group 3: Manosphere terms
        ['MGTOW', 'mens rights'],
        
        # Group 4: Gender-related controversial terms
        ['alpha male', 'beta male'],
        
        # Group 5: Anti-feminist terms
        ['feminism toxic', 'feminist agenda'],
        
        # Group 6: Dating/relationship terms
        ['hypergamous', 'female nature'],
        
        # Group 7: Pickup artist terms
        ['pickup artist', 'PUA'],
        
        # Group 8: Broader controversial terms
        ['traditional gender roles', 'gynocentrism']
    ]
    
    for i, keywords in enumerate(keyword_groups, 1):
        print(f"\n🔍 Group {i}: Searching for {keywords}...")
        
        try:
            df = scraper.collect_tweets_by_keywords(
                keywords=keywords,
                max_results=50  # Conservative per group
            )
            
            if not df.empty:
                print(f"✅ Collected {len(df)} tweets for {keywords}")
                all_data.append(df)
                
                # Save each group immediately in case of rate limits
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'twitter_group_{i}_{timestamp}'
                scraper.save_data(df, filename)
                
            else:
                print(f"⚠️ No tweets found for {keywords}")
                
        except Exception as e:
            print(f"❌ Error with {keywords}: {e}")
            if "rate limit" in str(e).lower():
                print("🚨 Hit rate limit. Waiting...")
                time.sleep(60)  # Wait 1 minute
                continue
            
        # Short delay between keyword groups
        print("⏱️ Waiting 15 seconds...")
        time.sleep(15)
    
    # Strategy 2: Single user timelines (if we have quota left)
    print(f"\n📊 Phase 2: User timelines (if quota available)...")
    
    # Target accounts that often have engagement around gender topics
    target_accounts = [
        'elonmusk',      # Often controversial takes
        'joerogan',      # Podcast host, often discusses gender
        'benshapiro',    # Conservative commentator
        'jordanbpeterson' # Psychologist, often discusses gender
    ]
    
    for account in target_accounts:
        print(f"\n👤 Collecting from @{account}...")
        
        try:
            timeline_df = scraper.collect_user_timeline(
                username=account,
                max_results=25  # Small number to conserve quota
            )
            
            if not timeline_df.empty:
                print(f"✅ Collected {len(timeline_df)} tweets from @{account}")
                all_data.append(timeline_df)
                
                # Save immediately
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'twitter_{account}_{timestamp}'
                scraper.save_data(timeline_df, filename)
                
            else:
                print(f"⚠️ No tweets from @{account}")
                
        except Exception as e:
            print(f"❌ Error with @{account}: {e}")
            if "rate limit" in str(e).lower():
                print("🚨 Hit user timeline rate limit. Stopping user collection.")
                break
            continue
        
        # Longer delay for user timelines (more expensive)
        print("⏱️ Waiting 30 seconds...")
        time.sleep(30)
    
    # Combine all collected data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['tweet_id'])
        final_count = len(combined_df)
        
        print(f"\n📈 Collection Summary:")
        print(f"Total tweets collected: {initial_count}")
        print(f"Unique tweets after deduplication: {final_count}")
        print(f"Removed {initial_count - final_count} duplicates")
        
        # Save final combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'twitter_strategic_collection_{timestamp}'
        scraper.save_data(combined_df, filename)
        
        print(f"💾 Saved combined dataset: {filename}.csv")
        
        # Show content breakdown
        if 'search_keywords' in combined_df.columns:
            print(f"\n📊 Content breakdown:")
            
            # Count keywords
            keyword_stats = {}
            for keywords in combined_df['search_keywords'].dropna():
                for keyword in keywords.split(', '):
                    keyword = keyword.strip()
                    if keyword:
                        keyword_stats[keyword] = keyword_stats.get(keyword, 0) + 1
            
            print("🔤 Top keywords:")
            for keyword, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {keyword}: {count} tweets")
        
        return combined_df
    else:
        print("❌ No Twitter data collected")
        return pd.DataFrame()

def check_rate_limit_status():
    """Check current rate limit status."""
    print("📊 Checking Twitter API rate limit status...")
    
    scraper = TwitterScraper()
    if scraper.client:
        try:
            # Try a simple API call to check status
            me = scraper.client.get_me()
            if me.data:
                print(f"✅ API working. Authenticated as: @{me.data.username}")
                return True
            else:
                print("⚠️ API connected but no user data")
                return False
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("🚨 Currently rate limited")
                return False
            else:
                print(f"❌ API error: {e}")
                return False
    else:
        print("❌ Twitter API not initialized")
        return False

def main():
    """Run strategic Twitter collection."""
    print("🚀 Strategic Twitter Data Collection")
    print("=" * 38)
    
    # Check if we can make API calls
    if not check_rate_limit_status():
        print("\n💡 If rate limited, try again in 15 minutes")
        print("   Rate limits reset every 15 minutes on Twitter API v2")
        return
    
    print("\n🎯 Starting strategic collection...")
    twitter_df = strategic_twitter_collection()
    
    if not twitter_df.empty:
        print(f"\n🎉 Successfully collected {len(twitter_df)} Twitter posts!")
        print(f"📁 Check data/raw/ for individual and combined files")
        
        # Show current total across all collections
        print(f"\n📊 Your Twitter data now includes:")
        print(f"   • Previous keywords: 92 tweets")
        print(f"   • New strategic collection: {len(twitter_df)} tweets")
        print(f"   • Total Twitter posts: {92 + len(twitter_df)}")
        
    else:
        print("\n😔 No new Twitter data collected")
        print("💡 This could be due to:")
        print("   • Rate limits (wait 15 minutes)")
        print("   • API quota exhausted")
        print("   • Network issues")

if __name__ == "__main__":
    main()
