"""
Twitter data collection using tweepy with age analysis capabilities.
"""
import tweepy
import pandas as pd
import datetime
import time
import logging
from typing import List, Dict, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import TWITTER_CONFIG, RAW_DATA_DIR
from text_processing import TextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterScraper:
    """Twitter data collection and processing with age analysis."""
    
    def __init__(self):
        """Initialize Twitter API connection."""
        try:
            # Twitter API v2 authentication
            self.client = tweepy.Client(
                bearer_token=TWITTER_CONFIG.get('bearer_token'),
                consumer_key=TWITTER_CONFIG.get('consumer_key'),
                consumer_secret=TWITTER_CONFIG.get('consumer_secret'),
                access_token=TWITTER_CONFIG.get('access_token'),
                access_token_secret=TWITTER_CONFIG.get('access_token_secret'),
                wait_on_rate_limit=True
            )
            
            # Test the connection
            try:
                me = self.client.get_me()
                if me.data:
                    logger.info("Twitter API connection established")
                else:
                    logger.warning("Twitter API connected but no user data returned")
            except Exception as e:
                logger.warning("Twitter API connection issue: {}".format(e))
                
        except Exception as e:
            logger.error("Failed to initialize Twitter API: {}".format(e))
            self.client = None
            
        self.text_processor = TextProcessor()
    
    def collect_tweets_by_keywords(self, 
                                  keywords: List[str],
                                  max_results: int = 100,
                                  lang: str = 'en') -> pd.DataFrame:
        """
        Collect tweets by searching for specific keywords.
        
        Args:
            keywords: List of search keywords
            max_results: Maximum number of tweets to collect
            lang: Language filter
            
        Returns:
            DataFrame with tweet data including user profile info for age analysis
        """
        if not self.client:
            logger.error("Twitter API not initialized")
            return pd.DataFrame()
        
        tweets_data = []
        query = ' OR '.join(['"{}"'.format(keyword) for keyword in keywords])
        
        try:
            logger.info("Searching for tweets with keywords: {}".format(keywords))
            
            # Search for tweets with user information
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=min(max_results, 100),  # API limit per request
                tweet_fields=['created_at', 'public_metrics', 'context_annotations', 'lang'],
                user_fields=['created_at', 'description', 'public_metrics', 'verified'],
                expansions=['author_id']
            ).flatten(limit=max_results)
            
            # Get user data
            users_dict = {}
            if hasattr(tweets, 'includes') and 'users' in tweets.includes:
                for user in tweets.includes['users']:
                    users_dict[user.id] = user
            
            for tweet in tweets:
                if tweet.lang != lang:
                    continue
                    
                # Get user information for age analysis
                user_data = users_dict.get(tweet.author_id, {})
                user_profile = self.get_user_profile_data(user_data) if user_data else {}
                
                tweet_data = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'retweet_count': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                    'like_count': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    'reply_count': tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0,
                    'quote_count': tweet.public_metrics.get('quote_count', 0) if tweet.public_metrics else 0,
                    'platform': 'twitter',
                    'search_keywords': ', '.join(keywords),
                    # User profile data for age analysis
                    'author_username': user_data.username if hasattr(user_data, 'username') else None,
                    'author_name': user_data.name if hasattr(user_data, 'name') else None,
                    'author_description': user_data.description if hasattr(user_data, 'description') else None,
                    'author_account_created': user_data.created_at if hasattr(user_data, 'created_at') else None,
                    'author_verified': user_data.verified if hasattr(user_data, 'verified') else False,
                    'author_followers_count': user_data.public_metrics.get('followers_count', 0) if hasattr(user_data, 'public_metrics') and user_data.public_metrics else 0,
                    'author_following_count': user_data.public_metrics.get('following_count', 0) if hasattr(user_data, 'public_metrics') and user_data.public_metrics else 0,
                    'author_tweet_count': user_data.public_metrics.get('tweet_count', 0) if hasattr(user_data, 'public_metrics') and user_data.public_metrics else 0
                }
                
                tweets_data.append(tweet_data)
                
                # Rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error("Error collecting tweets: {}".format(e))
        
        df = pd.DataFrame(tweets_data)
        logger.info("Collected {} tweets".format(len(df)))
        
        return df
    
    def collect_user_timeline(self, 
                             username: str,
                             max_results: int = 100) -> pd.DataFrame:
        """
        Collect tweets from a specific user's timeline.
        
        Args:
            username: Twitter username (without @)
            max_results: Maximum number of tweets
            
        Returns:
            DataFrame with user's tweets
        """
        if not self.client:
            logger.error("Twitter API not initialized")
            return pd.DataFrame()
        
        tweets_data = []
        
        try:
            # Get user information first
            user = self.client.get_user(username=username, 
                                       user_fields=['created_at', 'description', 'public_metrics', 'verified'])
            
            if not user.data:
                logger.error("User {} not found".format(username))
                return pd.DataFrame()
            
            user_profile = self.get_user_profile_data(user.data)
            
            # Get user's tweets
            tweets = tweepy.Paginator(
                self.client.get_users_tweets,
                id=user.data.id,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            ).flatten(limit=max_results)
            
            for tweet in tweets:
                tweet_data = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': user.data.id,
                    'author_username': user.data.username,
                    'author_name': user.data.name,
                    'retweet_count': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                    'like_count': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    'reply_count': tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0,
                    'quote_count': tweet.public_metrics.get('quote_count', 0) if tweet.public_metrics else 0,
                    'platform': 'twitter',
                    'collection_type': 'user_timeline',
                    # User profile data for age analysis
                    'author_description': user.data.description,
                    'author_account_created': user.data.created_at,
                    'author_verified': user.data.verified,
                    'author_followers_count': user.data.public_metrics.get('followers_count', 0) if user.data.public_metrics else 0,
                    'author_following_count': user.data.public_metrics.get('following_count', 0) if user.data.public_metrics else 0,
                    'author_tweet_count': user.data.public_metrics.get('tweet_count', 0) if user.data.public_metrics else 0
                }
                
                tweets_data.append(tweet_data)
                
        except Exception as e:
            logger.error("Error collecting timeline for {}: {}".format(username, e))
        
        df = pd.DataFrame(tweets_data)
        logger.info("Collected {} tweets from @{}".format(len(df), username))
        
        return df
    
    def get_user_profile_data(self, user_data) -> Dict:
        """
        Extract user profile information for age analysis.
        
        Args:
            user_data: Twitter user object
            
        Returns:
            Dictionary with user profile information
        """
        profile_data = {
            'account_created': None,
            'description': '',
            'verified': False,
            'followers_count': 0,
            'following_count': 0,
            'tweet_count': 0
        }
        
        if not user_data:
            return profile_data
        
        try:
            profile_data['account_created'] = user_data.created_at if hasattr(user_data, 'created_at') else None
            profile_data['description'] = user_data.description if hasattr(user_data, 'description') else ''
            profile_data['verified'] = user_data.verified if hasattr(user_data, 'verified') else False
            
            if hasattr(user_data, 'public_metrics') and user_data.public_metrics:
                profile_data['followers_count'] = user_data.public_metrics.get('followers_count', 0)
                profile_data['following_count'] = user_data.public_metrics.get('following_count', 0)
                profile_data['tweet_count'] = user_data.public_metrics.get('tweet_count', 0)
                
        except Exception as e:
            logger.debug("Could not extract full profile data: {}".format(e))
        
        return profile_data
    
    def collect_misogyny_related_tweets(self, max_results: int = 500) -> pd.DataFrame:
        """
        Collect tweets related to misogyny and red-pill ideology.
        
        Args:
            max_results: Maximum number of tweets to collect
            
        Returns:
            DataFrame with relevant tweets
        """
        # Red-pill and misogyny-related keywords
        keywords = [
            'red pill', 'redpill', 'MGTOW', 'mens rights',
            'feminism toxic', 'women hypergamous', 'female nature',
            'alpha male', 'beta male', 'pickup artist',
            'manosphere', 'blackpill', 'incel',
            'traditional gender roles', 'gynocentrism'
        ]
        
        all_tweets = []
        tweets_per_keyword = max_results // len(keywords)
        
        for i in range(0, len(keywords), 3):  # Process in batches of 3 keywords
            batch_keywords = keywords[i:i+3]
            df = self.collect_tweets_by_keywords(
                batch_keywords, 
                max_results=tweets_per_keyword * len(batch_keywords)
            )
            if not df.empty:
                all_tweets.append(df)
            
            # Rate limiting between batches
            time.sleep(2)
        
        if all_tweets:
            combined_df = pd.concat(all_tweets, ignore_index=True)
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['tweet_id'])
            return combined_df
        else:
            return pd.DataFrame()
    
    def collect_influencer_mentions(self, 
                                   influencer_handles: List[str],
                                   max_results: int = 200) -> pd.DataFrame:
        """
        Collect tweets mentioning specific red-pill influencers.
        
        Args:
            influencer_handles: List of Twitter handles (without @)
            max_results: Maximum results per influencer
            
        Returns:
            DataFrame with tweets mentioning influencers
        """
        all_tweets = []
        
        for handle in influencer_handles:
            query = '@{} OR "{}"'.format(handle, handle)
            df = self.collect_tweets_by_keywords([query], max_results=max_results)
            
            if not df.empty:
                df['mentioned_influencer'] = handle
                all_tweets.append(df)
            
            # Rate limiting
            time.sleep(2)
        
        if all_tweets:
            return pd.concat(all_tweets, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV file."""
        if df.empty:
            logger.warning("No data to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        filepath = os.path.join(RAW_DATA_DIR, '{}.csv'.format(filename))
        df.to_csv(filepath, index=False)
        logger.info("Data saved to {}".format(filepath))

def main():
    """Example usage of TwitterScraper with age analysis."""
    scraper = TwitterScraper()
    
    if not scraper.client:
        print("Twitter API not available. Please check your credentials.")
        return
    
    # Test keyword search
    print("Collecting red-pill related tweets...")
    redpill_tweets = scraper.collect_misogyny_related_tweets(max_results=100)
    
    if not redpill_tweets.empty:
        print("Collected {} red-pill related tweets".format(len(redpill_tweets)))
        scraper.save_data(redpill_tweets, 'twitter_redpill_sample')
        
        # Show age-related data availability
        age_data_count = redpill_tweets['author_description'].notna().sum()
        print("Tweets with user descriptions (for age analysis): {}".format(age_data_count))
        
        # Show account creation dates (for age estimation)
        account_dates_count = redpill_tweets['author_account_created'].notna().sum()
        print("Tweets with account creation dates: {}".format(account_dates_count))
    else:
        print("No tweets collected")

if __name__ == "__main__":
    main()
