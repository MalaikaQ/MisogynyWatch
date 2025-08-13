"""
Data collection coordinator for MisogynyWatch project.
Orchestrates Reddit and Twitter data collection with misogyny detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import (
    REDDIT_COMMUNITIES, TWITTER_SEARCH_TERMS, RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, COLLECTION_LIMITS
)
from reddit_scraper import RedditScraper
from twitter_scraper import TwitterScraper
from text_processing import TextProcessor
from demographics_analyzer import EnhancedDemographicsAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollectionCoordinator:
    """Coordinates data collection from multiple sources."""
    
    def __init__(self):
        """Initialize the coordinator."""
        self.reddit_scraper = RedditScraper()
        self.twitter_scraper = TwitterScraper()
        self.text_processor = TextProcessor()
        self.demographics_analyzer = EnhancedDemographicsAnalyzer()
        
    def collect_reddit_data(self, 
                           communities: List[str] = None,
                           days_back: int = 30,
                           posts_per_community: int = 500) -> pd.DataFrame:
        """
        Collect Reddit data from specified communities.
        
        Args:
            communities: List of subreddit names to scrape
            days_back: How many days back to collect data
            posts_per_community: Maximum posts per subreddit
            
        Returns:
            Combined Reddit dataframe
        """
        if communities is None:
            communities = REDDIT_COMMUNITIES
            
        all_reddit_data = []
        
        for community in communities:
            logger.info(f"Collecting data from r/{community}")
            
            try:
                # Collect posts
                posts_df = self.reddit_scraper.collect_subreddit_comments(
                    subreddit_name=community,
                    limit=posts_per_community,
                    time_filter='month'
                )
                
                if not posts_df.empty:
                    posts_df['subreddit'] = community
                    posts_df['content_type'] = 'post'
                    all_reddit_data.append(posts_df)
                    
                    # Collect comments for top posts
                    top_posts = posts_df.head(50)  # Top 50 posts
                    for _, post in top_posts.iterrows():
                        try:
                            comments_df = self.reddit_scraper.collect_post_comments(
                                post_id=post['id'],
                                limit=COLLECTION_LIMITS['reddit_comments_per_post']
                            )
                            
                            if not comments_df.empty:
                                comments_df['subreddit'] = community
                                comments_df['content_type'] = 'comment'
                                comments_df['parent_post_id'] = post['id']
                                all_reddit_data.append(comments_df)
                                
                        except Exception as e:
                            logger.error(f"Error collecting comments for post {post['id']}: {e}")
                            continue
                
                # Rate limiting
                time.sleep(COLLECTION_LIMITS['rate_limit_delay'])
                
            except Exception as e:
                logger.error(f"Error collecting data from r/{community}: {e}")
                continue
        
        if all_reddit_data:
            combined_df = pd.concat(all_reddit_data, ignore_index=True)
            logger.info(f"Collected {len(combined_df)} Reddit items")
            return combined_df
        else:
            logger.warning("No Reddit data collected")
            return pd.DataFrame()
    
    def collect_twitter_data(self, 
                           search_terms: List[str] = None,
                           days_back: int = 30,
                           tweets_per_term: int = 200) -> pd.DataFrame:
        """
        Collect Twitter data for specified search terms.
        
        Args:
            search_terms: List of search terms/hashtags
            days_back: How many days back to collect data
            tweets_per_term: Maximum tweets per search term
            
        Returns:
            Combined Twitter dataframe
        """
        if search_terms is None:
            search_terms = TWITTER_SEARCH_TERMS
            
        all_twitter_data = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for term in search_terms:
            logger.info(f"Collecting tweets for: {term}")
            
            try:
                tweets_df = self.twitter_scraper.collect_tweets_by_keywords(
                    keywords=[term],
                    max_results=tweets_per_term
                )
                
                if not tweets_df.empty:
                    tweets_df['search_term'] = term
                    all_twitter_data.append(tweets_df)
                
                # Rate limiting
                time.sleep(COLLECTION_LIMITS['rate_limit_delay'])
                
            except Exception as e:
                logger.error(f"Error collecting tweets for '{term}': {e}")
                continue
        
        if all_twitter_data:
            combined_df = pd.concat(all_twitter_data, ignore_index=True)
            # Remove duplicates based on tweet ID
            combined_df = combined_df.drop_duplicates(subset=['id'])
            logger.info(f"Collected {len(combined_df)} unique tweets")
            return combined_df
        else:
            logger.warning("No Twitter data collected")
            return pd.DataFrame()
    
    def process_and_analyze_data(self, 
                               reddit_df: pd.DataFrame, 
                               twitter_df: pd.DataFrame) -> tuple:
        """
        Process collected data with misogyny detection and age analysis.
        
        Args:
            reddit_df: Raw Reddit data
            twitter_df: Raw Twitter data
            
        Returns:
            Tuple of (processed_reddit_df, processed_twitter_df)
        """
        logger.info("Processing Reddit data...")
        if not reddit_df.empty:
            # Initialize enhanced text processor
            text_processor = TextProcessor(use_research_lexicons=True)
            
            # Text processing and misogyny detection
            reddit_df['cleaned_text'] = reddit_df['text'].apply(
                lambda x: text_processor.clean_text(x) if pd.notna(x) else ''
            )
            
            # Enhanced misogyny detection with research lexicons
            reddit_df['misogyny_score'] = reddit_df['text'].apply(
                lambda x: text_processor.detect_misogyny(x, use_research_boost=True) if pd.notna(x) else 0
            )
            
            # Get detailed detection information
            detection_details = reddit_df['text'].apply(
                lambda x: text_processor.get_detection_details(x) if pd.notna(x) else {}
            )
            
            reddit_df['misogyny_keywords'] = detection_details.apply(
                lambda x: x.get('matched_terms', [])
            )
            
            reddit_df['severity_level'] = detection_details.apply(
                lambda x: x.get('severity_level', 'none')
            )
            
            reddit_df['category_scores'] = detection_details.apply(
                lambda x: x.get('category_scores', {})
            )
            
            # Age analysis
            reddit_df['estimated_age'] = reddit_df['text'].apply(
                lambda x: self.demographics_analyzer.extract_age_from_text(x) if pd.notna(x) else None
            )
            
            reddit_df['age_group'] = reddit_df['estimated_age'].apply(
                lambda x: self.demographics_analyzer.assign_age_group(x) if pd.notna(x) else None
            )
            
            reddit_df['estimated_gender'] = reddit_df['text'].apply(
                lambda x: self.demographics_analyzer.extract_gender_from_text(x) if pd.notna(x) else None
            )
            
            # Add platform identifier
            reddit_df['platform'] = 'Reddit'
            
        logger.info("Processing Twitter data...")
        if not twitter_df.empty:
            # Use same enhanced text processor
            text_processor = TextProcessor(use_research_lexicons=True)
            
            # Text processing and misogyny detection
            twitter_df['cleaned_text'] = twitter_df['text'].apply(
                lambda x: text_processor.clean_text(x) if pd.notna(x) else ''
            )
            
            # Enhanced misogyny detection with research lexicons
            twitter_df['misogyny_score'] = twitter_df['text'].apply(
                lambda x: text_processor.detect_misogyny(x, use_research_boost=True) if pd.notna(x) else 0
            )
            
            # Get detailed detection information
            detection_details = twitter_df['text'].apply(
                lambda x: text_processor.get_detection_details(x) if pd.notna(x) else {}
            )
            
            twitter_df['misogyny_keywords'] = detection_details.apply(
                lambda x: x.get('matched_terms', [])
            )
            
            twitter_df['severity_level'] = detection_details.apply(
                lambda x: x.get('severity_level', 'none')
            )
            
            twitter_df['category_scores'] = detection_details.apply(
                lambda x: x.get('category_scores', {})
            )
            
            # Age analysis
            twitter_df['estimated_age'] = twitter_df['text'].apply(
                lambda x: self.demographics_analyzer.extract_age_from_text(x) if pd.notna(x) else None
            )
            
            twitter_df['age_group'] = twitter_df['estimated_age'].apply(
                lambda x: self.demographics_analyzer.assign_age_group(x) if pd.notna(x) else None
            )
            
            twitter_df['estimated_gender'] = twitter_df['text'].apply(
                lambda x: self.demographics_analyzer.extract_gender_from_text(x) if pd.notna(x) else None
            )
            
            # Add platform identifier
            twitter_df['platform'] = 'Twitter'
        
        return reddit_df, twitter_df
    
    def save_data(self, reddit_df: pd.DataFrame, twitter_df: pd.DataFrame):
        """
        Save processed data to files.
        
        Args:
            reddit_df: Processed Reddit data
            twitter_df: Processed Twitter data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        if not reddit_df.empty:
            reddit_raw_path = RAW_DATA_DIR / f'reddit_raw_{timestamp}.csv'
            reddit_df.to_csv(reddit_raw_path, index=False)
            logger.info(f"Reddit raw data saved to: {reddit_raw_path}")
            
            # Save processed data
            reddit_processed_path = PROCESSED_DATA_DIR / 'reddit_processed.csv'
            reddit_df.to_csv(reddit_processed_path, index=False)
            logger.info(f"Reddit processed data saved to: {reddit_processed_path}")
        
        if not twitter_df.empty:
            twitter_raw_path = RAW_DATA_DIR / f'twitter_raw_{timestamp}.csv'
            twitter_df.to_csv(twitter_raw_path, index=False)
            logger.info(f"Twitter raw data saved to: {twitter_raw_path}")
            
            # Save processed data
            twitter_processed_path = PROCESSED_DATA_DIR / 'twitter_processed.csv'
            twitter_df.to_csv(twitter_processed_path, index=False)
            logger.info(f"Twitter processed data saved to: {twitter_processed_path}")
    
    def run_full_collection(self, 
                          days_back: int = 30,
                          reddit_posts_per_community: int = 500,
                          twitter_tweets_per_term: int = 200) -> Dict:
        """
        Run complete data collection and processing pipeline.
        
        Args:
            days_back: Days to look back for data collection
            reddit_posts_per_community: Posts to collect per subreddit
            twitter_tweets_per_term: Tweets to collect per search term
            
        Returns:
            Summary statistics
        """
        logger.info("Starting full data collection pipeline...")
        
        # Collect Reddit data
        logger.info("=== REDDIT DATA COLLECTION ===")
        reddit_df = self.collect_reddit_data(
            days_back=days_back,
            posts_per_community=reddit_posts_per_community
        )
        
        # Collect Twitter data
        logger.info("=== TWITTER DATA COLLECTION ===")
        twitter_df = self.collect_twitter_data(
            days_back=days_back,
            tweets_per_term=twitter_tweets_per_term
        )
        
        # Process data
        logger.info("=== DATA PROCESSING ===")
        reddit_processed, twitter_processed = self.process_and_analyze_data(reddit_df, twitter_df)
        
        # Save data
        logger.info("=== SAVING DATA ===")
        self.save_data(reddit_processed, twitter_processed)
        
        # Generate summary
        summary = {
            'collection_date': datetime.now().isoformat(),
            'reddit_posts_collected': len(reddit_processed) if not reddit_processed.empty else 0,
            'twitter_posts_collected': len(twitter_processed) if not twitter_processed.empty else 0,
            'reddit_misogynistic_posts': len(reddit_processed[reddit_processed['misogyny_score'] > 0]) if not reddit_processed.empty else 0,
            'twitter_misogynistic_posts': len(twitter_processed[twitter_processed['misogyny_score'] > 0]) if not twitter_processed.empty else 0,
            'communities_collected': reddit_processed['subreddit'].nunique() if not reddit_processed.empty and 'subreddit' in reddit_processed.columns else 0,
            'search_terms_used': len(TWITTER_SEARCH_TERMS),
            'date_range_days': days_back
        }
        
        logger.info("=== COLLECTION SUMMARY ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        # Save summary
        summary_path = PROCESSED_DATA_DIR / 'collection_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")
        
        return summary

if __name__ == "__main__":
    coordinator = DataCollectionCoordinator()
    
    # Run collection with configurable parameters
    summary = coordinator.run_full_collection(
        days_back=30,  # Collect last 30 days
        reddit_posts_per_community=300,  # 300 posts per subreddit
        twitter_tweets_per_term=150  # 150 tweets per search term
    )
    
    print("\nData collection completed successfully!")
    print(f"Total posts collected: {summary['reddit_posts_collected'] + summary['twitter_posts_collected']}")
    print(f"Misogynistic content detected: {summary['reddit_misogynistic_posts'] + summary['twitter_misogynistic_posts']}")
