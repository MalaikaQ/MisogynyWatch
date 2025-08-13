#!/usr/bin/env python3
"""
Enhanced Reddit Data Collection for Research Questions

This script collects Reddit data specifically designed to answer:
1. Temporal trends in misogynistic language
2. Event correlation analysis  
3. Community impact assessment
4. Demographics analysis (age/gender)
"""

import praw
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedRedditCollector:
    def __init__(self):
        """Initialize Reddit API connection."""
        # Reddit API credentials (you'll need to set these up)
        try:
            self.reddit = praw.Reddit(
                client_id="your_client_id",      # Replace with your Reddit app credentials
                client_secret="your_client_secret",
                user_agent="MisogynyWatch Research v1.0",
                read_only=True
            )
            logger.info("Reddit API connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            self.reddit = None
    
    def collect_temporal_data(self, start_date=None, end_date=None):
        """
        Collect Reddit data across specific time periods for temporal analysis.
        """
        if not self.reddit:
            logger.error("Reddit API not available")
            return None
            
        # Define target subreddits for comprehensive analysis
        target_subreddits = [
            # Gender-focused communities
            'MensRights', 'TwoXChromosomes', 'AskWomen', 'AskMen',
            'FemaleDatingStrategy', 'MaleGrooming', 'Feminism',
            
            # Dating/Relationship communities  
            'dating_advice', 'relationship_advice', 'dating', 'relationships',
            'Tinder', 'Bumble', 'OnlineDating',
            
            # Red-pill/Manosphere (if accessible)
            'seduction', 'socialskills', 'confidence',
            
            # General discussion
            'unpopularopinion', 'changemyview', 'AskReddit', 'todayilearned',
            'offmychest', 'TrueOffMyChest',
            
            # Debate/Political
            'PurplePillDebate', 'FeminismUncensored', 'MensLib',
            
            # Age-specific communities (for demographics)
            'teenagers', 'college', 'adulting', 'RedditForGrownups',
            'GenZ', 'Millennials', 'GenX'
        ]
        
        # Keywords for targeted search
        search_keywords = [
            'misogyny', 'misogynistic', 'sexist', 'sexism',
            'red pill', 'MGTOW', 'incel', 'feminazi',
            'traditional gender roles', 'gender equality',
            'toxic masculinity', 'male privilege', 'female privilege',
            'dating market', 'hypergamy', 'alpha male', 'beta male',
            'body count', 'cock carousel', 'wall hitting',
            'simping', 'simp', 'white knight', 'pick me girl'
        ]
        
        collected_data = []
        
        for subreddit_name in target_subreddits:
            try:
                logger.info(f"Collecting from r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Collect recent posts
                for submission in subreddit.hot(limit=100):
                    submission_data = self.extract_submission_data(submission)
                    collected_data.append(submission_data)
                    
                    # Collect top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:20]:  # Top 20 comments
                        comment_data = self.extract_comment_data(comment, submission)
                        collected_data.append(comment_data)
                
                # Search for specific keywords
                for keyword in search_keywords[:5]:  # Limit to avoid rate limits
                    try:
                        for submission in subreddit.search(keyword, limit=50, sort='new'):
                            submission_data = self.extract_submission_data(submission, search_keyword=keyword)
                            collected_data.append(submission_data)
                    except Exception as e:
                        logger.warning(f"Search failed for '{keyword}' in r/{subreddit_name}: {e}")
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                continue
        
        return pd.DataFrame(collected_data)
    
    def extract_submission_data(self, submission, search_keyword=None):
        """Extract comprehensive data from Reddit submission."""
        return {
            'id': submission.id,
            'type': 'submission',
            'subreddit': submission.subreddit.display_name,
            'title': submission.title,
            'text': submission.selftext,
            'author': str(submission.author) if submission.author else '[deleted]',
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': datetime.fromtimestamp(submission.created_utc),
            'url': submission.url,
            'flair': submission.link_flair_text,
            'gilded': submission.gilded,
            'search_keyword': search_keyword,
            
            # Author information for demographics
            'author_account_age_days': self.get_account_age(submission.author),
            'author_karma': self.get_author_karma(submission.author),
            'author_verified': self.is_verified(submission.author),
            
            # Collection metadata
            'collected_timestamp': datetime.now(),
            'collection_method': 'keyword_search' if search_keyword else 'hot_posts'
        }
    
    def extract_comment_data(self, comment, submission):
        """Extract comprehensive data from Reddit comment."""
        return {
            'id': comment.id,
            'type': 'comment',
            'subreddit': submission.subreddit.display_name,
            'title': submission.title,  # Parent submission title
            'text': comment.body,
            'author': str(comment.author) if comment.author else '[deleted]',
            'score': comment.score,
            'upvote_ratio': None,  # Not available for comments
            'num_comments': len(comment.replies) if hasattr(comment, 'replies') else 0,
            'created_utc': datetime.fromtimestamp(comment.created_utc),
            'url': f"https://reddit.com{comment.permalink}",
            'flair': None,
            'gilded': comment.gilded,
            'search_keyword': None,
            'parent_submission_id': submission.id,
            'comment_depth': comment.depth if hasattr(comment, 'depth') else 0,
            
            # Author information for demographics
            'author_account_age_days': self.get_account_age(comment.author),
            'author_karma': self.get_author_karma(comment.author),
            'author_verified': self.is_verified(comment.author),
            
            # Collection metadata
            'collected_timestamp': datetime.now(),
            'collection_method': 'comment_collection'
        }
    
    def get_account_age(self, author):
        """Calculate account age in days."""
        try:
            if author and hasattr(author, 'created_utc'):
                account_created = datetime.fromtimestamp(author.created_utc)
                return (datetime.now() - account_created).days
        except:
            pass
        return None
    
    def get_author_karma(self, author):
        """Get author's karma scores."""
        try:
            if author:
                return {
                    'comment_karma': author.comment_karma,
                    'link_karma': author.link_karma,
                    'total_karma': author.comment_karma + author.link_karma
                }
        except:
            pass
        return {'comment_karma': None, 'link_karma': None, 'total_karma': None}
    
    def is_verified(self, author):
        """Check if author is verified."""
        try:
            if author and hasattr(author, 'is_gold'):
                return author.is_gold
        except:
            pass
        return False
    
    def collect_event_focused_data(self, events_list):
        """
        Collect data around specific events for correlation analysis.
        
        events_list: List of dicts with 'date', 'name', 'keywords'
        """
        event_data = []
        
        for event in events_list:
            event_date = pd.to_datetime(event['date'])
            
            # Search for posts around the event date
            for keyword in event.get('keywords', []):
                try:
                    # Use Reddit's search with time filters
                    search_results = self.reddit.subreddit('all').search(
                        f"{keyword} after:{event_date - timedelta(days=7)} before:{event_date + timedelta(days=7)}",
                        limit=200,
                        sort='new'
                    )
                    
                    for submission in search_results:
                        data = self.extract_submission_data(submission, search_keyword=keyword)
                        data['event_name'] = event['name']
                        data['event_date'] = event_date
                        data['days_from_event'] = (data['created_utc'] - event_date).days
                        event_data.append(data)
                        
                except Exception as e:
                    logger.error(f"Error collecting event data for {event['name']}: {e}")
        
        return pd.DataFrame(event_data)
    
    def collect_demographic_focused_data(self):
        """
        Collect data from age and gender-specific communities for demographic analysis.
        """
        demographic_subreddits = {
            'age_focused': {
                'teenagers': {'likely_age_group': '13-19'},
                'college': {'likely_age_group': '18-22'},
                'adulting': {'likely_age_group': '22-35'},
                'RedditForGrownups': {'likely_age_group': '35+'},
                'GenZ': {'likely_age_group': '18-26'},
                'Millennials': {'likely_age_group': '27-42'},
                'GenX': {'likely_age_group': '43-58'},
                'BoomersBeingFools': {'likely_age_group': '59+'}
            },
            'gender_focused': {
                'TwoXChromosomes': {'likely_gender': 'female'},
                'AskWomen': {'likely_gender': 'female'},
                'MensRights': {'likely_gender': 'male'},
                'AskMen': {'likely_gender': 'male'},
                'MaleGrooming': {'likely_gender': 'male'},
                'FemaleFashionAdvice': {'likely_gender': 'female'},
                'malefashionadvice': {'likely_gender': 'male'}
            }
        }
        
        demographic_data = []
        
        for category, subreddits in demographic_subreddits.items():
            for subreddit_name, attributes in subreddits.items():
                try:
                    logger.info(f"Collecting demographic data from r/{subreddit_name}")
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    for submission in subreddit.hot(limit=100):
                        data = self.extract_submission_data(submission)
                        data.update(attributes)
                        data['demographic_category'] = category
                        demographic_data.append(data)
                        
                        # Collect comments for more demographic indicators
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:10]:
                            comment_data = self.extract_comment_data(comment, submission)
                            comment_data.update(attributes)
                            comment_data['demographic_category'] = category
                            demographic_data.append(comment_data)
                    
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error collecting from r/{subreddit_name}: {e}")
        
        return pd.DataFrame(demographic_data)

def main():
    """Main collection function."""
    print("🔍 Enhanced Reddit Data Collection for Research Questions")
    print("=" * 60)
    
    collector = EnhancedRedditCollector()
    
    if not collector.reddit:
        print("❌ Reddit API connection failed. Please check your credentials.")
        print("\nTo set up Reddit API access:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new application")
        print("3. Update the credentials in this script")
        return
    
    # Create data directory
    data_dir = Path("/Users/malaikarashid/Documents/353/MisogynyWatch/data/enhanced_reddit")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Major events for correlation analysis
    major_events = [
        {
            'date': '2016-11-08',
            'name': '2016_US_Election',
            'keywords': ['trump', 'election', 'Hillary Clinton', 'women voters']
        },
        {
            'date': '2017-10-05', 
            'name': 'MeToo_Movement',
            'keywords': ['metoo', 'sexual harassment', 'Weinstein', 'believe women']
        },
        {
            'date': '2018-04-23',
            'name': 'Toronto_Van_Attack',
            'keywords': ['incel', 'Elliot Rodger', 'Toronto attack', 'violent misogyny']
        },
        {
            'date': '2022-06-24',
            'name': 'Roe_v_Wade_Overturned', 
            'keywords': ['abortion', 'Roe v Wade', 'reproductive rights', 'women rights']
        },
        {
            'date': '2023-01-01',
            'name': 'Andrew_Tate_Arrest',
            'keywords': ['Andrew Tate', 'red pill', 'male influencer', 'misogyny arrest']
        }
    ]
    
    try:
        # 1. Collect general temporal data
        print("📊 Collecting temporal trend data...")
        temporal_data = collector.collect_temporal_data()
        if temporal_data is not None and len(temporal_data) > 0:
            temporal_data.to_csv(data_dir / 'temporal_reddit_data.csv', index=False)
            print(f"✅ Collected {len(temporal_data):,} posts for temporal analysis")
        
        # 2. Collect event-focused data
        print("⚡ Collecting event correlation data...")
        event_data = collector.collect_event_focused_data(major_events)
        if len(event_data) > 0:
            event_data.to_csv(data_dir / 'event_reddit_data.csv', index=False)
            print(f"✅ Collected {len(event_data):,} posts for event analysis")
        
        # 3. Collect demographic-focused data
        print("👥 Collecting demographic data...")
        demographic_data = collector.collect_demographic_focused_data()
        if len(demographic_data) > 0:
            demographic_data.to_csv(data_dir / 'demographic_reddit_data.csv', index=False)
            print(f"✅ Collected {len(demographic_data):,} posts for demographic analysis")
        
        print("\n" + "=" * 60)
        print("✅ Enhanced data collection complete!")
        print(f"📁 Data saved to: {data_dir}")
        print("\nNext steps:")
        print("1. Run the research_focused_analysis.py script")
        print("2. Process the new data with misogyny detection")
        print("3. Generate comprehensive research visualizations")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
