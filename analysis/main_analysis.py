"""
Main analysis module for MisogynyWatch project.
Addresses the four main research questions:
1. Has misogynistic language increased over time?
2. Correlation with red-pill influencer events?
3. Which communities are most affected?
4. Which age groups and gender are most affected?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RED_PILL_EVENTS, PROCESSED_DATA_DIR
from text_processing import TextProcessor
from demographics_analyzer import EnhancedDemographicsAnalyzer

warnings.filterwarnings('ignore')

class MisogynyAnalyzer:
    """Main analysis class for misogyny research."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.text_processor = TextProcessor()
        self.demographics_analyzer = EnhancedDemographicsAnalyzer()
        self.red_pill_events = RED_PILL_EVENTS
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed Reddit and Twitter data.
        
        Returns:
            Tuple of (reddit_df, twitter_df)
        """
        try:
            reddit_df = pd.read_csv(PROCESSED_DATA_DIR / 'reddit_processed.csv')
            twitter_df = pd.read_csv(PROCESSED_DATA_DIR / 'twitter_processed.csv')
            
            # Convert date columns
            reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'])
            twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'])
            
            print(f"Loaded {len(reddit_df)} Reddit posts and {len(twitter_df)} Twitter posts")
            return reddit_df, twitter_df
            
        except FileNotFoundError as e:
            print(f"Data files not found: {e}")
            print("Please run data collection first using reddit_scraper.py and twitter_scraper.py")
            return pd.DataFrame(), pd.DataFrame()
    
    def analyze_temporal_trends(self, reddit_df: pd.DataFrame, twitter_df: pd.DataFrame) -> Dict:
        """
        Research Question 1: Has misogynistic language increased over time?
        
        Args:
            reddit_df: Reddit data
            twitter_df: Twitter data
            
        Returns:
            Dictionary with temporal analysis results
        """
        results = {}
        
        # Combine datasets for overall trend
        reddit_df['platform'] = 'Reddit'
        twitter_df['platform'] = 'Twitter'
        
        # Standardize date column names
        reddit_df['date'] = reddit_df['created_utc']
        twitter_df['date'] = twitter_df['created_at']
        
        combined_df = pd.concat([
            reddit_df[['date', 'misogyny_score', 'platform', 'text']],
            twitter_df[['date', 'misogyny_score', 'platform', 'text']]
        ])
        
        # Group by month for trend analysis
        combined_df['year_month'] = combined_df['date'].dt.to_period('M')
        monthly_trends = combined_df.groupby(['year_month', 'platform']).agg({
            'misogyny_score': ['mean', 'count'],
            'text': 'count'
        }).reset_index()
        
        monthly_trends.columns = ['year_month', 'platform', 'avg_misogyny_score', 
                                'misogynistic_posts', 'total_posts']
        
        # Calculate percentage of misogynistic content
        monthly_trends['misogyny_percentage'] = (
            monthly_trends['misogynistic_posts'] / monthly_trends['total_posts'] * 100
        )
        
        results['monthly_trends'] = monthly_trends
        
        # Statistical trend analysis
        from scipy import stats
        
        for platform in ['Reddit', 'Twitter']:
            platform_data = monthly_trends[monthly_trends['platform'] == platform]
            if len(platform_data) > 3:
                # Time series as numeric for correlation
                time_numeric = np.arange(len(platform_data))
                
                # Test for increasing trend in misogyny score
                slope_score, intercept_score, r_score, p_score, _ = stats.linregress(
                    time_numeric, platform_data['avg_misogyny_score']
                )
                
                # Test for increasing trend in percentage
                slope_pct, intercept_pct, r_pct, p_pct, _ = stats.linregress(
                    time_numeric, platform_data['misogyny_percentage']
                )
                
                results[f'{platform.lower()}_trends'] = {
                    'score_slope': slope_score,
                    'score_p_value': p_score,
                    'score_correlation': r_score,
                    'percentage_slope': slope_pct,
                    'percentage_p_value': p_pct,
                    'percentage_correlation': r_pct,
                    'trend_direction': 'increasing' if slope_pct > 0 else 'decreasing',
                    'significant': p_pct < 0.05
                }
        
        return results
    
    def analyze_event_impact(self, reddit_df: pd.DataFrame, twitter_df: pd.DataFrame) -> Dict:
        """
        Research Question 2: Correlation with red-pill influencer events?
        
        Args:
            reddit_df: Reddit data
            twitter_df: Twitter data
            
        Returns:
            Dictionary with event impact analysis results
        """
        results = {}
        
        # Combine datasets
        reddit_df['platform'] = 'Reddit'
        twitter_df['platform'] = 'Twitter'
        reddit_df['date'] = reddit_df['created_utc']
        twitter_df['date'] = twitter_df['created_at']
        
        combined_df = pd.concat([
            reddit_df[['date', 'misogyny_score', 'platform']],
            twitter_df[['date', 'misogyny_score', 'platform']]
        ])
        
        event_impacts = []
        
        for event_date_str, event_description in self.red_pill_events.items():
            event_date = pd.to_datetime(event_date_str)
            
            # Define time windows
            before_window = (event_date - timedelta(days=7), event_date)
            after_window = (event_date, event_date + timedelta(days=7))
            
            # Get data for each window
            before_data = combined_df[
                (combined_df['date'] >= before_window[0]) & 
                (combined_df['date'] < before_window[1])
            ]
            
            after_data = combined_df[
                (combined_df['date'] >= after_window[0]) & 
                (combined_df['date'] < after_window[1])
            ]
            
            if len(before_data) > 10 and len(after_data) > 10:
                # Calculate statistics
                before_mean = before_data['misogyny_score'].mean()
                after_mean = after_data['misogyny_score'].mean()
                
                # Statistical test for difference
                from scipy.stats import ttest_ind
                statistic, p_value = ttest_ind(
                    before_data['misogyny_score'], 
                    after_data['misogyny_score']
                )
                
                event_impacts.append({
                    'event_date': event_date,
                    'event_description': event_description,
                    'before_mean': before_mean,
                    'after_mean': after_mean,
                    'change': after_mean - before_mean,
                    'percent_change': ((after_mean - before_mean) / before_mean) * 100,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'before_count': len(before_data),
                    'after_count': len(after_data)
                })
        
        results['event_impacts'] = pd.DataFrame(event_impacts)
        
        # Overall correlation with event density
        # Create event density timeline
        all_dates = combined_df['date'].dt.date.unique()
        event_timeline = pd.DataFrame({'date': all_dates})
        event_timeline['event_proximity'] = 0
        
        for event_date_str in self.red_pill_events.keys():
            event_date = pd.to_datetime(event_date_str).date()
            for i, date in enumerate(all_dates):
                days_diff = abs((date - event_date).days)
                if days_diff <= 14:  # Within 2 weeks of event
                    event_timeline.loc[i, 'event_proximity'] = max(
                        event_timeline.loc[i, 'event_proximity'],
                        1 / (1 + days_diff)  # Decay function
                    )
        
        # Correlate with daily misogyny scores
        daily_scores = combined_df.groupby(combined_df['date'].dt.date)['misogyny_score'].mean()
        event_timeline = event_timeline.set_index('date')
        
        correlation_data = event_timeline.join(daily_scores, how='inner')
        
        if len(correlation_data) > 30:
            from scipy.stats import pearsonr
            correlation, p_value = pearsonr(
                correlation_data['event_proximity'], 
                correlation_data['misogyny_score']
            )
            
            results['event_correlation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
    
    def analyze_community_differences(self, reddit_df: pd.DataFrame, twitter_df: pd.DataFrame) -> Dict:
        """
        Research Question 3: Which communities are most affected?
        
        Args:
            reddit_df: Reddit data
            twitter_df: Twitter data
            
        Returns:
            Dictionary with community analysis results
        """
        results = {}
        
        # Reddit community analysis
        if 'subreddit' in reddit_df.columns:
            reddit_community_stats = reddit_df.groupby('subreddit').agg({
                'misogyny_score': ['mean', 'std', 'count'],
                'text': 'count'
            }).reset_index()
            
            reddit_community_stats.columns = [
                'subreddit', 'avg_misogyny_score', 'std_misogyny_score', 
                'misogynistic_posts', 'total_posts'
            ]
            
            reddit_community_stats['misogyny_percentage'] = (
                reddit_community_stats['misogynistic_posts'] / 
                reddit_community_stats['total_posts'] * 100
            )
            
            # Sort by misogyny percentage
            reddit_community_stats = reddit_community_stats.sort_values(
                'misogyny_percentage', ascending=False
            )
            
            results['reddit_communities'] = reddit_community_stats
        
        # Twitter hashtag analysis (if available)
        if 'hashtags' in twitter_df.columns:
            # Extract and analyze hashtags
            hashtag_data = []
            for idx, row in twitter_df.iterrows():
                if pd.notna(row['hashtags']) and row['hashtags'] != '[]':
                    hashtags = eval(row['hashtags'])  # Assuming stored as string representation of list
                    for hashtag in hashtags:
                        hashtag_data.append({
                            'hashtag': hashtag.lower(),
                            'misogyny_score': row['misogyny_score']
                        })
            
            if hashtag_data:
                hashtag_df = pd.DataFrame(hashtag_data)
                hashtag_stats = hashtag_df.groupby('hashtag').agg({
                    'misogyny_score': ['mean', 'count']
                }).reset_index()
                
                hashtag_stats.columns = ['hashtag', 'avg_misogyny_score', 'count']
                hashtag_stats = hashtag_stats[hashtag_stats['count'] >= 5]  # Min 5 occurrences
                hashtag_stats = hashtag_stats.sort_values('avg_misogyny_score', ascending=False)
                
                results['twitter_hashtags'] = hashtag_stats
        
        # Cross-platform comparison
        reddit_df['platform'] = 'Reddit'
        twitter_df['platform'] = 'Twitter'
        
        platform_comparison = pd.concat([
            reddit_df[['platform', 'misogyny_score']],
            twitter_df[['platform', 'misogyny_score']]
        ]).groupby('platform').agg({
            'misogyny_score': ['mean', 'std', 'count']
        }).reset_index()
        
        platform_comparison.columns = ['platform', 'avg_misogyny_score', 'std_misogyny_score', 'count']
        results['platform_comparison'] = platform_comparison
        
        # Statistical test for platform differences
        from scipy.stats import ttest_ind
        reddit_scores = reddit_df['misogyny_score'].dropna()
        twitter_scores = twitter_df['misogyny_score'].dropna()
        
        if len(reddit_scores) > 10 and len(twitter_scores) > 10:
            statistic, p_value = ttest_ind(reddit_scores, twitter_scores)
            results['platform_difference_test'] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'higher_platform': 'Reddit' if reddit_scores.mean() > twitter_scores.mean() else 'Twitter'
            }
        
        return results
    
    def analyze_age_demographics(self, reddit_df: pd.DataFrame, twitter_df: pd.DataFrame) -> Dict:
        """
        Research Question 4: Which age groups and gender are most affected?
        
        Args:
            reddit_df: Reddit data
            twitter_df: Twitter data
            
        Returns:
            Dictionary with age demographics analysis results
        """
        results = {}
        
        # Combine datasets
        reddit_df['platform'] = 'Reddit'
        twitter_df['platform'] = 'Twitter'
        
        combined_df = pd.concat([
            reddit_df[['platform', 'misogyny_score', 'text', 'estimated_age', 'estimated_gender']],
            twitter_df[['platform', 'misogyny_score', 'text', 'estimated_age', 'estimated_gender']]
        ])
        
        # Age group analysis
        age_data = combined_df[combined_df['estimated_age'].notna()].copy()
        
        if len(age_data) > 0:
            # Categorize ages into groups
            age_data['age_group'] = age_data['estimated_age'].apply(self.age_analyzer.categorize_age)
            
            age_group_stats = age_data.groupby('age_group').agg({
                'misogyny_score': ['mean', 'std', 'count'],
                'text': 'count'
            }).reset_index()
            
            age_group_stats.columns = [
                'age_group', 'avg_misogyny_score', 'std_misogyny_score', 
                'misogynistic_posts', 'total_posts'
            ]
            
            age_group_stats['misogyny_percentage'] = (
                age_group_stats['misogynistic_posts'] / 
                age_group_stats['total_posts'] * 100
            )
            
            results['age_groups'] = age_group_stats.sort_values('avg_misogyny_score', ascending=False)
        
        # Gender analysis
        gender_data = combined_df[combined_df['estimated_gender'].notna()].copy()
        
        if len(gender_data) > 0:
            gender_stats = gender_data.groupby('estimated_gender').agg({
                'misogyny_score': ['mean', 'std', 'count'],
                'text': 'count'
            }).reset_index()
            
            gender_stats.columns = [
                'gender', 'avg_misogyny_score', 'std_misogyny_score', 
                'misogynistic_posts', 'total_posts'
            ]
            
            gender_stats['misogyny_percentage'] = (
                gender_stats['misogynistic_posts'] / 
                gender_stats['total_posts'] * 100
            )
            
            results['gender_analysis'] = gender_stats
        
        # Age-Gender intersection analysis
        if len(age_data) > 0 and 'estimated_gender' in age_data.columns:
            intersection_data = age_data[age_data['estimated_gender'].notna()]
            
            if len(intersection_data) > 0:
                intersection_stats = intersection_data.groupby(['age_group', 'estimated_gender']).agg({
                    'misogyny_score': ['mean', 'count']
                }).reset_index()
                
                intersection_stats.columns = ['age_group', 'gender', 'avg_misogyny_score', 'count']
                intersection_stats = intersection_stats[intersection_stats['count'] >= 5]  # Min 5 posts
                
                results['age_gender_intersection'] = intersection_stats.sort_values(
                    'avg_misogyny_score', ascending=False
                )
        
        return results
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive analysis report addressing all research questions.
        
        Returns:
            Dictionary with all analysis results
        """
        print("Loading data...")
        reddit_df, twitter_df = self.load_data()
        
        if reddit_df.empty or twitter_df.empty:
            return {"error": "No data available for analysis"}
        
        print("Analyzing temporal trends...")
        temporal_results = self.analyze_temporal_trends(reddit_df, twitter_df)
        
        print("Analyzing event impacts...")
        event_results = self.analyze_event_impact(reddit_df, twitter_df)
        
        print("Analyzing community differences...")
        community_results = self.analyze_community_differences(reddit_df, twitter_df)
        
        print("Analyzing age demographics...")
        age_results = self.analyze_age_demographics(reddit_df, twitter_df)
        
        # Combine all results
        comprehensive_results = {
            'temporal_trends': temporal_results,
            'event_impacts': event_results,
            'community_differences': community_results,
            'age_demographics': age_results,
            'data_summary': {
                'reddit_posts': len(reddit_df),
                'twitter_posts': len(twitter_df),
                'date_range': {
                    'start': min(reddit_df['created_utc'].min(), twitter_df['created_at'].min()),
                    'end': max(reddit_df['created_utc'].max(), twitter_df['created_at'].max())
                }
            }
        }
        
        return comprehensive_results

if __name__ == "__main__":
    analyzer = MisogynyAnalyzer()
    results = analyzer.generate_comprehensive_report()
    
    # Print summary
    if "error" not in results:
        print("\n=== MISOGYNY ANALYSIS SUMMARY ===")
        print(f"Data analyzed: {results['data_summary']['reddit_posts']} Reddit posts, "
              f"{results['data_summary']['twitter_posts']} Twitter posts")
        print(f"Date range: {results['data_summary']['date_range']['start']} to "
              f"{results['data_summary']['date_range']['end']}")
        
        # Save results
        import json
        output_path = PROCESSED_DATA_DIR / 'analysis_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    else:
        print(f"Analysis failed: {results['error']}")
