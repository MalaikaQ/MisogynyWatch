"""
Enhanced demographics analyzer with comprehensive age and gender extraction.
Supports both comment-based and account-based demographic extraction.
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

class EnhancedDemographicsAnalyzer:
    """Enhanced analyzer for age and gender demographics in misogynistic content."""
    
    def __init__(self):
        """Initialize the enhanced demographics analyzer."""
        
        # Age extraction patterns (comprehensive)
        self.age_patterns = [
            r'\b(?:i\'?m|am)\s+(\d{1,2})\s*(?:years?\s*old|y\.?o\.?|yr)\b',
            r'\b(\d{1,2})\s*(?:year\s*old|y\.?o\.?|yr\s*old)\b',
            r'\bas\s+a\s+(\d{1,2})\s*(?:year\s*old|y\.?o\.?)\b',
            r'\bage\s*[:=]?\s*(\d{1,2})\b',
            r'\b(\d{1,2})\s*(?:male|female|m|f)\b',
            r'\b(?:turning|turned)\s+(\d{1,2})\b',
            r'\bborn\s+in\s+(\d{4})\b',  # Birth year
            r'\b(\d{1,2})\s*(?:yo|y/o)\b',  # Short forms
            r'\bat\s+(\d{1,2})\s*(?:years?\s*old)?\b'
        ]
        
        # Gender extraction patterns (comprehensive)
        self.gender_patterns = {
            'male': [
                r'\b(?:i\'?m|am)\s+(?:a\s+)?(?:guy|man|male|boy|dude)\b',
                r'\bas\s+a\s+(?:guy|man|male|boy)\b',
                r'\b(?:straight|gay|bi)\s+(?:guy|man|male)\b',
                r'\b\d+\s*(?:m|male)\b',
                r'\b(?:he/him|he\/him)\b',
                r'\bmale\s+here\b',
                r'\bguy\s+here\b',
                r'\bman\s+here\b'
            ],
            'female': [
                r'\b(?:i\'?m|am)\s+(?:a\s+)?(?:girl|woman|female|lady|gal)\b',
                r'\bas\s+a\s+(?:girl|woman|female|lady)\b',
                r'\b(?:straight|gay|bi|lesbian)\s+(?:girl|woman|female)\b',
                r'\b\d+\s*(?:f|female)\b',
                r'\b(?:she/her|she\/her)\b',
                r'\bfemale\s+here\b',
                r'\bgirl\s+here\b',
                r'\bwoman\s+here\b'
            ],
            'non_binary': [
                r'\b(?:non-binary|nonbinary|nb|enby)\b',
                r'\b(?:they/them|they\/them)\b',
                r'\bgenderfluid\b',
                r'\bagender\b',
                r'\bgenderqueer\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_age_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.age_patterns]
        self.compiled_gender_patterns = {
            gender: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for gender, patterns in self.gender_patterns.items()
        }
        
        # Age group definitions
        self.age_groups = {
            'Gen Z (13-18)': (13, 18),
            'Young Adults (19-25)': (19, 25), 
            'Millennials (26-35)': (26, 35),
            'Gen X (36-45)': (36, 45),
            'Middle Age (46-55)': (46, 55),
            'Older Adults (56+)': (56, 100)
        }
        
        # Platform-specific joining age patterns
        self.platform_join_patterns = {
            'reddit': (16, 25),
            'twitter': (16, 28),
            'youtube': (14, 22),
            'instagram': (14, 24),
            'tiktok': (13, 19),
            'facebook': (18, 35),
            'default': (16, 25)
        }
    
    def extract_age_from_text(self, text: str) -> Optional[int]:
        """
        Extract age information from text content using multiple patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Extracted age or None if not found
        """
        if not isinstance(text, str):
            return None
            
        text_lower = text.lower()
        
        # Try each age pattern
        for pattern in self.compiled_age_patterns:
            matches = pattern.findall(text_lower)
            if matches:
                for match in matches:
                    try:
                        # Handle birth year conversion
                        if len(match) == 4:  # Birth year
                            age = datetime.now().year - int(match)
                        else:
                            age = int(match)
                        
                        # Validate age range
                        if 13 <= age <= 80:
                            return age
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def extract_gender_from_text(self, text: str) -> Optional[str]:
        """
        Extract gender information from text content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected gender ('male', 'female', 'non_binary') or None
        """
        if not isinstance(text, str):
            return None
            
        text_lower = text.lower()
        
        # Check each gender category
        for gender, patterns in self.compiled_gender_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return gender
        
        return None
    
    def extract_demographics_from_profile(self, user_data: Dict) -> Dict[str, Any]:
        """
        Extract age and gender from user profile data.
        
        Args:
            user_data: Dictionary containing user profile information
            
        Returns:
            Dictionary with 'age' and 'gender' keys
        """
        demographics = {'age': None, 'gender': None}
        
        # Direct age/gender if provided
        if 'age' in user_data and user_data['age']:
            demographics['age'] = user_data['age']
        if 'gender' in user_data and user_data['gender']:
            demographics['gender'] = user_data['gender']
        
        # Extract from profile text fields
        text_fields = ['bio', 'description', 'about', 'profile_text', 'display_name', 'username']
        
        for field in text_fields:
            if field in user_data and user_data[field]:
                field_text = str(user_data[field])
                
                # Extract age if not already found
                if not demographics['age']:
                    demographics['age'] = self.extract_age_from_text(field_text)
                
                # Extract gender if not already found
                if not demographics['gender']:
                    demographics['gender'] = self.extract_gender_from_text(field_text)
                
                # Stop if both found
                if demographics['age'] and demographics['gender']:
                    break
        
        return demographics
    
    def extract_demographics_from_reddit_profile(self, reddit_data: Dict) -> Dict[str, Any]:
        """
        Extract demographics specifically from Reddit user data.
        
        Args:
            reddit_data: Dictionary containing Reddit user information
            
        Returns:
            Dictionary with 'age' and 'gender' keys
        """
        demographics = {'age': None, 'gender': None}
        
        # Reddit profile fields to check
        reddit_fields = [
            'author_flair_text',      # Subreddit flair often contains age/gender
            'author_description',      # User bio
            'author_subreddit',       # User's profile subreddit
            'author_public_description'
        ]
        
        # Check Reddit-specific fields
        for field in reddit_fields:
            if field in reddit_data and reddit_data[field]:
                field_text = str(reddit_data[field])
                
                if not demographics['age']:
                    demographics['age'] = self.extract_age_from_text(field_text)
                
                if not demographics['gender']:
                    demographics['gender'] = self.extract_gender_from_text(field_text)
        
        # Estimate age from account creation if available
        if not demographics['age'] and 'author_account_created' in reddit_data:
            try:
                if isinstance(reddit_data['author_account_created'], str):
                    account_date = datetime.strptime(reddit_data['author_account_created'], '%Y-%m-%d %H:%M:%S')
                else:
                    account_date = reddit_data['author_account_created']
                
                demographics['age'] = self.estimate_age_from_account_age(
                    account_date, 'reddit'
                )
            except Exception:
                pass
        
        # Check recent comment history if available
        if 'author_recent_comments' in reddit_data:
            for comment in reddit_data['author_recent_comments'][:10]:
                if not demographics['age']:
                    demographics['age'] = self.extract_age_from_text(comment)
                if not demographics['gender']:
                    demographics['gender'] = self.extract_gender_from_text(comment)
                
                if demographics['age'] and demographics['gender']:
                    break
        
        return demographics
    
    def extract_demographics_from_twitter_profile(self, twitter_data: Dict) -> Dict[str, Any]:
        """
        Extract demographics specifically from Twitter user data.
        
        Args:
            twitter_data: Dictionary containing Twitter user information
            
        Returns:
            Dictionary with 'age' and 'gender' keys
        """
        demographics = {'age': None, 'gender': None}
        
        # Twitter profile fields
        twitter_fields = [
            'author_description',     # Bio
            'author_name',           # Display name
            'author_location',       # Location (sometimes contains age info)
            'author_username'        # Username might contain hints
        ]
        
        # Check Twitter-specific fields
        for field in twitter_fields:
            if field in twitter_data and twitter_data[field]:
                field_text = str(twitter_data[field])
                
                if not demographics['age']:
                    demographics['age'] = self.extract_age_from_text(field_text)
                
                if not demographics['gender']:
                    demographics['gender'] = self.extract_gender_from_text(field_text)
        
        # Estimate age from account creation
        if not demographics['age'] and 'author_account_created' in twitter_data:
            try:
                if isinstance(twitter_data['author_account_created'], str):
                    account_date = datetime.strptime(twitter_data['author_account_created'], '%Y-%m-%d %H:%M:%S')
                else:
                    account_date = twitter_data['author_account_created']
                
                demographics['age'] = self.estimate_age_from_account_age(
                    account_date, 'twitter'
                )
            except Exception:
                pass
        
        return demographics
    
    def estimate_age_from_account_age(self, account_created: datetime, 
                                     platform: str = 'default') -> Optional[int]:
        """
        Estimate user age based on account creation date and platform demographics.
        
        Args:
            account_created: When the account was created
            platform: Platform name for specific patterns
            
        Returns:
            Estimated current age or None
        """
        years_since_creation = (datetime.now() - account_created).days / 365.25
        
        # Get platform-specific joining age patterns
        join_range = self.platform_join_patterns.get(platform, self.platform_join_patterns['default'])
        avg_join_age = sum(join_range) / 2
        
        estimated_current_age = avg_join_age + years_since_creation
        
        # Return reasonable age estimate
        if 13 <= estimated_current_age <= 80:
            return int(estimated_current_age)
        
        return None
    
    def assign_age_group(self, age: int) -> str:
        """
        Assign age to demographic group.
        
        Args:
            age: User age
            
        Returns:
            Age group name
        """
        if age is None:
            return 'Unknown'
            
        for group_name, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group_name
        return 'Unknown'
    
    def analyze_demographics_comprehensive(self, df: pd.DataFrame, 
                                         text_column: str = 'text',
                                         platform_column: str = 'platform',
                                         misogyny_column: str = 'misogyny_score') -> pd.DataFrame:
        """
        Comprehensive demographic analysis combining all extraction methods.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text content
            platform_column: Column indicating platform
            misogyny_column: Column containing misogyny scores
            
        Returns:
            DataFrame with comprehensive demographic analysis
        """
        df_demo = df.copy()
        
        print("Starting comprehensive demographic analysis...")
        
        # Initialize demographic columns
        df_demo['extracted_age_text'] = None
        df_demo['extracted_gender_text'] = None
        df_demo['extracted_age_profile'] = None
        df_demo['extracted_gender_profile'] = None
        df_demo['estimated_age_account'] = None
        
        # 1. Extract from text content
        print("  Extracting demographics from text content...")
        df_demo['extracted_age_text'] = df_demo[text_column].apply(self.extract_age_from_text)
        df_demo['extracted_gender_text'] = df_demo[text_column].apply(self.extract_gender_from_text)
        
        # 2. Platform-specific profile extraction
        if platform_column in df_demo.columns:
            print("  Applying platform-specific extraction...")
            
            # Reddit extraction
            reddit_mask = df_demo[platform_column] == 'reddit'
            if reddit_mask.any():
                print("    Processing Reddit profiles...")
                for idx in df_demo[reddit_mask].index:
                    reddit_demo = self.extract_demographics_from_reddit_profile(df_demo.loc[idx].to_dict())
                    if reddit_demo['age']:
                        df_demo.loc[idx, 'extracted_age_profile'] = reddit_demo['age']
                    if reddit_demo['gender']:
                        df_demo.loc[idx, 'extracted_gender_profile'] = reddit_demo['gender']
            
            # Twitter extraction
            twitter_mask = df_demo[platform_column] == 'twitter'
            if twitter_mask.any():
                print("    Processing Twitter profiles...")
                for idx in df_demo[twitter_mask].index:
                    twitter_demo = self.extract_demographics_from_twitter_profile(df_demo.loc[idx].to_dict())
                    if twitter_demo['age']:
                        df_demo.loc[idx, 'extracted_age_profile'] = twitter_demo['age']
                    if twitter_demo['gender']:
                        df_demo.loc[idx, 'extracted_gender_profile'] = twitter_demo['gender']
        
        # 3. Account age estimation
        account_date_columns = ['author_account_created', 'account_created', 'created_at']
        for date_col in account_date_columns:
            if date_col in df_demo.columns:
                print(f"    Estimating ages from {date_col}...")
                for idx, row in df_demo.iterrows():
                    if pd.notna(row[date_col]) and not df_demo.loc[idx, 'estimated_age_account']:
                        platform = row.get(platform_column, 'default')
                        estimated_age = self.estimate_age_from_account_age(row[date_col], platform)
                        if estimated_age:
                            df_demo.loc[idx, 'estimated_age_account'] = estimated_age
                break
        
        # 4. Combine sources (prioritize: text > profile > account estimation)
        print("  Combining demographic sources...")
        df_demo['final_age'] = (
            df_demo['extracted_age_text']
            .fillna(df_demo['extracted_age_profile'])
            .fillna(df_demo['estimated_age_account'])
        )
        
        df_demo['final_gender'] = (
            df_demo['extracted_gender_text']
            .fillna(df_demo['extracted_gender_profile'])
        )
        
        # 5. Assign categories
        df_demo['age_group'] = df_demo['final_age'].apply(self.assign_age_group)
        
        # 6. Create extraction method tracking
        df_demo['age_extraction_method'] = 'unknown'
        df_demo.loc[df_demo['extracted_age_text'].notna(), 'age_extraction_method'] = 'text'
        df_demo.loc[(df_demo['extracted_age_text'].isna()) & 
                   (df_demo['extracted_age_profile'].notna()), 'age_extraction_method'] = 'profile'
        df_demo.loc[(df_demo['extracted_age_text'].isna()) & 
                   (df_demo['extracted_age_profile'].isna()) & 
                   (df_demo['estimated_age_account'].notna()), 'age_extraction_method'] = 'account_estimate'
        
        df_demo['gender_extraction_method'] = 'unknown'
        df_demo.loc[df_demo['extracted_gender_text'].notna(), 'gender_extraction_method'] = 'text'
        df_demo.loc[(df_demo['extracted_gender_text'].isna()) & 
                   (df_demo['extracted_gender_profile'].notna()), 'gender_extraction_method'] = 'profile'
        
        print("  Demographic analysis complete!")
        
        return df_demo
    
    def generate_extraction_effectiveness_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate report on extraction method effectiveness.
        
        Args:
            df: DataFrame with demographic analysis
            
        Returns:
            Dictionary containing effectiveness metrics
        """
        total_posts = len(df)
        
        # Age extraction effectiveness
        age_by_method = df['age_extraction_method'].value_counts()
        age_coverage = (total_posts - age_by_method.get('unknown', 0)) / total_posts * 100
        
        # Gender extraction effectiveness
        gender_by_method = df['gender_extraction_method'].value_counts()
        gender_coverage = (total_posts - gender_by_method.get('unknown', 0)) / total_posts * 100
        
        # Platform-specific effectiveness
        platform_effectiveness = {}
        if 'platform' in df.columns:
            for platform in df['platform'].unique():
                platform_data = df[df['platform'] == platform]
                platform_age_coverage = (len(platform_data) - platform_data['age_extraction_method'].value_counts().get('unknown', 0)) / len(platform_data) * 100
                platform_gender_coverage = (len(platform_data) - platform_data['gender_extraction_method'].value_counts().get('unknown', 0)) / len(platform_data) * 100
                
                platform_effectiveness[platform] = {
                    'age_coverage': platform_age_coverage,
                    'gender_coverage': platform_gender_coverage,
                    'total_posts': len(platform_data)
                }
        
        report = {
            'total_posts': total_posts,
            'age_extraction': {
                'overall_coverage': age_coverage,
                'by_method': age_by_method.to_dict(),
                'text_extraction_rate': age_by_method.get('text', 0) / total_posts * 100,
                'profile_extraction_rate': age_by_method.get('profile', 0) / total_posts * 100,
                'account_estimation_rate': age_by_method.get('account_estimate', 0) / total_posts * 100
            },
            'gender_extraction': {
                'overall_coverage': gender_coverage,
                'by_method': gender_by_method.to_dict(),
                'text_extraction_rate': gender_by_method.get('text', 0) / total_posts * 100,
                'profile_extraction_rate': gender_by_method.get('profile', 0) / total_posts * 100
            },
            'platform_effectiveness': platform_effectiveness
        }
        
        return report

def test_extraction_methods():
    """Test the different extraction methods."""
    analyzer = EnhancedDemographicsAnalyzer()
    
    print("=== TESTING DEMOGRAPHIC EXTRACTION METHODS ===\n")
    
    # Test text extraction
    print("1. TEXT-BASED EXTRACTION:")
    test_texts = [
        "I'm 22 years old male and I think women are...",
        "As a 19 year old girl, I believe...",
        "I am 25M and recently discovered the red pill",
        "30F here, been dealing with misogyny for years",
        "Age 28, non-binary, thoughts on gender dynamics"
    ]
    
    for text in test_texts:
        age = analyzer.extract_age_from_text(text)
        gender = analyzer.extract_gender_from_text(text)
        age_group = analyzer.assign_age_group(age) if age else 'Unknown'
        print(f"Text: '{text[:50]}...'")
        print(f"  Age: {age}, Gender: {gender}, Group: {age_group}\n")
    
    # Test profile extraction
    print("2. PROFILE-BASED EXTRACTION:")
    test_profiles = [
        {
            'bio': '24 year old software engineer, he/him pronouns',
            'display_name': 'Tech_Guy_24'
        },
        {
            'author_description': 'Mother of two, 32, she/her',
            'author_flair_text': '32F'
        },
        {
            'about': 'College student, 20, they/them, studying psychology'
        }
    ]
    
    for i, profile in enumerate(test_profiles, 1):
        demographics = analyzer.extract_demographics_from_profile(profile)
        print(f"Profile {i}: {profile}")
        print(f"  Extracted: Age={demographics['age']}, Gender={demographics['gender']}\n")
    
    # Test account age estimation
    print("3. ACCOUNT AGE ESTIMATION:")
    test_accounts = [
        {'created': datetime(2020, 1, 1), 'platform': 'reddit'},
        {'created': datetime(2018, 6, 15), 'platform': 'twitter'},
        {'created': datetime(2022, 3, 20), 'platform': 'youtube'}
    ]
    
    for account in test_accounts:
        estimated_age = analyzer.estimate_age_from_account_age(
            account['created'], account['platform']
        )
        years_since = (datetime.now() - account['created']).days / 365.25
        print(f"Account created: {account['created'].strftime('%Y-%m-%d')} ({account['platform']})")
        print(f"  Years since creation: {years_since:.1f}")
        print(f"  Estimated current age: {estimated_age}\n")

if __name__ == "__main__":
    test_extraction_methods()
