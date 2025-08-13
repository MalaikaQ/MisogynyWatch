#!/usr/bin/env python3
"""
Enhanced Reddit Analysis with Proper Lexicon-Based Detection
Focuses on answering specific research questions with improved misogyny detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import re
from typing import Dict, List, Tuple
import nltk
from collections import Counter

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class EnhancedRedditAnalyzer:
    """Enhanced Reddit analyzer with improved misogyny detection."""
    
    def __init__(self, data_path="/Users/malaikarashid/Documents/353/MisogynyWatch/data/processed/reddit_processed.csv"):
        """Initialize the enhanced analyzer."""
        self.data_path = data_path
        self.df = None
        
        # Enhanced misogyny lexicons (research-based)
        self.misogyny_lexicons = {
            'explicit_slurs': [
                'bitch', 'slut', 'whore', 'cunt', 'thot', 'hoe', 'skank', 'tramp',
                'gold digger', 'cum dumpster', 'walking vagina', 'sperm receptacle'
            ],
            'red_pill_manosphere': [
                'hypergamy', 'cock carousel', 'alpha fucks beta bucks', 'awalt',
                'sexual market value', 'smv', 'the wall', 'post wall', 'hit the wall',
                'branch swinging', 'monkey branching', 'dual mating strategy',
                'chad', 'stacy', 'becky', 'roastie', 'femoid', 'foid'
            ],
            'objectification': [
                'piece of meat', 'sex object', 'holes', 'baby factory',
                'breeding', 'property', 'belong in kitchen', 'dishwasher',
                'sandwich maker', 'walking womb', 'fuck toy', 'meat hole'
            ],
            'dismissive_gender': [
                'women logic', 'female brain', 'emotional creature', 'irrational',
                'can\'t think logically', 'inferior sex', 'weaker sex',
                'women are stupid', 'female nature', 'women drivers',
                'typical woman', 'like all women', 'women moment'
            ],
            'control_dominance': [
                'should obey', 'need to be controlled', 'put in place',
                'know their place', 'barefoot and pregnant', 'submission',
                'female submission', 'traditional roles', 'stay in kitchen',
                'make me sandwich', 'serve men', 'please men'
            ],
            'incel_terminology': [
                'incel', 'blackpill', 'looksmaxing', 'looksmatched',
                'heightpill', 'juggernaut law', 'sui fuel', 'rope',
                'it\'s over', 'beta uprising', 'going er'
            ],
            'mgtow_terminology': [
                'mgtow', 'gynocentrism', 'false accusations', 'divorce rape',
                'family court', 'alimony slavery', 'male disposability',
                'misandry', 'anti-male bias', 'men going their own way'
            ]
        }
        
        # Gender indicators from usernames and text
        self.gender_indicators = {
            'male': [
                r'\bman\b', r'\bmale\b', r'\bguy\b', r'\bdude\b', r'\bbro\b',
                r'\bhusband\b', r'\bboyfriend\b', r'\bfather\b', r'\bdad\b',
                r'\bson\b', r'\bbrother\b', r'mr[\.\s]', r'\bking\b'
            ],
            'female': [
                r'\bwoman\b', r'\bfemale\b', r'\bgirl\b', r'\blady\b', r'\bsis\b',
                r'\bwife\b', r'\bgirlfriend\b', r'\bmother\b', r'\bmom\b',
                r'\bdaughter\b', r'\bsister\b', r'mrs?[\.\s]', r'\bqueen\b'
            ]
        }
        
        # Age indicators
        self.age_patterns = [
            r'\b(?:i\'?m|am)\s+(\d{1,2})\s*(?:years?\s*old|y\.?o\.?|yr)\b',
            r'\b(\d{1,2})\s*(?:year\s*old|y\.?o\.?|yr\s*old)\b',
            r'\bage\s*[:=]?\s*(\d{1,2})\b',
            r'\b(\d{1,2})\s*(?:male|female|m|f)\b'
        ]
        
        # Load and enhance data
        self.load_and_enhance_data()
    
    def load_and_enhance_data(self):
        """Load Reddit data and enhance with proper detection."""
        print("📊 Loading Reddit data...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df):,} Reddit posts")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return
        
        # Preprocess
        self.df['created_utc'] = pd.to_datetime(self.df['created_utc'])
        self.df['date'] = self.df['created_utc'].dt.date
        self.df['year'] = self.df['created_utc'].dt.year
        
        # Clean original data
        self.df['original_misogyny_score'] = pd.to_numeric(self.df['misogyny_score'], errors='coerce').fillna(0)
        self.df['original_is_misogynistic'] = self.df['is_misogynistic'].fillna(False)
        
        print(f"📅 Data range: {self.df['created_utc'].min().strftime('%Y-%m-%d')} to {self.df['created_utc'].max().strftime('%Y-%m-%d')}")
        print(f"🏛️ Communities: {self.df['subreddit'].nunique()} unique subreddits")
        print(f"⚠️ Original misogyny rate: {self.df['original_is_misogynistic'].mean()*100:.2f}%")
        
        # Enhanced processing
        print("\n🔍 Enhancing misogyny detection with research lexicons...")
        self.enhance_misogyny_detection()
        
        print("\n👥 Extracting demographic information...")
        self.extract_demographics()
        
        # Final statistics
        enhanced_rate = self.df['enhanced_is_misogynistic'].mean()
        print(f"\n✅ Enhancement complete!")
        print(f"📈 Enhanced misogyny rate: {enhanced_rate*100:.2f}%")
        print(f"📊 Detection improvement: {((enhanced_rate - self.df['original_is_misogynistic'].mean()) / self.df['original_is_misogynistic'].mean() * 100):+.1f}%")
    
    def enhance_misogyny_detection(self):
        """Enhanced misogyny detection using research lexicons."""
        enhanced_scores = []
        enhanced_binary = []
        detection_details = []
        
        total_posts = len(self.df)
        
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                progress = (idx / total_posts) * 100
                print(f"\rProcessing: {progress:.1f}% ({idx:,}/{total_posts:,})", end="", flush=True)
            
            # Get text content
            text = str(row.get('body', '')) if pd.notna(row.get('body')) else ''
            title = str(row.get('submission_title', '')) if pd.notna(row.get('submission_title')) else ''
            full_text = f"{title} {text}".strip().lower()
            
            if not full_text or len(full_text) < 10:
                enhanced_scores.append(0.0)
                enhanced_binary.append(False)
                detection_details.append({'method': 'no_content'})
                continue
            
            # Calculate enhanced score using lexicons
            score, details = self.calculate_enhanced_misogyny_score(full_text)
            
            enhanced_scores.append(score)
            enhanced_binary.append(score >= 0.3)  # Threshold for classification
            detection_details.append(details)
        
        print()  # New line after progress
        
        # Add enhanced columns
        self.df['enhanced_misogyny_score'] = enhanced_scores
        self.df['enhanced_is_misogynistic'] = enhanced_binary
        self.df['detection_details'] = detection_details
    
    def calculate_enhanced_misogyny_score(self, text: str) -> Tuple[float, Dict]:
        """Calculate enhanced misogyny score using research lexicons."""
        if not text:
            return 0.0, {'method': 'no_text'}
        
        score = 0.0
        triggered_categories = []
        category_scores = {}
        
        # Weight different categories based on severity
        category_weights = {
            'explicit_slurs': 1.0,
            'red_pill_manosphere': 0.9,
            'objectification': 0.9,
            'dismissive_gender': 0.7,
            'control_dominance': 0.8,
            'incel_terminology': 0.8,
            'mgtow_terminology': 0.7
        }
        
        # Check each category
        for category, keywords in self.misogyny_lexicons.items():
            category_score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                # Use word boundaries for exact matches, but also partial for compound terms
                if len(keyword.split()) > 1:
                    # Multi-word phrases
                    if keyword in text:
                        category_score += 0.3
                        matched_keywords.append(keyword)
                else:
                    # Single words with word boundaries
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                        category_score += 0.3
                        matched_keywords.append(keyword)
            
            if category_score > 0:
                weighted_score = min(category_score * category_weights[category], 1.0)
                score += weighted_score
                triggered_categories.append(category)
                category_scores[category] = {
                    'score': weighted_score,
                    'matches': matched_keywords
                }
        
        # Context analysis for quoted vs direct content
        quote_indicators = ['"', "'", 'said:', 'wrote:', 'posted:', 'rt @', 'quote']
        has_quotes = any(indicator in text for indicator in quote_indicators)
        
        # Reduce score if likely quoted content
        if has_quotes and score > 0:
            score *= 0.7  # 30% reduction for potentially quoted content
        
        # Cap the score at 1.0
        final_score = min(score, 1.0)
        
        return final_score, {
            'method': 'enhanced_lexicon',
            'total_score': final_score,
            'triggered_categories': triggered_categories,
            'category_scores': category_scores,
            'has_quotes': has_quotes
        }
    
    def extract_demographics(self):
        """Extract gender and age demographics from text and usernames."""
        genders = []
        age_groups = []
        
        total_posts = len(self.df)
        
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                progress = (idx / total_posts) * 100
                print(f"\rDemographic extraction: {progress:.1f}%", end="", flush=True)
            
            # Extract gender
            gender = self.extract_gender(row)
            genders.append(gender)
            
            # Extract age
            age_group = self.extract_age_group(row)
            age_groups.append(age_group)
        
        print()  # New line after progress
        
        self.df['gender'] = genders
        self.df['age_group'] = age_groups
        
        # Log distributions
        gender_dist = pd.Series(genders).value_counts()
        age_dist = pd.Series(age_groups).value_counts()
        
        print(f"🚻 Gender distribution: {dict(gender_dist)}")
        print(f"📅 Age distribution: {dict(age_dist)}")
    
    def extract_gender(self, row) -> str:
        """Extract gender from username, text, and subreddit context."""
        author = str(row.get('author', '')).lower()
        text = str(row.get('body', '')).lower()
        subreddit = str(row.get('subreddit', '')).lower()
        
        gender_scores = {'male': 0, 'female': 0}
        
        # Check username
        for gender, patterns in self.gender_indicators.items():
            for pattern in patterns:
                if re.search(pattern, author):
                    gender_scores[gender] += 2
        
        # Check text content
        for gender, patterns in self.gender_indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                gender_scores[gender] += matches * 0.5
        
        # Subreddit-based inference
        gender_subreddits = {
            'male': ['mensrights', 'askmen', 'malefashionadvice', 'malegrooming'],
            'female': ['twoxchromosomes', 'askwomen', 'femalefashionadvice', 'femaledatingstrategy']
        }
        
        for gender, subs in gender_subreddits.items():
            if subreddit in subs:
                gender_scores[gender] += 3
        
        # Self-identification patterns
        self_id_patterns = {
            'male': [r'\bi\'?m a (?:man|guy|male)\b', r'\bas a (?:man|guy|male)\b'],
            'female': [r'\bi\'?m a (?:woman|girl|female)\b', r'\bas a (?:woman|girl|female)\b']
        }
        
        for gender, patterns in self_id_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    gender_scores[gender] += 5
        
        # Determine gender
        if gender_scores['male'] > gender_scores['female'] and gender_scores['male'] > 1:
            return 'male'
        elif gender_scores['female'] > gender_scores['male'] and gender_scores['female'] > 1:
            return 'female'
        else:
            return 'unknown'
    
    def extract_age_group(self, row) -> str:
        """Extract age group from text content and subreddit."""
        text = str(row.get('body', '')).lower()
        subreddit = str(row.get('subreddit', '')).lower()
        
        # Extract explicit age mentions
        for pattern in self.age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    age = int(match)
                    if 13 <= age <= 19:
                        return 'Teen (13-19)'
                    elif 20 <= age <= 25:
                        return 'Young Adult (20-25)'
                    elif 26 <= age <= 35:
                        return 'Adult (26-35)'
                    elif 36 <= age <= 50:
                        return 'Middle Age (36-50)'
                    elif age > 50:
                        return 'Older Adult (50+)'
                except ValueError:
                    continue
        
        # Subreddit-based age inference
        age_subreddits = {
            'Teen (13-19)': ['teenagers', 'genz'],
            'Young Adult (20-25)': ['college', 'university'],
            'Adult (26-35)': ['millennials'],
            'Middle Age (36-50)': ['genx', 'askmenover30'],
            'Older Adult (50+)': ['redditforgrownups']
        }
        
        for age_group, subs in age_subreddits.items():
            if subreddit in subs:
                return age_group
        
        return 'unknown'
    
    def analyze_gender_patterns(self):
        """Analyze misogyny patterns by gender."""
        print("\n🚻 GENDER ANALYSIS")
        print("=" * 50)
        
        gender_data = self.df[self.df['gender'] != 'unknown'].copy()
        
        if len(gender_data) == 0:
            print("❌ No gender data available")
            return None
        
        gender_analysis = gender_data.groupby('gender').agg({
            'enhanced_is_misogynistic': ['count', 'sum', 'mean'],
            'enhanced_misogyny_score': 'mean'
        }).round(4)
        
        gender_analysis.columns = ['total_posts', 'misogynistic_posts', 'misogyny_rate', 'avg_score']
        gender_analysis = gender_analysis.sort_values('misogyny_rate', ascending=False)
        
        print("🏆 MISOGYNY BY GENDER:")
        for gender, row in gender_analysis.iterrows():
            print(f"   {gender.capitalize():<12} | {row['misogyny_rate']:.3f} rate | {row['total_posts']:,} posts | {row['avg_score']:.3f} avg score")
        
        most_misogynistic_gender = gender_analysis.index[0]
        highest_rate = gender_analysis.loc[most_misogynistic_gender, 'misogyny_rate']
        
        print(f"\n🎯 RESULT: {most_misogynistic_gender.upper()} users show highest misogyny rate ({highest_rate:.3f})")
        
        return gender_analysis
    
    def analyze_subreddit_patterns(self):
        """Analyze misogyny patterns by subreddit."""
        print("\n🏛️ SUBREDDIT ANALYSIS")
        print("=" * 50)
        
        subreddit_analysis = self.df.groupby('subreddit').agg({
            'enhanced_is_misogynistic': ['count', 'sum', 'mean'],
            'enhanced_misogyny_score': 'mean'
        }).round(4)
        
        subreddit_analysis.columns = ['total_posts', 'misogynistic_posts', 'misogyny_rate', 'avg_score']
        subreddit_analysis = subreddit_analysis[subreddit_analysis['total_posts'] >= 100].copy()
        subreddit_analysis = subreddit_analysis.sort_values('misogyny_rate', ascending=False)
        
        print("🏆 TOP 15 MOST AFFECTED SUBREDDITS (min 100 posts):")
        for i, (subreddit, row) in enumerate(subreddit_analysis.head(15).iterrows(), 1):
            print(f"   {i:2d}. r/{subreddit:<25} | {row['misogyny_rate']:.3f} rate | {row['total_posts']:,} posts")
        
        most_affected = subreddit_analysis.index[0]
        highest_rate = subreddit_analysis.loc[most_affected, 'misogyny_rate']
        
        print(f"\n🎯 RESULT: r/{most_affected} is most affected with {highest_rate:.3f} misogyny rate")
        
        return subreddit_analysis
    
    def analyze_age_patterns(self):
        """Analyze misogyny patterns by age group."""
        print("\n📅 AGE GROUP ANALYSIS")
        print("=" * 50)
        
        age_data = self.df[self.df['age_group'] != 'unknown'].copy()
        
        if len(age_data) == 0:
            print("❌ No age data available")
            return None
        
        age_analysis = age_data.groupby('age_group').agg({
            'enhanced_is_misogynistic': ['count', 'sum', 'mean'],
            'enhanced_misogyny_score': 'mean'
        }).round(4)
        
        age_analysis.columns = ['total_posts', 'misogynistic_posts', 'misogyny_rate', 'avg_score']
        age_analysis = age_analysis.sort_values('misogyny_rate', ascending=False)
        
        print("🏆 MISOGYNY BY AGE GROUP:")
        for age_group, row in age_analysis.iterrows():
            print(f"   {age_group:<20} | {row['misogyny_rate']:.3f} rate | {row['total_posts']:,} posts")
        
        most_misogynistic_age = age_analysis.index[0]
        highest_rate = age_analysis.loc[most_misogynistic_age, 'misogyny_rate']
        
        print(f"\n🎯 RESULT: {most_misogynistic_age} shows highest misogyny rate ({highest_rate:.3f})")
        
        return age_analysis
    
    def analyze_event_impact(self):
        """Analyze event impact using the comprehensive event list."""
        print("\n⚡ EVENT IMPACT ANALYSIS")
        print("=" * 50)
        
        # Red-pill events from your list
        events = {
            '2014-05-23': 'Isla Vista killings (Elliot Rodger)',
            '2016-07-07': 'Roosh V returns to blogging',
            '2017-10-15': 'Harvey Weinstein allegations (#MeToo)',
            '2018-09-27': 'Brett Kavanaugh hearings',
            '2019-04-23': 'Toronto van attack sentencing',
            '2020-08-05': 'Fresh and Fit podcast launch',
            '2020-11-25': 'Violence against Women Day',
            '2021-03-08': 'International Women\'s Day backlash',
            '2021-06-15': 'Andrew Tate TikTok viral period',
            '2021-09-13': 'OnlyFans policy changes',
            '2022-01-20': 'Joe Rogan hosts Jordan Peterson',
            '2022-06-24': 'Roe v. Wade overturned',
            '2022-08-19': 'Andrew Tate arrest coverage',
            '2022-12-29': 'Andrew Tate detained in Romania',
            '2023-01-04': 'Greta Thunberg vs Andrew Tate',
            '2023-03-15': 'Red Pill subreddit shutdown',
            '2023-05-09': 'Jordan Peterson viral content',
            '2023-07-21': 'Barbie movie gender debates',
            '2024-02-10': 'TikTok bans red-pill accounts',
            '2024-05-03': 'YouTube demonetization wave'
        }
        
        baseline_rate = self.df['enhanced_is_misogynistic'].mean()
        event_impacts = []
        
        for date_str, event_name in events.items():
            event_date = pd.to_datetime(date_str)
            
            # Skip if outside data range
            if event_date < self.df['created_utc'].min() or event_date > self.df['created_utc'].max():
                continue
            
            # Get 14 days before and after
            before_period = self.df[
                (self.df['created_utc'] >= event_date - timedelta(days=14)) &
                (self.df['created_utc'] < event_date)
            ]
            
            after_period = self.df[
                (self.df['created_utc'] >= event_date) &
                (self.df['created_utc'] <= event_date + timedelta(days=14))
            ]
            
            if len(before_period) >= 20 and len(after_period) >= 20:
                before_rate = before_period['enhanced_is_misogynistic'].mean()
                after_rate = after_period['enhanced_is_misogynistic'].mean()
                
                # Calculate changes from baseline
                before_change = ((before_rate - baseline_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
                after_change = ((after_rate - baseline_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
                event_impact = after_change - before_change
                
                event_impacts.append({
                    'event': event_name,
                    'date': event_date,
                    'before_rate': before_rate,
                    'after_rate': after_rate,
                    'before_change': before_change,
                    'after_change': after_change,
                    'event_impact': event_impact,
                    'before_posts': len(before_period),
                    'after_posts': len(after_period)
                })
                
                print(f"📅 {event_name[:45]:<45} | {event_impact:+6.1f}% impact")
        
        if event_impacts:
            # Sort by impact
            event_impacts.sort(key=lambda x: abs(x['event_impact']), reverse=True)
            
            print(f"\n🏆 HIGHEST IMPACT EVENTS:")
            for i, event in enumerate(event_impacts[:5], 1):
                print(f"   {i}. {event['event'][:45]:<45} | {event['event_impact']:+6.1f}%")
            
            print(f"\n📊 Baseline misogyny rate: {baseline_rate:.3f}")
            avg_impact = np.mean([abs(e['event_impact']) for e in event_impacts])
            print(f"📊 Average absolute impact: {avg_impact:.1f}%")
        else:
            print("❌ No events found in current data range")
        
        return event_impacts, baseline_rate
    
    def create_visualizations(self, gender_analysis, subreddit_analysis, age_analysis, event_impacts, baseline_rate):
        """Create comprehensive visualizations."""
        print("\n📊 CREATING VISUALIZATIONS")
        print("=" * 50)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        fig.suptitle('Enhanced Misogyny Analysis - Reddit Data\n(Using Research Lexicons & Contextual Detection)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Gender analysis
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        if gender_analysis is not None and len(gender_analysis) > 0:
            genders = gender_analysis.index
            rates = gender_analysis['misogyny_rate']
            colors = ['#e74c3c', '#3498db', '#f39c12']
            
            bars = ax1.bar(genders, rates, color=colors[:len(genders)], alpha=0.8)
            ax1.set_title('Misogyny Rate by Gender', fontweight='bold')
            ax1.set_ylabel('Misogyny Rate')
            
            for bar, rate in zip(bars, rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Gender Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Misogyny Rate by Gender', fontweight='bold')
        
        # 2. Top subreddits
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        top_subs = subreddit_analysis.head(8)
        y_pos = np.arange(len(top_subs))
        
        bars = ax2.barh(y_pos, top_subs['misogyny_rate'], color='darkred', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"r/{sub[:15]}" for sub in top_subs.index], fontsize=9)
        ax2.set_title('Most Affected Subreddits', fontweight='bold')
        ax2.set_xlabel('Misogyny Rate')
        
        for i, rate in enumerate(top_subs['misogyny_rate']):
            ax2.text(rate + 0.002, i, f'{rate:.3f}', va='center', fontsize=8, fontweight='bold')
        
        # 3. Age groups
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        if age_analysis is not None and len(age_analysis) > 0:
            ages = [age.replace(' ', '\n') for age in age_analysis.index]  # Line breaks for readability
            rates = age_analysis['misogyny_rate']
            colors = plt.cm.viridis(np.linspace(0, 1, len(ages)))
            
            bars = ax3.bar(range(len(ages)), rates, color=colors, alpha=0.8)
            ax3.set_xticks(range(len(ages)))
            ax3.set_xticklabels(ages, fontsize=8)
            ax3.set_title('Misogyny Rate by Age Group', fontweight='bold')
            ax3.set_ylabel('Misogyny Rate')
            
            for bar, rate in zip(bars, rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No Age Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Misogyny Rate by Age Group', fontweight='bold')
        
        # 4. Event impact timeline
        ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        if event_impacts:
            events_df = pd.DataFrame(event_impacts)
            events_df = events_df.sort_values('date')
            
            # Create timeline plot
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline')
            
            # Color code by impact direction
            colors = ['red' if x > 0 else 'blue' for x in events_df['event_impact']]
            bars = ax4.bar(range(len(events_df)), events_df['event_impact'], 
                          color=colors, alpha=0.7, width=0.8)
            
            # Customize x-axis
            ax4.set_xticks(range(len(events_df)))
            labels = []
            for _, row in events_df.iterrows():
                event_short = row['event'][:25] + '...' if len(row['event']) > 25 else row['event']
                labels.append(f"{row['date'].strftime('%m/%y')}\n{event_short}")
            
            ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax4.set_title('Event Impact on Misogyny Rates (% Change from Baseline)', fontweight='bold')
            ax4.set_ylabel('Change from Baseline (%)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, impact in zip(bars, events_df['event_impact']):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        height + (2 if height >= 0 else -4),
                        f'{impact:+.0f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', 
                        fontsize=8, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Events in Data Range', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Event Impact Analysis', fontweight='bold')
        
        plt.tight_layout()
        output_path = '/Users/malaikarashid/Documents/353/MisogynyWatch/enhanced_misogyny_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_path}")
        plt.close()
        
        return output_path

def main():
    """Main analysis function."""
    print("🔍 ENHANCED REDDIT MISOGYNY ANALYSIS")
    print("🎯 Using Research Lexicons & Contextual Detection")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = EnhancedRedditAnalyzer()
    
    if analyzer.df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    try:
        # Run all analyses
        gender_analysis = analyzer.analyze_gender_patterns()
        subreddit_analysis = analyzer.analyze_subreddit_patterns()
        age_analysis = analyzer.analyze_age_patterns()
        event_impacts, baseline_rate = analyzer.analyze_event_impact()
        
        # Create visualizations
        output_path = analyzer.create_visualizations(
            gender_analysis, subreddit_analysis, age_analysis, event_impacts, baseline_rate
        )
        
        # Summary
        print("\n" + "="*80)
        print("📋 FINAL RESEARCH ANSWERS")
        print("="*80)
        
        enhanced_rate = analyzer.df['enhanced_is_misogynistic'].mean()
        print(f"📊 Enhanced misogyny detection rate: {enhanced_rate*100:.2f}%")
        
        if gender_analysis is not None and len(gender_analysis) > 0:
            most_misogynistic_gender = gender_analysis.index[0]
            gender_rate = gender_analysis.loc[most_misogynistic_gender, 'misogyny_rate']
            print(f"🚻 Most misogynistic gender: {most_misogynistic_gender.upper()} ({gender_rate:.3f} rate)")
        else:
            print("🚻 Gender analysis: Insufficient data for reliable analysis")
        
        most_affected_subreddit = subreddit_analysis.index[0]
        sub_rate = subreddit_analysis.loc[most_affected_subreddit, 'misogyny_rate']
        print(f"🏛️ Most affected subreddit: r/{most_affected_subreddit} ({sub_rate:.3f} rate)")
        
        if age_analysis is not None and len(age_analysis) > 0:
            most_misogynistic_age = age_analysis.index[0]
            age_rate = age_analysis.loc[most_misogynistic_age, 'misogyny_rate']
            print(f"📅 Most misogynistic age group: {most_misogynistic_age} ({age_rate:.3f} rate)")
        else:
            print("📅 Age analysis: Insufficient data for reliable analysis")
        
        if event_impacts:
            highest_impact = max(event_impacts, key=lambda x: abs(x['event_impact']))
            print(f"⚡ Highest impact event: {highest_impact['event']} ({highest_impact['event_impact']:+.1f}%)")
            print(f"📊 Total events analyzed: {len(event_impacts)}")
        else:
            print("⚡ Event analysis: No events found in current data range")
        
        print(f"\n📊 Comprehensive visualization: {output_path}")
        print("\n✅ ENHANCED ANALYSIS COMPLETE!")
        print("🔬 Analysis used research-based lexicons and contextual detection")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
