"""
Configuration file for MisogynyWatch project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LEXICONS_DIR = DATA_DIR / 'lexicons'

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LEXICONS_DIR.mkdir(parents=True, exist_ok=True)

# Reddit API Configuration
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'MisogynyWatch/1.0'),
    'username': os.getenv('REDDIT_USERNAME', ''),
    'password': os.getenv('REDDIT_PASSWORD', '')
}

# Twitter API Configuration
TWITTER_CONFIG = {
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', ''),
    'consumer_key': os.getenv('TWITTER_API_KEY', ''),  # Updated to match .env
    'consumer_secret': os.getenv('TWITTER_API_SECRET', ''),  # Updated to match .env
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
}

# Reddit communities to monitor
REDDIT_COMMUNITIES = [
    # Red-pill/manosphere communities
    'TheRedPill',
    'MGTOW',
    'MensRights',
    'seduction',
    'relationship_advice',
    'dating_advice',
    'AskMen',
    'unpopularopinion',
    
    # General communities where misogyny might appear
    'AmItheAsshole',
    'TwoXChromosomes',
    'dating',
    'relationships',
    'AskReddit',
    
    # Age-specific communities
    'teenagers',
    'college',
    'AskWomen',
    'AskMenOver30'
]

# Twitter search terms and hashtags
TWITTER_SEARCH_TERMS = [
    # Red-pill terminology
    'red pill',
    'alpha male',
    'beta male',
    'hypergamy',
    'cock carousel',
    'wall hit',
    'feminism destroyed',
    'women logic',
    'female nature',
    'AWALT',  # All Women Are Like That
    'SMV',    # Sexual Market Value
    
    # MGTOW terminology
    'MGTOW',
    'men going their own way',
    'pump and dump',
    'false accusations',
    
    # General misogynistic terms
    'women are',
    'females should',
    'dishwasher joke',
    'kitchen joke',
    'make me sandwich'
]

# Influential red-pill figures and events to track
RED_PILL_EVENTS = {
    '2014-05-23': 'Isla Vista killings (Elliot Rodger manifesto, incel radicalization symbol)',
    '2016-07-07': 'Roosh V announces return to blogging, boosts manosphere content',
    '2017-10-15': 'Harvey Weinstein allegations (start of #MeToo movement)',
    '2018-09-27': 'Brett Kavanaugh hearings',
    '2019-04-23': 'Toronto van attack sentencing coverage (incel-related)',
    '2020-08-05': 'Fresh and Fit podcast launch',
    '2020-11-25': 'International Day for Elimination of Violence against Women',
    '2021-03-08': 'International Women\'s Day backlash in manosphere forums',
    '2021-06-15': 'Andrew Tate viral TikTok period begins',
    '2021-09-13': 'OnlyFans announces (and reverses) ban on explicit content',
    '2022-01-20': 'Joe Rogan hosts Jordan Peterson on gender/climate comments',
    '2022-06-24': 'Roe v. Wade overturned (massive gender discourse spike)',
    '2022-08-19': 'Andrew Tate arrest coverage begins circulating',
    '2022-12-29': 'Andrew Tate and brother detained in Romania',
    '2023-01-04': 'Greta Thunberg vs. Andrew Tate Twitter exchange',
    '2023-03-15': 'Red Pill subreddit temporary shutdown after policy warnings',
    '2023-05-09': 'Jordan Peterson viral content resurgence',
    '2023-07-21': 'Barbie movie release sparks online gender debates',
    '2024-02-10': 'TikTok bans several red-pill influencer accounts',
    '2024-05-03': 'Major manosphere YouTube demonetization wave'
}

# Misogyny detection keywords (expanded)
MISOGYNY_KEYWORDS = {
    'explicit': [
        'bitch', 'slut', 'whore', 'cunt', 'thot', 'hoe', 'gold digger',
        'dishwasher', 'kitchen', 'sandwich', 'breeding', 'property'
    ],
    'red_pill': [
        'hypergamy', 'cock carousel', 'alpha fucks beta bucks', 'AWALT',
        'sexual market value', 'SMV', 'the wall', 'post wall', 'hit the wall',
        'branch swinging', 'monkey branching', 'dual mating strategy'
    ],
    'objectification': [
        'piece of meat', 'sex object', 'walking vagina', 'holes',
        'cum dumpster', 'sperm receptacle', 'baby factory'
    ],
    'dismissive': [
        'women logic', 'female brain', 'emotional creature', 'irrational',
        'can\'t think logically', 'inferior sex', 'weaker sex'
    ],
    'control': [
        'should obey', 'need to be controlled', 'submissive', 'dominant',
        'put in place', 'know their place', 'barefoot and pregnant'
    ]
}

# Age-related patterns for analysis
AGE_INDICATORS = [
    r'\b(?:i\'?m|am)\s+(\d{1,2})\s*(?:years?\s*old|y\.?o\.?|yr)\b',
    r'\b(\d{1,2})\s*(?:year\s*old|y\.?o\.?|yr\s*old)\b',
    r'\bas\s+a\s+(\d{1,2})\s*(?:year\s*old|y\.?o\.?)\b',
    r'\bage\s*[:=]?\s*(\d{1,2})\b',
    r'\b(\d{1,2})\s*(?:male|female|m|f)\b'
]

# Data collection limits
COLLECTION_LIMITS = {
    'reddit_posts_per_subreddit': 1000,
    'reddit_comments_per_post': 100,
    'twitter_tweets_per_search': 1000,
    'days_lookback': 365,
    'rate_limit_delay': 1  # seconds
}

# Research lexicon file paths
RESEARCH_LEXICONS = {
    'hateval_2019': LEXICONS_DIR / 'hateval_2019.txt',
    'davidson_2017': LEXICONS_DIR / 'davidson_2017.txt',
    'founta_2018': LEXICONS_DIR / 'founta_2018.txt'
}

# Misogyny detection configuration
DETECTION_CONFIG = {
    'use_research_lexicons': True,
    'minimum_score_threshold': 0.1,
    'high_confidence_threshold': 0.7,
    'category_weights': {
        'explicit_slurs': 1.0,
        'red_pill_manosphere': 0.9,
        'incel_terminology': 0.8,
        'mgtow_terminology': 0.8,
        'objectification': 0.9,
        'dismissive_language': 0.6,
        'control_dominance': 0.7,
        'violence_threats': 1.0
    }
}
