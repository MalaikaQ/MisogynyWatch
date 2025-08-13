"""
Text processing utilities for misogyny detection.
"""
import os
import re
import string
import nltk
from typing import List, Dict, Set, Tuple
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from utils.config import MISOGYNY_KEYWORDS
except ImportError:
    # Fallback if config not available
    MISOGYNY_KEYWORDS = {
        'explicit': ['bitch', 'slut', 'whore', 'cunt', 'thot', 'hoe'],
        'red_pill': ['hypergamy', 'cock carousel', 'AWALT', 'the wall'],
        'objectification': ['piece of meat', 'sex object', 'holes'],
        'dismissive': ['women logic', 'female brain', 'emotional creature'],
        'control': ['should obey', 'need to be controlled', 'submissive']
    }

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextProcessor:
    """Text processing utilities for social media content with research-based lexicons."""
    
    def __init__(self, use_research_lexicons: bool = True):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Misogyny detection setup with research lexicons
        if use_research_lexicons:
            self.misogyny_keywords = create_enhanced_misogyny_lexicon()
            print(f"✅ Loaded enhanced lexicon with {sum(len(terms) for terms in self.misogyny_keywords.values())} terms")
        else:
            self.misogyny_keywords = MISOGYNY_KEYWORDS
            print(f"✅ Loaded basic lexicon with {sum(len(terms) for terms in self.misogyny_keywords.values())} terms")
            
        self.misogyny_patterns = self._compile_misogyny_patterns()
        
        # Load additional research lexicons
        self.research_lexicons = load_research_lexicons()
        self.hate_speech_terms = load_hate_speech_lexicon()
        
    def _compile_misogyny_patterns(self) -> Dict[str, List]:
        """Compile regex patterns for misogyny detection."""
        patterns = {}
        
        for category, keywords in self.misogyny_keywords.items():
            patterns[category] = []
            for keyword in keywords:
                # Create case-insensitive patterns with word boundaries
                # Handle multi-word phrases
                if ' ' in keyword:
                    # For phrases, use word boundaries around the whole phrase
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                else:
                    # For single words, use standard word boundaries
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                patterns[category].append((keyword, pattern))
        
        return patterns
        
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_mentions: bool = True,
                   remove_hashtags: bool = False,
                   remove_numbers: bool = True,
                   lowercase: bool = True,
                   remove_punctuation: bool = True,
                   remove_stopwords: bool = False,
                   lemmatize: bool = False) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Input text to clean
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_numbers: Remove numbers
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove English stopwords
            lemmatize: Apply lemmatization
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)
            
        # Remove mentions
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
            
        # Remove hashtags (but keep the text if remove_hashtags is False)
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            text = re.sub(r'#(\w+)', r'\1', text)  # Remove # but keep word
            
        # Remove numbers
        if remove_numbers:
            text = self.number_pattern.sub('', text)
            
        # Convert to lowercase
        if lowercase:
            text = text.lower()
            
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Join tokens back to string
        text = ' '.join(tokens)
        
        # Clean up whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic text features for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        if not isinstance(text, str):
            return {}
            
        # Basic metrics
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'url_count': len(self.url_pattern.findall(text)),
            'mention_count': len(self.mention_pattern.findall(text)),
            'hashtag_count': len(self.hashtag_pattern.findall(text))
        }
        
        return features
    
    def batch_process(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of text strings
            **kwargs: Arguments to pass to clean_text
            
        Returns:
            List of processed texts
        """
        return [self.clean_text(text, **kwargs) for text in texts]
    
    def filter_by_length(self, df: pd.DataFrame, 
                        text_column: str,
                        min_length: int = 10, 
                        max_length: int = 1000) -> pd.DataFrame:
        """
        Filter DataFrame by text length.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        df['text_length'] = df[text_column].str.len()
        return df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         text_column: str,
                         similarity_threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove near-duplicate texts based on Jaccard similarity.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            similarity_threshold: Similarity threshold for removal
            
        Returns:
            DataFrame with duplicates removed
        """
        def jaccard_similarity(text1: str, text2: str) -> float:
            """Calculate Jaccard similarity between two texts."""
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0
        
        df = df.copy()
        to_remove = set()
        
        for i in range(len(df)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(df)):
                if j in to_remove:
                    continue
                similarity = jaccard_similarity(df.iloc[i][text_column], df.iloc[j][text_column])
                if similarity > similarity_threshold:
                    to_remove.add(j)
        
        return df.drop(df.index[list(to_remove)])
    
    def detect_misogyny(self, text: str, use_research_boost: bool = True) -> float:
        """
        Detect misogynistic content in text using enhanced keyword matching and research lexicons.
        
        Args:
            text: Input text to analyze
            use_research_boost: Whether to boost scores using research lexicons
            
        Returns:
            Misogyny score (0-1, where 1 is highly misogynistic)
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
            
        text_lower = text.lower()
        total_score = 0.0
        
        # Enhanced category weights based on research severity
        category_weights = {
            'explicit_slurs': 1.0,
            'red_pill_manosphere': 0.9,
            'incel_terminology': 0.8,
            'mgtow_terminology': 0.8,
            'objectification': 0.9,
            'dismissive_language': 0.6,
            'control_dominance': 0.7,
            'violence_threats': 1.0,
            # Legacy categories for backward compatibility
            'explicit': 1.0,
            'red_pill': 0.8,
            'dismissive': 0.6,
            'control': 0.7
        }
        
        # Check each category
        for category, patterns in self.misogyny_patterns.items():
            category_score = 0.0
            category_weight = category_weights.get(category, 0.5)
            
            for keyword, pattern in patterns:
                matches = len(pattern.findall(text))
                if matches > 0:
                    # Score based on frequency and category weight
                    category_score += matches * category_weight
            
            total_score += category_score
        
        # Research lexicon boost
        if use_research_boost:
            research_boost = 0.0
            
            # Check against each research lexicon
            for lexicon_name, terms in self.research_lexicons.items():
                for term in terms:
                    if term in text_lower:
                        # Weight boost based on research source
                        lexicon_weights = {
                            'hateval': 0.3,
                            'davidson': 0.25,
                            'founta': 0.2,
                            'incel': 0.35,
                            'mgtow': 0.3
                        }
                        research_boost += lexicon_weights.get(lexicon_name, 0.2)
            
            total_score += research_boost
        
        # Normalize score with improved algorithm
        text_length = len(text_lower.split())
        if text_length > 0:
            # Apply length normalization with diminishing returns for longer texts
            length_factor = min(text_length / 10, 2.0)  # Cap length factor at 2.0
            normalized_score = min(total_score / length_factor, 1.0)
        else:
            normalized_score = 0.0
            
        return normalized_score
    
    def get_detection_details(self, text: str) -> Dict[str, any]:
        """
        Get detailed breakdown of misogyny detection results.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed detection information
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'overall_score': 0.0,
                'category_scores': {},
                'matched_terms': [],
                'research_matches': {},
                'severity_level': 'none'
            }
        
        text_lower = text.lower()
        category_scores = {}
        matched_terms = []
        research_matches = {}
        
        # Analyze by category
        for category, patterns in self.misogyny_patterns.items():
            category_score = 0.0
            category_matches = []
            
            for keyword, pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    category_score += len(matches)
                    category_matches.extend(matches)
                    matched_terms.append(keyword)
            
            if category_score > 0:
                category_scores[category] = category_score
        
        # Check research lexicons
        for lexicon_name, terms in self.research_lexicons.items():
            lexicon_matches = [term for term in terms if term in text_lower]
            if lexicon_matches:
                research_matches[lexicon_name] = lexicon_matches
        
        # Calculate overall score
        overall_score = self.detect_misogyny(text)
        
        # Determine severity level
        if overall_score >= 0.8:
            severity = 'severe'
        elif overall_score >= 0.5:
            severity = 'high'
        elif overall_score >= 0.3:
            severity = 'moderate'
        elif overall_score >= 0.1:
            severity = 'low'
        else:
            severity = 'none'
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'matched_terms': list(set(matched_terms)),
            'research_matches': research_matches,
            'severity_level': severity,
            'text_length': len(text.split()),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def extract_misogyny_keywords(self, text: str) -> List[str]:
        """
        Extract misogynistic keywords found in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of misogynistic keywords found
        """
        if not isinstance(text, str) or not text.strip():
            return []
            
        found_keywords = []
        
        for category, patterns in self.misogyny_patterns.items():
            for keyword, pattern in patterns:
                if pattern.search(text):
                    found_keywords.append(keyword)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def analyze_misogyny_severity(self, text: str) -> Dict[str, float]:
        """
        Analyze misogyny severity by category.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with scores for each misogyny category
        """
        if not isinstance(text, str) or not text.strip():
            return {category: 0.0 for category in self.misogyny_keywords.keys()}
            
        category_scores = {}
        text_length = len(text.lower().split())
        
        for category, patterns in self.misogyny_patterns.items():
            category_score = 0.0
            for keyword, pattern in patterns:
                matches = len(pattern.findall(text))
                category_score += matches
            
            # Normalize by text length
            if text_length > 0:
                category_scores[category] = min(category_score / text_length, 1.0)
            else:
                category_scores[category] = 0.0
                
        return category_scores

def create_misogyny_lexicon() -> Set[str]:
    """
    Create a lexicon of misogynistic terms and phrases.
    
    Returns:
        Set of misogynistic terms
    """
    # Base misogynistic terms (this is a starter set - you should expand this)
    base_terms = {
        # Derogatory terms for women
        'bitch', 'slut', 'whore', 'cunt', 'skank', 'hoe', 'thot',
        
        # Red-pill/manosphere specific terms
        'femoid', 'foid', 'roastie', 'becky', 'stacy', 'hypergamy',
        'cock carousel', 'alpha widow', 'beta bux', 'riding the cock carousel',
        
        # Incel terminology
        'blackpill', 'looksmaxing', 'chad', 'virgin shaming', 
        'heightpill', 'gymcel', 'mentalcel',
        
        # MGTOW terms
        'awalt', 'gynocentrism', 'simp', 'white knight', 'beta male',
        'pussy pass', 'false accusation',
        
        # General misogynistic concepts
        'women are', 'females are', 'all women', 'typical female',
        'attention whore', 'gold digger', 'daddy issues'
    }
    
    # Add variations and common misspellings
    expanded_terms = set(base_terms)
    
    # Add plural forms
    for term in base_terms:
        if not term.endswith('s'):
            expanded_terms.add(term + 's')
    
    return expanded_terms

def load_research_lexicons() -> Dict[str, Set[str]]:
    """
    Load multiple research-based hate speech and misogyny lexicons.
    
    Returns:
        Dictionary with lexicon names as keys and term sets as values
    """
    lexicons = {}
    
    # HatEval 2019 lexicon (simulated - based on real research)
    hateval_terms = {
        'bitch', 'slut', 'whore', 'cunt', 'hoe', 'thot', 'skank',
        'femoid', 'foid', 'roastie', 'becky', 'stacy', 'karen',
        'hypergamy', 'cock carousel', 'alpha widow', 'beta bux',
        'awalt', 'gynocentrism', 'simp', 'white knight', 'beta male',
        'pussy pass', 'false accusation', 'gold digger', 'attention whore',
        'riding the cock carousel', 'hit the wall', 'post wall',
        'branch swinging', 'monkey branching', 'dual mating strategy'
    }
    lexicons['hateval'] = hateval_terms
    
    # Davidson et al. (2017) hate speech lexicon (simulated)
    davidson_terms = {
        'stupid bitch', 'dumb slut', 'worthless whore', 'useless cunt',
        'lying bitch', 'crazy bitch', 'psycho bitch', 'evil bitch',
        'feminist nazi', 'feminazi', 'man hater', 'ball buster',
        'ice queen', 'frigid bitch', 'prude', 'cocktease',
        'easy lay', 'town bicycle', 'village bicycle', 'pump and dump'
    }
    lexicons['davidson'] = davidson_terms
    
    # Founta et al. (2018) abusive language lexicon (simulated)
    founta_terms = {
        'dishwasher', 'kitchen appliance', 'baby factory', 'breeding machine',
        'sex object', 'piece of meat', 'walking vagina', 'holes',
        'cum dumpster', 'sperm receptacle', 'breeding stock',
        'property', 'owned', 'submissive', 'obedient', 'barefoot and pregnant'
    }
    lexicons['founta'] = founta_terms
    
    # Incel-specific terminology (research-based)
    incel_terms = {
        'blackpill', 'looksmaxing', 'chad', 'stacy', 'becky', 'normie',
        'heightpill', 'gymcel', 'mentalcel', 'volcel', 'truecel',
        'inceldom', 'virgin shaming', 'heightism', 'lookism',
        'facial aesthetics', 'canthal tilt', 'hunter eyes', 'rope',
        'ldar', 'neet', 'wagecuck', 'betabux', 'hypergamous'
    }
    lexicons['incel'] = incel_terms
    
    # MGTOW-specific terminology
    mgtow_terms = {
        'mgtow', 'men going their own way', 'gynocentric', 'gynocentrism',
        'blue pill', 'red pill', 'purple pill', 'black pill',
        'the wall', 'hitting the wall', 'post wall', 'expired',
        'alpha fucks beta bucks', 'dual mating strategy', 'branch swinging',
        'monkey branching', 'divorce rape', 'alimony slavery'
    }
    lexicons['mgtow'] = mgtow_terms
    
    return lexicons

def load_hate_speech_lexicon(file_path: str = None) -> Set[str]:
    """
    Load additional hate speech terms from external sources.
    
    Args:
        file_path: Path to external lexicon file
        
    Returns:
        Set of hate speech terms
    """
    # Load from file if provided
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return set(line.strip().lower() for line in f if line.strip())
    
    # Load research lexicons if no file provided
    research_lexicons = load_research_lexicons()
    combined_terms = set()
    
    for lexicon_name, terms in research_lexicons.items():
        combined_terms.update(terms)
    
    return combined_terms

def create_enhanced_misogyny_lexicon() -> Dict[str, Set[str]]:
    """
    Create an enhanced lexicon combining manual curation with research datasets.
    
    Returns:
        Dictionary with categorized misogynistic terms
    """
    # Load research lexicons
    research_lexicons = load_research_lexicons()
    
    # Enhanced categorized lexicon
    enhanced_lexicon = {
        'explicit_slurs': {
            'bitch', 'slut', 'whore', 'cunt', 'thot', 'hoe', 'skank',
            'twat', 'slag', 'tramp', 'tart', 'strumpet'
        },
        
        'red_pill_manosphere': {
            'hypergamy', 'cock carousel', 'awalt', 'the wall', 'alpha widow',
            'beta bux', 'branch swinging', 'monkey branching', 'dual mating strategy',
            'sexual market value', 'smv', 'post wall', 'hit the wall',
            'alpha fucks beta bucks', 'riding the cock carousel'
        },
        
        'incel_terminology': research_lexicons.get('incel', set()),
        
        'mgtow_terminology': research_lexicons.get('mgtow', set()),
        
        'objectification': {
            'piece of meat', 'sex object', 'walking vagina', 'holes',
            'cum dumpster', 'sperm receptacle', 'baby factory', 'breeding machine',
            'dishwasher', 'kitchen appliance', 'breeding stock', 'property'
        },
        
        'dismissive_language': {
            'women logic', 'female brain', 'emotional creature', 'irrational',
            'hysterical', 'crazy', 'psycho', 'mental', 'unstable',
            'hormonal', 'pms', 'time of the month', 'typical female'
        },
        
        'control_dominance': {
            'should obey', 'need to be controlled', 'submissive', 'obedient',
            'know their place', 'put in place', 'barefoot and pregnant',
            'domestic', 'housewife', 'stay home', 'traditional role'
        },
        
        'violence_threats': {
            'beat', 'slap', 'punch', 'hit', 'violence', 'force',
            'rape', 'assault', 'abuse', 'hurt', 'pain', 'suffer'
        }
    }
    
    # Add research terms to appropriate categories
    for category, terms in enhanced_lexicon.items():
        if category in ['incel_terminology', 'mgtow_terminology']:
            continue  # Already assigned
        # Add overlapping terms from research lexicons
        for research_terms in research_lexicons.values():
            enhanced_lexicon[category].update(
                term for term in research_terms 
                if any(keyword in term for keyword in terms)
            )
    
    return enhanced_lexicon

if __name__ == "__main__":
    # Example usage with enhanced research lexicons
    print("🔬 Testing Enhanced Misogyny Detection with Research Lexicons")
    print("=" * 60)
    
    # Initialize processor with research lexicons
    processor = TextProcessor(use_research_lexicons=True)
    
    # Test cases representing different categories
    test_cases = [
        {
            'text': "Women are just emotional creatures who can't think logically",
            'category': 'Dismissive Language'
        },
        {
            'text': "AWALT - all women are like that, hypergamy is real",
            'category': 'Red-pill/Manosphere'
        },
        {
            'text': "She hit the wall at 30, now she's looking for a beta bux",
            'category': 'MGTOW Terminology'
        },
        {
            'text': "Stupid bitch should know her place in the kitchen",
            'category': 'Multiple Categories'
        },
        {
            'text': "Supporting women's rights and gender equality",
            'category': 'Non-misogynistic'
        },
        {
            'text': "Femoids only want Chad, blackpill is the truth",
            'category': 'Incel Terminology'
        }
    ]
    
    print(f"Testing {len(test_cases)} examples:\n")
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        expected_category = test_case['category']
        
        print(f"Test {i}: {expected_category}")
        print(f"Text: \"{text}\"")
        
        # Get detailed analysis
        details = processor.get_detection_details(text)
        
        print(f"Overall Score: {details['overall_score']:.3f}")
        print(f"Severity Level: {details['severity_level']}")
        
        if details['matched_terms']:
            print(f"Matched Terms: {', '.join(details['matched_terms'])}")
        
        if details['research_matches']:
            print("Research Lexicon Matches:")
            for lexicon, matches in details['research_matches'].items():
                print(f"  {lexicon}: {', '.join(matches)}")
        
        if details['category_scores']:
            print("Category Breakdown:")
            for category, score in details['category_scores'].items():
                print(f"  {category}: {score}")
        
        print("-" * 40)
    
    # Test research lexicon loading
    print("\n🔬 RESEARCH LEXICON STATISTICS")
    print("=" * 40)
    
    research_lexicons = load_research_lexicons()
    total_terms = 0
    
    for lexicon_name, terms in research_lexicons.items():
        print(f"{lexicon_name}: {len(terms)} terms")
        total_terms += len(terms)
    
    print(f"\nTotal research terms: {total_terms}")
    
    # Test file-based lexicon loading
    print(f"\n📁 FILE-BASED LEXICON LOADING")
    print("=" * 40)
    
    lexicon_dir = Path(__file__).parent / 'data' / 'lexicons'
    
    for lexicon_file in ['hateval_2019.txt', 'davidson_2017.txt', 'founta_2018.txt']:
        file_path = lexicon_dir / lexicon_file
        if file_path.exists():
            terms = load_hate_speech_lexicon(str(file_path))
            print(f"{lexicon_file}: {len(terms)} terms loaded")
        else:
            print(f"{lexicon_file}: File not found")
    
    print(f"\n✅ Enhanced misogyny detection system ready!")
    print(f"📊 Total lexicon coverage: {sum(len(terms) for terms in processor.misogyny_keywords.values())} terms")
    print(f"🔬 Research lexicons: {len(research_lexicons)} sources")
