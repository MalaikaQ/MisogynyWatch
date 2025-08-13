"""
Contextual Misogyny Detector
Enhanced misogyny detection that distinguishes between:
1. Direct misogynistic statements
2. Quoted or discussed misogynistic content  
3. Critical analysis of misogyny
4. Context-aware intent analysis
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

class ContextualMisogynyDetector:
    """Enhanced misogyny detector with context awareness."""
    
    def __init__(self):
        """Initialize the contextual detector."""
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Contextual indicators for different types of content
        self.quote_indicators = [
            r'"[^"]*"',  # Text in quotes
            r"'[^']*'",  # Text in single quotes
            r'\b(?:said|says|wrote|posted|tweeted|commented):\s*',
            r'\b(?:quote|quoting)\b',
            r'\bRT\s*@',  # Retweets
            r'\b(?:he|she|they)\s+(?:said|wrote|posted|claimed)\b',
            r'\b(?:according to|as per|reported that)\b',
            r'\b(?:example|instance|case)\s+(?:of|where|when)\b',
            r'\b(?:someone|somebody)\s+(?:told|said|wrote)\b'
        ]
        
        self.critical_analysis_indicators = [
            r'\b(?:this is|this kind of)\s+(?:problematic|wrong|sexist|misogyn)',
            r'\b(?:why is|how is|what makes)\s+(?:this|that)\s+(?:sexist|misogyn|problematic)',
            r'\b(?:tired of|sick of|fed up with)\s+(?:this|these|men who|people who)',
            r'\b(?:not okay|unacceptable|disgusting|awful)\b',
            r'\b(?:call(?:ing)?\s+out|pointing out|highlighting)\b',
            r'\b(?:experience(?:d|s)?|deal(?:ing)?\s+with|face(?:d|s)?|encounter(?:ed|s)?)\s+(?:sexism|misogyny)',
            r'\b(?:as a (?:woman|female))[,\s]',
            r'\b(?:speaking as|from my experience as)\b',
            r'\b(?:this happens to|this is common for)\s+(?:women|females|us)\b',
            r'\b(?:we need to|society needs to|men need to)\s+(?:stop|change|address)',
            r'\b(?:awareness|education|discussion)\s+(?:about|of|regarding)\b',
            r'\b(?:share|sharing)\s+(?:my|our|this)\s+(?:story|experience)\b'
        ]
        
        self.support_indicators = [
            r'\b(?:support|solidarity|standing with)\s+(?:women|victims|survivors)\b',
            r'\b(?:believe|supporting|validating)\s+(?:women|her|them)\b',
            r'\b(?:sending|offering)\s+(?:love|support|hugs)\b',
            r'\b(?:you(?:\'re| are)\s+(?:not alone|brave|strong))\b',
            r'\b(?:thank you for|grateful for)\s+(?:sharing|speaking out)\b'
        ]
        
        self.discussion_indicators = [
            r'\b(?:what do you think|thoughts on|opinions on)\b',
            r'\b(?:has anyone|does anyone|anyone else)\s+(?:experienced|dealt with|noticed)\b',
            r'\b(?:discussion|conversation|dialogue)\s+(?:about|on|regarding)\b',
            r'\b(?:research|study|studies|data)\s+(?:shows|indicates|suggests)\b',
            r'\b(?:statistics|numbers|findings)\s+(?:on|about|regarding)\b'
        ]
        
        self.personal_experience_indicators = [
            r'\b(?:I|my)\s+(?:experienced?|dealt with|faced|encountered)\b',
            r'\b(?:happened to me|told me|said to me)\b',
            r'\b(?:my story|my experience|what I went through)\b',
            r'\b(?:as a (?:woman|female|girl))[,\s]',
            r'\b(?:when I was|I remember when)\b'
        ]
        
        # Direct misogyny indicators (more explicit)
        self.direct_misogyny_indicators = [
            r'\b(?:women are|females are|girls are)\s+(?:all|only|just|nothing but)\b',
            r'\b(?:all women|every woman|most women)\s+(?:want|need|deserve|should)\b',
            r'\b(?:women should|females should|girls should)\s+(?:stay|remain|be|not)\b',
            r'\b(?:I hate|can\'t stand|despise)\s+(?:women|females|girls)\b',
            r'\b(?:women|females)\s+(?:belong in|are only good for)\b',
            r'\b(?:shut up|be quiet),?\s+(?:woman|female|girl|bitch)\b',
            r'\b(?:know your place|stay in your lane),?\s+(?:woman|female)\b'
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            'quotes': [re.compile(pattern, re.IGNORECASE) for pattern in self.quote_indicators],
            'critical': [re.compile(pattern, re.IGNORECASE) for pattern in self.critical_analysis_indicators],
            'support': [re.compile(pattern, re.IGNORECASE) for pattern in self.support_indicators],
            'discussion': [re.compile(pattern, re.IGNORECASE) for pattern in self.discussion_indicators],
            'personal': [re.compile(pattern, re.IGNORECASE) for pattern in self.personal_experience_indicators],
            'direct': [re.compile(pattern, re.IGNORECASE) for pattern in self.direct_misogyny_indicators]
        }
    
    def analyze_context_indicators(self, text: str) -> Dict[str, float]:
        """
        Analyze text for different types of contextual indicators.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with scores for each context type
        """
        if not isinstance(text, str):
            return {key: 0.0 for key in self.compiled_patterns.keys()}
        
        text_lower = text.lower()
        scores = {}
        
        for context_type, patterns in self.compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(pattern.findall(text_lower))
            
            # Normalize by text length (per 100 characters)
            scores[context_type] = matches * (100 / max(len(text), 1))
        
        return scores
    
    def analyze_sentiment_context(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment to help determine context.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not isinstance(text, str):
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores
    
    def detect_quoted_content(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect if text contains quoted misogynistic content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (has_quotes, list_of_quoted_text)
        """
        if not isinstance(text, str):
            return False, []
        
        # Find quoted text
        quote_patterns = [
            r'"([^"]*)"',
            r"'([^']*)'",
            r'«([^»]*)»',
            r'„([^"]*)"'
        ]
        
        quoted_texts = []
        for pattern in quote_patterns:
            matches = re.findall(pattern, text)
            quoted_texts.extend(matches)
        
        return len(quoted_texts) > 0, quoted_texts
    
    def analyze_surrounding_context(self, text: str, misogyny_keywords: List[str]) -> Dict[str, any]:
        """
        Analyze the context surrounding misogynistic keywords.
        
        Args:
            text: Input text to analyze
            misogyny_keywords: List of detected misogynistic terms
            
        Returns:
            Dictionary with context analysis
        """
        if not isinstance(text, str) or not misogyny_keywords:
            return {'before_context': [], 'after_context': [], 'context_sentiment': 'neutral'}
        
        sentences = sent_tokenize(text)
        context_analysis = {'before_context': [], 'after_context': [], 'context_sentiment': 'neutral'}
        
        for keyword in misogyny_keywords:
            for i, sentence in enumerate(sentences):
                if keyword.lower() in sentence.lower():
                    # Get surrounding sentences
                    before = sentences[max(0, i-1):i] if i > 0 else []
                    after = sentences[i+1:min(len(sentences), i+2)] if i < len(sentences)-1 else []
                    
                    context_analysis['before_context'].extend(before)
                    context_analysis['after_context'].extend(after)
        
        # Analyze sentiment of surrounding context
        context_text = ' '.join(context_analysis['before_context'] + context_analysis['after_context'])
        if context_text:
            sentiment = self.sentiment_analyzer.polarity_scores(context_text)
            if sentiment['compound'] > 0.1:
                context_analysis['context_sentiment'] = 'positive'
            elif sentiment['compound'] < -0.1:
                context_analysis['context_sentiment'] = 'negative'
            else:
                context_analysis['context_sentiment'] = 'neutral'
        
        return context_analysis
    
    def classify_misogyny_type(self, text: str, original_misogyny_score: float) -> Dict[str, any]:
        """
        Classify the type of misogynistic content based on context.
        
        Args:
            text: Input text to analyze
            original_misogyny_score: Original misogyny score from basic detection
            
        Returns:
            Dictionary with classification results
        """
        if not isinstance(text, str) or original_misogyny_score == 0:
            return {
                'misogyny_type': 'none',
                'confidence': 1.0,
                'adjusted_score': 0.0,
                'context_factors': {},
                'reasoning': 'No misogynistic content detected'
            }
        
        # Analyze context indicators
        context_scores = self.analyze_context_indicators(text)
        sentiment_scores = self.analyze_sentiment_context(text)
        has_quotes, quoted_texts = self.detect_quoted_content(text)
        
        # Classification logic
        classification = {
            'misogyny_type': 'direct',
            'confidence': 0.5,
            'adjusted_score': original_misogyny_score,
            'context_factors': {
                'context_scores': context_scores,
                'sentiment': sentiment_scores,
                'has_quotes': has_quotes,
                'quoted_count': len(quoted_texts)
            },
            'reasoning': '',
            'has_quotes': has_quotes,
            'quoted_count': len(quoted_texts)
        }
        
        # Determine type based on context indicators
        if context_scores['quotes'] > 1.0 or has_quotes:
            if context_scores['critical'] > 0.5 or sentiment_scores['compound'] < -0.1:
                classification['misogyny_type'] = 'quoted_critical'
                classification['adjusted_score'] = original_misogyny_score * 0.3  # Reduce score
                classification['reasoning'] = 'Quoted misogynistic content with critical analysis'
                classification['confidence'] = 0.8
            else:
                classification['misogyny_type'] = 'quoted_neutral'
                classification['adjusted_score'] = original_misogyny_score * 0.5  # Moderately reduce
                classification['reasoning'] = 'Quoted misogynistic content without clear stance'
                classification['confidence'] = 0.7
        
        elif context_scores['personal'] > 0.5:
            if sentiment_scores['compound'] < -0.2:
                classification['misogyny_type'] = 'personal_experience'
                classification['adjusted_score'] = original_misogyny_score * 0.2  # Significantly reduce
                classification['reasoning'] = 'Personal experience sharing with negative sentiment'
                classification['confidence'] = 0.8
            else:
                classification['misogyny_type'] = 'personal_neutral'
                classification['adjusted_score'] = original_misogyny_score * 0.4
                classification['reasoning'] = 'Personal experience sharing'
                classification['confidence'] = 0.7
        
        elif context_scores['critical'] > 1.0:
            classification['misogyny_type'] = 'critical_analysis'
            classification['adjusted_score'] = original_misogyny_score * 0.2  # Significantly reduce
            classification['reasoning'] = 'Critical analysis of misogyny'
            classification['confidence'] = 0.8
        
        elif context_scores['support'] > 0.5:
            classification['misogyny_type'] = 'supportive'
            classification['adjusted_score'] = original_misogyny_score * 0.1  # Almost eliminate
            classification['reasoning'] = 'Supportive content for victims/survivors'
            classification['confidence'] = 0.9
        
        elif context_scores['discussion'] > 0.5:
            classification['misogyny_type'] = 'academic_discussion'
            classification['adjusted_score'] = original_misogyny_score * 0.3
            classification['reasoning'] = 'Academic or research-based discussion'
            classification['confidence'] = 0.7
        
        elif context_scores['direct'] > 0.5:
            classification['misogyny_type'] = 'direct'
            classification['adjusted_score'] = original_misogyny_score * 1.2  # Increase for direct statements
            classification['reasoning'] = 'Direct misogynistic statement'
            classification['confidence'] = 0.9
        
        else:
            # Default case - ambiguous
            classification['misogyny_type'] = 'ambiguous'
            classification['adjusted_score'] = original_misogyny_score * 0.7
            classification['reasoning'] = 'Ambiguous context, moderately reducing score'
            classification['confidence'] = 0.5
        
        # Ensure adjusted score doesn't exceed 1.0
        classification['adjusted_score'] = min(classification['adjusted_score'], 1.0)
        
        return classification
    
    def process_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                       misogyny_score_column: str = 'misogyny_score') -> pd.DataFrame:
        """
        Process entire dataset with contextual misogyny detection.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text content
            misogyny_score_column: Column containing original misogyny scores
            
        Returns:
            DataFrame with additional contextual analysis columns
        """
        print("Processing dataset with contextual misogyny detection...")
        
        df_contextual = df.copy()
        
        # Initialize new columns
        df_contextual['misogyny_type'] = 'none'
        df_contextual['context_confidence'] = 0.0
        df_contextual['adjusted_misogyny_score'] = 0.0
        df_contextual['context_reasoning'] = ''
        df_contextual['has_quotes'] = False
        df_contextual['quote_count'] = 0
        df_contextual['context_sentiment'] = 0.0
        
        # Process each row
        for idx, row in df_contextual.iterrows():
            if idx % 5000 == 0:
                print(f"  Processed {idx:,} posts...")
            
            text = row[text_column]
            original_score = row[misogyny_score_column]
            
            # Classify misogyny type
            classification = self.classify_misogyny_type(text, original_score)
            
            # Update columns safely
            df_contextual.loc[idx, 'misogyny_type'] = classification.get('misogyny_type', 'none')
            df_contextual.loc[idx, 'context_confidence'] = classification.get('confidence', 0.0)
            df_contextual.loc[idx, 'adjusted_misogyny_score'] = classification.get('adjusted_score', original_score)
            df_contextual.loc[idx, 'context_reasoning'] = classification.get('reasoning', '')
            
            # Handle context factors safely
            context_factors = classification.get('context_factors', {})
            df_contextual.loc[idx, 'has_quotes'] = context_factors.get('has_quotes', False)
            df_contextual.loc[idx, 'quote_count'] = context_factors.get('quoted_count', 0)
            
            sentiment = context_factors.get('sentiment', {})
            df_contextual.loc[idx, 'context_sentiment'] = sentiment.get('compound', 0.0) if isinstance(sentiment, dict) else 0.0
        
        # Update binary misogyny classification based on adjusted scores
        df_contextual['is_misogynistic_adjusted'] = df_contextual['adjusted_misogyny_score'] > 0.3
        
        print(f"Contextual analysis complete! Processed {len(df_contextual):,} posts")
        
        return df_contextual
    
    def generate_context_analysis_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate a report on the contextual analysis results.
        
        Args:
            df: DataFrame with contextual analysis
            
        Returns:
            Dictionary with analysis report
        """
        # Misogyny type distribution
        type_distribution = df['misogyny_type'].value_counts()
        
        # Compare original vs adjusted scores
        original_misogyny_rate = (df['misogyny_score'] > 0.3).mean() * 100
        adjusted_misogyny_rate = (df['adjusted_misogyny_score'] > 0.3).mean() * 100
        
        # Score changes by type
        score_changes = df.groupby('misogyny_type').agg({
            'misogyny_score': 'mean',
            'adjusted_misogyny_score': 'mean',
            'context_confidence': 'mean'
        }).round(3)
        
        # Quotes analysis
        quotes_analysis = {
            'posts_with_quotes': (df['has_quotes'] == True).sum(),
            'quote_percentage': (df['has_quotes'] == True).mean() * 100,
            'avg_quotes_per_post': df['quote_count'].mean()
        }
        
        report = {
            'total_posts': len(df),
            'misogyny_type_distribution': type_distribution.to_dict(),
            'misogyny_rate_comparison': {
                'original_rate': original_misogyny_rate,
                'adjusted_rate': adjusted_misogyny_rate,
                'reduction': original_misogyny_rate - adjusted_misogyny_rate
            },
            'score_changes_by_type': score_changes.to_dict(),
            'quotes_analysis': quotes_analysis,
            'confidence_distribution': df['context_confidence'].describe().to_dict()
        }
        
        return report

def test_contextual_detector():
    """Test the contextual misogyny detector with sample texts."""
    detector = ContextualMisogynyDetector()
    
    test_cases = [
        {
            'text': 'Women are just emotional and can\'t handle leadership roles.',
            'original_score': 0.8,
            'expected_type': 'direct'
        },
        {
            'text': 'Someone told me "women belong in the kitchen" yesterday. This kind of thinking is so problematic.',
            'original_score': 0.6,
            'expected_type': 'quoted_critical'
        },
        {
            'text': 'As a woman, I experienced workplace discrimination when my boss said I was "too emotional" for the job.',
            'original_score': 0.4,
            'expected_type': 'personal_experience'
        },
        {
            'text': 'Research shows that women face significant barriers in STEM fields due to persistent stereotypes.',
            'original_score': 0.3,
            'expected_type': 'academic_discussion'
        },
        {
            'text': 'Sending support to all the women sharing their stories. You are brave and not alone.',
            'original_score': 0.2,
            'expected_type': 'supportive'
        }
    ]
    
    print("=" * 60)
    print("TESTING CONTEXTUAL MISOGYNY DETECTOR")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {case['text']}")
        print(f"Original Score: {case['original_score']}")
        print(f"Expected Type: {case['expected_type']}")
        
        classification = detector.classify_misogyny_type(case['text'], case['original_score'])
        
        print(f"Detected Type: {classification['misogyny_type']}")
        print(f"Adjusted Score: {classification['adjusted_score']:.3f}")
        print(f"Confidence: {classification['confidence']:.3f}")
        print(f"Reasoning: {classification['reasoning']}")
        print(f"Correct: {'✓' if classification['misogyny_type'] == case['expected_type'] else '✗'}")

if __name__ == "__main__":
    test_contextual_detector()
