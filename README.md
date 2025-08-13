# MisogynyWatch

A comprehensive data science project analyzing misogynistic language trends across Reddit and Twitter platforms, with focus on temporal patterns, event correlations, community differences, and age demographics.

## Research Questions Addressed

1. **Has misogynistic language increased over time?** → Time series trend analysis
2. **Correlation with red-pill influencer events?** → Event impact analysis  
3. **Which communities are most affected?** → Cross-platform comparison
4. **Which age groups and gender are most affected?** → Demographic analysis

## Project Structure

```
MisogynyWatch/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                    # Raw scraped data
│   └── processed/              # Cleaned and analyzed data
├── utils/
│   ├── __init__.py
│   └── config.py              # Configuration and constants
├── analysis/
│   ├── main_analysis.py       # Main analysis module
│   └── visualizations.py     # Plotting and dashboard creation
├── notebooks/                 # Jupyter notebooks for exploration
├── reddit_scraper.py         # Reddit data collection
├── twitter_scraper.py        # Twitter data collection  
├── text_processing.py        # NLP and misogyny detection
├── demographics_analyzer.py  # Age and gender demographic analysis
└── data_collection_coordinator.py  # Orchestrates data collection
```

## Features

### Data Collection
- **Reddit**: Scrapes posts and comments from red-pill/manosphere communities
- **Twitter**: Collects tweets using misogyny-related keywords and hashtags
- **Rate limiting**: Respects API limits and implements delays
- **Deduplication**: Removes duplicate content across platforms

### Text Analysis
- **Misogyny Detection**: Multi-category keyword-based scoring system
  - Explicit derogatory terms
  - Red-pill/manosphere terminology  
  - Objectification language
  - Dismissive language
  - Control/dominance themes
- **Age Extraction**: Identifies age mentions in user content
- **Gender Inference**: Basic gender identification from text patterns
- **Text Preprocessing**: Comprehensive cleaning and normalization

### Analysis Capabilities
- **Temporal Trends**: Statistical analysis of misogyny changes over time
- **Event Impact**: Correlation with red-pill influencer events and viral content
- **Community Comparison**: Cross-platform and cross-community analysis
- **Demographic Analysis**: Age group and gender-based patterns
- **Statistical Testing**: Significance testing for all comparisons

### Visualizations
- Interactive Plotly dashboards
- Temporal trend charts with event markers
- Community comparison bar charts
- Age demographic breakdowns
- Platform comparison analyses
- Comprehensive summary dashboard

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd MisogynyWatch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Create a `.env` file in the project root with your API credentials:

```bash
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=MisogynyWatch/1.0
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password

# Twitter API v2 (https://developer.twitter.com/)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_CONSUMER_KEY=your_twitter_consumer_key
TWITTER_CONSUMER_SECRET=your_twitter_consumer_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Quick Start

1. **Collect Data** (30-60 minutes depending on API limits):
```bash
python data_collection_coordinator.py
```

2. **Run Analysis** (5-10 minutes):
```bash
python analysis/main_analysis.py
```

3. **Generate Visualizations**:
```bash
python analysis/visualizations.py
```

### Individual Components

**Reddit Data Collection**:
```bash
python reddit_scraper.py
```

**Twitter Data Collection**:
```bash
python twitter_scraper.py
```

**Text Processing Example**:
```python
from text_processing import TextProcessor
processor = TextProcessor()

text = "Sample social media post text"
misogyny_score = processor.detect_misogyny(text)
keywords = processor.extract_misogyny_keywords(text)
```

**Age Analysis Example**:
```python
from demographics_analyzer import EnhancedDemographicsAnalyzer
analyzer = EnhancedDemographicsAnalyzer()

age = analyzer.extract_age_from_text("I'm 22 years old")
gender = analyzer.extract_gender_from_text("I'm 22M")
age_group = analyzer.assign_age_group(22)
```

## Data Sources

### Reddit Communities Monitored
- Red-pill/manosphere: TheRedPill, MGTOW, MensRights, seduction
- General discussion: relationship_advice, dating_advice, AskMen
- Women-focused: TwoXChromosomes, AskWomen
- Age-specific: teenagers, college, AskMenOver30

### Twitter Search Terms
- Red-pill terminology: "red pill", "alpha male", "hypergamy", "AWALT"
- MGTOW content: "MGTOW", "men going their own way"
- General misogynistic patterns: "women are", "females should"

### Event Timeline Tracking
- Major red-pill influencer events (Andrew Tate, Fresh & Fit, etc.)
- Social movements and backlash events
- Legal/political events affecting gender discussions

## Analysis Methods

### Statistical Approaches
- **Trend Analysis**: Linear regression on time series data
- **Event Impact**: Before/after comparison with t-tests
- **Community Differences**: ANOVA and post-hoc comparisons
- **Correlation Analysis**: Pearson correlation with event proximity

### Misogyny Scoring Algorithm
1. **Keyword Matching**: Multi-category weighted scoring
2. **Frequency Normalization**: Adjusted for text length
3. **Category Weights**: Explicit terms weighted higher than implicit
4. **Composite Scoring**: Combined score across all categories

## Output Files

- `data/processed/reddit_processed.csv` - Processed Reddit data
- `data/processed/twitter_processed.csv` - Processed Twitter data  
- `data/processed/analysis_results.json` - Complete analysis results
- `analysis/plots/` - All visualization files (HTML and PNG)
- `data/processed/collection_summary.json` - Data collection statistics

## Ethical Considerations

- **Privacy**: No personal identifying information collected
- **Content Warning**: Deals with harmful and offensive language
- **Research Purpose**: Academic analysis of harmful online behavior
- **Data Handling**: Secure storage and processing of sensitive content

## Limitations

- **Keyword-based Detection**: May miss subtle or evolving language
- **Platform Bias**: Reddit and Twitter user demographics
- **Temporal Scope**: Limited to recent data due to API restrictions
- **Age Inference**: Based on self-reported information in text
- **Sample Size**: Dependent on API rate limits and data availability

## Future Improvements

- Machine learning models for misogyny detection
- Sentiment analysis integration
- Network analysis of user interactions
- Expanded platform coverage (TikTok, Instagram, etc.)
- Real-time monitoring capabilities

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- `pandas`, `numpy`, `scipy` - Data manipulation and analysis
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `nltk` - Natural language processing
- `praw` - Reddit API wrapper
- `tweepy` - Twitter API wrapper

## License

This project is for academic research purposes. Please ensure compliance with platform Terms of Service and applicable data protection regulations.

## Contact

For questions about methodology, data collection, or analysis techniques, please create an issue in this repository.

---

**Note**: This tool is designed for academic research into online misogyny patterns. The content analyzed may be disturbing or offensive. Proper precautions should be taken when reviewing collected data.