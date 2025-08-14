# MisogynyWatch

A comprehensive research analysis of misogynistic language patterns on Reddit using lexicon-based detection and contextual analysis. This project provides academic-grade insights into gender-based online harassment, demographic patterns, and event-driven discourse changes.

## Research Questions Answered

1. **Which gender is most misogynistic? Male vs Female** → **Male users are 3x more misogynistic** (0.3% vs 0.1% rate)
2. **Which subreddit is most affected?** → **r/relationship_advice** has highest rate (2.1%)  
3. **Which age group is most affected?** → not enough significant data
4. **Event impact visualization?** → **Andrew Tate detention caused +930% spike**

## Key Findings

- **Gender Patterns**: Male users 3x more likely to engage in misogynistic discourse
- **Platform Insights**: Reddit relationship advice contexts especially problematic
- **Event Analysis**: Legal consequences for prominent figures trigger largest spikes
- **Temporal Trends**: 2017-2018 and 2022 were peak years for misogynistic content
- **Research Methodology**: Comprehensive lexicon-based analysis with contextual filtering

## Project Structure

```
MisogynyWatch/
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
├── .env.example                          # API configuration template
├── data/
│   ├── raw/                             # Raw collected data (CSV files)
│   │   ├── reddit_processed.csv         # Sample Reddit dataset
│   │   ├── reddit_MensRights_*.csv      # Subreddit-specific data
│   │   ├── reddit_relationship_advice_*.csv
│   │   └── twitter_keywords_*.csv       # Twitter data samples
│   ├── processed/                       # Analysis-ready datasets
│   │   ├── reddit_processed.csv         # Enhanced Reddit analysis data
│   │   └── twitter_processed.csv        # Twitter analysis data
│   ├── lexicons/                        # Research-based misogyny lexicons
│   └── analysis/                        # Research outputs and reports
│       ├── FINAL_RESEARCH_ANSWERS.md    # Complete research findings
│       ├── research_summary_report.md   # Technical analysis details
│       └── plots/                       # Generated visualizations
├── utils/
│   ├── config.py                        # Configuration settings & event data
│   └── __init__.py
├── enhanced_reddit_analysis.py          # Main analysis system
├── research_summary.py                  # Research findings generator
├── event_impact_analysis.py             # Event correlation analysis
├── yearly_event_analysis.py             # Longitudinal analysis (2014-2024)
├── text_processing.py                   # Lexicon definitions & text preprocessing
├── contextual_misogyny_detector.py      # Detection logic & algorithms
├── enhanced_demographics.py             # Age/gender extraction methods
├── enhanced_reddit_collector.py         # Data collection system
├── reddit_scraper.py                    # Reddit API data collection
├── twitter_scraper.py                   # Twitter API data collection
└── venv/                                # Virtual environment (created during setup)
```

## Features

### Research-Based Misogyny Detection
- **Multi-Category Lexicons**: 7 categories of misogynistic language
  - Explicit slurs and derogatory terms
  - Red-pill/manosphere terminology  
  - Objectification language
  - Dismissive gender language
  - Control/dominance themes
  - Incel terminology
  - MGTOW terminology
- **Contextual Analysis**: Quote detection and educational context filtering
- **Academic Methodology**: Research-grade lexicon-based detection system

### Comprehensive Data Analysis
- **53,069 Reddit Posts**: Large-scale dataset analysis
- **Gender Demographics**: 50% identification success rate (26,535 users)
- **Event Correlation**: Impact analysis of 20 red-pill events (2014-2024)
- **Temporal Analysis**: Year-over-year trend identification
- **Statistical Validation**: Cross-validation and significance testing
### Advanced Demographics Analysis
- **Multi-Method Gender Inference**: Username patterns, content analysis, subreddit participation
- **Age Extraction**: Pattern matching for age mentions in text
- **Enhanced Processing**: NLTK-based text preprocessing and normalization

### Research-Grade Visualizations
- **Comprehensive Analysis Dashboard**: Overview of all findings
- **Event Impact Timeline**: Detailed correlation with red-pill events
- **Longitudinal Analysis**: 10-year trend visualization (2014-2024)
- **Statistical Validation**: Confidence intervals and significance testing
- **Academic-Quality Outputs**: Publication-ready visualizations

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/MalaikaQ/MisogynyWatch.git
cd MisogynyWatch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. API Configuration (Optional - for new data collection)

Create a `.env` file in the project root with Reddit API credentials:

```bash
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=MisogynyWatch/1.0
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
```

*Note: Twitter API configuration removed as analysis focuses on Reddit data only.*

## Data Files & Input Format

This repository includes sample input data files to demonstrate the expected data format:

### Sample Data Files Included
- **`data/raw/reddit_processed.csv`** - Sample Reddit dataset (53,069 posts)
- **`data/raw/reddit_MensRights_*.csv`** - Subreddit-specific sample data
- **`data/raw/reddit_relationship_advice_*.csv`** - Relationship advice subreddit data
- **`data/processed/reddit_processed.csv`** - Analysis-ready processed dataset

### Expected Input Data Format
Reddit CSV files should contain these columns:
```
id, title, selftext, author, subreddit, created_utc, score, num_comments, url
```

Sample row:
```csv
"abc123","Help with relationship","My girlfriend and I are having issues...","user123","relationship_advice","2024-01-15 10:30:00",45,23,"https://reddit.com/r/..."
```

### Data Collection Commands
**Collect new Reddit data:**
```bash
python reddit_scraper.py --subreddit relationship_advice --limit 1000
python enhanced_reddit_collector.py --keywords "dating advice" --days 30
```

**Process collected data:**
```bash
python text_processing.py --input data/raw/ --output data/processed/
```

## Usage

### Quick Analysis (Using Existing Data)

**Run Complete Analysis** (5-10 minutes):
```bash
python enhanced_reddit_analysis.py
# Analyzes data/processed/reddit_processed.csv
# Outputs: enhanced_misogyny_analysis.png, console results
```

**Generate Research Summary**:
```bash
python research_summary.py
# Creates: data/analysis/research_summary_report.md
# Creates: research_summary_visualization.png
```

**Create Event Impact Analysis**:
```bash
python event_impact_analysis.py
# Analyzes 20 red-pill events (2014-2024)
# Creates: event_impact_visualization.png
```

**Generate Yearly Analysis**:
```bash
python yearly_event_analysis.py
# Creates: yearly_misogyny_analysis.png
# Shows longitudinal trends with event correlation
```

### Command-Line Options

**Enhanced Reddit Analysis:**
```bash
python enhanced_reddit_analysis.py [--data_path PATH] [--output_dir PATH]
# --data_path: Path to input CSV file (default: data/processed/reddit_processed.csv)
# --output_dir: Output directory for results (default: current directory)
```

**Data Collection:**
```bash
python reddit_scraper.py --subreddit SUBREDDIT [--limit NUM] [--timeframe DAYS]
# --subreddit: Target subreddit name (required)
# --limit: Number of posts to collect (default: 1000)
# --timeframe: Days back to collect (default: 30)
```

### Expected Output Files

**Analysis Results:**
- `enhanced_misogyny_analysis.png` - Main research dashboard
- `research_summary_visualization.png` - Key findings overview  
- `event_impact_visualization.png` - Event correlation timeline
- `yearly_misogyny_analysis.png` - 10-year longitudinal analysis

**Data Outputs:**
- `data/analysis/research_summary_report.md` - Technical report
- `data/analysis/FINAL_RESEARCH_ANSWERS.md` - Complete findings
- Console output with statistical summaries and key metrics

### Individual Analysis Components

**Misogyny Detection Example**:
```python
from text_processing import TextProcessor
from contextual_misogyny_detector import ContextualMisogynyDetector

processor = TextProcessor()
detector = ContextualMisogynyDetector()

text = "Sample Reddit post text"
misogyny_score = detector.detect_misogyny_enhanced(text)
categories = detector.categorize_misogyny(text)
```

**Demographics Analysis Example**:
```python
from enhanced_demographics import EnhancedDemographicsAnalyzer

analyzer = EnhancedDemographicsAnalyzer()
gender = analyzer.extract_gender_comprehensive("username", "post text", "subreddit")
age = analyzer.extract_age_from_text("I'm 22 years old and...")
```

## Research Data & Methodology

### Detection Algorithm
1. **Research Lexicons**: 7 categories of misogynistic language based on academic research
2. **Contextual Filtering**: Exclude quoted content and educational discussions  
3. **Composite Scoring**: Weighted scoring across multiple misogyny categories
4. **Statistical Analysis**: Cross-validation and significance testing

### Dataset Specifications
- **Reddit Posts Analyzed**: 53,069 posts from targeted subreddits
- **Gender Identification**: 26,535 users (50% success rate)
- **Age Identification**: 452 posts with age indicators (0.85% success rate)  
- **Subreddits Covered**: r/relationship_advice, r/MensRights, r/TwoXChromosomes, etc.
- **Time Range**: Focus on 2020-2024 with historical event analysis back to 2014

### Event Timeline Analysis (20 Major Events)
- **2014**: Isla Vista killings (Elliot Rodger manifesto)
- **2017**: Harvey Weinstein allegations (#MeToo start)
- **2018**: Brett Kavanaugh Supreme Court hearings
- **2022**: Roe v. Wade overturned, Andrew Tate detained
- **2024**: Platform crackdowns and demonetization waves

*Complete event list and impact analysis in data/analysis/FINAL_RESEARCH_ANSWERS.md*

## Research Outputs

### Generated Visualizations
- **`data/analysis/plots/research_summary_visualization.png`** - Comprehensive findings overview
- **`data/analysis/plots/event_impact_visualization.png`** - Timeline of event-driven spikes  
- **`data/analysis/plots/yearly_misogyny_analysis.png`** - Longitudinal analysis (2014-2024)

### Research Documents
- **`data/analysis/FINAL_RESEARCH_ANSWERS.md`** - Complete research findings and methodology
- **`data/analysis/research_summary_report.md`** - Technical analysis details

### Data Files
- **`data/processed/enhanced_reddit_analysis.csv`** - Processed dataset with scores
- **`data/processed/demographic_analysis.json`** - Gender and age statistics
- **`data/processed/event_impact_results.json`** - Event correlation analysis

## Key Research Insights

### Gender Analysis  
- **Male users**: 3x more likely to engage in misogynistic discourse
- **Pattern**: Male-dominated spaces show higher rates
- **Insight**: Gender role discussions trigger elevated misogyny

### Subreddit Analysis
- **r/relationship_advice**: Highest rate (2.1%) - relationship contexts problematic
- **r/MensRights**: Second highest (0.4%) - ideological discussions  
- **Pattern**: Advice-seeking contexts vulnerable to misogynistic responses

### Temporal Patterns
- **2017-2018**: Peak period due to #MeToo backlash and political events
- **2022**: Second peak from Roe v. Wade and Andrew Tate events
- **2024**: Decline due to platform enforcement actions

### Event Impact
- **Legal consequences**: Highest impact (Andrew Tate detention +930%)
- **Political events**: Significant spikes (Roe v. Wade, Kavanaugh hearings)
- **Platform actions**: Effective at reducing rates (bans, demonetization)

## Ethical Considerations & Limitations

### Ethics
- **Academic Research Purpose**: Analysis of harmful online behavior patterns
- **Privacy Protection**: No personal identifying information collected or stored
- **Content Warning**: Analysis involves disturbing and offensive language
- **Responsible Usage**: Results intended for harm reduction and policy research

### Technical Limitations
- **Detection Method**: Lexicon-based approach may miss evolving language patterns
- **Platform Scope**: Reddit-focused analysis (Twitter analysis removed)
- **Age Demographics**: Limited success rate (0.85%) due to sparse age indicators
- **Temporal Coverage**: API limitations restrict historical data collection
- **Sample Bias**: Reddit user demographics may not represent general population

### Research Limitations
- **Causality**: Correlation analysis cannot establish causal relationships
- **Cultural Context**: Analysis focused on English-language content
- **Detection Scope**: Lexicon-based approach captures specific language patterns
- **Event Attribution**: Multiple confounding factors may influence trends

## Future Research Directions

- **Machine Learning Models**: Advanced NLP for context-aware detection
- **Real-Time Monitoring**: Live dashboard for ongoing trend analysis  
- **Cross-Platform Analysis**: Integration with TikTok, Instagram, YouTube
- **Enhanced Demographics**: Improved age and geographic inference
- **Intervention Studies**: Testing effectiveness of counter-messaging strategies
- **Longitudinal User Studies**: Tracking individual behavior changes over time

## Dependencies & Technical Requirements

### Core Dependencies
```
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0           # Numerical computing
scipy>=1.9.0            # Statistical analysis
matplotlib>=3.5.0       # Basic plotting
seaborn>=0.11.0         # Statistical visualization  
plotly>=5.10.0          # Interactive visualizations
nltk>=3.7               # Natural language processing
textblob>=0.17.1        # Sentiment analysis and NLP
praw>=7.6.0             # Reddit API wrapper
tweepy>=4.12.0          # Twitter API wrapper
requests>=2.28.0        # HTTP requests (web scraping backup)
python-dotenv>=0.19.0   # Environment variable management
```

### System Requirements
- **Python**: 3.8+ recommended
- **Memory**: 4GB+ RAM for large dataset processing
- **Storage**: 1GB+ for datasets and visualizations
- **Network**: Internet connection for data collection (optional)


