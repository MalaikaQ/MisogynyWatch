# MisogynyWatch

A comprehensive research analysis of misogynistic language patterns on Reddit using lexicon-based detection and contextual analysis. This project provides academic-grade insights into gender-based online harassment, demographic patterns, and event-driven discourse changes.

## ✅ Research Questions Answered

1. **Which gender is most misogynistic?** → **Male users are 3x more misogynistic** (0.3% vs 0.1% rate)
2. **Which subreddit is most affected?** → **r/relationship_advice** has highest rate (2.1%)  
3. **Which age group is most affected?** → **Young adult males (20-30)** most represented
4. **Event impact visualization?** → **Andrew Tate detention caused +930% spike**

## 🎯 Key Findings

- **Gender Patterns**: Male users 3x more likely to engage in misogynistic discourse
- **Platform Insights**: Reddit relationship advice contexts especially problematic
- **Event Analysis**: Legal consequences for prominent figures trigger largest spikes
- **Temporal Trends**: 2017-2018 and 2022 were peak years for misogynistic content
- **Research Methodology**: Comprehensive lexicon-based analysis with contextual filtering

## 📁 Project Structure

```
MisogynyWatch/
├── README.md
├── requirements.txt
├── .env.example                           # API configuration template
├── FINAL_RESEARCH_ANSWERS.md             # Complete research findings
├── data/
│   ├── raw/                              # Raw Reddit data 
│   ├── processed/                        # Enhanced processed datasets
│   └── lexicons/                         # Research-based misogyny lexicons
├── utils/
│   ├── config.py                         # Configuration settings
│   └── __init__.py
├── enhanced_reddit_analysis.py           # Main analysis system
├── research_summary.py                   # Research findings generator
├── event_impact_analysis.py              # Event correlation analysis
├── yearly_event_analysis.py              # Longitudinal analysis (2014-2024)
├── text_processing.py                    # Lexicon definitions
├── contextual_misogyny_detector.py       # Enhanced detection logic
├── enhanced_demographics.py              # Age/gender extraction
├── enhanced_reddit_collector.py          # Data collection system
└── visualizations/                       # Generated research outputs
    ├── research_summary_visualization.png
    ├── event_impact_visualization.png
    └── yearly_misogyny_analysis.png
```

## 🚀 Features

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

## 🛠️ Setup Instructions

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

## 🔬 Usage

### Quick Analysis (Using Existing Data)

**Run Complete Analysis** (5-10 minutes):
```bash
python enhanced_reddit_analysis.py
```

**Generate Research Summary**:
```bash
python research_summary.py
```

**Create Event Impact Analysis**:
```bash
python event_impact_analysis.py
```

**Generate Yearly Analysis**:
```bash
python yearly_event_analysis.py
```

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

## 📊 Research Data & Methodology

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

*Complete event list and impact analysis in FINAL_RESEARCH_ANSWERS.md*

## 📈 Research Outputs

### Generated Visualizations
- **`research_summary_visualization.png`** - Comprehensive findings overview
- **`event_impact_visualization.png`** - Timeline of event-driven spikes  
- **`yearly_misogyny_analysis.png`** - Longitudinal analysis (2014-2024)

### Research Documents
- **`FINAL_RESEARCH_ANSWERS.md`** - Complete research findings and methodology
- **`research_summary_report.md`** - Technical analysis details

### Data Files
- **`data/processed/enhanced_reddit_analysis.csv`** - Processed dataset with scores
- **`data/processed/demographic_analysis.json`** - Gender and age statistics
- **`data/processed/event_impact_results.json`** - Event correlation analysis

## 🎯 Key Research Insights

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

## ⚖️ Ethical Considerations & Limitations

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

## 🔮 Future Research Directions

- **Machine Learning Models**: Advanced NLP for context-aware detection
- **Real-Time Monitoring**: Live dashboard for ongoing trend analysis  
- **Cross-Platform Analysis**: Integration with TikTok, Instagram, YouTube
- **Enhanced Demographics**: Improved age and geographic inference
- **Intervention Studies**: Testing effectiveness of counter-messaging strategies
- **Longitudinal User Studies**: Tracking individual behavior changes over time

## 📚 Dependencies & Technical Requirements

### Core Dependencies
```python
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.20.0           # Numerical computing
matplotlib>=3.5.0       # Basic plotting
seaborn>=0.11.0         # Statistical visualization  
nltk>=3.7               # Natural language processing
scipy>=1.8.0            # Statistical analysis
praw>=7.0.0             # Reddit API wrapper
python-dotenv>=0.19.0   # Environment variable management
```

### System Requirements
- **Python**: 3.8+ recommended
- **Memory**: 4GB+ RAM for large dataset processing
- **Storage**: 1GB+ for datasets and visualizations
- **Network**: Internet connection for data collection (optional)

## 📄 License & Citation

### License
This project is licensed for academic research purposes. Please ensure compliance with:
- Platform Terms of Service (Reddit API)
- Applicable data protection regulations (GDPR, CCPA)
- Institutional Review Board (IRB) requirements for human subjects research

### Citation
If you use this research or codebase, please cite:
```
MisogynyWatch: Enhanced Lexicon-Based Analysis of Online Misogyny Patterns
GitHub Repository: https://github.com/MalaikaQ/MisogynyWatch
Year: 2025
```

## 🤝 Contributing & Contact

### Contributing
- Fork the repository and create feature branches
- Follow academic research ethics and methodology standards
- Submit pull requests with detailed documentation
- Report issues or suggest improvements via GitHub Issues

### Contact & Support
- **Research Questions**: Create an issue with the "research" label
- **Technical Issues**: Use GitHub Issues for bug reports
- **Methodology Discussions**: Open a discussion thread
- **Academic Collaboration**: Contact via repository discussions

---

## ⚠️ Important Disclaimers

**Content Warning**: This research analyzes disturbing and offensive language patterns. Users should be prepared to encounter misogynistic content during analysis.

**Research Ethics**: This tool is designed for academic research into online harassment patterns. All analysis should be conducted with appropriate ethical oversight and for harm reduction purposes.

**Platform Compliance**: Ensure all data collection and analysis complies with platform Terms of Service and applicable legal requirements.

**Academic Integrity**: Results should be interpreted within the context of the methodology's limitations and used responsibly in academic and policy contexts.

