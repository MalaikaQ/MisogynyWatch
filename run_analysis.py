#!/usr/bin/env python3
"""
Main execution script for MisogynyWatch project.
Run complete pipeline: data collection → analysis → visualization
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_collection_coordinator import DataCollectionCoordinator
from analysis.main_analysis import MisogynyAnalyzer
from analysis.visualizations import MisogynyVisualizer


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='MisogynyWatch: Analyze misogyny trends across social media')
    
    parser.add_argument('--collect-only', action='store_true', 
                       help='Only collect data, skip analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing data, skip collection')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only create visualizations from existing analysis')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days of historical data to collect (default: 30)')
    parser.add_argument('--reddit-posts', type=int, default=300,
                       help='Posts per Reddit community (default: 300)')
    parser.add_argument('--twitter-tweets', type=int, default=150,
                       help='Tweets per search term (default: 150)')
    parser.add_argument('--quick-run', action='store_true',
                       help='Quick run with reduced data collection')
    
    args = parser.parse_args()
    
    print("🔍 MisogynyWatch: Social Media Misogyny Analysis")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick_run:
        args.days_back = 7
        args.reddit_posts = 50
        args.twitter_tweets = 50
        print("⚡ Quick run mode enabled - reduced data collection")
    
    # Data Collection Phase
    if not args.analyze_only and not args.visualize_only:
        print("\n📊 Phase 1: Data Collection")
        print("-" * 30)
        
        try:
            coordinator = DataCollectionCoordinator()
            summary = coordinator.run_full_collection(
                days_back=args.days_back,
                reddit_posts_per_community=args.reddit_posts,
                twitter_tweets_per_term=args.twitter_tweets
            )
            
            print(f"✅ Data collection completed!")
            print(f"   Reddit posts: {summary['reddit_posts_collected']}")
            print(f"   Twitter posts: {summary['twitter_posts_collected']}")
            print(f"   Misogynistic content: {summary['reddit_misogynistic_posts'] + summary['twitter_misogynistic_posts']}")
            
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
            if args.collect_only:
                return 1
            print("⚠️  Proceeding with existing data...")
    
    if args.collect_only:
        print("\n✅ Data collection completed. Use --analyze-only to run analysis.")
        return 0
    
    # Analysis Phase  
    if not args.visualize_only:
        print("\n🧮 Phase 2: Analysis")
        print("-" * 30)
        
        try:
            analyzer = MisogynyAnalyzer()
            results = analyzer.generate_comprehensive_report()
            
            if "error" in results:
                print(f"❌ Analysis failed: {results['error']}")
                return 1
            
            print("✅ Analysis completed!")
            print(f"   Temporal trends: {'✓' if 'temporal_trends' in results else '✗'}")
            print(f"   Event impacts: {'✓' if 'event_impacts' in results else '✗'}")
            print(f"   Community analysis: {'✓' if 'community_differences' in results else '✗'}")
            print(f"   Age demographics: {'✓' if 'age_demographics' in results else '✗'}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            if args.analyze_only:
                return 1
            print("⚠️  Skipping visualization...")
            return 1
    
    if args.analyze_only:
        print("\n✅ Analysis completed. Use --visualize-only to generate plots.")
        return 0
    
    # Visualization Phase
    print("\n📈 Phase 3: Visualization")
    print("-" * 30)
    
    try:
        # Load results if not already available
        if not 'results' in locals():
            import json
            from utils.config import PROCESSED_DATA_DIR
            
            results_path = PROCESSED_DATA_DIR / 'analysis_results.json'
            with open(results_path, 'r') as f:
                results = json.load(f)
        
        visualizer = MisogynyVisualizer()
        visualizer.generate_all_plots(results)
        
        print("✅ Visualizations completed!")
        print(f"   Plots saved to: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return 1
    
    # Summary
    print("\n🎉 Pipeline Completed Successfully!")
    print("=" * 50)
    print("📁 Output Files:")
    print("   • data/processed/reddit_processed.csv")
    print("   • data/processed/twitter_processed.csv") 
    print("   • data/processed/analysis_results.json")
    print("   • analysis/plots/ (interactive visualizations)")
    print("\n💡 Next Steps:")
    print("   • Open analysis/plots/comprehensive_dashboard.html in your browser")
    print("   • Review the analysis_results.json for detailed findings")
    print("   • Check the notebooks/ directory for further exploration")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
