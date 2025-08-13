"""
Visualization module for MisogynyWatch project.
Creates charts and plots for research findings.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, RED_PILL_EVENTS

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MisogynyVisualizer:
    """Create visualizations for misogyny analysis."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer."""
        self.output_dir = output_dir or Path(__file__).parent / 'plots'
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_temporal_trends(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Plot temporal trends in misogynistic content.
        
        Args:
            results: Results from temporal analysis
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if 'monthly_trends' not in results:
            return None
            
        monthly_trends = results['monthly_trends']
        
        fig = go.Figure()
        
        # Add traces for each platform
        for platform in monthly_trends['platform'].unique():
            platform_data = monthly_trends[monthly_trends['platform'] == platform]
            
            fig.add_trace(go.Scatter(
                x=platform_data['year_month'].astype(str),
                y=platform_data['misogyny_percentage'],
                mode='lines+markers',
                name=f'{platform} - Misogyny %',
                line=dict(width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=platform_data['year_month'].astype(str),
                y=platform_data['avg_misogyny_score'] * 100,  # Scale to percentage
                mode='lines+markers',
                name=f'{platform} - Avg Score',
                line=dict(dash='dash', width=2),
                yaxis='y2'
            ))
        
        # Add event markers (only if x-axis contains actual dates)
        try:
            if len(monthly_trends) > 0:
                # Check if we can convert x values to dates
                x_values = monthly_trends['year_month'].astype(str).values
                for event_date, event_desc in RED_PILL_EVENTS.items():
                    # Only add marker if event date is within our data range
                    event_year_month = event_date[:7]  # Get YYYY-MM format
                    if event_year_month in x_values:
                        fig.add_vline(
                            x=event_year_month,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=event_desc[:30] + "...",
                            annotation_position="top"
                        )
        except Exception:
            # Skip event markers if there are compatibility issues
            pass
        
        fig.update_layout(
            title='Temporal Trends in Misogynistic Content',
            xaxis_title='Time Period',
            yaxis_title='Misogynistic Content (%)',
            yaxis2=dict(
                title='Average Misogyny Score',
                overlaying='y',
                side='right'
            ),
            height=600,
            showlegend=True
        )
        
        if save:
            fig.write_html(self.output_dir / 'temporal_trends.html')
            
        return fig
    
    def plot_event_impact(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Plot impact of red-pill events on misogynistic content.
        
        Args:
            results: Results from event impact analysis
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if 'event_impacts' not in results:
            return None
            
        event_impacts = results['event_impacts']
        
        # Create subplot figure
        fig = go.Figure()
        
        # Bar chart of event impacts
        colors = ['red' if x > 0 else 'blue' for x in event_impacts['percent_change']]
        
        fig.add_trace(go.Bar(
            x=event_impacts['event_description'],
            y=event_impacts['percent_change'],
            marker_color=colors,
            name='Percent Change in Misogyny',
            text=[f"{x:.1f}%" for x in event_impacts['percent_change']],
            textposition='auto'
        ))
        
        # Add significance markers
        significant_events = event_impacts[event_impacts['significant']]
        fig.add_trace(go.Scatter(
            x=significant_events['event_description'],
            y=significant_events['percent_change'],
            mode='markers',
            marker=dict(
                symbol='star',
                size=15,
                color='yellow',
                line=dict(color='black', width=2)
            ),
            name='Statistically Significant',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Impact of Red-Pill Events on Misogynistic Content',
            xaxis_title='Event',
            yaxis_title='Percent Change in Misogyny (%)',
            xaxis_tickangle=-45,
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / 'event_impact.html')
            
        return fig
    
    def plot_community_comparison(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Plot comparison of misogyny across communities.
        
        Args:
            results: Results from community analysis
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Reddit communities
        if 'reddit_communities' in results:
            reddit_data = results['reddit_communities'].head(15)  # Top 15
            
            fig.add_trace(go.Bar(
                x=reddit_data['subreddit'],
                y=reddit_data['misogyny_percentage'],
                name='Reddit Communities',
                marker_color='orange',
                text=[f"{x:.1f}%" for x in reddit_data['misogyny_percentage']],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Misogyny Levels Across Reddit Communities',
            xaxis_title='Community',
            yaxis_title='Misogyny Percentage (%)',
            xaxis_tickangle=-45,
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / 'community_comparison.html')
            
        return fig
    
    def plot_age_demographics(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Plot age demographics analysis.
        
        Args:
            results: Results from age analysis
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if 'age_groups' not in results:
            return None
            
        age_groups = results['age_groups']
        
        fig = go.Figure()
        
        # Age group misogyny levels
        fig.add_trace(go.Bar(
            x=age_groups['age_group'],
            y=age_groups['avg_misogyny_score'],
            name='Average Misogyny Score',
            marker_color='red',
            yaxis='y'
        ))
        
        # Add count as secondary axis
        fig.add_trace(go.Scatter(
            x=age_groups['age_group'],
            y=age_groups['total_posts'],
            mode='lines+markers',
            name='Total Posts',
            line=dict(color='blue', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Misogyny Levels by Age Group',
            xaxis_title='Age Group',
            yaxis_title='Average Misogyny Score',
            yaxis2=dict(
                title='Total Posts',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / 'age_demographics.html')
            
        return fig
    
    def plot_platform_comparison(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Plot comparison between Reddit and Twitter.
        
        Args:
            results: Results from platform analysis
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if 'platform_comparison' not in results:
            return None
            
        platform_data = results['platform_comparison']
        
        fig = go.Figure()
        
        # Bar chart comparing platforms
        fig.add_trace(go.Bar(
            x=platform_data['platform'],
            y=platform_data['avg_misogyny_score'],
            name='Average Misogyny Score',
            marker_color=['#FF6B35', '#1DA1F2'],  # Reddit orange, Twitter blue
            text=[f"{x:.3f}" for x in platform_data['avg_misogyny_score']],
            textposition='auto'
        ))
        
        # Add error bars (standard deviation)
        fig.update_traces(
            error_y=dict(
                type='data',
                array=platform_data['std_misogyny_score'],
                visible=True
            )
        )
        
        fig.update_layout(
            title='Platform Comparison: Misogyny Levels',
            xaxis_title='Platform',
            yaxis_title='Average Misogyny Score',
            height=500
        )
        
        if save:
            fig.write_html(self.output_dir / 'platform_comparison.html')
            
        return fig
    
    def create_comprehensive_dashboard(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            results: Complete analysis results
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure with subplots
        """
        from plotly.subplots import make_subplots
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Temporal Trends',
                'Community Comparison',
                'Age Demographics',
                'Platform Comparison'
            ),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add temporal trends (simplified)
        if 'temporal_trends' in results and 'monthly_trends' in results['temporal_trends']:
            monthly_data = results['temporal_trends']['monthly_trends']
            for platform in monthly_data['platform'].unique():
                platform_data = monthly_data[monthly_data['platform'] == platform]
                fig.add_trace(
                    go.Scatter(
                        x=platform_data['year_month'].astype(str),
                        y=platform_data['misogyny_percentage'],
                        mode='lines+markers',
                        name=f'{platform}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add community comparison
        if 'community_differences' in results and 'reddit_communities' in results['community_differences']:
            reddit_data = results['community_differences']['reddit_communities'].head(10)
            fig.add_trace(
                go.Bar(
                    x=reddit_data['subreddit'],
                    y=reddit_data['misogyny_percentage'],
                    name='Reddit Communities',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add age demographics
        if 'age_demographics' in results and 'age_groups' in results['age_demographics']:
            age_data = results['age_demographics']['age_groups']
            fig.add_trace(
                go.Bar(
                    x=age_data['age_group'],
                    y=age_data['avg_misogyny_score'],
                    name='Age Groups',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Add platform comparison
        if 'community_differences' in results and 'platform_comparison' in results['community_differences']:
            platform_data = results['community_differences']['platform_comparison']
            fig.add_trace(
                go.Bar(
                    x=platform_data['platform'],
                    y=platform_data['avg_misogyny_score'],
                    name='Platforms',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="MisogynyWatch: Comprehensive Analysis Dashboard",
            height=800
        )
        
        if save:
            fig.write_html(self.output_dir / 'comprehensive_dashboard.html')
            
        return fig
    
    def generate_all_plots(self, results: Dict):
        """
        Generate all available plots from analysis results.
        
        Args:
            results: Complete analysis results dictionary
        """
        print("Generating visualizations...")
        
        # Generate individual plots
        if 'temporal_trends' in results:
            self.plot_temporal_trends(results['temporal_trends'])
            print("✓ Temporal trends plot created")
        
        if 'event_impacts' in results:
            self.plot_event_impact(results['event_impacts'])
            print("✓ Event impact plot created")
        
        if 'community_differences' in results:
            self.plot_community_comparison(results['community_differences'])
            self.plot_platform_comparison(results['community_differences'])
            print("✓ Community comparison plots created")
        
        if 'age_demographics' in results:
            self.plot_age_demographics(results['age_demographics'])
            print("✓ Age demographics plot created")
        
        # Generate comprehensive dashboard
        self.create_comprehensive_dashboard(results)
        print("✓ Comprehensive dashboard created")
        
        print(f"\nAll plots saved to: {self.output_dir}")


def create_static_plots(results: Dict, output_dir: Path = None):
    """
    Create static matplotlib plots as backup/alternative to Plotly.
    
    Args:
        results: Analysis results
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'static_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    # Temporal trends plot
    if 'temporal_trends' in results and 'monthly_trends' in results['temporal_trends']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        monthly_data = results['temporal_trends']['monthly_trends']
        
        for platform in monthly_data['platform'].unique():
            platform_data = monthly_data[monthly_data['platform'] == platform]
            x_vals = range(len(platform_data))
            
            ax1.plot(x_vals, platform_data['misogyny_percentage'], 
                    marker='o', label=f'{platform} - Misogyny %', linewidth=2)
            ax2.plot(x_vals, platform_data['avg_misogyny_score'], 
                    marker='s', label=f'{platform} - Avg Score', linewidth=2)
        
        ax1.set_title('Misogynistic Content Percentage Over Time')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Average Misogyny Score Over Time')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Average Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Static plots saved to: {output_dir}")


if __name__ == "__main__":
    # Load analysis results and create visualizations
    import json
    
    try:
        results_path = PROCESSED_DATA_DIR / 'analysis_results.json'
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        visualizer = MisogynyVisualizer()
        visualizer.generate_all_plots(results)
        
        # Also create static plots
        create_static_plots(results)
        
    except FileNotFoundError:
        print("Analysis results not found. Please run main_analysis.py first.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
