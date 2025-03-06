#!/usr/bin/env python3
"""
Script to analyze coverage trends over time.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.coverage.utils import parse_coverage_xml, calculate_coverage_stats
from src.coverage import COVERAGE_THRESHOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class CoverageTrendAnalyzer:
    """Analyzes coverage trends over time."""
    
    def __init__(self, data_dir: str = 'tests/reports/coverage'):
        """Initialize analyzer with data directory."""
        self.data_dir = Path(data_dir)
        self.trends_file = self.data_dir / 'trends.json'
        
    def add_coverage_data(self, coverage_xml: str) -> None:
        """Add new coverage data point."""
        try:
            # Parse coverage data
            coverage_data = parse_coverage_xml(coverage_xml)
            if not coverage_data:
                logger.error("Failed to parse coverage data")
                return
                
            # Calculate statistics
            stats = calculate_coverage_stats(coverage_data)
            
            # Load existing trends
            trends = self.load_trends()
            
            # Add new data point
            trends.append({
                'timestamp': datetime.now().isoformat(),
                'coverage': stats,
                'files': len(coverage_data['files'])
            })
            
            # Save updated trends
            self.save_trends(trends)
            
        except Exception as e:
            logger.error(f"Error adding coverage data: {str(e)}")
            
    def load_trends(self) -> List[Dict[str, Any]]:
        """Load historical trends data."""
        if self.trends_file.exists():
            try:
                with open(self.trends_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading trends: {str(e)}")
        return []
        
    def save_trends(self, trends: List[Dict[str, Any]]) -> None:
        """Save trends data."""
        try:
            self.trends_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trends_file, 'w') as f:
                json.dump(trends, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trends: {str(e)}")
            
    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze coverage trends for specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            trends = self.load_trends()
            if not trends:
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(trends)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Filter for specified period
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff]
            
            if df.empty:
                return {}
                
            # Calculate statistics
            analysis = {
                'start_date': df.index.min().isoformat(),
                'end_date': df.index.max().isoformat(),
                'data_points': len(df),
                'current_coverage': df.iloc[-1]['coverage'],
                'coverage_change': (
                    df.iloc[-1]['coverage']['total_coverage'] -
                    df.iloc[0]['coverage']['total_coverage']
                ),
                'trend_direction': self._get_trend_direction(df),
                'below_threshold_days': self._count_below_threshold_days(df),
                'stability_score': self._calculate_stability_score(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {}
            
    def generate_report(self, output_dir: str = 'reports') -> None:
        """Generate trend analysis report."""
        try:
            analysis = self.analyze_trends()
            if not analysis:
                logger.error("No trend data available")
                return
                
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            self._plot_coverage_trend(output_path / 'coverage_trend.png')
            self._plot_stability_trend(output_path / 'stability_trend.png')
            
            # Generate markdown report
            self._generate_markdown_report(analysis, output_path / 'trends_report.md')
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            
    def _get_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction."""
        first = df.iloc[0]['coverage']['total_coverage']
        last = df.iloc[-1]['coverage']['total_coverage']
        
        if last > first:
            return 'improving'
        elif last < first:
            return 'declining'
        return 'stable'
        
    def _count_below_threshold_days(self, df: pd.DataFrame) -> int:
        """Count days below minimum threshold."""
        min_coverage = COVERAGE_THRESHOLDS['acceptable']
        return len(df[df['coverage'].apply(
            lambda x: x['total_coverage'] < min_coverage
        )])
        
    def _calculate_stability_score(self, df: pd.DataFrame) -> float:
        """Calculate coverage stability score (0-100)."""
        try:
            coverage_values = df['coverage'].apply(
                lambda x: x['total_coverage']
            ).values
            
            # Calculate volatility
            volatility = pd.Series(coverage_values).std()
            
            # Calculate trend consistency
            diffs = np.diff(coverage_values)
            direction_changes = np.sum(np.abs(np.diff(np.sign(diffs))))
            
            # Combine into score (higher is better)
            max_volatility = 10  # Maximum expected volatility
            max_changes = len(coverage_values) - 2  # Maximum possible direction changes
            
            volatility_score = max(0, 100 * (1 - volatility / max_volatility))
            consistency_score = 100 * (1 - direction_changes / max_changes)
            
            return (volatility_score + consistency_score) / 2
            
        except Exception:
            return 0.0
            
    def _plot_coverage_trend(self, output_path: Path) -> None:
        """Generate coverage trend plot."""
        trends = self.load_trends()
        if not trends:
            return
            
        df = pd.DataFrame(trends)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['coverage'] = df['coverage'].apply(lambda x: x['total_coverage'])
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        plt.plot(df['timestamp'], df['coverage'], marker='o')
        
        # Add threshold lines
        for level, value in COVERAGE_THRESHOLDS.items():
            plt.axhline(y=value, color='gray', linestyle='--', alpha=0.5)
            plt.text(
                df['timestamp'].min(),
                value,
                f' {level} ({value}%)',
                verticalalignment='bottom'
            )
            
        plt.title('Coverage Trend')
        plt.xlabel('Date')
        plt.ylabel('Coverage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()
        
    def _plot_stability_trend(self, output_path: Path) -> None:
        """Generate stability trend plot."""
        trends = self.load_trends()
        if not trends:
            return
            
        df = pd.DataFrame(trends)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate rolling stability scores
        window = min(7, len(df))  # 7-day window or less if not enough data
        df['stability'] = df['coverage'].rolling(window).apply(
            lambda x: self._calculate_stability_score(pd.DataFrame({'coverage': x}))
        )
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        plt.plot(df['timestamp'], df['stability'], marker='o')
        plt.title('Coverage Stability Trend')
        plt.xlabel('Date')
        plt.ylabel('Stability Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()
        
    def _generate_markdown_report(
        self,
        analysis: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Generate markdown report."""
        report = f"""# Coverage Trend Analysis

## Summary
- Period: {analysis['start_date']} to {analysis['end_date']}
- Data Points: {analysis['data_points']}
- Current Coverage: {analysis['current_coverage']['total_coverage']:.1f}%
- Coverage Change: {analysis['coverage_change']:+.1f}%
- Trend Direction: {analysis['trend_direction'].title()}
- Days Below Threshold: {analysis['below_threshold_days']}
- Stability Score: {analysis['stability_score']:.1f}/100

## Trend Visualization
![Coverage Trend](coverage_trend.png)
![Stability Trend](stability_trend.png)

## Details
### Current Coverage Metrics
- Line Coverage: {analysis['current_coverage']['line_coverage']:.1f}%
- Branch Coverage: {analysis['current_coverage']['branch_coverage']:.1f}%
- Total Coverage: {analysis['current_coverage']['total_coverage']:.1f}%

### Thresholds
{chr(10).join([f'- {level.title()}: {value}%' for level, value in COVERAGE_THRESHOLDS.items()])}

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        output_path.write_text(report)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze coverage trends"
    )
    
    parser.add_argument(
        '--coverage-file',
        help='Path to new coverage XML file'
    )
    
    parser.add_argument(
        '--data-dir',
        default='tests/reports/coverage',
        help='Directory containing coverage data'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports',
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = CoverageTrendAnalyzer(args.data_dir)
        
        if args.coverage_file:
            analyzer.add_coverage_data(args.coverage_file)
            
        analyzer.generate_report(args.output_dir)
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())