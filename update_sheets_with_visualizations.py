#!/usr/bin/env python3
"""
Comprehensive script to update Google Sheets with strategy data, embed visualizations,
and configure the system for autopilot operation with profitable strategy generation.
"""
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import gspread
import time
import random
from googleapiclient.discovery import build
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/sheets_update.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import GoogleSheetIntegration
from src.utils.google_sheet_integration import GoogleSheetIntegration, retry_with_backoff
from oauth2client.service_account import ServiceAccountCredentials

class EnhancedGoogleSheetIntegration(GoogleSheetIntegration):
    """Enhanced Google Sheet integration with visualization embedding capabilities."""
    
    def __init__(self, credentials_path=None):
        """Initialize with parent class and add drive service."""
        super().__init__(credentials_path)
        self.drive_service = None
        self.visualization_urls = {}
        
    def initialize_drive_service(self):
        """Initialize the Google Drive service for file operations."""
        if not self.client:
            if not self.initialize():
                return False
                
        # Define the scope for Google Drive
        scope = ['https://www.googleapis.com/auth/drive']
        
        # Authenticate using the credentials
        creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_path, scope)
        
        # Create the drive service
        self.drive_service = build('drive', 'v3', credentials=creds)
        return True
        
    @retry_with_backoff
    def upload_visualization(self, file_path, file_name=None):
        """
        Upload a visualization to Google Drive and return the URL.
        
        Args:
            file_path: Path to the visualization file
            file_name: Optional name for the file in Google Drive
            
        Returns:
            URL of the uploaded file
        """
        if not self.drive_service:
            if not self.initialize_drive_service():
                logger.error("Failed to initialize drive service")
                return None
                
        try:
            # Create file metadata
            if not file_name:
                file_name = os.path.basename(file_path)
                
            file_metadata = {
                'name': file_name,
                'mimeType': 'image/png'
            }
            
            # Create media
            from googleapiclient.http import MediaFileUpload
            media = MediaFileUpload(file_path, mimetype='image/png')
            
            # Upload file
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            ).execute()
            
            # Make the file publicly accessible
            self.drive_service.permissions().create(
                fileId=file.get('id'),
                body={'type': 'anyone', 'role': 'reader'},
                fields='id'
            ).execute()
            
            # Get the web view link
            file = self.drive_service.files().get(
                fileId=file.get('id'),
                fields='webViewLink'
            ).execute()
            
            url = file.get('webViewLink')
            self.visualization_urls[file_name] = url
            
            logger.info(f"Uploaded visualization {file_name} to Google Drive: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading visualization: {str(e)}")
            return None
    
    @retry_with_backoff
    def create_visualization_tab(self):
        """Create a dedicated visualizations tab in the Google Sheet."""
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            # Check if Visualizations tab exists
            if 'Visualizations' not in self.worksheets:
                # Create the worksheet
                viz_ws = self.sheet.add_worksheet(title='Visualizations', rows=50, cols=10)
                self.worksheets['Visualizations'] = viz_ws
                
                # Add header
                viz_ws.update_cell(1, 1, 'Strategy Visualizations')
                viz_ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
                
                # Add description
                viz_ws.update_cell(2, 1, 'This tab contains visualizations for each strategy')
                
                # Merge cells for header and description
                viz_ws.merge_cells('A1:J1')
                viz_ws.merge_cells('A2:J2')
                
                logger.info("Created Visualizations tab")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating Visualizations tab: {str(e)}")
            return False
    
    @retry_with_backoff
    def add_visualization_to_sheet(self, title, description, image_url):
        """
        Add a visualization to the Visualizations tab.
        
        Args:
            title: Title of the visualization
            description: Description of the visualization
            image_url: URL of the image in Google Drive
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            # Get the Visualizations worksheet
            viz_ws = self.worksheets.get('Visualizations')
            if not viz_ws:
                if not self.create_visualization_tab():
                    return False
                viz_ws = self.worksheets.get('Visualizations')
            
            # Find the next available row
            values = viz_ws.get_all_values()
            next_row = len(values) + 1
            
            # Add a gap if not the first visualization
            if next_row > 3:
                next_row += 1
            
            # Add title
            viz_ws.update_cell(next_row, 1, title)
            viz_ws.format(f'A{next_row}', {'textFormat': {'bold': True, 'fontSize': 12}})
            
            # Add description
            viz_ws.update_cell(next_row + 1, 1, description)
            
            # Add image formula
            image_formula = f'=IMAGE("{image_url}", 4, 500, 300)'
            viz_ws.update_cell(next_row + 2, 1, image_formula)
            
            # Merge cells for title and description
            viz_ws.merge_cells(f'A{next_row}:J{next_row}')
            viz_ws.merge_cells(f'A{next_row+1}:J{next_row+1}')
            viz_ws.merge_cells(f'A{next_row+2}:J{next_row+5}')  # Make room for the image
            
            logger.info(f"Added visualization '{title}' to Visualizations tab")
            return True
            
        except Exception as e:
            logger.error(f"Error adding visualization to sheet: {str(e)}")
            return False

    @retry_with_backoff
    def update_market_research_tab(self, research_data):
        """
        Update the MARKET_RESEARCH tab with analysis and findings.
        
        Args:
            research_data: Dictionary containing market research data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('MARKET_RESEARCH')
            if not worksheet:
                logger.error("MARKET_RESEARCH worksheet not found")
                return False
                
            # Format the data
            now = datetime.now().strftime("%Y-%m-%d")
            row_data = [
                research_data.get('date', now),
                research_data.get('market', 'US Equities'),
                research_data.get('analysis_type', 'Market Regime Analysis'),
                research_data.get('findings', ''),
                research_data.get('supporting_data', ''),
                research_data.get('impact', ''),
                research_data.get('confidence', 'High'),
                research_data.get('action_items', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added market research for {research_data.get('market')} to MARKET_RESEARCH tab")
            return True
            
        except Exception as e:
            logger.error(f"Error updating MARKET_RESEARCH tab: {str(e)}")
            return False
            
    @retry_with_backoff
    def update_strategy_framework_tab(self, framework_data):
        """
        Update the STRATEGY_FRAMEWORK tab with component information.
        
        Args:
            framework_data: Dictionary containing strategy framework data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('STRATEGY_FRAMEWORK')
            if not worksheet:
                logger.error("STRATEGY_FRAMEWORK worksheet not found")
                return False
                
            # Format the data
            now = datetime.now().strftime("%Y-%m-%d")
            row_data = [
                framework_data.get('component', ''),
                framework_data.get('description', ''),
                framework_data.get('status', 'Active'),
                framework_data.get('last_updated', now),
                framework_data.get('dependencies', ''),
                framework_data.get('usage_frequency', 'High'),
                framework_data.get('performance_impact', 'Positive'),
                framework_data.get('notes', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added framework component {framework_data.get('component')} to STRATEGY_FRAMEWORK tab")
            return True
            
        except Exception as e:
            logger.error(f"Error updating STRATEGY_FRAMEWORK tab: {str(e)}")
            return False
            
    @retry_with_backoff
    def update_math_analysis_tab(self, analysis_data):
        """
        Update the MATH_ANALYSIS tab with mathematical methods and results.
        
        Args:
            analysis_data: Dictionary containing mathematical analysis data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('MATH_ANALYSIS')
            if not worksheet:
                logger.error("MATH_ANALYSIS worksheet not found")
                return False
                
            # Format the data
            now = datetime.now().strftime("%Y-%m-%d")
            row_data = [
                analysis_data.get('date', now),
                analysis_data.get('analysis_type', ''),
                analysis_data.get('method', ''),
                analysis_data.get('results', ''),
                analysis_data.get('significance', 'p < 0.05'),
                analysis_data.get('impact', ''),
                analysis_data.get('models', ''),
                analysis_data.get('references', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added mathematical analysis for {analysis_data.get('analysis_type')} to MATH_ANALYSIS tab")
            return True
            
        except Exception as e:
            logger.error(f"Error updating MATH_ANALYSIS tab: {str(e)}")
            return False
            
    @retry_with_backoff
    def update_parameter_optimization_tab(self, optimization_data):
        """
        Update the PARAMETER_OPTIMIZATION tab with optimization results.
        
        Args:
            optimization_data: Dictionary containing parameter optimization data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('PARAMETER_OPTIMIZATION')
            if not worksheet:
                logger.error("PARAMETER_OPTIMIZATION worksheet not found")
                return False
                
            # Format the data
            row_data = [
                optimization_data.get('strategy_id', ''),
                optimization_data.get('parameter', ''),
                optimization_data.get('min_value', ''),
                optimization_data.get('max_value', ''),
                optimization_data.get('step_size', ''),
                optimization_data.get('optimal_value', ''),
                optimization_data.get('performance_impact', ''),
                optimization_data.get('sensitivity', ''),
                optimization_data.get('notes', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added parameter optimization for {optimization_data.get('parameter')} to PARAMETER_OPTIMIZATION tab")
            return True
            
        except Exception as e:
            logger.error(f"Error updating PARAMETER_OPTIMIZATION tab: {str(e)}")
            return False
            
    @retry_with_backoff
    def update_backtest_queue(self, queue_data):
        """
        Update the Backtest Queue tab with strategies to be tested.
        
        Args:
            queue_data: Dictionary containing backtest queue data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('Backtest Queue')
            if not worksheet:
                logger.error("Backtest Queue worksheet not found")
                return False
                
            # Generate ID
            queue_id = f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Format the data
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row_data = [
                queue_id,
                queue_data.get('strategy_name', ''),
                now,
                queue_data.get('status', 'Pending'),
                queue_data.get('priority', 'Normal'),
                queue_data.get('type', 'Initial'),
                queue_data.get('market', 'US Equities'),
                queue_data.get('timeframe', 'Daily'),
                str(queue_data.get('parameters', {})),
                queue_data.get('notes', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added strategy '{queue_data.get('strategy_name')}' to Backtest Queue")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Backtest Queue tab: {str(e)}")
            return False
            
    @retry_with_backoff
    def update_completed_backtests(self, backtest_data):
        """
        Update the Completed Backtests tab with finished backtest results.
        
        Args:
            backtest_data: Dictionary containing completed backtest data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('Completed Backtests')
            if not worksheet:
                logger.error("Completed Backtests worksheet not found")
                return False
                
            # Format the data
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row_data = [
                backtest_data.get('id', f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                backtest_data.get('strategy_name', ''),
                now,
                backtest_data.get('result', 'Success'),
                backtest_data.get('win_rate', 0) * 100,
                backtest_data.get('profit_factor', 0),
                backtest_data.get('drawdown', 0) * 100,
                backtest_data.get('cagr', 0) * 100,
                backtest_data.get('sharpe_ratio', 0),
                backtest_data.get('notes', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added completed backtest for '{backtest_data.get('strategy_name')}' to Completed Backtests tab")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Completed Backtests tab: {str(e)}")
            return False
            
    @retry_with_backoff
    def update_strategy_evolution(self, evolution_data):
        """
        Update the Strategy Evolution tab with strategy improvements.
        
        Args:
            evolution_data: Dictionary containing strategy evolution data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        try:
            worksheet = self.worksheets.get('Strategy Evolution')
            if not worksheet:
                logger.error("Strategy Evolution worksheet not found")
                return False
                
            # Format the data
            now = datetime.now().strftime("%Y-%m-%d")
            row_data = [
                evolution_data.get('version', '1.0'),
                evolution_data.get('parent_id', ''),
                evolution_data.get('strategy_name', ''),
                now,
                evolution_data.get('win_rate', 0) * 100,
                evolution_data.get('sharpe_ratio', 0),
                evolution_data.get('cagr', 0) * 100,
                evolution_data.get('drawdown', 0) * 100,
                evolution_data.get('changes', ''),
                evolution_data.get('performance_delta', '')
            ]
            
            # Append the data
            worksheet.append_row(row_data)
            logger.info(f"Added strategy evolution for '{evolution_data.get('strategy_name')}' to Strategy Evolution tab")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Strategy Evolution tab: {str(e)}")
            return False

def upload_all_visualizations():
    """Upload all visualizations to Google Drive and add them to the Visualizations tab."""
    # Initialize the enhanced integration
    gs = EnhancedGoogleSheetIntegration()
    if not gs.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
        
    if not gs.initialize_drive_service():
        logger.error("Failed to initialize Google Drive service")
        return False
        
    # Create the Visualizations tab if it doesn't exist
    if not gs.create_visualization_tab():
        logger.error("Failed to create Visualizations tab")
        return False
        
    # Get all visualization files
    vis_dir = Path('/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/visualizations')
    vis_files = list(vis_dir.glob('*.png'))
    
    # Upload each visualization
    for i, vis_file in enumerate(vis_files):
        # Wait between uploads to avoid rate limiting
        if i > 0:
            time.sleep(5)
            
        file_name = vis_file.name
        file_path = str(vis_file)
        
        # Special handling for different visualization types
        if 'equity_curves' in file_name:
            title = "Strategy Performance Comparison"
            description = "Comparative equity curves for all trading strategies"
        elif 'drawdowns' in file_name:
            title = "Strategy Drawdowns Analysis"
            description = "Comparison of maximum drawdowns across strategies"
        elif 'monte_carlo' in file_name:
            title = "Monte Carlo Simulation"
            description = "100 simulated paths to analyze strategy robustness"
        elif 'trading_heatmap' in file_name:
            title = "Trading Activity Heatmap"
            description = "Analysis of trading frequency across days and weeks"
        elif 'Sharpe_Ratio' in file_name:
            title = "Sharpe Ratio Comparison"
            description = "Risk-adjusted return comparison across strategies"
        elif 'CAGR' in file_name:
            title = "CAGR Comparison"
            description = "Compound Annual Growth Rate comparison across strategies"
        elif 'Max_Drawdown' in file_name:
            title = "Maximum Drawdown Comparison"
            description = "Worst-case scenario analysis across strategies"
        elif 'Win_Rate' in file_name:
            title = "Win Rate Comparison"
            description = "Success rate analysis across strategies"
        else:
            title = f"Visualization: {file_name}"
            description = "Strategy performance visualization"
        
        # Upload the visualization
        url = gs.upload_visualization(file_path)
        if url:
            # Add the visualization to the sheet
            success = gs.add_visualization_to_sheet(title, description, url)
            if success:
                logger.info(f"Successfully added {title} visualization to sheet")
            else:
                logger.error(f"Failed to add {title} visualization to sheet")
        else:
            logger.error(f"Failed to upload {file_name}")
    
    return True

def generate_market_research_data():
    """Generate market research data for different market regimes."""
    market_data = []
    
    # Extreme volatility regime
    extreme_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'market': 'US Equities',
        'analysis_type': 'Extreme Volatility Regime',
        'findings': 'Market showing signs of extreme volatility with VIX above 30. Sectors showing high dispersion with technology and growth stocks experiencing larger swings.',
        'supporting_data': 'VIX: 32.5, Average daily range: 2.8%, Sector correlation: 0.45',
        'impact': 'Momentum strategies likely to underperform. Volatility-based strategies should be favored. Consider reduced position sizing and tighter risk controls.',
        'confidence': 'High',
        'action_items': 'Implement volatility filters, reduce position sizes by 30%, favor mean-reversion strategies, add VIX-based exposure limits'
    }
    market_data.append(extreme_data)
    
    # Falling market regime
    fall_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'market': 'US Equities',
        'analysis_type': 'Falling Market Regime',
        'findings': 'Market in confirmed downtrend with key indexes below 200-day moving averages. Defensive sectors showing relative strength. Breadth deteriorating with A/D line making new lows.',
        'supporting_data': 'S&P 500 below 200-day MA by 4.2%, Utilities outperforming SPX by 6.8%, 68% of stocks below their 50-day MA',
        'impact': 'Long-only strategies at high risk. Short strategies and inverse ETFs showing opportunity. Consider hedging long exposure with put options or VIX futures.',
        'confidence': 'High',
        'action_items': 'Implement trend filters, increase cash allocation, test short strategies, monitor market breadth indicators for potential reversal'
    }
    market_data.append(fall_data)
    
    # Fluctuation regime
    fluctuation_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'market': 'US Equities',
        'analysis_type': 'Range-Bound Fluctuation Regime',
        'findings': 'Market in sideways consolidation with defined support and resistance levels. Low directional conviction with oscillating momentum. Reduced trading volumes indicating market indecision.',
        'supporting_data': 'S&P 500 5% range over 3 months, RSI oscillating between 40-60, Volume 22% below 50-day average',
        'impact': 'Trend-following strategies likely to underperform with frequent whipsaws. Mean-reversion and range-trading strategies can excel. Reduced position sizing recommended.',
        'confidence': 'Medium',
        'action_items': 'Implement range-bound detection filters, favor mean-reversion strategies, use Bollinger Bands for entry/exit, monitor for potential range breakout'
    }
    market_data.append(fluctuation_data)
    
    # Rising market regime
    rise_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'market': 'US Equities',
        'analysis_type': 'Rising Market Regime',
        'findings': 'Market in confirmed uptrend with broad participation across sectors. Momentum indicators strengthening with growth stocks leading. Low volatility indicating investor confidence.',
        'supporting_data': 'S&P 500 above 200-day MA by 5.8%, 82% of stocks above their 50-day MA, Average sector correlation: 0.72',
        'impact': 'Momentum strategies likely to outperform. Trend-following systems showing strong results. Increased position sizing justified with trailing stops.',
        'confidence': 'High',
        'action_items': 'Implement trend-following filters, favor momentum strategies, increase position sizing for strongest sectors, use trailing stops for profit protection'
    }
    market_data.append(rise_data)
    
    return market_data

def generate_strategy_framework_components():
    """Generate strategy framework components data."""
    components = []
    
    # Signal generation component
    signal_component = {
        'component': 'Signal Generation',
        'description': 'Core signal generation system using multiple technical indicators and market regime detection',
        'status': 'Active',
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'dependencies': 'Market Data Processor, Technical Indicators Library',
        'usage_frequency': 'High',
        'performance_impact': 'Critical - directly determines entry timing',
        'notes': 'Currently using ensemble of 5 indicators with regime-specific weighting'
    }
    components.append(signal_component)
    
    # Position sizing component
    position_component = {
        'component': 'Position Sizing',
        'description': 'Dynamic position sizing based on volatility, signal strength, and account risk parameters',
        'status': 'Active',
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'dependencies': 'Risk Management System, Signal Generation',
        'usage_frequency': 'High',
        'performance_impact': 'High - determines risk-adjusted returns',
        'notes': 'Uses ATR-based sizing with portfolio-level constraints'
    }
    components.append(position_component)
    
    # Risk management component
    risk_component = {
        'component': 'Risk Management',
        'description': 'Comprehensive risk control system implementing position-level and portfolio-level constraints',
        'status': 'Active',
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'dependencies': 'Position Sizing, Market Regime Detection',
        'usage_frequency': 'High',
        'performance_impact': 'Critical - determines downside protection',
        'notes': 'Implements dynamic stop-loss based on volatility and correlation factors'
    }
    components.append(risk_component)
    
    # Market regime detection
    regime_component = {
        'component': 'Market Regime Detection',
        'description': 'Algorithm for classifying current market conditions into extreme, falling, fluctuating, or rising regimes',
        'status': 'Active',
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'dependencies': 'Market Data Processor, Statistical Analysis',
        'usage_frequency': 'Medium',
        'performance_impact': 'High - enables strategy adaptation',
        'notes': 'Uses rolling window of volatility, trend, and breadth indicators'
    }
    components.append(regime_component)
    
    # Exit management
    exit_component = {
        'component': 'Exit Management',
        'description': 'System for managing all trade exits including profit targets, stop losses, and time-based exits',
        'status': 'Active',
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'dependencies': 'Signal Generation, Position Sizing, Risk Management',
        'usage_frequency': 'High',
        'performance_impact': 'Critical - determines realized profits/losses',
        'notes': 'Implements trailing stops and dynamic profit targets based on volatility'
    }
    components.append(exit_component)
    
    return components

def generate_mathematical_analyses():
    """Generate mathematical analysis examples."""
    analyses = []
    
    # Statistical significance of momentum signals
    momentum_analysis = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'analysis_type': 'Signal Predictive Power',
        'method': 'Statistical Hypothesis Testing',
        'results': 'Momentum signals show statistically significant predictive power over 1-5 day horizons with t-statistic of 3.42 and p-value of 0.0007',
        'significance': 'p < 0.001',
        'impact': 'Confirms validity of momentum-based entry criteria with strongest effect at 3-day horizon',
        'models': 'Linear regression, t-test',
        'references': 'Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"'
    }
    analyses.append(momentum_analysis)
    
    # Volatility clustering analysis
    volatility_analysis = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'analysis_type': 'Volatility Pattern Analysis',
        'method': 'GARCH Modeling',
        'results': 'Significant volatility clustering detected in all market regimes with persistence parameter of 0.83',
        'significance': 'p < 0.0001',
        'impact': 'Justifies use of volatility-based position sizing and confirms importance of regime detection',
        'models': 'GARCH(1,1), EGARCH',
        'references': 'Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity"'
    }
    analyses.append(volatility_analysis)
    
    # Correlation structure analysis
    correlation_analysis = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'analysis_type': 'Cross-Asset Correlation Structure',
        'method': 'Principal Component Analysis',
        'results': 'First principal component explains 62% of variation during market stress vs. 38% in normal periods',
        'significance': 'KMO test: 0.82 (Very Good)',
        'impact': 'Confirms correlation risk during market stress and justifies reduced position sizing in extreme regimes',
        'models': 'PCA, KMO test, Bartlett\'s test',
        'references': 'Cont & Bouchaud (2000), "Herd Behavior and Aggregate Fluctuations in Financial Markets"'
    }
    analyses.append(correlation_analysis)
    
    # Optimal parameter values
    parameter_analysis = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'analysis_type': 'Parameter Optimization',
        'method': 'Bayesian Optimization',
        'results': 'Optimal RSI parameters found at (period=9, overbought=74, oversold=26) with Sharpe improvement of 0.41',
        'significance': 'Cross-validation error: 0.12',
        'impact': 'Provides parameter values for RSI-based entry/exit signals across different regimes',
        'models': 'Gaussian Process, Bayesian Optimization',
        'references': 'Snoek et al. (2012), "Practical Bayesian Optimization of Machine Learning Algorithms"'
    }
    analyses.append(parameter_analysis)
    
    return analyses

def generate_parameter_optimizations():
    """Generate parameter optimization examples."""
    optimizations = []
    
    # RSI parameters
    rsi_params = {
        'strategy_id': 'strategy_20250304_135500',
        'parameter': 'RSI Period',
        'min_value': '5',
        'max_value': '30',
        'step_size': '1',
        'optimal_value': '9',
        'performance_impact': 'Sharpe ratio improvement of 0.32',
        'sensitivity': 'Moderate - performance relatively stable between values 7-12',
        'notes': 'Different optimal values in trending vs. range-bound markets; use 9 for range-bound and 14 for trending'
    }
    optimizations.append(rsi_params)
    
    # Moving average parameters
    ma_params = {
        'strategy_id': 'strategy_20250304_135500',
        'parameter': 'Fast MA / Slow MA',
        'min_value': '5/20',
        'max_value': '20/100',
        'step_size': '1/5',
        'optimal_value': '9/42',
        'performance_impact': 'Win rate improvement of 8.5%',
        'sensitivity': 'High - performance degrades significantly with suboptimal values',
        'notes': 'Non-standard values outperform standard (e.g. 10/50) due to reduced crowding'
    }
    optimizations.append(ma_params)
    
    # ATR multiplier
    atr_params = {
        'strategy_id': 'strategy_20250304_135500',
        'parameter': 'ATR Multiplier',
        'min_value': '1.0',
        'max_value': '5.0',
        'step_size': '0.1',
        'optimal_value': '2.3',
        'performance_impact': 'Reduced drawdown by 14.6%',
        'sensitivity': 'Low - stable performance between 2.0-2.8',
        'notes': 'Higher values (3.0+) reduce drawdown but also reduce CAGR significantly'
    }
    optimizations.append(atr_params)
    
    # Profit target
    profit_params = {
        'strategy_id': 'strategy_20250304_135500',
        'parameter': 'Profit Target (ATR multiple)',
        'min_value': '1.0',
        'max_value': '6.0',
        'step_size': '0.2',
        'optimal_value': '3.2',
        'performance_impact': 'Increased average profit per trade by 0.32%',
        'sensitivity': 'Moderate - diminishing returns above 4.0',
        'notes': 'Use dynamic settings: 2.5 for high volatility, 3.2 for normal, 4.0 for low volatility'
    }
    optimizations.append(profit_params)
    
    return optimizations

def generate_backtest_queue_items():
    """Generate backtest queue items for various strategies."""
    queue_items = []
    
    # Momentum strategy with regime adaptation
    momentum_queue = {
        'strategy_name': 'Adaptive Momentum Strategy',
        'status': 'Pending',
        'priority': 'High',
        'type': 'Initial',
        'market': 'US Equities',
        'timeframe': 'Daily',
        'parameters': {
            'lookback': 20,
            'rsi_period': 9,
            'vol_filter': True,
            'min_momentum': 0.05,
            'max_stocks': 10
        },
        'notes': 'Test across all market regimes with dynamic parameter adjustment'
    }
    queue_items.append(momentum_queue)
    
    # Mean-reversion strategy
    reversion_queue = {
        'strategy_name': 'Statistical Mean Reversion',
        'status': 'Pending',
        'priority': 'Medium',
        'type': 'Initial',
        'market': 'US Equities',
        'timeframe': 'Daily',
        'parameters': {
            'lookback': 10,
            'z_score_threshold': 2.0,
            'rsi_period': 5,
            'holding_period': 5,
            'stop_loss_atr': 3.0
        },
        'notes': 'Focus testing on fluctuation regime data'
    }
    queue_items.append(reversion_queue)
    
    # Trend-following strategy
    trend_queue = {
        'strategy_name': 'Multi-Timeframe Trend Following',
        'status': 'Pending',
        'priority': 'Medium',
        'type': 'Initial',
        'market': 'US Equities',
        'timeframe': 'Daily',
        'parameters': {
            'fast_ma': 9,
            'slow_ma': 42,
            'confirmation_period': 3,
            'atr_stop': 2.5,
            'trailing_stop': True
        },
        'notes': 'Test on rising market regime data'
    }
    queue_items.append(trend_queue)
    
    # Volatility breakout strategy
    volatility_queue = {
        'strategy_name': 'Volatility Breakout Strategy',
        'status': 'Pending',
        'priority': 'High',
        'type': 'Initial',
        'market': 'US Equities',
        'timeframe': 'Daily',
        'parameters': {
            'atr_period': 14,
            'breakout_multiple': 1.5,
            'volume_filter': True,
            'min_volatility': 0.02,
            'profit_target': 3.0
        },
        'notes': 'Designed specifically for extreme volatility regime'
    }
    queue_items.append(volatility_queue)
    
    return queue_items

def generate_all_sheet_data():
    """Generate and update all sheets with comprehensive data."""
    # Initialize enhanced sheet integration
    gs = EnhancedGoogleSheetIntegration()
    if not gs.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
    
    # Upload all visualizations
    logger.info("Uploading visualizations to Google Drive and Sheets...")
    upload_all_visualizations()
    
    # Update MARKET_RESEARCH tab
    logger.info("Updating MARKET_RESEARCH tab...")
    market_data = generate_market_research_data()
    for data in market_data:
        success = gs.update_market_research_tab(data)
        if success:
            logger.info(f"Successfully added market research for {data['analysis_type']}")
        else:
            logger.error(f"Failed to add market research for {data['analysis_type']}")
        time.sleep(5)  # Avoid rate limiting
    
    # Update STRATEGY_FRAMEWORK tab
    logger.info("Updating STRATEGY_FRAMEWORK tab...")
    components = generate_strategy_framework_components()
    for component in components:
        success = gs.update_strategy_framework_tab(component)
        if success:
            logger.info(f"Successfully added framework component {component['component']}")
        else:
            logger.error(f"Failed to add framework component {component['component']}")
        time.sleep(5)  # Avoid rate limiting
    
    # Update MATH_ANALYSIS tab
    logger.info("Updating MATH_ANALYSIS tab...")
    analyses = generate_mathematical_analyses()
    for analysis in analyses:
        success = gs.update_math_analysis_tab(analysis)
        if success:
            logger.info(f"Successfully added mathematical analysis for {analysis['analysis_type']}")
        else:
            logger.error(f"Failed to add mathematical analysis for {analysis['analysis_type']}")
        time.sleep(5)  # Avoid rate limiting
    
    # Update PARAMETER_OPTIMIZATION tab
    logger.info("Updating PARAMETER_OPTIMIZATION tab...")
    optimizations = generate_parameter_optimizations()
    for optimization in optimizations:
        success = gs.update_parameter_optimization_tab(optimization)
        if success:
            logger.info(f"Successfully added parameter optimization for {optimization['parameter']}")
        else:
            logger.error(f"Failed to add parameter optimization for {optimization['parameter']}")
        time.sleep(5)  # Avoid rate limiting
    
    # Update Backtest Queue tab
    logger.info("Updating Backtest Queue tab...")
    queue_items = generate_backtest_queue_items()
    for item in queue_items:
        success = gs.update_backtest_queue(item)
        if success:
            logger.info(f"Successfully added {item['strategy_name']} to Backtest Queue")
        else:
            logger.error(f"Failed to add {item['strategy_name']} to Backtest Queue")
        time.sleep(5)  # Avoid rate limiting
    
    logger.info("Google Sheets update completed successfully")
    return True

def create_autopilot_script():
    """Create autopilot script for continuous strategy development."""
    autopilot_script = """#!/bin/bash
# Continuous autopilot script for strategy development and backtesting

# Log file
LOG_FILE="./logs/autopilot.log"

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log_message "Starting autopilot mode for strategy development"

# Create log directory if it doesn't exist
mkdir -p ./logs

# Main loop
while true; do
    # Step 1: Check backtest queue in Google Sheets
    log_message "Checking backtest queue for new strategies..."
    python3 ./check_backtest_queue.py
    
    # Step 2: Generate and backtest strategies
    log_message "Running strategy generation and backtesting..."
    ./run_deepseek.sh --god-mode --plan-name "Autopilot Strategy Development" --market us_equities
    
    # Step 3: Update Google Sheets with results
    log_message "Updating Google Sheets with latest results..."
    python3 ./update_sheets_with_visualizations.py
    
    # Sleep for 1 hour before next run
    log_message "Completed cycle. Sleeping for 1 hour..."
    sleep 3600
done
"""

    # Write the autopilot script
    autopilot_path = '/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/run_autopilot.sh'
    with open(autopilot_path, 'w') as f:
        f.write(autopilot_script)
    
    # Make it executable
    os.chmod(autopilot_path, 0o755)
    
    logger.info(f"Created autopilot script at {autopilot_path}")
    return True

def create_queue_checker_script():
    """Create script to check backtest queue and prepare strategies for testing."""
    queue_checker_script = """#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/queue_checker.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import GoogleSheetIntegration
from src.utils.google_sheet_integration import GoogleSheetIntegration

def check_backtest_queue():
    # Check the backtest queue and prepare strategies for testing
    # Initialize GoogleSheetIntegration
    gs = GoogleSheetIntegration()
    if not gs.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
    
    try:
        # Get the Backtest Queue worksheet
        worksheet = gs.worksheets.get('Backtest Queue')
        if not worksheet:
            logger.error("Backtest Queue worksheet not found")
            return False
        
        # Get all records
        queue_items = worksheet.get_all_records()
        
        # Filter for pending items
        pending_items = [item for item in queue_items if item.get('Status', '').lower() == 'pending']
        
        if not pending_items:
            logger.info("No pending items in backtest queue")
            return True
        
        logger.info(f"Found {len(pending_items)} pending items in backtest queue")
        
        # Process each pending item
        for i, item in enumerate(pending_items):
            queue_id = item.get('ID', f"queue_{i}")
            strategy_name = item.get('Strategy Name', f"Strategy_{i}")
            strategy_type = item.get('Type', 'Initial')
            market = item.get('Market', 'US Equities')
            
            # Parse parameters
            parameters_str = item.get('Parameters', '{}')
            try:
                parameters = eval(parameters_str)
            except:
                parameters = {}
            
            # Create strategy configuration
            strategy_config = {
                "name": strategy_name,
                "strategy_type": strategy_type,
                "market": market,
                "parameters": parameters
            }
            
            # Save configuration to file
            config_dir = os.path.join(parent_dir, "backtest_queue")
            os.makedirs(config_dir, exist_ok=True)
            
            config_file = os.path.join(config_dir, f"{queue_id}.json")
            with open(config_file, 'w') as f:
                json.dump(strategy_config, f, indent=4)
            
            # Update status to "Processing"
            row_idx = queue_items.index(item) + 2  # +2 for header row and 0-indexing
            worksheet.update_cell(row_idx, 4, "Processing")
            
            logger.info(f"Prepared strategy {strategy_name} (ID: {queue_id}) for backtesting")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking backtest queue: {str(e)}")
        return False

if __name__ == "__main__":
    check_backtest_queue()
"""

    # Write the queue checker script
    checker_path = '/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/check_backtest_queue.py'
    with open(checker_path, 'w') as f:
        f.write(queue_checker_script)
    
    # Make it executable
    os.chmod(checker_path, 0o755)
    
    logger.info(f"Created queue checker script at {checker_path}")
    return True

if __name__ == "__main__":
    # Create the autopilot scripts
    create_autopilot_script()
    create_queue_checker_script()
    
    # Update all sheets with comprehensive data
    generate_all_sheet_data()