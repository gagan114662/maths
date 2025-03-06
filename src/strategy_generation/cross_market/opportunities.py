"""
Cross-market opportunity detection module.

This module provides functionality to detect trading opportunities
between different markets and asset classes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from .correlations import CrossMarketCorrelationAnalyzer

logger = logging.getLogger(__name__)

class OpportunityDetector:
    """
    Detects trading opportunities across different markets and asset classes.
    
    This class analyzes relationships between different asset classes and
    markets to identify potential trading opportunities such as lead-lag,
    pairs trading, and other cross-market relationships.
    """
    
    def __init__(self, min_significance: float = 0.05, min_lookback: int = 252):
        """
        Initialize the opportunity detector.
        
        Args:
            min_significance: Minimum statistical significance level (max p-value)
            min_lookback: Minimum lookback period for opportunity detection
        """
        self.min_significance = min_significance
        self.min_lookback = min_lookback
        self.correlation_analyzer = None
        self.opportunity_types = [
            'lead_lag',
            'pairs_trading',
            'cross_asset_correlation',
            'risk_on_off',
            'rotation',
            'regime_specific'
        ]
        
    def set_correlation_analyzer(self, analyzer: CrossMarketCorrelationAnalyzer) -> None:
        """
        Set the correlation analyzer to use.
        
        Args:
            analyzer: CrossMarketCorrelationAnalyzer instance
        """
        self.correlation_analyzer = analyzer
        
    def find_lead_lag_opportunities(self, 
                                  min_correlation: float = 0.5,
                                  min_lag: int = 1,
                                  max_lag: int = 10,
                                  restrict_to_asset_classes: List[str] = None) -> List[Dict]:
        """
        Find lead-lag trading opportunities.
        
        Args:
            min_correlation: Minimum absolute correlation
            min_lag: Minimum lag (in days) to consider meaningful
            max_lag: Maximum lag (in days) to analyze
            restrict_to_asset_classes: Restrict analysis to these asset classes
            
        Returns:
            List of dictionaries with lead-lag opportunities
        """
        if not self.correlation_analyzer:
            logger.error("Correlation analyzer not set")
            return []
            
        # Get available symbols
        symbols = list(self.correlation_analyzer.asset_data.keys())
        
        # Filter by asset class if requested
        if restrict_to_asset_classes:
            filtered_symbols = []
            for symbol in symbols:
                asset_class = self.correlation_analyzer.asset_classes.get(symbol)
                if asset_class and asset_class in restrict_to_asset_classes:
                    filtered_symbols.append(symbol)
                    
            symbols = filtered_symbols
            
        # Find all possible pairs
        pairs = []
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:  # Avoid duplicates and self-pairs
                    continue
                    
                # Skip if same asset class and we want different ones
                if restrict_to_asset_classes:
                    class1 = self.correlation_analyzer.asset_classes.get(symbol1)
                    class2 = self.correlation_analyzer.asset_classes.get(symbol2)
                    
                    if class1 == class2:
                        continue
                        
                pairs.append((symbol1, symbol2))
                
        # Analyze lead-lag for each pair
        opportunities = []
        
        for symbol1, symbol2 in pairs:
            lead_lag = self.correlation_analyzer.analyze_lead_lag(
                symbol1, symbol2, max_lags=max_lag
            )
            
            if not lead_lag:
                continue
                
            # Check if there's a significant lead-lag relationship
            optimal_lag = lead_lag.get('optimal_lag', 0)
            significance = lead_lag.get('significance', False)
            correlation = lead_lag.get('max_correlation', 0)
            
            if (abs(optimal_lag) >= min_lag and 
                significance and 
                abs(correlation) >= min_correlation):
                
                # Add to opportunities
                opportunities.append({
                    'type': 'lead_lag',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'lead_asset': lead_lag.get('lead_asset'),
                    'lag_asset': lead_lag.get('lag_asset'),
                    'optimal_lag': optimal_lag,
                    'correlation': correlation,
                    'description': lead_lag.get('relationship'),
                    'asset_class1': self.correlation_analyzer.asset_classes.get(symbol1, 'unknown'),
                    'asset_class2': self.correlation_analyzer.asset_classes.get(symbol2, 'unknown'),
                    'p_value': lead_lag.get('lag_correlations')[max_lag]['p_value'] if 'lag_correlations' in lead_lag else None,
                    'confidence': 'high' if abs(correlation) > 0.7 else 'medium'
                })
                
        # Sort by absolute correlation
        opportunities.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return opportunities
    
    def find_pairs_trading_opportunities(self,
                                       min_correlation: float = 0.7,
                                       min_half_life: int = 5,
                                       max_half_life: int = 100,
                                       restrict_to_asset_classes: List[str] = None) -> List[Dict]:
        """
        Find pairs trading opportunities.
        
        Args:
            min_correlation: Minimum correlation between pairs
            min_half_life: Minimum half-life of mean reversion (in days)
            max_half_life: Maximum half-life of mean reversion (in days)
            restrict_to_asset_classes: Restrict analysis to these asset classes
            
        Returns:
            List of dictionaries with pairs trading opportunities
        """
        if not self.correlation_analyzer:
            logger.error("Correlation analyzer not set")
            return []
            
        # Get available symbols
        symbols = list(self.correlation_analyzer.asset_data.keys())
        
        # Filter by asset class if requested
        if restrict_to_asset_classes:
            filtered_symbols = []
            for symbol in symbols:
                asset_class = self.correlation_analyzer.asset_classes.get(symbol)
                if asset_class and asset_class in restrict_to_asset_classes:
                    filtered_symbols.append(symbol)
                    
            symbols = filtered_symbols
            
        # Find correlated pairs
        correlated_pairs = self.correlation_analyzer.find_correlated_pairs(
            symbols=symbols,
            min_correlation=min_correlation
        )
        
        # Check each pair for cointegration and mean reversion
        opportunities = []
        
        for pair in correlated_pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            # Check cointegration
            coint_results = self.correlation_analyzer.check_cointegration(symbol1, symbol2)
            
            if not coint_results:
                continue
                
            # Check if pair is cointegrated with suitable half-life
            is_cointegrated = coint_results.get('is_cointegrated', False)
            half_life = coint_results.get('half_life', float('inf'))
            
            if (is_cointegrated and 
                not np.isnan(half_life) and 
                half_life >= min_half_life and 
                half_life <= max_half_life):
                
                # Add to opportunities
                opportunities.append({
                    'type': 'pairs_trading',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation': pair['correlation'],
                    'hedge_ratio': coint_results.get('hedge_ratio', 1.0),
                    'half_life': half_life,
                    'p_value': coint_results.get('p_value'),
                    'spread_std': coint_results.get('spread_std'),
                    'asset_class1': self.correlation_analyzer.asset_classes.get(symbol1, 'unknown'),
                    'asset_class2': self.correlation_analyzer.asset_classes.get(symbol2, 'unknown'),
                    'description': f"Pairs trading: {symbol1} vs {symbol2}, Half-life: {half_life:.1f} days",
                    'confidence': 'high' if half_life < 30 else 'medium'
                })
                
        # Sort by half-life (lower is better)
        opportunities.sort(key=lambda x: x['half_life'])
        
        return opportunities
    
    def find_cross_asset_correlations(self,
                                    min_correlation: float = 0.5,
                                    require_diff_asset_class: bool = True,
                                    lookback_days: int = 252) -> List[Dict]:
        """
        Find significant correlations between different asset classes.
        
        Args:
            min_correlation: Minimum absolute correlation
            require_diff_asset_class: Whether to require different asset classes
            lookback_days: Lookback period for correlation analysis
            
        Returns:
            List of dictionaries with cross-asset correlation opportunities
        """
        if not self.correlation_analyzer:
            logger.error("Correlation analyzer not set")
            return []
            
        # Find correlated pairs
        correlated_pairs = self.correlation_analyzer.find_correlated_pairs(
            min_correlation=min_correlation,
            require_diff_asset_class=require_diff_asset_class
        )
        
        # Compute rolling correlations for each pair
        opportunities = []
        
        for pair in correlated_pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            # Compute rolling correlation (60-day window)
            rolling_corr = self.correlation_analyzer.compute_rolling_correlations(
                symbol1, symbol2, window=60
            )
            
            if rolling_corr.empty:
                continue
                
            # Limit to lookback period
            if len(rolling_corr) > lookback_days:
                recent_corr = rolling_corr[-lookback_days:]
            else:
                recent_corr = rolling_corr
                
            # Check if correlation is stable or trending
            if len(recent_corr) < 30:
                continue
                
            # Calculate correlation stability
            corr_std = recent_corr.std()
            
            # Calculate correlation trend
            from scipy import stats
            x = np.arange(len(recent_corr))
            y = recent_corr.values
            slope, _, r_value, p_value, _ = stats.linregress(x, y)
            
            # Add to opportunities
            opportunities.append({
                'type': 'cross_asset_correlation',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': pair['correlation'],
                'correlation_stability': corr_std,
                'correlation_trend': slope * len(recent_corr),  # Scale to full period
                'trend_significance': p_value < 0.05,
                'asset_class1': pair['asset_class1'],
                'asset_class2': pair['asset_class2'],
                'description': f"Cross-asset correlation: {symbol1} ({pair['asset_class1']}) vs {symbol2} ({pair['asset_class2']})",
                'confidence': 'high' if abs(pair['correlation']) > 0.7 and corr_std < 0.2 else 'medium'
            })
            
        # Sort by absolute correlation
        opportunities.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return opportunities
    
    def find_risk_on_off_indicators(self,
                                  reference_symbols: List[str] = None,
                                  min_correlation: float = 0.7,
                                  lookback_days: int = 252) -> List[Dict]:
        """
        Find indicators of risk-on/risk-off market regimes.
        
        Args:
            reference_symbols: Reference symbols representing risk assets
            min_correlation: Minimum absolute correlation
            lookback_days: Lookback period for analysis
            
        Returns:
            List of dictionaries with risk-on/off indicator opportunities
        """
        if not self.correlation_analyzer:
            logger.error("Correlation analyzer not set")
            return []
            
        # Use default reference symbols if none provided
        if not reference_symbols:
            # Try to identify risk assets (equities, high-yield bonds, etc.)
            reference_symbols = []
            
            for symbol, asset_class in self.correlation_analyzer.asset_classes.items():
                if asset_class in ['equity', 'equity_index', 'high_yield_bond']:
                    reference_symbols.append(symbol)
                    
            # If still no reference symbols, use all
            if not reference_symbols:
                reference_symbols = list(self.correlation_analyzer.asset_data.keys())
                
        # For each reference symbol, find correlated assets
        risk_indicators = []
        
        for ref_symbol in reference_symbols:
            # Skip if no data
            if ref_symbol not in self.correlation_analyzer.asset_data:
                continue
                
            # Compute correlations with all other assets
            for symbol in self.correlation_analyzer.asset_data.keys():
                if symbol == ref_symbol:
                    continue
                    
                # Compute rolling correlation (60-day window)
                rolling_corr = self.correlation_analyzer.compute_rolling_correlations(
                    ref_symbol, symbol, window=60
                )
                
                if rolling_corr.empty:
                    continue
                    
                # Limit to lookback period
                if len(rolling_corr) > lookback_days:
                    recent_corr = rolling_corr[-lookback_days:]
                else:
                    recent_corr = rolling_corr
                    
                # Calculate average correlation
                avg_corr = recent_corr.mean()
                
                # Skip if correlation is too low
                if abs(avg_corr) < min_correlation:
                    continue
                    
                # Determine risk relationship
                if avg_corr > 0:
                    risk_relationship = 'aligned'  # Moves with risk asset
                else:
                    risk_relationship = 'inverse'  # Moves against risk asset
                    
                # Add to indicators
                risk_indicators.append({
                    'type': 'risk_on_off',
                    'reference_symbol': ref_symbol,
                    'indicator_symbol': symbol,
                    'correlation': avg_corr,
                    'risk_relationship': risk_relationship,
                    'reference_asset_class': self.correlation_analyzer.asset_classes.get(ref_symbol, 'unknown'),
                    'indicator_asset_class': self.correlation_analyzer.asset_classes.get(symbol, 'unknown'),
                    'description': f"Risk-{'on' if risk_relationship == 'aligned' else 'off'} indicator: {symbol} vs {ref_symbol}",
                    'confidence': 'high' if abs(avg_corr) > 0.8 else 'medium'
                })
                
        # Sort by absolute correlation
        risk_indicators.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return risk_indicators
    
    def find_all_opportunities(self, min_confidence: str = 'medium') -> Dict[str, List[Dict]]:
        """
        Find all trading opportunities across different categories.
        
        Args:
            min_confidence: Minimum confidence level ('high' or 'medium')
            
        Returns:
            Dictionary mapping opportunity types to lists of opportunities
        """
        all_opportunities = {}
        
        # Find lead-lag opportunities
        lead_lag = self.find_lead_lag_opportunities()
        if lead_lag:
            all_opportunities['lead_lag'] = [
                opp for opp in lead_lag 
                if opp.get('confidence', 'low') >= min_confidence
            ]
            
        # Find pairs trading opportunities
        pairs_trading = self.find_pairs_trading_opportunities()
        if pairs_trading:
            all_opportunities['pairs_trading'] = [
                opp for opp in pairs_trading 
                if opp.get('confidence', 'low') >= min_confidence
            ]
            
        # Find cross-asset correlations
        cross_asset = self.find_cross_asset_correlations()
        if cross_asset:
            all_opportunities['cross_asset_correlation'] = [
                opp for opp in cross_asset 
                if opp.get('confidence', 'low') >= min_confidence
            ]
            
        # Find risk-on/off indicators
        risk_indicators = self.find_risk_on_off_indicators()
        if risk_indicators:
            all_opportunities['risk_on_off'] = [
                opp for opp in risk_indicators 
                if opp.get('confidence', 'low') >= min_confidence
            ]
            
        return all_opportunities
    
    def generate_opportunity_report(self, 
                                  opportunities: Dict[str, List[Dict]] = None,
                                  top_n: int = 5) -> Dict:
        """
        Generate a comprehensive report of detected opportunities.
        
        Args:
            opportunities: Dictionary of opportunities (from find_all_opportunities)
            top_n: Number of top opportunities to include for each type
            
        Returns:
            Dictionary with opportunity report
        """
        if not opportunities:
            opportunities = self.find_all_opportunities()
            
        if not opportunities:
            logger.error("No opportunities found")
            return {}
            
        # Create report
        report = {
            'summary': {
                'total_opportunities': sum(len(opps) for opps in opportunities.values()),
                'opportunity_types': list(opportunities.keys()),
                'top_opportunities': [],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'opportunities': opportunities
        }
        
        # Add top opportunities across all types
        all_opps = []
        for opp_type, opps in opportunities.items():
            for opp in opps:
                opp_copy = opp.copy()
                opp_copy['opportunity_type'] = opp_type
                all_opps.append(opp_copy)
                
        # Sort by confidence and correlation/other metrics
        def opp_sort_key(opp):
            confidence_score = 2 if opp.get('confidence') == 'high' else 1
            corr_score = abs(opp.get('correlation', 0))
            return (confidence_score, corr_score)
            
        all_opps.sort(key=opp_sort_key, reverse=True)
        
        # Add top N to report
        report['summary']['top_opportunities'] = all_opps[:top_n]
        
        # Add statistics by asset class
        asset_class_stats = {}
        
        for opp in all_opps:
            asset_class1 = opp.get('asset_class1', 'unknown')
            asset_class2 = opp.get('asset_class2', 'unknown')
            
            for asset_class in [asset_class1, asset_class2]:
                if asset_class not in asset_class_stats:
                    asset_class_stats[asset_class] = {
                        'total_opportunities': 0,
                        'by_type': {}
                    }
                    
                asset_class_stats[asset_class]['total_opportunities'] += 1
                
                opp_type = opp.get('opportunity_type', 'unknown')
                if opp_type not in asset_class_stats[asset_class]['by_type']:
                    asset_class_stats[asset_class]['by_type'][opp_type] = 0
                    
                asset_class_stats[asset_class]['by_type'][opp_type] += 1
                
        report['asset_class_stats'] = asset_class_stats
        
        return report
    
    def generate_trading_ideas(self, 
                             opportunities: Dict[str, List[Dict]] = None,
                             min_confidence: str = 'medium',
                             max_ideas: int = 10) -> List[Dict]:
        """
        Generate actionable trading ideas from opportunities.
        
        Args:
            opportunities: Dictionary of opportunities (from find_all_opportunities)
            min_confidence: Minimum confidence level ('high' or 'medium')
            max_ideas: Maximum number of ideas to generate
            
        Returns:
            List of dictionaries with trading ideas
        """
        if not opportunities:
            opportunities = self.find_all_opportunities(min_confidence=min_confidence)
            
        if not opportunities:
            logger.error("No opportunities found")
            return []
            
        # Collect all opportunities
        all_opps = []
        for opp_type, opps in opportunities.items():
            for opp in opps:
                if opp.get('confidence', 'low') >= min_confidence:
                    opp_copy = opp.copy()
                    opp_copy['opportunity_type'] = opp_type
                    all_opps.append(opp_copy)
                    
        # Sort by confidence and correlation/other metrics
        def opp_sort_key(opp):
            confidence_score = 2 if opp.get('confidence') == 'high' else 1
            corr_score = abs(opp.get('correlation', 0))
            return (confidence_score, corr_score)
            
        all_opps.sort(key=opp_sort_key, reverse=True)
        
        # Generate trading ideas
        trading_ideas = []
        
        for opp in all_opps[:max_ideas]:
            opp_type = opp.get('opportunity_type')
            
            if opp_type == 'lead_lag':
                # Create lead-lag trading idea
                lead_asset = opp.get('lead_asset')
                lag_asset = opp.get('lag_asset')
                optimal_lag = opp.get('optimal_lag')
                correlation = opp.get('correlation')
                
                if not lead_asset or not lag_asset:
                    continue
                    
                # Determine entry/exit logic based on correlation sign
                if correlation > 0:
                    entry_logic = f"Enter LONG {lag_asset} when {lead_asset} rises over past {optimal_lag} days"
                    exit_logic = f"Exit when {lead_asset} falls or after holding for {optimal_lag*2} days"
                else:
                    entry_logic = f"Enter SHORT {lag_asset} when {lead_asset} rises over past {optimal_lag} days"
                    exit_logic = f"Exit when {lead_asset} falls or after holding for {optimal_lag*2} days"
                    
                # Create idea
                idea = {
                    'type': 'lead_lag',
                    'title': f"{lead_asset} Leads {lag_asset} by {optimal_lag} Days",
                    'assets': [lead_asset, lag_asset],
                    'entry_logic': entry_logic,
                    'exit_logic': exit_logic,
                    'rationale': f"{lead_asset} tends to lead {lag_asset} by {optimal_lag} days with {abs(correlation):.2f} correlation",
                    'confidence': opp.get('confidence', 'medium'),
                    'timeframe': f"{optimal_lag} - {optimal_lag*2} days",
                    'position_type': 'long' if correlation > 0 else 'short'
                }
                
                trading_ideas.append(idea)
                
            elif opp_type == 'pairs_trading':
                # Create pairs trading idea
                symbol1 = opp.get('symbol1')
                symbol2 = opp.get('symbol2')
                hedge_ratio = opp.get('hedge_ratio', 1.0)
                half_life = opp.get('half_life')
                
                if not symbol1 or not symbol2 or not half_life:
                    continue
                    
                # Create idea
                idea = {
                    'type': 'pairs_trading',
                    'title': f"Mean-Reverting Pair: {symbol1}/{symbol2}",
                    'assets': [symbol1, symbol2],
                    'entry_logic': f"When spread (={symbol1} - {hedge_ratio:.2f}*{symbol2}) exceeds 2 standard deviations from mean",
                    'exit_logic': f"When spread reverts to mean or after {int(half_life*3)} days",
                    'rationale': f"Cointegrated pair with {half_life:.1f}-day half-life for mean reversion",
                    'confidence': opp.get('confidence', 'medium'),
                    'timeframe': f"{int(half_life)} - {int(half_life*3)} days",
                    'position_type': 'long/short pair'
                }
                
                trading_ideas.append(idea)
                
            elif opp_type == 'cross_asset_correlation':
                # Create cross-asset correlation idea
                symbol1 = opp.get('symbol1')
                symbol2 = opp.get('symbol2')
                correlation = opp.get('correlation')
                asset_class1 = opp.get('asset_class1')
                asset_class2 = opp.get('asset_class2')
                
                if not symbol1 or not symbol2 or not correlation:
                    continue
                    
                # Determine trading logic based on correlation
                if correlation > 0:
                    entry_logic = f"LONG {symbol1} and LONG {symbol2} during trending markets"
                    exit_logic = f"Exit when correlation breaks down or trend reverses"
                else:
                    entry_logic = f"LONG {symbol1} and SHORT {symbol2} to create market-neutral position"
                    exit_logic = f"Exit when correlation breaks down"
                    
                # Create idea
                idea = {
                    'type': 'cross_asset_correlation',
                    'title': f"Cross-Asset Relationship: {symbol1}/{symbol2}",
                    'assets': [symbol1, symbol2],
                    'entry_logic': entry_logic,
                    'exit_logic': exit_logic,
                    'rationale': f"Strong {'positive' if correlation > 0 else 'negative'} correlation ({correlation:.2f}) between {asset_class1} and {asset_class2}",
                    'confidence': opp.get('confidence', 'medium'),
                    'timeframe': "Medium-term (1-3 months)",
                    'position_type': 'long/long' if correlation > 0 else 'long/short'
                }
                
                trading_ideas.append(idea)
                
            elif opp_type == 'risk_on_off':
                # Create risk-on/off idea
                ref_symbol = opp.get('reference_symbol')
                indicator_symbol = opp.get('indicator_symbol')
                risk_relationship = opp.get('risk_relationship')
                
                if not ref_symbol or not indicator_symbol:
                    continue
                    
                # Determine trading logic based on risk relationship
                if risk_relationship == 'aligned':
                    entry_logic = f"LONG {ref_symbol} when {indicator_symbol} shows strong uptrend"
                    exit_logic = f"Exit when {indicator_symbol} trend reverses"
                else:
                    entry_logic = f"SHORT {ref_symbol} when {indicator_symbol} shows strong uptrend"
                    exit_logic = f"Exit when {indicator_symbol} trend reverses"
                    
                # Create idea
                idea = {
                    'type': 'risk_on_off',
                    'title': f"Risk {'On' if risk_relationship == 'aligned' else 'Off'} Signal: {indicator_symbol} for {ref_symbol}",
                    'assets': [ref_symbol, indicator_symbol],
                    'entry_logic': entry_logic,
                    'exit_logic': exit_logic,
                    'rationale': f"{indicator_symbol} serves as a {'positive' if risk_relationship == 'aligned' else 'negative'} risk sentiment indicator for {ref_symbol}",
                    'confidence': opp.get('confidence', 'medium'),
                    'timeframe': "Medium-term (1-3 months)",
                    'position_type': 'long' if risk_relationship == 'aligned' else 'short'
                }
                
                trading_ideas.append(idea)
                
        return trading_ideas