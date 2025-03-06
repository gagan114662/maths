#!/usr/bin/env python
"""
Options-Based Hedging Strategy Module

This module implements sophisticated options-based hedging strategies for portfolio
protection and risk management, including dynamic delta hedging, tail risk hedging,
and cost-optimized protection strategies.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

class OptionsHedgeStrategy:
    """Implements sophisticated options-based hedging strategies."""
    
    def __init__(self,
                max_hedge_cost: float = 0.02,  # 2% max annual cost
                min_protection_level: float = 0.95,  # 95% protection
                rebalance_threshold: float = 0.1,  # 10% delta deviation
                vol_window: int = 63):
        """
        Initialize the options hedging strategy.

        Args:
            max_hedge_cost: Maximum annual cost of hedging as fraction of portfolio
            min_protection_level: Minimum downside protection level
            rebalance_threshold: Delta deviation threshold for rebalancing
            vol_window: Window for volatility calculation
        """
        self.max_hedge_cost = max_hedge_cost
        self.min_protection_level = min_protection_level
        self.rebalance_threshold = rebalance_threshold
        self.vol_window = vol_window
        self.logger = logging.getLogger(__name__)
        
    def design_hedge_strategy(self, 
                            portfolio_value: float,
                            current_positions: Dict[str, float],
                            market_data: pd.DataFrame,
                            options_data: Dict[str, pd.DataFrame],
                            risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Design optimal hedging strategy using options.
        
        Args:
            portfolio_value: Current portfolio value
            current_positions: Dictionary of current positions
            market_data: Market data for underlyings
            options_data: Options chain data for available hedging instruments
            risk_metrics: Dictionary of portfolio risk metrics
            
        Returns:
            Dictionary containing hedge strategy details
        """
        # Calculate portfolio sensitivities
        portfolio_delta = self._calculate_portfolio_delta(
            current_positions, market_data
        )
        portfolio_beta = risk_metrics.get('portfolio_beta', 1.0)
        
        # Estimate tail risk
        tail_risk = risk_metrics.get('var_99', 0.0)
        
        # Design protective put strategy
        put_strategy = self._design_protective_puts(
            portfolio_value,
            portfolio_delta,
            tail_risk,
            options_data
        )
        
        # Design collar strategy if cost reduction needed
        if put_strategy['annual_cost'] > self.max_hedge_cost * portfolio_value:
            collar_strategy = self._design_collar(
                portfolio_value,
                portfolio_delta,
                put_strategy,
                options_data
            )
            
            # Choose better strategy based on cost-benefit
            if collar_strategy['cost_benefit_ratio'] > put_strategy['cost_benefit_ratio']:
                hedge_strategy = collar_strategy
            else:
                hedge_strategy = put_strategy
        else:
            hedge_strategy = put_strategy
            
        return hedge_strategy
    
    def calculate_dynamic_hedge_adjustments(self,
                                         current_hedges: Dict[str, Dict],
                                         market_data: pd.DataFrame,
                                         options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate required adjustments to maintain hedge effectiveness.
        
        Args:
            current_hedges: Current hedge positions
            market_data: Current market data
            options_data: Current options data
            
        Returns:
            Dictionary of required hedge adjustments
        """
        adjustments = {}
        
        for instrument, hedge in current_hedges.items():
            # Calculate current delta
            current_delta = self._calculate_position_delta(
                hedge, market_data, options_data
            )
            
            # Calculate target delta
            target_delta = hedge['target_delta']
            
            # Check if adjustment needed
            delta_deviation = abs(current_delta - target_delta)
            if delta_deviation > self.rebalance_threshold:
                adjustments[instrument] = {
                    'current_delta': current_delta,
                    'target_delta': target_delta,
                    'required_adjustment': target_delta - current_delta,
                    'suggested_trades': self._suggest_hedge_trades(
                        instrument, target_delta - current_delta, options_data
                    )
                }
                
        return adjustments
    
    def _design_protective_puts(self,
                             portfolio_value: float,
                             portfolio_delta: float,
                             tail_risk: float,
                             options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Design protective put strategy."""
        strategy = {
            'type': 'protective_put',
            'positions': [],
            'annual_cost': 0.0,
            'protection_level': 0.0,
            'cost_benefit_ratio': 0.0
        }
        
        # Find optimal strike price
        for underlying, options in options_data.items():
            puts = options[options['type'] == 'put']
            puts = puts.sort_values('strike')
            
            # Find puts that provide required protection
            viable_puts = puts[
                puts['strike'] >= portfolio_value * self.min_protection_level
            ]
            
            if viable_puts.empty:
                continue
                
            # Calculate cost and benefit metrics
            for _, put in viable_puts.iterrows():
                cost = put['price'] * portfolio_delta * 100  # Cost per contract
                protection_level = put['strike'] / portfolio_value
                
                # Calculate cost-benefit ratio
                cost_benefit = (1 - protection_level) / (cost / portfolio_value)
                
                strategy['positions'].append({
                    'type': 'put',
                    'underlying': underlying,
                    'strike': put['strike'],
                    'expiration': put['expiration'],
                    'contracts': int(portfolio_delta * 100),
                    'cost': cost,
                    'protection_level': protection_level
                })
                
                strategy['annual_cost'] += cost
                strategy['protection_level'] = max(
                    strategy['protection_level'],
                    protection_level
                )
                strategy['cost_benefit_ratio'] = cost_benefit
                
        return strategy
    
    def _design_collar(self,
                     portfolio_value: float,
                     portfolio_delta: float,
                     put_strategy: Dict[str, Any],
                     options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Design collar strategy (protective put + covered call)."""
        strategy = {
            'type': 'collar',
            'positions': put_strategy['positions'].copy(),
            'annual_cost': put_strategy['annual_cost'],
            'protection_level': put_strategy['protection_level'],
            'cost_benefit_ratio': put_strategy['cost_benefit_ratio'],
            'upside_cap': float('inf')
        }
        
        # Find optimal covered calls to reduce cost
        for underlying, options in options_data.items():
            calls = options[options['type'] == 'call']
            calls = calls.sort_values('strike')
            
            # Find calls that reduce cost while maintaining acceptable upside
            target_premium = strategy['annual_cost'] - (
                self.max_hedge_cost * portfolio_value
            )
            
            viable_calls = calls[
                calls['price'] * portfolio_delta * 100 >= target_premium
            ]
            
            if viable_calls.empty:
                continue
                
            # Select call with highest strike that meets premium target
            selected_call = viable_calls.iloc[-1]
            
            # Add call position to strategy
            strategy['positions'].append({
                'type': 'call',
                'underlying': underlying,
                'strike': selected_call['strike'],
                'expiration': selected_call['expiration'],
                'contracts': -int(portfolio_delta * 100),  # Short call
                'premium': selected_call['price'] * portfolio_delta * 100
            })
            
            # Update strategy metrics
            strategy['annual_cost'] -= selected_call['price'] * portfolio_delta * 100
            strategy['upside_cap'] = min(
                strategy['upside_cap'],
                selected_call['strike'] / portfolio_value
            )
            
            # Update cost-benefit ratio
            protection_range = strategy['upside_cap'] - strategy['protection_level']
            strategy['cost_benefit_ratio'] = protection_range / (
                strategy['annual_cost'] / portfolio_value
            )
            
        return strategy
    
    def _calculate_portfolio_delta(self,
                                current_positions: Dict[str, float],
                                market_data: pd.DataFrame) -> float:
        """Calculate total portfolio delta."""
        portfolio_delta = 0.0
        
        for asset, position in current_positions.items():
            if asset in market_data.columns:
                price = market_data[asset].iloc[-1]
                portfolio_delta += position * price
                
        return portfolio_delta
    
    def _calculate_position_delta(self,
                               position: Dict[str, Any],
                               market_data: pd.DataFrame,
                               options_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate delta for a single position."""
        if position['type'] in ['put', 'call']:
            # Get current price and volatility
            underlying = position['underlying']
            current_price = market_data[underlying].iloc[-1]
            volatility = market_data[underlying].pct_change().std() * np.sqrt(252)
            
            # Calculate time to expiration
            dte = (position['expiration'] - datetime.now()).days / 365
            
            # Calculate option delta
            delta = self._calculate_option_delta(
                position['type'],
                current_price,
                position['strike'],
                dte,
                volatility
            )
            
            return delta * position['contracts']
        else:
            return 1.0  # Stock position
    
    def _calculate_option_delta(self,
                             option_type: str,
                             spot: float,
                             strike: float,
                             time: float,
                             volatility: float,
                             rate: float = 0.02) -> float:
        """Calculate option delta using Black-Scholes."""
        d1 = (np.log(spot/strike) + (rate + volatility**2/2)*time) / (volatility*np.sqrt(time))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    def _suggest_hedge_trades(self,
                           instrument: str,
                           delta_adjustment: float,
                           options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Suggest specific trades to adjust hedge position."""
        suggestions = []
        
        if instrument in options_data:
            options = options_data[instrument]
            
            # Find options closest to target delta
            target_delta = abs(delta_adjustment)
            options['delta_distance'] = abs(abs(options['delta']) - target_delta)
            best_matches = options.nsmallest(3, 'delta_distance')
            
            for _, option in best_matches.iterrows():
                suggestions.append({
                    'instrument': instrument,
                    'type': option['type'],
                    'strike': option['strike'],
                    'expiration': option['expiration'],
                    'contracts': int(delta_adjustment / option['delta']),
                    'estimated_cost': option['price'] * abs(delta_adjustment)
                })
                
        return suggestions

def calculate_hedge_effectiveness(hedge_positions: Dict[str, Dict],
                               market_data: pd.DataFrame,
                               risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate effectiveness metrics for current hedges.
    
    Args:
        hedge_positions: Current hedge positions
        market_data: Market data history
        risk_free_rate: Risk-free rate for calculations
        
    Returns:
        Dictionary of hedge effectiveness metrics
    """
    # Calculate hedged vs unhedged performance
    unhedged_returns = market_data.pct_change().dropna()
    hedged_returns = unhedged_returns.copy()
    
    # Add hedge position returns
    for _, hedge in hedge_positions.items():
        if hedge['type'] in ['put', 'call']:
            # Calculate option returns
            option_returns = _calculate_option_returns(
                hedge, market_data, risk_free_rate
            )
            hedged_returns += option_returns
            
    # Calculate effectiveness metrics
    metrics = {
        'beta_reduction': _calculate_beta_reduction(
            unhedged_returns, hedged_returns
        ),
        'volatility_reduction': _calculate_volatility_reduction(
            unhedged_returns, hedged_returns
        ),
        'tail_risk_reduction': _calculate_tail_risk_reduction(
            unhedged_returns, hedged_returns
        ),
        'hedge_cost': _calculate_total_hedge_cost(
            hedge_positions, market_data
        )
    }
    
    return metrics

def _calculate_option_returns(hedge: Dict[str, Any],
                           market_data: pd.DataFrame,
                           risk_free_rate: float) -> pd.Series:
    """Calculate historical returns for option position."""
    spot_price = market_data[hedge['underlying']]
    returns = pd.Series(0, index=market_data.index)
    
    # Simplified option return calculation
    if hedge['type'] == 'put':
        returns = np.maximum(hedge['strike'] - spot_price, 0) - hedge['cost']
    else:  # call
        returns = np.maximum(spot_price - hedge['strike'], 0) - hedge['cost']
        
    return returns * hedge['contracts'] / spot_price

def _calculate_beta_reduction(unhedged_returns: pd.Series,
                           hedged_returns: pd.Series) -> float:
    """Calculate reduction in beta from hedging."""
    market_returns = unhedged_returns  # Assuming market returns are the same as unhedged returns
    unhedged_beta = unhedged_returns.cov(market_returns) / market_returns.var()
    hedged_beta = hedged_returns.cov(market_returns) / market_returns.var()
    
    return (unhedged_beta - hedged_beta) / unhedged_beta

def _calculate_volatility_reduction(unhedged_returns: pd.Series,
                                 hedged_returns: pd.Series) -> float:
    """Calculate reduction in volatility from hedging."""
    unhedged_vol = unhedged_returns.std() * np.sqrt(252)
    hedged_vol = hedged_returns.std() * np.sqrt(252)
    
    return (unhedged_vol - hedged_vol) / unhedged_vol

def _calculate_tail_risk_reduction(unhedged_returns: pd.Series,
                                hedged_returns: pd.Series) -> float:
    """Calculate reduction in tail risk from hedging."""
    unhedged_var = np.percentile(unhedged_returns, 1)
    hedged_var = np.percentile(hedged_returns, 1)
    
    return (unhedged_var - hedged_var) / unhedged_var

def _calculate_total_hedge_cost(hedge_positions: Dict[str, Dict],
                             market_data: pd.DataFrame) -> float:
    """Calculate total cost of hedge positions."""
    total_cost = 0.0
    
    for _, hedge in hedge_positions.items():
        if 'cost' in hedge:
            total_cost += hedge['cost']
            
    return total_cost