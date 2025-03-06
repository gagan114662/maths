"""
Strategy evaluation module incorporating comprehensive metrics and ethical guidelines.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm, t

class StrategyEvaluator:
    def __init__(self, criteria: Optional[Dict[str, float]] = None):
        """
        Initialize the strategy evaluator with comprehensive metrics.
        
        Args:
            criteria: Optional dictionary with target criteria for evaluation metrics
                     (e.g., {'cagr': 0.25, 'sharpe_ratio': 1.0})
        """
        self.logger = self._setup_logger()
        self.risk_free_rate = 0.05  # 5% as specified
        
        # Default criteria
        self.criteria = {
            'cagr': 0.25,              # 25% annual return
            'sharpe_ratio': 1.0,       # Sharpe ratio > 1.0
            'sortino_ratio': 1.2,      # Sortino ratio > 1.2
            'max_drawdown': -0.20,     # Maximum drawdown < 20%
            'win_rate': 0.55,          # Win rate > 55%
            'profit_factor': 1.5,      # Profit factor > 1.5
            'calmar_ratio': 1.0,       # Calmar ratio > 1.0
            'information_ratio': 0.5,  # Information ratio > 0.5
            'ulcer_index': 0.10,       # Ulcer index < 10%
            'recovery_efficiency': 0.3  # Recovery efficiency > 0.3
        }
        
        # Override with user-specified criteria if provided
        if criteria is not None:
            self.criteria.update(criteria)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the evaluator."""
        logger = logging.getLogger("StrategyEvaluator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def evaluate_strategy(self, 
                         predictions: pd.DataFrame,
                         actuals: pd.DataFrame,
                         returns: pd.DataFrame,
                         trades: pd.DataFrame,
                         benchmark_returns: Optional[pd.DataFrame] = None,
                         beta: Optional[float] = None,
                         train_test_split: Optional[float] = None,
                         market_regimes: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Comprehensive strategy evaluation using multiple metric categories.
        
        Args:
            predictions: DataFrame with strategy predictions
            actuals: DataFrame with actual values
            returns: DataFrame with strategy returns
            trades: DataFrame with trade information
            benchmark_returns: Optional DataFrame with benchmark returns for beta-based metrics
            beta: Optional pre-calculated beta value
            train_test_split: Optional float that determines the split between training and test data
                             (e.g., 0.7 means 70% train, 30% test). If provided, out-of-sample evaluation is performed.
            market_regimes: Optional dictionary mapping regime names to boolean masks of the returns DataFrame
                          (e.g., {'bull_market': mask1, 'bear_market': mask2, 'high_volatility': mask3})
                          If provided, the strategy will be evaluated separately for each regime.
            
        Returns:
            Dictionary containing evaluation results across all categories
        """
        # Standard in-sample evaluation
        results = {
            'ranking_metrics': self._calculate_ranking_metrics(predictions, actuals),
            'portfolio_metrics': self._calculate_portfolio_metrics(returns, benchmark_returns, beta),
            'error_metrics': self._calculate_error_metrics(predictions, actuals),
            'ethical_compliance': self._check_ethical_compliance(trades, returns),
            'robustness_metrics': self._calculate_robustness_metrics(returns, trades),
            'drawdown_analysis': self._calculate_comprehensive_drawdown_metrics(returns),
            'risk_metrics': self._calculate_advanced_risk_metrics(returns, benchmark_returns),
            'statistical_significance': self._test_statistical_significance(returns, benchmark_returns)
        }
        
        # Out-of-sample evaluation if train_test_split is provided
        if train_test_split is not None:
            self.logger.info(f"Performing out-of-sample evaluation with {train_test_split*100:.0f}% train, {(1-train_test_split)*100:.0f}% test split")
            
            # Calculate split indices
            if isinstance(returns, pd.DataFrame):
                total_samples = len(returns)
            else:
                total_samples = len(returns.index)
                
            train_size = int(total_samples * train_test_split)
            
            # Split data
            train_indices = list(range(train_size))
            test_indices = list(range(train_size, total_samples))
            
            # Create train/test datasets
            train_returns = returns.iloc[train_indices]
            test_returns = returns.iloc[test_indices]
            
            train_predictions = predictions.iloc[train_indices]
            test_predictions = predictions.iloc[test_indices]
            
            train_actuals = actuals.iloc[train_indices]
            test_actuals = actuals.iloc[test_indices]
            
            # Get the in-sample and out-of-sample trades
            if not trades.empty:
                # This assumes trades DataFrame has a timestamp or date index
                train_end_date = returns.index[train_size - 1]
                train_trades = trades[trades.index <= train_end_date]
                test_trades = trades[trades.index > train_end_date]
            else:
                train_trades = pd.DataFrame()
                test_trades = pd.DataFrame()
            
            # Split benchmark returns if available
            train_benchmark = None if benchmark_returns is None else benchmark_returns.iloc[train_indices]
            test_benchmark = None if benchmark_returns is None else benchmark_returns.iloc[test_indices]
            
            # Perform separate evaluations for in-sample and out-of-sample
            in_sample_results = {
                'ranking_metrics': self._calculate_ranking_metrics(train_predictions, train_actuals),
                'portfolio_metrics': self._calculate_portfolio_metrics(train_returns, train_benchmark),
                'error_metrics': self._calculate_error_metrics(train_predictions, train_actuals),
                'robustness_metrics': self._calculate_robustness_metrics(train_returns, train_trades) if not train_trades.empty else {},
                'drawdown_analysis': self._calculate_comprehensive_drawdown_metrics(train_returns),
                'risk_metrics': self._calculate_advanced_risk_metrics(train_returns, train_benchmark),
                'statistical_significance': self._test_statistical_significance(train_returns, train_benchmark)
            }
            
            out_of_sample_results = {
                'ranking_metrics': self._calculate_ranking_metrics(test_predictions, test_actuals),
                'portfolio_metrics': self._calculate_portfolio_metrics(test_returns, test_benchmark),
                'error_metrics': self._calculate_error_metrics(test_predictions, test_actuals),
                'robustness_metrics': self._calculate_robustness_metrics(test_returns, test_trades) if not test_trades.empty else {},
                'drawdown_analysis': self._calculate_comprehensive_drawdown_metrics(test_returns),
                'risk_metrics': self._calculate_advanced_risk_metrics(test_returns, test_benchmark),
                'statistical_significance': self._test_statistical_significance(test_returns, test_benchmark)
            }
            
            # Add train/test results to the main results dictionary
            results['train_test_evaluation'] = {
                'train_results': in_sample_results,
                'test_results': out_of_sample_results,
                'performance_degradation': self._calculate_performance_degradation(in_sample_results, out_of_sample_results),
                'overfitting_metrics': self._calculate_overfitting_metrics(in_sample_results, out_of_sample_results),
                'walkforward_metrics': self._calculate_walkforward_metrics(returns, train_size)
            }
        
        # Overall strategy score
        results['overall_score'] = self._calculate_overall_score(results)
        
        # Evaluate against criteria
        results['criteria_evaluation'] = self._evaluate_against_criteria(results)
        
        # Market regime analysis if provided
        if market_regimes is not None:
            self.logger.info(f"Performing market regime analysis across {len(market_regimes)} regimes")
            results['market_regime_analysis'] = self._analyze_market_regimes(
                returns, predictions, actuals, trades, benchmark_returns, market_regimes
            )
        
        return results

    def _calculate_ranking_metrics(self, predictions: pd.DataFrame, 
                                 actuals: pd.DataFrame) -> Dict[str, float]:
        """Calculate ranking-based metrics including IC and RankICIR."""
        metrics = {}
        
        # Information Coefficient (IC)
        rank_pred = predictions.rank(axis=1)
        rank_actual = actuals.rank(axis=1)
        ic_series = rank_pred.corrwith(rank_actual, axis=1)
        
        metrics['ic_mean'] = ic_series.mean()
        metrics['ic_std'] = ic_series.std()
        metrics['rankicir'] = metrics['ic_mean'] / metrics['ic_std']
        
        # Spearman rank correlation
        metrics['spearman_corr'] = stats.spearmanr(
            predictions.values.flatten(),
            actuals.values.flatten()
        )[0]
        
        return metrics

    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, 
                                  benchmark_returns: Optional[pd.DataFrame] = None,
                                  beta: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio-based metrics including risk-adjusted returns.
        
        Args:
            returns: DataFrame of strategy returns
            benchmark_returns: Optional DataFrame of benchmark returns for beta-based metrics
            beta: Optional beta value if pre-calculated
            
        Returns:
            Dictionary of portfolio metrics
        """
        metrics = {}
        
        # Annualized return and volatility
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        max_dd = self._calculate_max_drawdown(returns)
        
        # Basic metrics
        metrics.update({
            'cagr': self._calculate_cagr(returns),
            'annualized_volatility': ann_vol,
            'annualized_return': ann_return,
            'max_drawdown': max_dd,
            'avg_profit': returns[returns > 0].mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf'),
            'win_rate': len(returns[returns > 0]) / len(returns)
        })
        
        # Risk-adjusted return metrics
        # 1. Sharpe Ratio
        metrics['sharpe_ratio'] = (ann_return - self.risk_free_rate) / ann_vol if ann_vol != 0 else 0
        
        # 2. Sortino Ratio
        metrics['sortino_ratio'] = self._calculate_sortino(returns)
        
        # 3. Calmar Ratio
        metrics['calmar_ratio'] = abs(ann_return / max_dd) if max_dd != 0 else float('inf')
        
        # 4. STARR Ratio (Stable Tail Adjusted Return Ratio)
        metrics['starr_ratio'] = self._calculate_starr_ratio(returns)
        
        # 5. Omega Ratio
        metrics['omega_ratio'] = self._calculate_omega_ratio(returns)
        
        # Beta-dependent metrics if benchmark is provided
        if benchmark_returns is not None:
            # Align dates
            aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
            aligned_data.columns = ['strategy', 'benchmark']
            
            # Calculate beta if not provided
            if beta is None:
                covariance = aligned_data.cov().iloc[0, 1]
                benchmark_variance = aligned_data['benchmark'].var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # 6. Treynor Ratio
            excess_return = ann_return - self.risk_free_rate
            metrics['treynor_ratio'] = excess_return / beta if beta != 0 else 0
            
            # 7. Jensen's Alpha
            benchmark_return = (1 + aligned_data['benchmark'].mean()) ** 252 - 1
            expected_return = self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate)
            metrics['jensens_alpha'] = ann_return - expected_return
            
            # 8. Information Ratio
            tracking_error = (aligned_data['strategy'] - aligned_data['benchmark']).std() * np.sqrt(252)
            metrics['information_ratio'] = (ann_return - benchmark_return) / tracking_error if tracking_error != 0 else 0
            
            # Store beta for reference
            metrics['beta'] = beta
            
        return metrics

    def _calculate_error_metrics(self, predictions: pd.DataFrame, 
                               actuals: pd.DataFrame) -> Dict[str, float]:
        """Calculate error-based metrics."""
        metrics = {}
        
        # Mean squared error
        mse = ((predictions - actuals) ** 2).mean().mean()
        metrics['rmse'] = np.sqrt(mse)
        
        # Mean absolute error
        metrics['mae'] = (predictions - actuals).abs().mean().mean()
        
        # Mean absolute percentage error
        metrics['mape'] = (((predictions - actuals) / actuals).abs() * 100).mean().mean()
        
        return metrics

    def _check_ethical_compliance(self, trades: pd.DataFrame, 
                                returns: pd.DataFrame) -> Dict[str, bool]:
        """
        Check compliance with ethical guidelines.
        
        Ethical Guidelines:
        1. No market manipulation
        2. Fair trading practices
        3. Risk management compliance
        4. Transaction size limits
        5. Trading frequency limits
        """
        compliance = {}
        
        # Check for potential market manipulation
        compliance['no_market_manipulation'] = self._check_manipulation_patterns(trades)
        
        # Check fair trading practices
        compliance['fair_trading'] = self._check_fair_trading(trades)
        
        # Check risk management compliance
        compliance['risk_compliant'] = self._check_risk_compliance(returns)
        
        # Check transaction size limits
        compliance['size_compliant'] = self._check_size_limits(trades)
        
        # Check trading frequency
        compliance['frequency_compliant'] = self._check_frequency_limits(trades)
        
        return compliance

    def _calculate_robustness_metrics(self, returns: pd.DataFrame, 
                                    trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate strategy robustness metrics."""
        metrics = {}
        
        # Stability of returns
        metrics['return_stability'] = 1 - returns.std() / abs(returns.mean())
        
        # Strategy consistency
        metrics['consistency_score'] = self._calculate_consistency(returns)
        
        # Drawdown recovery efficiency
        metrics['recovery_efficiency'] = self._calculate_recovery_efficiency(returns)
        
        # Trade efficiency
        metrics['trade_efficiency'] = self._calculate_trade_efficiency(trades)
        
        return metrics

    def _calculate_cagr(self, returns: pd.DataFrame) -> float:
        """Calculate Compound Annual Growth Rate."""
        total_return = (1 + returns).prod()
        years = len(returns) / 252
        return total_return ** (1 / years) - 1

    def _calculate_sortino(self, returns: pd.DataFrame) -> float:
        """Calculate Sortino Ratio using downside deviation."""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(252) * np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        return np.mean(excess_returns * 252) / downside_std if downside_std != 0 else 0
        
    def _calculate_starr_ratio(self, returns: pd.DataFrame, 
                             confidence_level: float = 0.95) -> float:
        """
        Calculate STARR Ratio (Stable Tail Adjusted Return Ratio).
        STARR = Expected Return / Expected Tail Loss
        
        Args:
            returns: DataFrame of strategy returns
            confidence_level: Confidence level for Expected Tail Loss calculation
            
        Returns:
            STARR ratio
        """
        # Calculate expected return (annualized)
        expected_return = (1 + returns.mean()) ** 252 - 1
        
        # Calculate Expected Tail Loss (ETL) or Conditional Value at Risk (CVaR)
        alpha = 1 - confidence_level
        var = np.percentile(returns, alpha * 100)  # Value at Risk
        
        # Expected Tail Loss = average of returns below VaR
        tail_losses = returns[returns <= var]
        etl = tail_losses.mean() * np.sqrt(252) if len(tail_losses) > 0 else 0
        
        # STARR ratio
        return expected_return / abs(etl) if etl != 0 else 0
        
    def _calculate_omega_ratio(self, returns: pd.DataFrame, 
                            threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.
        Omega = (Sum of returns above threshold) / (Sum of returns below threshold)
        
        Args:
            returns: DataFrame of strategy returns
            threshold: Return threshold (default: 0)
            
        Returns:
            Omega ratio
        """
        # Adjust returns by threshold
        excess = returns - threshold
        
        # Calculate positive and negative sums
        positive_sum = excess[excess > 0].sum()
        negative_sum = abs(excess[excess < 0].sum())
        
        # Omega ratio
        return positive_sum / negative_sum if negative_sum != 0 else float('inf')

    def _calculate_max_drawdown(self, returns: pd.DataFrame) -> float:
        """Calculate Maximum Drawdown as a percentage."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
        
    def _calculate_comprehensive_drawdown_metrics(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive drawdown analysis including underwater periods.
        
        Args:
            returns: DataFrame of strategy returns
            
        Returns:
            Dictionary with comprehensive drawdown metrics
        """
        # Calculate cumulative returns and rolling maximum
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        
        # Find drawdown periods
        is_drawdown = drawdowns < 0
        
        # Initialize metrics dictionary
        metrics = {}
        
        # 1. Maximum drawdown and its date
        max_dd = drawdowns.min()
        max_dd_date = drawdowns.idxmin() if not drawdowns.empty else None
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_date'] = max_dd_date
        
        # 2. Calculate drawdown duration metrics
        drawdown_info = self._calculate_drawdown_periods(drawdowns)
        
        # Add drawdown duration metrics
        metrics['max_drawdown_duration'] = drawdown_info['max_duration'] if drawdown_info else 0
        metrics['avg_drawdown_duration'] = drawdown_info['avg_duration'] if drawdown_info else 0
        metrics['total_drawdown_periods'] = drawdown_info['total_periods'] if drawdown_info else 0
        metrics['drawdown_periods'] = drawdown_info['periods'] if drawdown_info else []
        
        # 3. Recovery metrics
        metrics['avg_recovery_time'] = drawdown_info['avg_recovery'] if drawdown_info else 0
        metrics['max_recovery_time'] = drawdown_info['max_recovery'] if drawdown_info else 0
        
        # 4. Ulcer Index (UI) - square root of the mean of squared drawdowns
        metrics['ulcer_index'] = np.sqrt(np.mean(drawdowns[drawdowns < 0] ** 2)) if np.any(drawdowns < 0) else 0
        
        # 5. Pain Index - average of absolute drawdowns
        metrics['pain_index'] = np.abs(drawdowns[drawdowns < 0]).mean() if np.any(drawdowns < 0) else 0
        
        # 6. Drawdown to Volatility Ratio
        ann_vol = returns.std() * np.sqrt(252)
        metrics['drawdown_to_volatility'] = abs(max_dd) / ann_vol if ann_vol != 0 else 0
        
        # 7. Time underwater percentage
        metrics['time_underwater_pct'] = np.mean(is_drawdown) if len(is_drawdown) > 0 else 0
        
        # 8. Gain to Pain Ratio - ratio of sum of returns to sum of absolute drawdowns
        sum_abs_dd = np.abs(drawdowns[drawdowns < 0]).sum()
        metrics['gain_to_pain_ratio'] = returns.sum() / sum_abs_dd if sum_abs_dd != 0 else float('inf')
        
        # 9. Martin Ratio - annualized return divided by Ulcer Index
        ann_return = (1 + returns.mean()) ** 252 - 1
        metrics['martin_ratio'] = ann_return / metrics['ulcer_index'] if metrics['ulcer_index'] != 0 else 0
        
        # 10. Pain Ratio - annualized return divided by Pain Index
        metrics['pain_ratio'] = ann_return / metrics['pain_index'] if metrics['pain_index'] != 0 else 0
        
        # 11. Top 5 drawdowns
        top_drawdowns = self._get_top_drawdowns(drawdowns, n=5)
        metrics['top_drawdowns'] = top_drawdowns
        
        return metrics
    
    def _calculate_drawdown_periods(self, drawdowns: pd.Series) -> Dict[str, Any]:
        """
        Calculate drawdown periods and their statistics.
        
        Args:
            drawdowns: Series of drawdown values
            
        Returns:
            Dictionary with drawdown period statistics
        """
        if len(drawdowns) == 0:
            return {}
            
        # Initialize tracking variables
        periods = []
        in_drawdown = False
        start_idx = None
        max_dd_during_period = 0
        
        # Analyze the drawdown series
        for i, (date, value) in enumerate(drawdowns.items()):
            if value < 0 and not in_drawdown:
                # Start of a new drawdown period
                in_drawdown = True
                start_idx = i
                max_dd_during_period = value
            elif value < 0 and in_drawdown:
                # Update max drawdown during this period
                max_dd_during_period = min(max_dd_during_period, value)
            elif value >= 0 and in_drawdown:
                # End of a drawdown period
                end_idx = i - 1
                duration = end_idx - start_idx + 1
                recovery = i - end_idx - 1  # Days until recovery
                
                # Record this period
                periods.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration': duration,
                    'recovery': recovery,
                    'max_drawdown': max_dd_during_period,
                    'start_date': drawdowns.index[start_idx],
                    'end_date': drawdowns.index[end_idx],
                    'recovery_date': drawdowns.index[i-1] if i > 0 else None
                })
                
                # Reset tracking
                in_drawdown = False
                start_idx = None
                max_dd_during_period = 0
        
        # Handle case where we're still in a drawdown at the end of the series
        if in_drawdown:
            end_idx = len(drawdowns) - 1
            duration = end_idx - start_idx + 1
            
            # Record this period (recovery is None as we haven't recovered yet)
            periods.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': duration,
                'recovery': None,  # Still in drawdown
                'max_drawdown': max_dd_during_period,
                'start_date': drawdowns.index[start_idx],
                'end_date': drawdowns.index[end_idx],
                'recovery_date': None
            })
        
        # Calculate summary statistics
        if periods:
            durations = [p['duration'] for p in periods]
            recoveries = [p['recovery'] for p in periods if p['recovery'] is not None]
            
            return {
                'periods': periods,
                'total_periods': len(periods),
                'max_duration': max(durations) if durations else 0,
                'avg_duration': np.mean(durations) if durations else 0,
                'max_recovery': max(recoveries) if recoveries else 0,
                'avg_recovery': np.mean(recoveries) if recoveries else 0
            }
        else:
            return {
                'periods': [],
                'total_periods': 0,
                'max_duration': 0,
                'avg_duration': 0,
                'max_recovery': 0,
                'avg_recovery': 0
            }
    
    def _get_top_drawdowns(self, drawdowns: pd.Series, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N drawdowns by magnitude.
        
        Args:
            drawdowns: Series of drawdown values
            n: Number of top drawdowns to return
            
        Returns:
            List of dictionaries with top drawdown information
        """
        # Initialize tracking variables
        periods = []
        in_drawdown = False
        start_idx = None
        max_dd_during_period = 0
        max_dd_idx = None
        
        # Analyze the drawdown series
        for i, (date, value) in enumerate(drawdowns.items()):
            if value < 0 and not in_drawdown:
                # Start of a new drawdown period
                in_drawdown = True
                start_idx = i
                max_dd_during_period = value
                max_dd_idx = i
            elif value < 0 and in_drawdown:
                # Update max drawdown during this period
                if value < max_dd_during_period:
                    max_dd_during_period = value
                    max_dd_idx = i
            elif value >= 0 and in_drawdown:
                # End of a drawdown period
                end_idx = i - 1
                
                # Record this period
                periods.append({
                    'start_date': drawdowns.index[start_idx],
                    'end_date': drawdowns.index[end_idx],
                    'max_drawdown_date': drawdowns.index[max_dd_idx],
                    'max_drawdown': max_dd_during_period,
                    'duration': end_idx - start_idx + 1,
                    'recovery': i - end_idx - 1
                })
                
                # Reset tracking
                in_drawdown = False
                start_idx = None
                max_dd_during_period = 0
                max_dd_idx = None
        
        # Handle case where we're still in a drawdown at the end of the series
        if in_drawdown:
            end_idx = len(drawdowns) - 1
            
            # Record this period
            periods.append({
                'start_date': drawdowns.index[start_idx],
                'end_date': drawdowns.index[end_idx],
                'max_drawdown_date': drawdowns.index[max_dd_idx],
                'max_drawdown': max_dd_during_period,
                'duration': end_idx - start_idx + 1,
                'recovery': None  # Still in drawdown
            })
        
        # Sort by drawdown magnitude and take top n
        periods.sort(key=lambda x: x['max_drawdown'])
        return periods[:n]
        
    def _calculate_advanced_risk_metrics(self, returns: pd.DataFrame,
                                      benchmark_returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate advanced risk metrics including VaR, CVaR, tail risk, etc.
        
        Args:
            returns: DataFrame of strategy returns
            benchmark_returns: Optional DataFrame of benchmark returns
            
        Returns:
            Dictionary with advanced risk metrics
        """
        metrics = {}
        
        # Convert to numpy array for calculations
        returns_array = returns.values.flatten()
        
        # 1. Value at Risk (VaR) at different confidence levels
        confidence_levels = [0.99, 0.95, 0.90]
        for conf in confidence_levels:
            # Historical VaR
            historical_var = np.percentile(returns_array, (1-conf) * 100)
            metrics[f'historical_var_{int(conf*100)}'] = historical_var
            
            # Parametric VaR (assuming normal distribution)
            mean = returns_array.mean()
            std = returns_array.std()
            z_score = norm.ppf(1-conf)
            parametric_var = mean + z_score * std
            metrics[f'parametric_var_{int(conf*100)}'] = parametric_var
            
            # Conditional VaR (CVaR) / Expected Shortfall (ES)
            cvar_values = returns_array[returns_array <= historical_var]
            cvar = cvar_values.mean() if len(cvar_values) > 0 else historical_var
            metrics[f'cvar_{int(conf*100)}'] = cvar
        
        # 2. Tail Risk Metrics
        
        # Skewness
        metrics['skewness'] = stats.skew(returns_array)
        
        # Kurtosis (excess kurtosis, normal = 0)
        metrics['kurtosis'] = stats.kurtosis(returns_array)
        
        # 3. Downside Risk Metrics
        
        # Downside Deviation
        downside_returns = returns_array[returns_array < 0]
        metrics['downside_deviation'] = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        
        # Downside Capture Ratio (if benchmark provided)
        if benchmark_returns is not None:
            # Align dates
            aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
            aligned_data.columns = ['strategy', 'benchmark']
            
            # Get negative benchmark periods
            negative_benchmark = aligned_data[aligned_data['benchmark'] < 0]
            
            if len(negative_benchmark) > 0:
                # Downside Capture
                downside_capture = negative_benchmark['strategy'].mean() / negative_benchmark['benchmark'].mean()
                metrics['downside_capture'] = downside_capture
                
                # Upside Capture
                positive_benchmark = aligned_data[aligned_data['benchmark'] > 0]
                if len(positive_benchmark) > 0:
                    upside_capture = positive_benchmark['strategy'].mean() / positive_benchmark['benchmark'].mean()
                    metrics['upside_capture'] = upside_capture
                    
                    # Capture Ratio
                    metrics['capture_ratio'] = abs(upside_capture / downside_capture) if downside_capture != 0 else float('inf')
        
        # 4. Tail Risk Measures
        
        # Modified VaR (Cornish-Fisher VaR)
        # Accounts for skewness and kurtosis
        skew = metrics['skewness']
        kurt = metrics['kurtosis']
        
        for conf in confidence_levels:
            z = norm.ppf(1-conf)
            # Cornish-Fisher expansion
            cf_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
            modified_var = returns_array.mean() + cf_z * returns_array.std()
            metrics[f'modified_var_{int(conf*100)}'] = modified_var
        
        # 5. Return/Risk Efficiency Metrics
        
        # Annualized Return
        ann_return = (1 + returns_array.mean()) ** 252 - 1
        
        # Annualized Volatility
        ann_vol = returns_array.std() * np.sqrt(252)
        
        # Return/Risk Ratio (Alternative to Sharpe when not using risk-free rate)
        metrics['return_risk_ratio'] = ann_return / ann_vol if ann_vol != 0 else 0
        
        # 6. Drawdown-adjusted Return Measures
        
        # Sterling Ratio (using maximum drawdown)
        max_dd = self._calculate_max_drawdown(returns)
        metrics['sterling_ratio'] = ann_return / abs(max_dd) if max_dd != 0 else float('inf')
        
        # Burke Ratio (uses sum of squared drawdowns)
        drawdowns = self._calculate_drawdown_series(returns)
        top_drawdowns = drawdowns.nsmallest(5)  # Top 5 drawdowns
        sum_sq_dd = np.sum(np.square(top_drawdowns))
        metrics['burke_ratio'] = ann_return / np.sqrt(sum_sq_dd) if sum_sq_dd != 0 else float('inf')
        
        # 7. Semi-variance and Target Semi-variance
        target_return = 0  # Can be customized based on strategy goals
        semi_variance = np.mean((returns_array[returns_array < target_return] - target_return) ** 2)
        metrics['semi_variance'] = semi_variance
        
        # R-Squared (if benchmark provided)
        if benchmark_returns is not None:
            # Correlation with benchmark
            correlation = aligned_data.corr().iloc[0, 1]
            metrics['correlation_with_benchmark'] = correlation
            
            # R-Squared
            metrics['r_squared'] = correlation ** 2
            
            # Tracking Error
            tracking_diff = aligned_data['strategy'] - aligned_data['benchmark']
            tracking_error = tracking_diff.std() * np.sqrt(252)
            metrics['tracking_error'] = tracking_error
        
        # 8. Higher moment risk metrics
        
        # Value at Risk adjusted for higher moments
        # Using Cornish-Fisher expansion for adjusted quantiles
        z_99 = norm.ppf(0.01)
        cf_adj_99 = z_99 + (z_99**2 - 1) * skew / 6 + (z_99**3 - 3*z_99) * kurt / 24 - (2*z_99**3 - 5*z_99) * skew**2 / 36
        metrics['adjusted_var_99'] = returns_array.mean() + cf_adj_99 * returns_array.std()
        
        # Maximum Likelihood Estimation of t-distribution parameters for fat tails
        try:
            t_params = t.fit(returns_array)
            metrics['t_dist_df'] = t_params[0]  # Degrees of freedom
            metrics['t_dist_loc'] = t_params[1]  # Location
            metrics['t_dist_scale'] = t_params[2]  # Scale
            
            # Lower degree of freedom = fatter tails
            metrics['fat_tail_measure'] = 1 / metrics['t_dist_df'] if metrics['t_dist_df'] != 0 else float('inf')
        except:
            # If fitting fails, skip these metrics
            pass
        
        return metrics

    def _check_manipulation_patterns(self, trades: pd.DataFrame) -> bool:
        """Check for potential market manipulation patterns."""
        # Implementation of specific manipulation detection rules
        return True  # Placeholder

    def _check_fair_trading(self, trades: pd.DataFrame) -> bool:
        """Check for fair trading practices."""
        # Implementation of fair trading checks
        return True  # Placeholder

    def _check_risk_compliance(self, returns: pd.DataFrame) -> bool:
        """Check compliance with risk management rules."""
        max_drawdown = self._calculate_max_drawdown(returns)
        daily_var = self._calculate_var(returns)
        
        return max_drawdown > -0.20 and daily_var > -0.05

    def _check_size_limits(self, trades: pd.DataFrame) -> bool:
        """Check compliance with position size limits."""
        # Implementation of size limit checks
        return True  # Placeholder

    def _check_frequency_limits(self, trades: pd.DataFrame) -> bool:
        """Check compliance with trading frequency limits."""
        # Implementation of frequency limit checks
        return True  # Placeholder

    def _calculate_consistency(self, returns: pd.DataFrame) -> float:
        """Calculate strategy consistency score."""
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=63)  # Quarter
        return 1 - rolling_sharpe.std()

    def _calculate_recovery_efficiency(self, returns: pd.DataFrame) -> float:
        """Calculate drawdown recovery efficiency."""
        drawdowns = self._calculate_drawdown_series(returns)
        recoveries = self._calculate_recovery_times(drawdowns)
        return 1 / np.mean(recoveries) if recoveries.any() else 0

    def _calculate_trade_efficiency(self, trades: pd.DataFrame) -> float:
        """Calculate trading efficiency score."""
        # Implementation of trade efficiency calculation
        return 0.0  # Placeholder

    def _calculate_var(self, returns: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_rolling_sharpe(self, returns: pd.DataFrame, 
                                window: int = 63) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        rolling_ret = excess_returns.rolling(window=window).mean() * 252
        rolling_vol = excess_returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_ret / rolling_vol

    def _calculate_drawdown_series(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate drawdown series."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        return cum_returns / rolling_max - 1

    def _calculate_recovery_times(self, drawdowns: pd.Series) -> pd.Series:
        """Calculate recovery times from drawdowns."""
        is_drawdown = drawdowns < 0
        recovery_times = pd.Series(index=drawdowns.index, dtype=float)
        
        current_drawdown_start = None
        for i, is_down in enumerate(is_drawdown):
            if is_down and current_drawdown_start is None:
                current_drawdown_start = i
            elif not is_down and current_drawdown_start is not None:
                recovery_times[i] = i - current_drawdown_start
                current_drawdown_start = None
                
        return recovery_times.dropna()

    def _test_statistical_significance(self, returns: pd.DataFrame, 
                                benchmark_returns: Optional[pd.DataFrame] = None,
                                significance_level: float = 0.05,
                                num_bootstrap_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform statistical significance testing on strategy returns.
        
        Args:
            returns: DataFrame of strategy returns
            benchmark_returns: Optional DataFrame of benchmark returns (e.g., S&P 500)
            significance_level: Alpha level for statistical tests (default: 0.05)
            num_bootstrap_samples: Number of bootstrap samples for simulation tests
            
        Returns:
            Dictionary with statistical significance test results
        """
        results = {}
        
        # 1. Test if returns are significantly different from zero (t-test)
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        results['returns_different_from_zero'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < significance_level
        }
        
        # 2. Test if returns are normally distributed (Jarque-Bera test)
        jb_stat, jb_p_value, skew, kurtosis = jarque_bera(returns)
        results['returns_normality_test'] = {
            'jb_statistic': jb_stat,
            'p_value': jb_p_value,
            'skewness': skew,
            'kurtosis': kurtosis,
            'is_normal': jb_p_value > significance_level
        }
        
        # 3. Test stationarity of returns (Augmented Dickey-Fuller test)
        adf_result = adfuller(returns, regression='c')
        results['returns_stationarity_test'] = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < significance_level,
            'critical_values': adf_result[4]
        }
        
        # 4. Bootstrap significance test (compare to random strategies)
        bootstrap_results = self._bootstrap_significance_test(
            returns, num_samples=num_bootstrap_samples, alpha=significance_level
        )
        results['bootstrap_test'] = bootstrap_results
        
        # 5. Compare to benchmark if provided
        if benchmark_returns is not None:
            # Ensure alignment of dates
            aligned_returns = pd.concat([returns, benchmark_returns], axis=1, join='inner')
            strategy_returns = aligned_returns.iloc[:, 0]
            benchmark = aligned_returns.iloc[:, 1]
            
            # Test if strategy outperforms benchmark
            outperformance = strategy_returns - benchmark
            t_stat, p_value = stats.ttest_1samp(outperformance, 0)
            results['benchmark_outperformance'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < significance_level
            }
            
            # Information ratio
            tracking_error = outperformance.std() * np.sqrt(252)  # Annualized
            info_ratio = (outperformance.mean() * 252) / tracking_error
            results['information_ratio'] = info_ratio
        
        return results
    
    def _bootstrap_significance_test(self, returns: pd.DataFrame, 
                                  num_samples: int = 1000, 
                                  alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform bootstrap significance testing by comparing strategy Sharpe ratio
        to bootstrapped random strategies.
        
        Args:
            returns: DataFrame of strategy returns
            num_samples: Number of bootstrap samples
            alpha: Significance level
            
        Returns:
            Dictionary with bootstrap test results
        """
        # Calculate actual Sharpe ratio
        actual_sharpe = self._calculate_sharpe_ratio(returns)
        
        # Generate bootstrap samples by randomly reordering returns
        bootstrap_sharpes = []
        for _ in range(num_samples):
            # Randomly reorder returns to create random strategy
            random_returns = returns.sample(frac=1, replace=False)
            random_sharpe = self._calculate_sharpe_ratio(random_returns)
            bootstrap_sharpes.append(random_sharpe)
        
        # Calculate p-value as proportion of random strategies with Sharpe higher than actual
        p_value = sum(1 for x in bootstrap_sharpes if x > actual_sharpe) / num_samples
        
        # Calculate confidence intervals
        bootstrap_sharpes.sort()
        lower_ci = bootstrap_sharpes[int(alpha/2 * num_samples)]
        upper_ci = bootstrap_sharpes[int((1-alpha/2) * num_samples)]
        
        return {
            'actual_sharpe': actual_sharpe,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'confidence_interval': (lower_ci, upper_ci),
            'bootstrap_mean': np.mean(bootstrap_sharpes),
            'bootstrap_median': np.median(bootstrap_sharpes),
            'bootstrap_std': np.std(bootstrap_sharpes)
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.DataFrame) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return excess_returns.mean() * 252 / (excess_returns.std() * np.sqrt(252))
        
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall strategy score."""
        weights = {
            'ranking_metrics': 0.15,
            'portfolio_metrics': 0.20,
            'error_metrics': 0.10,
            'ethical_compliance': 0.10,
            'robustness_metrics': 0.10,
            'drawdown_analysis': 0.10,
            'risk_metrics': 0.10,
            'statistical_significance': 0.10,
            'train_test_evaluation': 0.05  # Only used if train/test split was performed
        }
        
        score = 0.0
        
        # Ranking metrics contribution
        ranking_score = (
            results['ranking_metrics']['ic_mean'] * 0.4 +
            results['ranking_metrics']['rankicir'] * 0.6
        )
        score += ranking_score * weights['ranking_metrics']
        
        # Portfolio metrics contribution
        portfolio_score = (
            min(1, results['portfolio_metrics']['sharpe_ratio'] / 2) * 0.3 +
            min(1, results['portfolio_metrics']['cagr'] / 0.25) * 0.3 +
            (1 + results['portfolio_metrics']['max_drawdown'] / 0.2) * 0.4
        )
        score += portfolio_score * weights['portfolio_metrics']
        
        # Error metrics contribution (lower is better)
        error_score = 1 - min(1, (
            results['error_metrics']['rmse'] * 0.4 +
            results['error_metrics']['mae'] * 0.3 +
            results['error_metrics']['mape'] / 100 * 0.3
        ))
        score += error_score * weights['error_metrics']
        
        # Ethical compliance contribution
        compliance_score = np.mean(list(results['ethical_compliance'].values()))
        score += compliance_score * weights['ethical_compliance']
        
        # Robustness metrics contribution
        robustness_score = np.mean(list(results['robustness_metrics'].values()))
        score += robustness_score * weights['robustness_metrics']
        
        # Drawdown analysis contribution
        if 'drawdown_analysis' in results:
            drawdown_metrics = results['drawdown_analysis']
            
            # Create normalized drawdown score from key metrics (lower drawdown is better)
            max_dd_score = min(1, max(0, (abs(drawdown_metrics.get('max_drawdown', 0)) < 0.20)))
            dd_duration_score = min(1, max(0, 1 - drawdown_metrics.get('avg_drawdown_duration', 0) / 63))  # Normalize to quarter
            underwater_score = min(1, max(0, 1 - drawdown_metrics.get('time_underwater_pct', 0)))
            recovery_score = min(1, max(0, 1 - drawdown_metrics.get('avg_recovery_time', 0) / 30))  # Normalize to month
            
            drawdown_score = (
                max_dd_score * 0.4 +
                dd_duration_score * 0.2 +
                underwater_score * 0.2 +
                recovery_score * 0.2
            )
            
            score += drawdown_score * weights['drawdown_analysis']
            
        # Risk metrics contribution
        if 'risk_metrics' in results:
            risk_metrics = results['risk_metrics']
            
            # Create normalized risk score (lower risk is better)
            # VaR component - normalize to expectation
            var_99_score = min(1, max(0, 1 + risk_metrics.get('historical_var_99', -0.02) / 0.02))
            # Tail risk component
            tail_score = min(1, max(0, 1 - abs(risk_metrics.get('skewness', 0)) / 2))  # Prefer near-zero skewness
            # Downside risk component
            downside_score = min(1, max(0, 1 - risk_metrics.get('downside_deviation', 0) / 0.02))
            # Return/risk efficiency
            efficiency_score = min(1, max(0, risk_metrics.get('return_risk_ratio', 0) / 1.5))
            
            risk_score = (
                var_99_score * 0.3 +
                tail_score * 0.2 +
                downside_score * 0.3 +
                efficiency_score * 0.2
            )
            
            score += risk_score * weights['risk_metrics']
            
        # Statistical significance contribution
        if 'statistical_significance' in results:
            # Calculate significance score based on p-values and bootstrap results
            significance_score = 0.0
            
            # Score from t-test (returns different from zero)
            if results['statistical_significance'].get('returns_different_from_zero'):
                p_value = results['statistical_significance']['returns_different_from_zero']['p_value']
                significance_score += max(0, 1 - p_value * 2) * 0.4  # Lower p-value = higher score
            
            # Score from bootstrap test
            if results['statistical_significance'].get('bootstrap_test'):
                bootstrap_p_value = results['statistical_significance']['bootstrap_test']['p_value']
                significance_score += max(0, 1 - bootstrap_p_value * 2) * 0.6  # Lower p-value = higher score
            
            score += significance_score * weights['statistical_significance']
            
        # Train/test evaluation contribution (out-of-sample performance)
        if 'train_test_evaluation' in results:
            train_test_score = 0.0
            train_test = results['train_test_evaluation']
            
            # Reward lower overfitting
            if 'overfitting_metrics' in train_test and 'overall_overfitting_score' in train_test['overfitting_metrics']:
                # Lower overfitting_score is better
                overfitting_score = train_test['overfitting_metrics']['overall_overfitting_score']
                train_test_score += (1 - overfitting_score) * 0.4
                
            # Reward consistency in walkforward testing
            if 'walkforward_metrics' in train_test and 'overall_consistency' in train_test['walkforward_metrics']:
                consistency_score = train_test['walkforward_metrics']['overall_consistency']
                train_test_score += consistency_score * 0.4
                
            # Reward lower degradation
            if 'performance_degradation' in train_test and 'overall_degradation_rate' in train_test['performance_degradation']:
                degradation_rate = train_test['performance_degradation']['overall_degradation_rate']
                train_test_score += (1 - degradation_rate) * 0.2
                
            score += train_test_score * weights['train_test_evaluation']
        
        return min(1.0, max(0.0, score))
        
    def _calculate_performance_degradation(self, 
                                        in_sample_results: Dict[str, Any],
                                        out_of_sample_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance degradation between in-sample and out-of-sample results.
        
        Args:
            in_sample_results: Dictionary with in-sample evaluation results
            out_of_sample_results: Dictionary with out-of-sample evaluation results
            
        Returns:
            Dictionary with performance degradation metrics
        """
        degradation = {}
        
        # Key metrics to compare between in-sample and out-of-sample
        key_metrics = [
            ('sharpe_ratio', 'portfolio_metrics'),
            ('sortino_ratio', 'portfolio_metrics'),
            ('cagr', 'portfolio_metrics'),
            ('max_drawdown', 'drawdown_analysis'),
            ('ic_mean', 'ranking_metrics'),
            ('rmse', 'error_metrics')
        ]
        
        for metric_name, metric_category in key_metrics:
            # Skip if category or metric is missing in either result
            if (metric_category not in in_sample_results or 
                metric_category not in out_of_sample_results or
                metric_name not in in_sample_results[metric_category] or 
                metric_name not in out_of_sample_results[metric_category]):
                continue
                
            # Get in-sample and out-of-sample values
            in_sample_value = in_sample_results[metric_category][metric_name]
            out_of_sample_value = out_of_sample_results[metric_category][metric_name]
            
            # Calculate absolute and relative changes
            absolute_change = out_of_sample_value - in_sample_value
            
            # Relative change (handle divide by zero)
            if in_sample_value != 0:
                relative_change = absolute_change / abs(in_sample_value)
            else:
                relative_change = 0 if out_of_sample_value == 0 else float('inf')
            
            # For metrics where higher is better (e.g., Sharpe, Sortino, CAGR, IC)
            if metric_name in ['sharpe_ratio', 'sortino_ratio', 'cagr', 'ic_mean']:
                degradation[f'{metric_name}_absolute_change'] = absolute_change
                degradation[f'{metric_name}_relative_change'] = relative_change
                degradation[f'{metric_name}_degradation'] = relative_change < 0  # True if performance degraded
            
            # For metrics where lower is better (e.g., max_drawdown, RMSE)
            elif metric_name in ['max_drawdown', 'rmse']:
                # Adjust signs for consistent interpretation (negative means degradation)
                degradation[f'{metric_name}_absolute_change'] = -absolute_change
                degradation[f'{metric_name}_relative_change'] = -relative_change
                degradation[f'{metric_name}_degradation'] = relative_change > 0  # True if performance degraded
        
        # Calculate overall performance degradation score
        if len(degradation) > 0:
            # Count metrics that showed degradation
            degradation_count = sum(1 for key in degradation if key.endswith('_degradation') and degradation[key])
            total_metrics = sum(1 for key in degradation if key.endswith('_degradation'))
            
            degradation['overall_degradation_rate'] = degradation_count / total_metrics if total_metrics > 0 else 0
            
            # Calculate average relative change across all metrics
            relative_changes = [
                degradation[key] for key in degradation 
                if key.endswith('_relative_change') and not np.isinf(degradation[key]) and not np.isnan(degradation[key])
            ]
            
            degradation['average_relative_change'] = np.mean(relative_changes) if relative_changes else 0
        
        return degradation
    
    def _calculate_overfitting_metrics(self,
                                    in_sample_results: Dict[str, Any],
                                    out_of_sample_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics to detect and quantify overfitting.
        
        Args:
            in_sample_results: Dictionary with in-sample evaluation results
            out_of_sample_results: Dictionary with out-of-sample evaluation results
            
        Returns:
            Dictionary with overfitting metrics
        """
        overfitting = {}
        
        # 1. Performance Ratio (Out-of-Sample / In-Sample)
        # Key metrics to compare
        key_metrics = [
            ('sharpe_ratio', 'portfolio_metrics'),
            ('sortino_ratio', 'portfolio_metrics'),
            ('cagr', 'portfolio_metrics')
        ]
        
        for metric_name, metric_category in key_metrics:
            # Skip if category or metric is missing in either result
            if (metric_category not in in_sample_results or 
                metric_category not in out_of_sample_results or
                metric_name not in in_sample_results[metric_category] or 
                metric_name not in out_of_sample_results[metric_category]):
                continue
                
            # Get in-sample and out-of-sample values
            in_sample_value = in_sample_results[metric_category][metric_name]
            out_of_sample_value = out_of_sample_results[metric_category][metric_name]
            
            # Calculate performance ratio
            if in_sample_value > 0:
                perf_ratio = out_of_sample_value / in_sample_value
            elif in_sample_value < 0 and out_of_sample_value < 0:
                # Both negative, higher ratio means less negative out-of-sample
                perf_ratio = in_sample_value / out_of_sample_value
            elif in_sample_value == 0:
                perf_ratio = 1.0 if out_of_sample_value == 0 else 0.0
            else:
                # in_sample < 0, out_of_sample >= 0
                perf_ratio = 0.0  # Significant divergence
                
            overfitting[f'{metric_name}_performance_ratio'] = perf_ratio
        
        # 2. Prediction Error Ratio (Out-of-Sample / In-Sample)
        error_metrics = ['rmse', 'mae', 'mape']
        
        for metric in error_metrics:
            if ('error_metrics' in in_sample_results and 
                'error_metrics' in out_of_sample_results and
                metric in in_sample_results['error_metrics'] and 
                metric in out_of_sample_results['error_metrics']):
                
                in_sample_error = in_sample_results['error_metrics'][metric]
                out_of_sample_error = out_of_sample_results['error_metrics'][metric]
                
                # Calculate error ratio (higher means more overfitting)
                if in_sample_error > 0:
                    error_ratio = out_of_sample_error / in_sample_error
                else:
                    error_ratio = 1.0 if out_of_sample_error == 0 else float('inf')
                    
                overfitting[f'{metric}_ratio'] = error_ratio
        
        # 3. Probabilistic Overfitting Score
        # Based on Bailey et al. "The Probability of Backtest Overfitting"
        # Simplified implementation
        if len(overfitting) >= 3:  # Need at least a few metrics to calculate
            # Calculate average performance ratio (should be close to 1.0 for no overfitting)
            perf_ratios = [
                overfitting[key] for key in overfitting 
                if key.endswith('_performance_ratio') and not np.isinf(overfitting[key]) and not np.isnan(overfitting[key])
            ]
            
            if perf_ratios:
                avg_perf_ratio = np.mean(perf_ratios)
                perf_ratio_std = np.std(perf_ratios)
                
                # Calculate probability of overfitting
                # Simplified: How far is avg_perf_ratio from 1.0
                overfitting_distance = max(0, 1 - avg_perf_ratio)
                prob_overfitting = min(1, overfitting_distance * 2)  # Scale to [0, 1]
                
                overfitting['probability_of_overfitting'] = prob_overfitting
                overfitting['performance_ratio_mean'] = avg_perf_ratio
                overfitting['performance_ratio_std'] = perf_ratio_std
        
        # 4. Deflated Sharpe Ratio
        # Based on Lopez de Prado's "Deflated Sharpe Ratio" - simplified implementation
        if ('portfolio_metrics' in in_sample_results and 
            'portfolio_metrics' in out_of_sample_results and
            'sharpe_ratio' in in_sample_results['portfolio_metrics'] and 
            'sharpe_ratio' in out_of_sample_results['portfolio_metrics']):
            
            in_sample_sharpe = in_sample_results['portfolio_metrics']['sharpe_ratio']
            out_of_sample_sharpe = out_of_sample_results['portfolio_metrics']['sharpe_ratio']
            
            # Simplified deflated Sharpe calculation
            deflated_sharpe = out_of_sample_sharpe / in_sample_sharpe if in_sample_sharpe > 0 else 0
            overfitting['deflated_sharpe_ratio'] = deflated_sharpe
        
        # 5. Overall overfitting score (0 = no overfitting, 1 = severe overfitting)
        if 'probability_of_overfitting' in overfitting:
            # Start with the probability of overfitting
            overfitting_score = overfitting['probability_of_overfitting']
            
            # Adjust based on error ratios if available
            error_ratios = [
                overfitting[key] for key in overfitting 
                if key.endswith('_ratio') and not key.endswith('_performance_ratio') 
                and not np.isinf(overfitting[key]) and not np.isnan(overfitting[key])
            ]
            
            if error_ratios:
                # Normalize error ratios (higher means more overfitting)
                norm_error_ratios = [min(2, ratio) / 2 for ratio in error_ratios]
                avg_error_score = np.mean(norm_error_ratios)
                overfitting_score = (overfitting_score + avg_error_score) / 2
            
            overfitting['overall_overfitting_score'] = overfitting_score
            
            # Qualitative assessment
            if overfitting_score < 0.25:
                overfitting['overfitting_assessment'] = "Low overfitting risk"
            elif overfitting_score < 0.5:
                overfitting['overfitting_assessment'] = "Moderate overfitting risk"
            elif overfitting_score < 0.75:
                overfitting['overfitting_assessment'] = "High overfitting risk"
            else:
                overfitting['overfitting_assessment'] = "Severe overfitting risk"
        
        return overfitting
    
    def _calculate_walkforward_metrics(self, returns: pd.DataFrame, train_size: int) -> Dict[str, Any]:
        """
        Calculate walkforward metrics to evaluate strategy consistency over time.
        
        Args:
            returns: DataFrame of strategy returns
            train_size: Size of the training set
            
        Returns:
            Dictionary with walkforward metrics
        """
        walkforward = {}
        
        # Calculate minimum required window size for reliable metrics
        min_window = min(63, len(returns) // 5)  # ~quarter or 1/5 of data, whichever is smaller
        
        # Skip if not enough data
        if len(returns) < train_size + min_window:
            return {'error': 'Not enough data for reliable walkforward analysis'}
        
        # Calculate expanding window metrics
        window_metrics = []
        window_sharpes = []
        window_returns = []
        window_drawdowns = []
        
        # Start from the end of training period
        for i in range(train_size, len(returns) - min_window + 1, min_window):
            # Take window of data
            window_end = min(i + min_window, len(returns))
            window_data = returns.iloc[i:window_end]
            
            # Skip if window is too small
            if len(window_data) < min_window // 2:
                continue
            
            # Calculate key metrics for this window
            ann_return = (1 + window_data.mean()) ** 252 - 1
            ann_vol = window_data.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol != 0 else 0
            
            # Calculate drawdown for this window
            cum_returns = (1 + window_data).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_dd = drawdowns.min()
            
            # Store metrics
            window_metrics.append({
                'start_idx': i,
                'end_idx': window_end,
                'start_date': returns.index[i],
                'end_date': returns.index[window_end - 1] if window_end < len(returns) else returns.index[-1],
                'annualized_return': ann_return,
                'annualized_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            })
            
            window_sharpes.append(sharpe)
            window_returns.append(ann_return)
            window_drawdowns.append(max_dd)
        
        # Add window metrics
        walkforward['window_metrics'] = window_metrics
        
        # Calculate consistency measures
        if window_sharpes:
            walkforward['sharpe_ratio_mean'] = np.mean(window_sharpes)
            walkforward['sharpe_ratio_std'] = np.std(window_sharpes)
            walkforward['return_mean'] = np.mean(window_returns)
            walkforward['return_std'] = np.std(window_returns)
            walkforward['max_drawdown_mean'] = np.mean(window_drawdowns)
            walkforward['max_drawdown_std'] = np.std(window_drawdowns)
            
            # Consistency scores (lower std relative to mean = more consistent)
            # For Sharpe and returns (higher is better)
            if walkforward['sharpe_ratio_mean'] > 0:
                walkforward['sharpe_consistency'] = 1 - min(1, walkforward['sharpe_ratio_std'] / max(0.01, abs(walkforward['sharpe_ratio_mean'])))
            else:
                walkforward['sharpe_consistency'] = 0
                
            if walkforward['return_mean'] > 0:
                walkforward['return_consistency'] = 1 - min(1, walkforward['return_std'] / max(0.01, abs(walkforward['return_mean'])))
            else:
                walkforward['return_consistency'] = 0
            
            # For drawdowns (lower is better, so invert the calculation)
            if walkforward['max_drawdown_mean'] < 0:
                walkforward['drawdown_consistency'] = 1 - min(1, walkforward['max_drawdown_std'] / max(0.01, abs(walkforward['max_drawdown_mean'])))
            else:
                walkforward['drawdown_consistency'] = 0
            
            # Overall consistency score
            walkforward['overall_consistency'] = (
                walkforward['sharpe_consistency'] * 0.4 +
                walkforward['return_consistency'] * 0.4 +
                walkforward['drawdown_consistency'] * 0.2
            )
            
            # Trend metrics
            if len(window_sharpes) > 1:
                # Calculate linear regression to detect trends
                x = np.arange(len(window_sharpes))
                
                # Sharpe trend
                sharpe_slope, _, sharpe_rvalue, sharpe_pvalue, _ = stats.linregress(x, window_sharpes)
                walkforward['sharpe_trend_slope'] = sharpe_slope
                walkforward['sharpe_trend_r_squared'] = sharpe_rvalue ** 2
                walkforward['sharpe_trend_significant'] = sharpe_pvalue < 0.05
                
                # Return trend
                return_slope, _, return_rvalue, return_pvalue, _ = stats.linregress(x, window_returns)
                walkforward['return_trend_slope'] = return_slope
                walkforward['return_trend_r_squared'] = return_rvalue ** 2
                walkforward['return_trend_significant'] = return_pvalue < 0.05
                
                # Drawdown trend
                dd_slope, _, dd_rvalue, dd_pvalue, _ = stats.linregress(x, window_drawdowns)
                walkforward['drawdown_trend_slope'] = dd_slope
                walkforward['drawdown_trend_r_squared'] = dd_rvalue ** 2
                walkforward['drawdown_trend_significant'] = dd_pvalue < 0.05
                
                # Overall trend assessment
                walkforward['performance_improving'] = (
                    (sharpe_slope > 0 and walkforward['sharpe_trend_significant']) or
                    (return_slope > 0 and walkforward['return_trend_significant']) or
                    (dd_slope > 0 and walkforward['drawdown_trend_significant'])  # For drawdowns, positive slope means improving (less negative)
                )
        
        return walkforward
        
    def _evaluate_against_criteria(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate strategy results against specified criteria.
        
        Args:
            results: Dictionary containing evaluation results
            
        Returns:
            Dictionary with criteria evaluation results
        """
        criteria_results = {}
        metrics_to_check = []
        
        # Collect metrics from all result categories
        for category in ['portfolio_metrics', 'drawdown_analysis', 'risk_metrics']:
            if category in results:
                for metric_name, metric_value in results[category].items():
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                        metrics_to_check.append((metric_name, metric_value, category))
        
        # Process criteria
        met_criteria = []
        unmet_criteria = []
        
        for criterion_name, criterion_value in self.criteria.items():
            # Find matching metric
            matching_metrics = [m for m in metrics_to_check if m[0] == criterion_name]
            
            if not matching_metrics:
                continue
                
            metric_name, metric_value, category = matching_metrics[0]
            
            # Determine if criterion is met
            criterion_met = False
            
            # For metrics where higher is better
            if metric_name in ['cagr', 'sharpe_ratio', 'sortino_ratio', 'win_rate', 
                              'profit_factor', 'calmar_ratio', 'information_ratio',
                              'recovery_efficiency', 'return_risk_ratio', 'starr_ratio',
                              'omega_ratio', 'gain_to_pain_ratio', 'martin_ratio',
                              'pain_ratio', 'sterling_ratio', 'burke_ratio']:
                criterion_met = metric_value >= criterion_value
                
            # For metrics where lower is better
            elif metric_name in ['max_drawdown', 'ulcer_index', 'pain_index',
                                'downside_deviation', 'fat_tail_measure',
                                'historical_var_99', 'parametric_var_99', 'cvar_99',
                                'time_underwater_pct']:
                criterion_met = metric_value <= criterion_value
            
            # Store result
            result = {
                'metric_name': metric_name,
                'category': category,
                'criterion_value': criterion_value,
                'actual_value': metric_value,
                'criterion_met': criterion_met,
                'distance': metric_value - criterion_value  # Positive = exceeds criterion (for better-is-higher metrics)
            }
            
            if criterion_met:
                met_criteria.append(result)
            else:
                unmet_criteria.append(result)
        
        # Add results
        criteria_results['total_criteria'] = len(met_criteria) + len(unmet_criteria)
        criteria_results['met_criteria_count'] = len(met_criteria)
        criteria_results['unmet_criteria_count'] = len(unmet_criteria)
        criteria_results['success_rate'] = len(met_criteria) / criteria_results['total_criteria'] if criteria_results['total_criteria'] > 0 else 0
        
        criteria_results['met_criteria'] = met_criteria
        criteria_results['unmet_criteria'] = unmet_criteria
        
        # Specific checks for key criteria
        for key_metric in ['cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            if key_metric in self.criteria:
                target = self.criteria[key_metric]
                
                # Find actual value
                actual = None
                for name, value, category in metrics_to_check:
                    if name == key_metric:
                        actual = value
                        break
                
                if actual is not None:
                    criteria_results[f'{key_metric}_target'] = target
                    criteria_results[f'{key_metric}_actual'] = actual
                    criteria_results[f'{key_metric}_met'] = (
                        actual >= target if key_metric != 'max_drawdown' else actual <= target
                    )
        
        # Performance grade based on success rate
        success_rate = criteria_results['success_rate']
        if success_rate >= 0.9:
            criteria_results['grade'] = 'A'
        elif success_rate >= 0.8:
            criteria_results['grade'] = 'B'
        elif success_rate >= 0.7:
            criteria_results['grade'] = 'C'
        elif success_rate >= 0.6:
            criteria_results['grade'] = 'D'
        else:
            criteria_results['grade'] = 'F'
            
        return criteria_results
        
    def _analyze_market_regimes(self,
                             returns: pd.DataFrame,
                             predictions: pd.DataFrame,
                             actuals: pd.DataFrame,
                             trades: pd.DataFrame,
                             benchmark_returns: Optional[pd.DataFrame],
                             market_regimes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze strategy performance across different market regimes.
        
        Args:
            returns: DataFrame of strategy returns
            predictions: DataFrame with strategy predictions
            actuals: DataFrame with actual values
            trades: DataFrame with trade information
            benchmark_returns: Optional DataFrame with benchmark returns
            market_regimes: Dictionary mapping regime names to boolean masks
            
        Returns:
            Dictionary with regime analysis results
        """
        regime_results = {}
        
        # For each market regime, evaluate the strategy on that subset of data
        for regime_name, regime_mask in market_regimes.items():
            self.logger.info(f"Analyzing regime: {regime_name}")
            
            # Skip if the regime has no data
            if not regime_mask.any():
                regime_results[regime_name] = {'error': 'No data in this regime'}
                continue
                
            # Filter data for this regime
            regime_returns = returns[regime_mask]
            regime_predictions = predictions[regime_mask]
            regime_actuals = actuals[regime_mask]
            
            # Filter trades for this regime
            if not trades.empty:
                # Filter trades by date (assumes index is datetime)
                regime_start = regime_returns.index[0]
                regime_end = regime_returns.index[-1]
                regime_trades = trades[(trades.index >= regime_start) & (trades.index <= regime_end)]
            else:
                regime_trades = pd.DataFrame()
            
            # Filter benchmark returns for this regime
            regime_benchmark = None if benchmark_returns is None else benchmark_returns[regime_mask]
            
            # Calculate metrics for this regime
            regime_metrics = {
                'portfolio_metrics': self._calculate_portfolio_metrics(regime_returns, regime_benchmark),
                'ranking_metrics': self._calculate_ranking_metrics(regime_predictions, regime_actuals),
                'error_metrics': self._calculate_error_metrics(regime_predictions, regime_actuals),
                'drawdown_analysis': self._calculate_comprehensive_drawdown_metrics(regime_returns),
                'risk_metrics': self._calculate_advanced_risk_metrics(regime_returns, regime_benchmark)
            }
            
            # Add number of samples in this regime
            regime_metrics['sample_count'] = len(regime_returns)
            regime_metrics['sample_percentage'] = len(regime_returns) / len(returns) * 100
            
            # Store results for this regime
            regime_results[regime_name] = regime_metrics
        
        # Calculate comparative metrics across regimes
        if len(regime_results) > 1:
            # Example metrics to compare: Sharpe ratio, returns, drawdowns
            comparative = {}
            
            key_metrics = [
                ('sharpe_ratio', 'portfolio_metrics'),
                ('sortino_ratio', 'portfolio_metrics'),
                ('cagr', 'portfolio_metrics'),
                ('max_drawdown', 'drawdown_analysis')
            ]
            
            for metric_name, category in key_metrics:
                metric_values = {}
                
                for regime_name, regime_data in regime_results.items():
                    if isinstance(regime_data, dict) and category in regime_data and metric_name in regime_data[category]:
                        metric_values[regime_name] = regime_data[category][metric_name]
                
                if metric_values:
                    best_regime = max(metric_values, key=lambda k: metric_values[k] if metric_name != 'max_drawdown' else -metric_values[k])
                    worst_regime = min(metric_values, key=lambda k: metric_values[k] if metric_name != 'max_drawdown' else -metric_values[k])
                    
                    comparative[f'{metric_name}_by_regime'] = metric_values
                    comparative[f'{metric_name}_best_regime'] = best_regime
                    comparative[f'{metric_name}_worst_regime'] = worst_regime
                    
                    # Calculate ratio of best to worst (for metrics where higher is better)
                    if metric_name != 'max_drawdown' and metric_values[worst_regime] != 0:
                        comparative[f'{metric_name}_best_to_worst_ratio'] = metric_values[best_regime] / metric_values[worst_regime]
            
            # Add consistency metrics
            consistency_scores = {}
            
            for metric_name, category in key_metrics:
                values = [regime_data[category][metric_name] 
                         for regime_name, regime_data in regime_results.items() 
                         if isinstance(regime_data, dict) and category in regime_data and metric_name in regime_data[category]]
                
                if values:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    
                    # For metrics where higher is better (except max_drawdown)
                    if metric_name != 'max_drawdown' and mean_value != 0:
                        consistency = 1 - min(1, std_value / abs(mean_value))
                    # For max_drawdown (lower is better)
                    elif metric_name == 'max_drawdown' and mean_value != 0:
                        consistency = 1 - min(1, std_value / abs(mean_value))
                    else:
                        consistency = 0
                        
                    consistency_scores[f'{metric_name}_consistency'] = consistency
            
            comparative['consistency_scores'] = consistency_scores
            comparative['overall_consistency'] = np.mean(list(consistency_scores.values())) if consistency_scores else 0
            
            # Add robustness score - how well the strategy performs across different regimes
            regime_results['comparative'] = comparative
        
        return regime_results