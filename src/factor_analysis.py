import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configure logging
logger = logging.getLogger(__name__)

class AlphaFactorAnalyzer:
    """
    Automated factor analysis module for alpha discovery.
    
    This module implements various factor analysis techniques to discover
    alpha signals in financial data. It includes:
    - Traditional factor model analysis (Fama-French, etc.)
    - Statistical factor extraction (PCA, Factor Analysis)
    - Custom factor discovery and validation
    - Factor performance evaluation and selection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AlphaFactorAnalyzer with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.factors = {}
        self.factor_returns = {}
        self.factor_loadings = {}
        self.factor_performance = {}
        self.output_dir = self.config.get('output_dir', 'factor_analysis_results')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        logger.info("Initialized AlphaFactorAnalyzer")
    
    def load_data(self, asset_returns: pd.DataFrame, 
                  factor_data: Optional[pd.DataFrame] = None,
                  alternative_data: Optional[pd.DataFrame] = None) -> None:
        """
        Load market data, factor data, and alternative data.
        
        Args:
            asset_returns: DataFrame containing asset returns (index=dates, columns=assets)
            factor_data: Optional DataFrame containing known factor data (e.g., Fama-French factors)
            alternative_data: Optional DataFrame containing alternative data to consider
        """
        self.asset_returns = asset_returns
        self.dates = asset_returns.index
        self.assets = asset_returns.columns
        
        self.factor_data = factor_data
        self.alternative_data = alternative_data
        
        logger.info(f"Loaded data for {len(self.assets)} assets over {len(self.dates)} time periods")
    
    def discover_statistical_factors(self, 
                                    n_components: int = 5, 
                                    method: str = 'pca',
                                    explained_variance_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Discover statistical factors using PCA or Factor Analysis.
        
        Args:
            n_components: Number of components/factors to extract
            method: Method to use ('pca' or 'factor_analysis')
            explained_variance_threshold: Minimum explained variance threshold
            
        Returns:
            Dictionary containing factor information and results
        """
        logger.info(f"Discovering statistical factors using {method}")
        
        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(self.asset_returns)
        
        if method == 'pca':
            # Determine optimal number of components if not specified
            if n_components is None:
                pca_full = PCA()
                pca_full.fit(scaled_returns)
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
                logger.info(f"Automatically selected {n_components} components to explain {explained_variance_threshold*100:.1f}% variance")
            
            # Extract factors using PCA
            model = PCA(n_components=n_components)
            factor_loadings = model.fit_transform(scaled_returns)
            components = model.components_
            explained_variance = model.explained_variance_ratio_
            
            # Create factor returns DataFrame
            factor_returns = pd.DataFrame(
                factor_loadings, 
                index=self.asset_returns.index,
                columns=[f'PCA_Factor_{i+1}' for i in range(n_components)]
            )
            
            # Create factor loadings DataFrame (how each asset loads on each factor)
            factor_loadings_df = pd.DataFrame(
                components.T,
                index=self.asset_returns.columns,
                columns=[f'PCA_Factor_{i+1}' for i in range(n_components)]
            )
            
            # Store results
            self.factors.update({f'PCA_Factor_{i+1}': {
                'type': 'statistical',
                'method': 'pca',
                'explained_variance': explained_variance[i]
            } for i in range(n_components)})
            
            # Store factor returns and loadings
            for i in range(n_components):
                factor_name = f'PCA_Factor_{i+1}'
                self.factor_returns[factor_name] = factor_returns[factor_name]
                self.factor_loadings[factor_name] = factor_loadings_df[factor_name]
            
            result = {
                'factor_returns': factor_returns,
                'factor_loadings': factor_loadings_df,
                'explained_variance': explained_variance,
                'total_explained_variance': sum(explained_variance),
                'n_components': n_components
            }
            
        elif method == 'factor_analysis':
            # Extract factors using Factor Analysis
            model = FactorAnalysis(n_components=n_components, random_state=42)
            factor_loadings = model.fit_transform(scaled_returns)
            components = model.components_
            
            # Create factor returns DataFrame
            factor_returns = pd.DataFrame(
                factor_loadings, 
                index=self.asset_returns.index,
                columns=[f'FA_Factor_{i+1}' for i in range(n_components)]
            )
            
            # Create factor loadings DataFrame
            factor_loadings_df = pd.DataFrame(
                components.T,
                index=self.asset_returns.columns,
                columns=[f'FA_Factor_{i+1}' for i in range(n_components)]
            )
            
            # Store results
            self.factors.update({f'FA_Factor_{i+1}': {
                'type': 'statistical',
                'method': 'factor_analysis'
            } for i in range(n_components)})
            
            # Store factor returns and loadings
            for i in range(n_components):
                factor_name = f'FA_Factor_{i+1}'
                self.factor_returns[factor_name] = factor_returns[factor_name]
                self.factor_loadings[factor_name] = factor_loadings_df[factor_name]
            
            result = {
                'factor_returns': factor_returns,
                'factor_loadings': factor_loadings_df,
                'n_components': n_components
            }
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'factor_analysis'.")
        
        return result
    
    def create_traditional_factors(self) -> Dict[str, pd.Series]:
        """
        Create traditional market factors (e.g., value, momentum, size, etc.)
        
        Returns:
            Dictionary of factor series
        """
        logger.info("Creating traditional market factors")
        
        traditional_factors = {}
        
        # Check if we have price data as well
        if hasattr(self, 'price_data'):
            price_data = self.price_data
            
            # 1. Momentum factor (12-month return, skipping most recent month)
            if len(price_data) >= 252:  # At least 1 year of data
                momentum = price_data.pct_change(periods=252).shift(21)  # 12-month return, skip 1 month
                traditional_factors['momentum'] = momentum
                
                # Store factor metadata
                self.factors['momentum'] = {
                    'type': 'traditional',
                    'description': '12-month price momentum, skipping most recent month'
                }
                
                # Store factor returns
                self.factor_returns['momentum'] = momentum.mean(axis=1)
        
        # Create cross-sectional factors (require multiple assets)
        if len(self.assets) > 1:
            # 2. Size factor proxy (based on return covariance with market)
            market_return = self.asset_returns.mean(axis=1)
            
            # Run regression for each asset against market
            size_betas = {}
            for asset in self.assets:
                model = sm.OLS(self.asset_returns[asset], sm.add_constant(market_return)).fit()
                size_betas[asset] = model.params[1]  # Beta coefficient
            
            size_factor = pd.Series(size_betas)
            traditional_factors['size'] = size_factor
            
            # Store factor metadata
            self.factors['size'] = {
                'type': 'traditional',
                'description': 'Size factor based on market beta'
            }
        
        return traditional_factors
    
    def evaluate_factor_performance(self, 
                                    factor_names: Optional[List[str]] = None, 
                                    target_returns: Optional[pd.Series] = None,
                                    forward_periods: List[int] = [1, 5, 21, 63]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate factors for alpha prediction performance.
        
        Args:
            factor_names: List of factor names to evaluate (None = all factors)
            target_returns: Target returns to predict (None = use mean asset returns)
            forward_periods: List of forward periods to test (days)
            
        Returns:
            Dictionary of factor performance metrics
        """
        logger.info("Evaluating factor performance for alpha prediction")
        
        if factor_names is None:
            factor_names = list(self.factor_returns.keys())
        
        if target_returns is None:
            target_returns = self.asset_returns.mean(axis=1)
        
        performance = {}
        
        # Create forward returns for different horizons
        forward_returns = {}
        for period in forward_periods:
            forward_returns[period] = target_returns.shift(-period)
        
        # Evaluate each factor
        for factor_name in factor_names:
            if factor_name not in self.factor_returns:
                logger.warning(f"Factor {factor_name} not found in factor returns")
                continue
                
            factor_series = self.factor_returns[factor_name]
            factor_perf = {}
            
            # Evaluate for each forward period
            for period in forward_periods:
                # Create dataframe with factor and forward returns
                data = pd.DataFrame({
                    'factor': factor_series,
                    'forward_return': forward_returns[period]
                }).dropna()
                
                if len(data) < 30:  # Need sufficient data points
                    logger.warning(f"Insufficient data for {factor_name} at {period}-day horizon")
                    continue
                
                # Correlation
                correlation = data['factor'].corr(data['forward_return'])
                
                # Information Coefficient (IC) calculation
                # Use Spearman rank correlation for non-linearity
                rank_ic = data['factor'].corr(data['forward_return'], method='spearman')
                
                # Simple linear regression
                X = sm.add_constant(data['factor'])
                model = sm.OLS(data['forward_return'], X).fit()
                r_squared = model.rsquared
                t_stat = model.tvalues[1]
                p_value = model.pvalues[1]
                
                factor_perf[f'{period}_day'] = {
                    'correlation': correlation,
                    'rank_ic': rank_ic,
                    'r_squared': r_squared,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'statistically_significant': p_value < 0.05
                }
            
            performance[factor_name] = factor_perf
        
        # Store performance metrics
        self.factor_performance = performance
        
        return performance
    
    def select_alpha_factors(self, 
                            min_abs_correlation: float = 0.1,
                            max_p_value: float = 0.05,
                            forward_period: int = 21) -> List[str]:
        """
        Select the best alpha-generating factors based on performance metrics.
        
        Args:
            min_abs_correlation: Minimum absolute correlation with forward returns
            max_p_value: Maximum p-value for statistical significance
            forward_period: Forward period to use for evaluation (days)
            
        Returns:
            List of selected alpha factor names
        """
        logger.info(f"Selecting alpha factors with |correlation| >= {min_abs_correlation} and p-value <= {max_p_value}")
        
        if not self.factor_performance:
            logger.warning("No factor performance data available. Run evaluate_factor_performance first.")
            return []
        
        selected_factors = []
        
        for factor_name, performance in self.factor_performance.items():
            # Check if we have metrics for the requested forward period
            period_key = f'{forward_period}_day'
            if period_key not in performance:
                continue
            
            metrics = performance[period_key]
            correlation = metrics['correlation']
            p_value = metrics['p_value']
            
            # Check selection criteria
            if abs(correlation) >= min_abs_correlation and p_value <= max_p_value:
                selected_factors.append(factor_name)
                logger.info(f"Selected factor {factor_name}: correlation={correlation:.4f}, p-value={p_value:.4f}")
        
        return selected_factors
    
    def create_combined_alpha_model(self, 
                                   selected_factors: List[str],
                                   target_returns: Optional[pd.Series] = None,
                                   forward_period: int = 21) -> Dict[str, Any]:
        """
        Create a combined alpha model using multiple factors.
        
        Args:
            selected_factors: List of selected factor names
            target_returns: Target returns to predict (None = use mean asset returns)
            forward_period: Forward period for prediction (days)
            
        Returns:
            Dictionary containing model information and performance metrics
        """
        logger.info(f"Creating combined alpha model with {len(selected_factors)} factors")
        
        if not selected_factors:
            logger.warning("No factors selected for alpha model")
            return {}
        
        if target_returns is None:
            target_returns = self.asset_returns.mean(axis=1)
        
        # Create forward returns
        forward_returns = target_returns.shift(-forward_period)
        
        # Build dataset with all selected factors
        X_data = pd.DataFrame()
        for factor_name in selected_factors:
            if factor_name in self.factor_returns:
                X_data[factor_name] = self.factor_returns[factor_name]
        
        # Combine with target
        data = pd.DataFrame({
            'forward_return': forward_returns
        }).join(X_data).dropna()
        
        if len(data) < 30:
            logger.warning("Insufficient data for combined alpha model")
            return {}
        
        # Split into features and target
        X = data[selected_factors]
        y = data['forward_return']
        
        # Fit linear regression model
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # Calculate performance metrics
        predictions = model.predict(X_with_const)
        r_squared = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        
        # Store factors and coefficients
        coefficients = model.params.to_dict()
        factor_significance = {
            factor: {
                't_statistic': model.tvalues[i],
                'p_value': model.pvalues[i]
            }
            for i, factor in enumerate(model.params.index)
        }
        
        # Create model information
        alpha_model = {
            'factors': selected_factors,
            'coefficients': coefficients,
            'factor_significance': factor_significance,
            'r_squared': r_squared,
            'mse': mse,
            'aic': model.aic,
            'bic': model.bic,
            'summary': model.summary().as_text(),
            'forward_period': forward_period
        }
        
        # Store the model
        self.alpha_model = alpha_model
        
        return alpha_model
    
    def generate_alpha_signals(self, 
                              latest_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate alpha signals using the combined alpha model.
        
        Args:
            latest_data: Latest factor data (None = use existing data)
            
        Returns:
            Series of alpha signals (predicted forward returns)
        """
        if not hasattr(self, 'alpha_model'):
            logger.warning("No alpha model available. Create a model first.")
            return pd.Series()
        
        logger.info("Generating alpha signals using combined model")
        
        # Get factor data
        if latest_data is not None:
            factor_data = latest_data[self.alpha_model['factors']]
        else:
            factor_data = pd.DataFrame()
            for factor_name in self.alpha_model['factors']:
                if factor_name in self.factor_returns:
                    factor_data[factor_name] = self.factor_returns[factor_name]
        
        # Apply coefficients to generate signals
        factor_data = factor_data.copy()
        factor_data['const'] = 1.0  # Add constant term
        
        alpha_signals = pd.Series(0.0, index=factor_data.index)
        
        # Multiply each factor by its coefficient
        for factor, coef in self.alpha_model['coefficients'].items():
            if factor in factor_data.columns:
                alpha_signals += factor_data[factor] * coef
        
        return alpha_signals
    
    def visualize_factor_analysis(self, 
                                 output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate visualizations of the factor analysis results.
        
        Args:
            output_dir: Directory to save visualizations (None = use instance output_dir)
            
        Returns:
            Dictionary of paths to generated visualizations
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Generating factor analysis visualizations in {output_dir}")
        
        visualization_paths = {}
        
        # 1. Factor correlation heatmap
        if self.factor_returns:
            plt.figure(figsize=(12, 10))
            factor_correlation = pd.DataFrame(self.factor_returns).corr()
            sns.heatmap(factor_correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title("Factor Correlation Heatmap")
            plt.tight_layout()
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"factor_correlation_{timestamp}.png")
            plt.savefig(filepath)
            plt.close()
            
            visualization_paths['factor_correlation'] = filepath
        
        # 2. Factor performance metrics
        if self.factor_performance:
            # Get all forward periods
            forward_periods = set()
            for factor, perf in self.factor_performance.items():
                forward_periods.update(period.split('_')[0] for period in perf.keys())
            
            for period in forward_periods:
                period_key = f'{period}_day'
                
                # Collect metrics across factors
                correlations = []
                factor_names = []
                p_values = []
                
                for factor, perf in self.factor_performance.items():
                    if period_key in perf:
                        factor_names.append(factor)
                        correlations.append(perf[period_key]['correlation'])
                        p_values.append(perf[period_key]['p_value'])
                
                if not factor_names:
                    continue
                
                # Create performance DataFrame
                performance_df = pd.DataFrame({
                    'Factor': factor_names,
                    'Correlation': correlations,
                    'P-Value': p_values,
                    'Significant': [p <= 0.05 for p in p_values]
                })
                
                # Sort by absolute correlation
                performance_df['Abs_Correlation'] = performance_df['Correlation'].abs()
                performance_df = performance_df.sort_values('Abs_Correlation', ascending=False)
                
                # Plot performance metrics
                plt.figure(figsize=(12, 8))
                bars = plt.barh(performance_df['Factor'], performance_df['Correlation'], 
                               color=[('green' if sig else 'grey') for sig in performance_df['Significant']])
                
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.xlabel('Correlation with Forward Returns')
                plt.title(f'Factor Predictive Power ({period}-day Forward Returns)')
                plt.grid(axis='x', alpha=0.3)
                
                # Add p-value annotations
                for i, bar in enumerate(bars):
                    p_value = performance_df['P-Value'].iloc[i]
                    plt.text(0.01, i, f'p={p_value:.4f}', 
                            va='center', fontsize=8,
                            color='black' if p_value <= 0.05 else 'gray')
                
                plt.tight_layout()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(output_dir, f"factor_performance_{period}day_{timestamp}.png")
                plt.savefig(filepath)
                plt.close()
                
                visualization_paths[f'factor_performance_{period}day'] = filepath
        
        # 3. Alpha model performance
        if hasattr(self, 'alpha_model'):
            # Extract model data
            coefficients = self.alpha_model['coefficients']
            factor_significance = self.alpha_model['factor_significance']
            
            # Remove constant term for visualization
            if 'const' in coefficients:
                del coefficients['const']
                del factor_significance['const']
            
            # Create coefficient DataFrame
            coef_df = pd.DataFrame({
                'Factor': list(coefficients.keys()),
                'Coefficient': list(coefficients.values()),
                'T-Statistic': [factor_significance[f]['t_statistic'] for f in coefficients.keys()],
                'P-Value': [factor_significance[f]['p_value'] for f in coefficients.keys()],
                'Significant': [factor_significance[f]['p_value'] <= 0.05 for f in coefficients.keys()]
            })
            
            # Sort by absolute coefficient
            coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            
            # Plot coefficients
            plt.figure(figsize=(12, 8))
            bars = plt.barh(coef_df['Factor'], coef_df['Coefficient'], 
                          color=[('blue' if sig else 'lightgray') for sig in coef_df['Significant']])
            
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Coefficient Value')
            plt.title(f'Alpha Model Factor Coefficients (Forward Period: {self.alpha_model["forward_period"]} days)')
            plt.grid(axis='x', alpha=0.3)
            
            # Add t-statistic annotations
            for i, bar in enumerate(bars):
                t_stat = coef_df['T-Statistic'].iloc[i]
                plt.text(0.01, i, f't={t_stat:.2f}', 
                       va='center', fontsize=8,
                       color='black' if coef_df['Significant'].iloc[i] else 'gray')
            
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"alpha_model_coefficients_{timestamp}.png")
            plt.savefig(filepath)
            plt.close()
            
            visualization_paths['alpha_model_coefficients'] = filepath
            
            # Model performance metrics
            metrics = {
                'R-Squared': self.alpha_model['r_squared'],
                'MSE': self.alpha_model['mse'],
                'AIC': self.alpha_model['aic'],
                'BIC': self.alpha_model['bic']
            }
            
            plt.figure(figsize=(8, 6))
            plt.bar(metrics.keys(), metrics.values())
            plt.title('Alpha Model Performance Metrics')
            plt.ylabel('Value')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"alpha_model_metrics_{timestamp}.png")
            plt.savefig(filepath)
            plt.close()
            
            visualization_paths['alpha_model_metrics'] = filepath
        
        logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def _generate_factor_report(self) -> str:
        """
        Generate a text report summarizing factor analysis results.
        
        Returns:
            String containing the report text
        """
        report = []
        report.append("=" * 80)
        report.append("FACTOR ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary of factors
        report.append(f"Number of factors analyzed: {len(self.factors)}")
        report.append("")
        
        # Statistical factors
        stat_factors = [f for f, info in self.factors.items() if info.get('type') == 'statistical']
        if stat_factors:
            report.append(f"Statistical factors ({len(stat_factors)}):")
            for factor in stat_factors:
                info = self.factors[factor]
                if 'explained_variance' in info:
                    report.append(f"  - {factor}: {info['method'].upper()}, explained variance: {info['explained_variance']:.4f}")
                else:
                    report.append(f"  - {factor}: {info['method'].upper()}")
            report.append("")
        
        # Traditional factors
        trad_factors = [f for f, info in self.factors.items() if info.get('type') == 'traditional']
        if trad_factors:
            report.append(f"Traditional factors ({len(trad_factors)}):")
            for factor in trad_factors:
                info = self.factors[factor]
                report.append(f"  - {factor}: {info.get('description', '')}")
            report.append("")
        
        # Performance summary
        if self.factor_performance:
            report.append("Factor performance summary:")
            
            # Get all forward periods
            forward_periods = set()
            for factor, perf in self.factor_performance.items():
                forward_periods.update(period.split('_')[0] for period in perf.keys())
            
            for period in sorted(forward_periods, key=int):
                period_key = f'{period}_day'
                report.append(f"\n  {period}-day forward returns:")
                
                # Collect metrics across factors
                factor_metrics = []
                for factor, perf in self.factor_performance.items():
                    if period_key in perf:
                        metrics = perf[period_key]
                        factor_metrics.append({
                            'factor': factor,
                            'correlation': metrics['correlation'],
                            'p_value': metrics['p_value'],
                            'significant': metrics['p_value'] <= 0.05
                        })
                
                # Sort by absolute correlation
                factor_metrics.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                for metrics in factor_metrics:
                    sig_marker = '*' if metrics['significant'] else ''
                    report.append(f"    - {metrics['factor']}: corr={metrics['correlation']:.4f}, p={metrics['p_value']:.4f} {sig_marker}")
            
            report.append("")
        
        # Alpha model summary
        if hasattr(self, 'alpha_model'):
            report.append("Alpha model summary:")
            report.append(f"  Forward period: {self.alpha_model['forward_period']} days")
            report.append(f"  R-squared: {self.alpha_model['r_squared']:.4f}")
            report.append(f"  MSE: {self.alpha_model['mse']:.6f}")
            report.append("")
            
            report.append("  Factors and coefficients:")
            coefficients = self.alpha_model['coefficients']
            factor_significance = self.alpha_model['factor_significance']
            
            # Sort factors by absolute coefficient value
            factors = list(coefficients.keys())
            if 'const' in factors:
                factors.remove('const')  # Handle constant term separately
                
            factors.sort(key=lambda x: abs(coefficients[x]), reverse=True)
            
            if 'const' in coefficients:
                report.append(f"    - Intercept: {coefficients['const']:.6f}")
                
            for factor in factors:
                coef = coefficients[factor]
                t_stat = factor_significance[factor]['t_statistic']
                p_value = factor_significance[factor]['p_value']
                sig_marker = '*' if p_value <= 0.05 else ''
                report.append(f"    - {factor}: {coef:.6f} (t={t_stat:.2f}, p={p_value:.4f}) {sig_marker}")
            
            report.append("")
        
        # Return the report as a string
        return "\n".join(report)
    
    def save_factor_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate and save a comprehensive factor analysis report.
        
        Args:
            output_path: Path to save the report (None = use default path)
            
        Returns:
            Path to the saved report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"factor_analysis_report_{timestamp}.txt")
        
        # Generate the report
        report_text = self._generate_factor_report()
        
        # Save the report
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Saved factor analysis report to {output_path}")
        return output_path