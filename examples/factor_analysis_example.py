#!/usr/bin/env python
"""
Example script demonstrating the usage of the AlphaFactorAnalyzer module
for automated factor analysis and alpha discovery.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the AlphaFactorAnalyzer
from src.factor_analysis import AlphaFactorAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data():
    """
    Load sample market data for demonstration.
    In a real application, this would load data from files or APIs.
    
    Returns:
        Tuple of (asset_returns, factor_data)
    """
    logger.info("Loading sample data")
    
    # Create a date range for the sample data
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='B')
    
    # Create simulated asset returns for 10 assets
    np.random.seed(42)  # For reproducibility
    asset_names = [f'Asset_{i+1}' for i in range(10)]
    
    # Create base factors that will drive returns
    n_samples = len(dates)
    
    # Create 3 underlying factors
    factor1 = np.random.normal(0, 1, n_samples)  # Market factor
    factor2 = np.random.normal(0, 1, n_samples)  # Size factor
    factor3 = np.random.normal(0, 1, n_samples)  # Value factor
    
    # Add some autocorrelation to factors
    for i in range(1, n_samples):
        factor1[i] = 0.8 * factor1[i] + 0.2 * factor1[i-1]
        factor2[i] = 0.7 * factor2[i] + 0.3 * factor2[i-1]
        factor3[i] = 0.6 * factor3[i] + 0.4 * factor3[i-1]
    
    # Create factor data DataFrame
    factor_data = pd.DataFrame({
        'Market': factor1,
        'Size': factor2,
        'Value': factor3
    }, index=dates)
    
    # Generate asset returns based on factors with different loadings
    asset_returns_data = {}
    
    for i, asset in enumerate(asset_names):
        # Each asset has different exposures to factors
        market_beta = 0.5 + np.random.rand() * 1.0  # Between 0.5 and 1.5
        size_beta = -0.5 + np.random.rand() * 1.0   # Between -0.5 and 0.5
        value_beta = -0.3 + np.random.rand() * 0.6  # Between -0.3 and 0.3
        
        # Generate returns with factor exposures plus idiosyncratic return
        asset_return = (market_beta * factor1 + 
                        size_beta * factor2 + 
                        value_beta * factor3 + 
                        np.random.normal(0, 0.02, n_samples))  # Idiosyncratic returns
        
        asset_returns_data[asset] = asset_return
    
    # Create asset returns DataFrame
    asset_returns = pd.DataFrame(asset_returns_data, index=dates)
    
    logger.info(f"Created sample data with {len(asset_returns)} days and {len(asset_names)} assets")
    
    return asset_returns, factor_data

def main():
    """
    Main function demonstrating factor analysis workflow.
    """
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(__file__), 'factor_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample data
    asset_returns, factor_data = load_sample_data()
    
    # Initialize the AlphaFactorAnalyzer
    config = {
        'output_dir': output_dir
    }
    analyzer = AlphaFactorAnalyzer(config)
    
    # Load data into the analyzer
    analyzer.load_data(asset_returns, factor_data)
    
    # 1. Discover statistical factors using PCA
    logger.info("Discovering statistical factors using PCA")
    pca_results = analyzer.discover_statistical_factors(
        n_components=5, 
        method='pca',
        explained_variance_threshold=0.9
    )
    
    # 2. Discover statistical factors using Factor Analysis
    logger.info("Discovering statistical factors using Factor Analysis")
    fa_results = analyzer.discover_statistical_factors(
        n_components=3, 
        method='factor_analysis'
    )
    
    # 3. Create traditional factors
    logger.info("Creating traditional factors")
    traditional_factors = analyzer.create_traditional_factors()
    
    # 4. Evaluate factor performance
    logger.info("Evaluating factor performance")
    performance = analyzer.evaluate_factor_performance(
        forward_periods=[1, 5, 21, 63]  # 1-day, 1-week, 1-month, 3-month
    )
    
    # 5. Select best alpha factors
    logger.info("Selecting best alpha factors")
    selected_factors = analyzer.select_alpha_factors(
        min_abs_correlation=0.1,
        max_p_value=0.05,
        forward_period=21  # 1-month forward returns
    )
    
    logger.info(f"Selected {len(selected_factors)} factors: {selected_factors}")
    
    # 6. Create combined alpha model
    if selected_factors:
        logger.info("Creating combined alpha model")
        alpha_model = analyzer.create_combined_alpha_model(
            selected_factors=selected_factors,
            forward_period=21
        )
        
        # 7. Generate alpha signals
        logger.info("Generating alpha signals")
        alpha_signals = analyzer.generate_alpha_signals()
        
        # Plot alpha signals
        plt.figure(figsize=(12, 6))
        alpha_signals.plot()
        plt.title('Alpha Signals')
        plt.xlabel('Date')
        plt.ylabel('Predicted Return')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alpha_plot_path = os.path.join(output_dir, f"alpha_signals_{timestamp}.png")
        plt.savefig(alpha_plot_path)
        plt.close()
        
        logger.info(f"Saved alpha signals plot to {alpha_plot_path}")
    
    # 8. Visualize factor analysis results
    logger.info("Visualizing factor analysis results")
    visualization_paths = analyzer.visualize_factor_analysis()
    
    # 9. Save factor analysis report
    logger.info("Saving factor analysis report")
    report_path = analyzer.save_factor_report()
    
    logger.info(f"Factor analysis complete. Results saved to {output_dir}")
    logger.info(f"Factor report saved to {report_path}")
    
    # Print summary of visualizations
    print("\nFactor Analysis Visualizations:")
    for name, path in visualization_paths.items():
        print(f"- {name}: {path}")
    
    # Print selected factors
    print(f"\nSelected Alpha Factors: {selected_factors}")
    
    # If we have an alpha model, print performance
    if hasattr(analyzer, 'alpha_model'):
        print("\nAlpha Model Performance:")
        print(f"R-squared: {analyzer.alpha_model['r_squared']:.4f}")
        print(f"MSE: {analyzer.alpha_model['mse']:.6f}")
        print("\nTop Factors by Importance:")
        
        # Get coefficients and sort by absolute value
        coefficients = analyzer.alpha_model['coefficients'].copy()
        if 'const' in coefficients:
            del coefficients['const']  # Remove intercept
            
        sorted_factors = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for factor, coef in sorted_factors[:5]:  # Show top 5
            print(f"- {factor}: {coef:.6f}")

if __name__ == "__main__":
    main()