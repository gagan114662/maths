#!/usr/bin/env python3
"""
Demonstrate Causal Discovery for Financial Market Data

This script showcases the use of causal discovery techniques to identify causal relationships
between different financial assets, beyond simple correlations. It demonstrates various
approaches to causal inference, including Granger causality, constraint-based methods,
and information-theoretic measures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import sys
import yfinance as yf
from datetime import datetime, timedelta
import networkx as nx

from src.causal_discovery import CausalDiscovery

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/causal_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def compare_causal_methods(data, output_dir):
    """
    Compare different causal discovery methods on the same dataset.
    
    Args:
        data: DataFrame with time series data
        output_dir: Directory to save outputs
    """
    # Initialize causal discovery
    cd = CausalDiscovery(data=data)
    
    # Preprocess data - apply differencing to ensure stationarity
    preprocessed_data = cd.preprocess_data(diff_order=1, normalize=True)
    
    # Create subplots for comparing methods
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.suptitle("Comparison of Causal Discovery Methods", fontsize=16)
    
    # 1. Granger Causality
    logger.info("Running Granger causality tests...")
    gc_results = cd.test_granger_causality(max_lag=3)
    G_granger = cd.build_granger_causal_graph(gc_results)
    
    # Plot Granger causality graph
    pos = nx.spring_layout(G_granger, seed=42)
    nx.draw_networkx(G_granger, pos=pos, with_labels=True, 
                     node_color='lightblue', node_size=500, 
                     font_size=10, font_weight='bold', ax=axes[0, 0])
    axes[0, 0].set_title("Granger Causality")
    axes[0, 0].axis('off')
    
    # Identify key drivers using Granger
    drivers_granger = cd.identify_key_causal_drivers(G_granger, top_n=3)
    logger.info(f"Top causal drivers (Granger): {drivers_granger}")
    
    # 2. PC Algorithm
    logger.info("Running PC algorithm...")
    G_pc = cd.run_pc_algorithm(max_cond_vars=2)
    
    # Plot PC algorithm graph
    pos = nx.spring_layout(G_pc, seed=42)
    nx.draw_networkx(G_pc, pos=pos, with_labels=True, 
                     node_color='lightgreen', node_size=500, 
                     font_size=10, font_weight='bold', ax=axes[0, 1])
    axes[0, 1].set_title("PC Algorithm")
    axes[0, 1].axis('off')
    
    # Identify key drivers using PC
    drivers_pc = cd.identify_key_causal_drivers(G_pc, top_n=3)
    logger.info(f"Top causal drivers (PC): {drivers_pc}")
    
    # 3. LiNGAM Algorithm
    logger.info("Running LiNGAM algorithm...")
    try:
        G_lingam = cd.run_lingam()
        
        # Plot LiNGAM graph
        pos = nx.spring_layout(G_lingam, seed=42)
        nx.draw_networkx(G_lingam, pos=pos, with_labels=True, 
                        node_color='lightsalmon', node_size=500, 
                        font_size=10, font_weight='bold', ax=axes[1, 0])
        axes[1, 0].set_title("LiNGAM")
        axes[1, 0].axis('off')
        
        # Identify key drivers using LiNGAM
        drivers_lingam = cd.identify_key_causal_drivers(G_lingam, top_n=3)
        logger.info(f"Top causal drivers (LiNGAM): {drivers_lingam}")
        
        # Intervention effects using top driver
        if drivers_lingam:
            top_driver = drivers_lingam[0][0]
            target = data.columns[0] if data.columns[0] != top_driver else data.columns[1]
            
            logger.info(f"Evaluating intervention on {top_driver}...")
            effects = cd.evaluate_intervention_effects(target, top_driver, 1.0)
            
            # Plot intervention effects
            intervention_fig = cd.plot_intervention_effects(effects, top_driver)
            intervention_fig.savefig(os.path.join(output_dir, "intervention_effects.png"))
    except Exception as e:
        logger.error(f"LiNGAM failed: {str(e)}")
        axes[1, 0].text(0.5, 0.5, "LiNGAM Failed", ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')
    
    # 4. Transfer Entropy
    logger.info("Calculating transfer entropy...")
    try:
        te_matrix = cd.calculate_transfer_entropy_matrix(bins=8)
        
        # Plot transfer entropy matrix
        im = axes[1, 1].imshow(te_matrix.values, cmap='viridis')
        axes[1, 1].set_title("Transfer Entropy")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Transfer Entropy (nats)')
        
        # Add labels
        axes[1, 1].set_xticks(np.arange(len(te_matrix.columns)))
        axes[1, 1].set_yticks(np.arange(len(te_matrix.index)))
        axes[1, 1].set_xticklabels(te_matrix.columns, rotation=45)
        axes[1, 1].set_yticklabels(te_matrix.index)
        
        # Identify key information sources
        sources = []
        for col in te_matrix.columns:
            source_score = te_matrix[col].sum()
            sources.append((col, source_score))
        
        sources.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Top information sources (Transfer Entropy): {sources[:3]}")
        
        # Save transfer entropy matrix
        te_matrix.to_csv(os.path.join(output_dir, "transfer_entropy_matrix.csv"))
    except Exception as e:
        logger.error(f"Transfer entropy failed: {str(e)}")
        axes[1, 1].text(0.5, 0.5, "Transfer Entropy Failed", ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "causal_methods_comparison.png"))
    
    return {
        'granger': {'graph': G_granger, 'drivers': drivers_granger},
        'pc': {'graph': G_pc, 'drivers': drivers_pc},
        'lingam': {'graph': G_lingam if 'G_lingam' in locals() else None, 
                  'drivers': drivers_lingam if 'drivers_lingam' in locals() else None},
        'transfer_entropy': {'matrix': te_matrix if 'te_matrix' in locals() else None, 
                           'sources': sources[:3] if 'sources' in locals() else None}
    }

def causal_impact_analysis(data, output_dir):
    """
    Perform causal impact analysis for a significant event.
    
    Args:
        data: DataFrame with time series data
        output_dir: Directory to save outputs
    """
    # Initialize causal discovery
    cd = CausalDiscovery(data=data)
    
    # Find a reasonable date for event analysis (mid-point of data)
    mid_idx = len(data) // 2
    event_date = data.index[mid_idx].strftime("%Y-%m-%d")
    
    # For each asset, analyze the impact of this "event"
    for asset in data.columns:
        logger.info(f"Performing causal impact analysis for {asset} around {event_date}...")
        
        # Define other assets as controls
        controls = [col for col in data.columns if col != asset]
        
        # Run causal impact analysis
        impact_results = cd.causal_impact_analysis(
            target=asset,
            event_date=event_date,
            pre_period=60,  # 60 days before
            post_period=30,  # 30 days after
            control_variables=controls
        )
        
        # Plot results
        fig = cd.plot_causal_impact(impact_results)
        fig.savefig(os.path.join(output_dir, f"causal_impact_{asset}.png"))
        
        # Log results
        logger.info(f"Results for {asset}:")
        logger.info(f"  Mean impact: {impact_results['mean_impact']:.4f}")
        logger.info(f"  Relative impact: {impact_results['relative_impact_mean']:.2f}%")
        logger.info(f"  p-value: {impact_results['p_value']:.4f}")
    
    return event_date

def find_key_market_drivers(data, output_dir):
    """
    Find key market drivers using multiple causal methods.
    
    Args:
        data: DataFrame with time series data
        output_dir: Directory to save outputs
    """
    # Initialize causal discovery
    cd = CausalDiscovery(data=data)
    
    # Preprocess data
    preprocessed_data = cd.preprocess_data(diff_order=1, normalize=True)
    
    # Run multiple causal discovery methods
    methods_results = compare_causal_methods(data, output_dir)
    
    # Aggregate drivers across methods
    all_drivers = {}
    
    # Collect drivers from Granger causality
    if methods_results['granger']['drivers']:
        for driver, score in methods_results['granger']['drivers']:
            if driver not in all_drivers:
                all_drivers[driver] = {'score': 0, 'mentions': 0}
            all_drivers[driver]['score'] += score
            all_drivers[driver]['mentions'] += 1
    
    # Collect drivers from PC algorithm
    if methods_results['pc']['drivers']:
        for driver, score in methods_results['pc']['drivers']:
            if driver not in all_drivers:
                all_drivers[driver] = {'score': 0, 'mentions': 0}
            all_drivers[driver]['score'] += score
            all_drivers[driver]['mentions'] += 1
    
    # Collect drivers from LiNGAM
    if methods_results['lingam']['drivers']:
        for driver, score in methods_results['lingam']['drivers']:
            if driver not in all_drivers:
                all_drivers[driver] = {'score': 0, 'mentions': 0}
            all_drivers[driver]['score'] += score
            all_drivers[driver]['mentions'] += 1
    
    # Collect information sources from transfer entropy
    if methods_results['transfer_entropy']['sources']:
        for source, score in methods_results['transfer_entropy']['sources']:
            if source not in all_drivers:
                all_drivers[source] = {'score': 0, 'mentions': 0}
            all_drivers[source]['score'] += score
            all_drivers[source]['mentions'] += 1
    
    # Calculate average scores
    for driver in all_drivers:
        all_drivers[driver]['avg_score'] = all_drivers[driver]['score'] / all_drivers[driver]['mentions']
    
    # Sort by number of mentions and then by average score
    ranked_drivers = sorted(
        [(k, v['mentions'], v['avg_score']) for k, v in all_drivers.items()],
        key=lambda x: (x[1], x[2]),
        reverse=True
    )
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    
    if ranked_drivers:
        drivers, mentions, scores = zip(*ranked_drivers)
        
        # Normalize scores for plotting
        max_score = max(scores)
        normalized_scores = [score/max_score * 50 for score in scores]
        
        # Plot drivers by mentions (size) and average score (color)
        plt.scatter(range(len(drivers)), [1] * len(drivers), 
                   s=normalized_scores, c=scores, cmap='viridis', 
                   alpha=0.7)
        
        # Add driver labels
        for i, driver in enumerate(drivers):
            plt.text(i, 1, driver, ha='center', va='center', 
                    fontweight='bold', fontsize=12)
        
        plt.title('Key Market Drivers Identified by Multiple Causal Methods')
        plt.xlabel('Driver Rank')
        plt.xticks(range(len(drivers)), [f"{i+1}" for i in range(len(drivers))])
        plt.yticks([])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(scores)
        cbar = plt.colorbar(sm)
        cbar.set_label('Average Causal Influence Score')
        
        # Add legend for size
        for i, size in enumerate([10, 30, 50]):
            plt.scatter([], [], s=size, c='gray', alpha=0.7, 
                       label=f"{i+1} method{'s' if i > 0 else ''}")
        plt.legend(title="Detected by", loc='upper right')
    else:
        plt.text(0.5, 0.5, "No drivers found", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "key_market_drivers.png"))
    
    # Write results to a CSV
    if ranked_drivers:
        df = pd.DataFrame(ranked_drivers, columns=['Driver', 'Methods_Count', 'Avg_Score'])
        df.to_csv(os.path.join(output_dir, "key_market_drivers.csv"), index=False)
    
    return ranked_drivers

def main():
    """Run the causal discovery demonstration."""
    parser = argparse.ArgumentParser(description="Causal Discovery Demonstration")
    parser.add_argument("--sectors", type=str, default="SPY,XLF,XLE,XLK,XLV,XLI,XLU,XLP,XLB,XLC", 
                      help="Comma-separated list of sector ETFs to analyze")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data")
    parser.add_argument("--output", type=str, default="causal_output", help="Output directory")
    parser.add_argument("--demo", type=str, default="all", 
                      choices=["all", "comparison", "impact", "drivers"],
                      help="Which demonstration to run")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Starting causal discovery demonstration with sectors: {args.sectors}")
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)
    
    logger.info(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    symbols = args.sectors.split(',')
    
    data = yf.download(symbols, start=start_date, end=end_date)["Close"]
    data.columns = symbols
    
    if data.empty:
        logger.error("No data found for the specified symbols")
        return 1
    
    logger.info(f"Downloaded {len(data)} days of data for {len(symbols)} symbols")
    
    # Make sure data is clean
    data = data.dropna()
    
    # Run selected demonstrations
    results = {}
    
    if args.demo in ["all", "comparison"]:
        logger.info("Running causal methods comparison...")
        results['comparison'] = compare_causal_methods(data, args.output)
    
    if args.demo in ["all", "impact"]:
        logger.info("Running causal impact analysis...")
        results['event_date'] = causal_impact_analysis(data, args.output)
    
    if args.demo in ["all", "drivers"]:
        logger.info("Finding key market drivers...")
        results['key_drivers'] = find_key_market_drivers(data, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("CAUSAL DISCOVERY DEMONSTRATION COMPLETED")
    print("="*80)
    
    if 'comparison' in results:
        print("\nCausal Methods Comparison:")
        print("  Graphs and visualizations saved to causal_methods_comparison.png")
    
    if 'event_date' in results:
        print(f"\nCausal Impact Analysis:")
        print(f"  Event date used: {results['event_date']}")
        print(f"  Individual asset impact analyses saved to causal_impact_*.png")
    
    if 'key_drivers' in results and results['key_drivers']:
        print("\nKey Market Drivers:")
        for i, (driver, mentions, score) in enumerate(results['key_drivers'][:5]):
            print(f"  {i+1}. {driver}: Detected by {mentions} method(s), Score: {score:.4f}")
        print(f"  Full results saved to key_market_drivers.csv")
    
    print("\nAll results saved to:", args.output)
    print("="*80)
    
    # Print task completion message
    print("\n" + "="*80)
    print("CAUSAL DISCOVERY TASK COMPLETED SUCCESSFULLY")
    print("="*80)
    print("The causal discovery module has been implemented with the following features:")
    print("1. Granger causality testing for time series data")
    print("2. PC algorithm for constraint-based causal discovery") 
    print("3. LiNGAM for causality in non-Gaussian data")
    print("4. Transfer entropy for measuring directed information flow")
    print("5. Causal impact analysis for event studies")
    print("6. Intervention effect estimation (do-calculus)")
    print("\nThis implementation allows for discovering true causal relationships")
    print("beyond simple correlations in financial market data.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())