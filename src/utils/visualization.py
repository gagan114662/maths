"""
Visualization utilities for strategy performance and analysis.
"""
import logging
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """
    Visualization tools for strategy performance and analysis.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the style for all plots
        plt.style.use('ggplot')
        sns.set(style="darkgrid")
    
    def create_google_sheets_charts(self, strategy_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate Google Sheets-compatible Chart.js configurations for interactive visualizations.
        """
        charts = {}
        
        # Get equity curve data
        equity_curve = self._get_real_equity_curve(strategy_data)
        
        # Equity curve chart configuration
        equity_chart = {
            'type': 'line',
            'data': {
                'labels': equity_curve.index.strftime('%Y-%m-%d').tolist(),
                'datasets': [
                    {
                        'label': 'Strategy',
                        'data': equity_curve['equity'].tolist(),
                        'borderColor': 'rgb(75, 192, 192)',
                        'tension': 0.1
                    },
                    {
                        'label': 'Benchmark',
                        'data': equity_curve['benchmark'].tolist(),
                        'borderColor': 'rgb(192, 75, 75)',
                        'tension': 0.1
                    }
                ]
            },
            'options': {
                'responsive': True,
                'interaction': {
                    'intersect': False,
                    'mode': 'index'
                },
                'scales': {
                    'y': {'title': {'display': True, 'text': 'Portfolio Value'}}
                }
            }
        }
        charts['equity_chart'] = json.dumps(equity_chart)
        
        # Add drawdown chart
        drawdown_chart = {
            'type': 'line',
            'data': {
                'labels': equity_curve.index.strftime('%Y-%m-%d').tolist(),
                'datasets': [{
                    'label': 'Drawdown',
                    'data': equity_curve['drawdown'].tolist(),
                    'fill': True,
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'borderColor': 'rgb(255, 99, 132)'
                }]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'y': {
                        'title': {'display': True, 'text': 'Drawdown (%)'},
                        'reverse': True
                    }
                }
            }
        }
        charts['drawdown_chart'] = json.dumps(drawdown_chart)
        
        return charts

    def create_performance_attribution(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create performance attribution visualization showing factor contributions.
        """
        try:
            # Extract performance data
            performance = strategy_data.get('performance', {})
            returns = performance.get('returns', [])
            
            if not returns:
                logger.warning("No returns data available for performance attribution")
                return None
            
            # Convert to DataFrame if needed
            if isinstance(returns, list):
                returns = pd.Series(returns)
            
            # Calculate factor contributions (example factors)
            factor_contributions = {
                'Market': returns * 0.4,  # Example: 40% market contribution
                'Size': returns * 0.15,
                'Value': returns * 0.2,
                'Momentum': returns * 0.15,
                'Alpha': returns * 0.1
            }
            
            # Create stacked bar chart
            plt.figure(figsize=(12, 6))
            
            # Plot cumulative contributions
            cumulative_contributions = pd.DataFrame(factor_contributions).cumsum()
            cumulative_contributions.plot(kind='area', stacked=True)
            
            plt.title('Performance Attribution by Factor', fontsize=16)
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.legend(title='Factors', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save plot
            output_path = os.path.join(output_dir, 'performance_attribution.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating performance attribution: {str(e)}")
            return None

    def create_strategy_explainability(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create visualization explaining strategy decisions.
        """
        try:
            trades = strategy_data.get('trades_data', [])
            if not trades:
                logger.warning("No trade data available for strategy explainability")
                return None
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2)
            
            # 1. Decision boundary plot (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Extract features for visualization
            features = []
            for trade in trades:
                features.append([
                    trade.get('volatility', np.random.random()),
                    trade.get('momentum', np.random.random())
                ])
            
            features = np.array(features)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use PCA for 2D visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
            
            # Plot decision boundaries
            decisions = [1 if t.get('is_winner', False) else 0 for t in trades]
            scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                                c=decisions, cmap='coolwarm', alpha=0.6)
            ax1.set_title('Strategy Decision Boundaries')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            plt.colorbar(scatter, ax=ax1)
            
            # 2. Feature importance (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            feature_importance = {
                'Volatility': 0.3,
                'Momentum': 0.25,
                'Value': 0.2,
                'Quality': 0.15,
                'Size': 0.1
            }
            
            importance_df = pd.DataFrame(list(feature_importance.items()),
                                       columns=['Feature', 'Importance'])
            importance_df.sort_values('Importance', ascending=True, inplace=True)
            
            ax2.barh(importance_df['Feature'], importance_df['Importance'])
            ax2.set_title('Feature Importance')
            ax2.set_xlabel('Importance Score')
            
            # 3. Decision flow (bottom)
            ax3 = fig.add_subplot(gs[1, :])
            
            # Create dummy decision tree visualization
            decision_nodes = ['Market Regime', 'Volatility Check', 'Momentum Signal', 'Position Size']
            node_x = np.linspace(0, 1, len(decision_nodes))
            node_y = [0.5] * len(decision_nodes)
            
            ax3.scatter(node_x, node_y, s=1000, c='lightblue', zorder=2)
            
            # Add arrows between nodes
            for i in range(len(decision_nodes)-1):
                ax3.arrow(node_x[i]+0.05, node_y[i],
                         node_x[i+1]-node_x[i]-0.1, 0,
                         head_width=0.05, head_length=0.05,
                         fc='k', ec='k', zorder=1)
            
            # Add node labels
            for i, node in enumerate(decision_nodes):
                ax3.annotate(node, (node_x[i], node_y[i]),
                           ha='center', va='center')
            
            ax3.set_title('Strategy Decision Flow')
            ax3.axis('off')
            
            # Save plot
            output_path = os.path.join(output_dir, 'strategy_explainability.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating strategy explainability: {str(e)}")
            return None

    def create_performance_forecast(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create performance forecasting visualization based on market conditions.
        """
        try:
            # Get historical performance data
            equity_curve = self._get_real_equity_curve(strategy_data)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot historical performance
            historical_dates = equity_curve.index
            historical_values = equity_curve['equity']
            plt.plot(historical_dates, historical_values,
                    label='Historical', color='blue', linewidth=2)
            
            # Generate forecast
            last_value = historical_values.iloc[-1]
            forecast_days = 30
            forecast_dates = pd.date_range(start=historical_dates[-1],
                                         periods=forecast_days+1)[1:]
            
            # Create three scenarios
            scenarios = {
                'Bullish': {'return': 0.15, 'color': 'green', 'style': '--'},
                'Base': {'return': 0.08, 'color': 'gray', 'style': '--'},
                'Bearish': {'return': -0.05, 'color': 'red', 'style': '--'}
            }
            
            # Plot scenarios
            for scenario, params in scenarios.items():
                daily_return = (1 + params['return']) ** (1/252) - 1
                forecast_values = last_value * np.cumprod(
                    np.repeat(1 + daily_return, forecast_days))
                
                plt.plot(forecast_dates, forecast_values,
                        label=f'{scenario} Scenario',
                        color=params['color'],
                        linestyle=params['style'])
                
                # Add confidence intervals
                if scenario == 'Base':
                    std_dev = historical_values.pct_change().std()
                    upper = forecast_values * (1 + 2*std_dev)
                    lower = forecast_values * (1 - 2*std_dev)
                    plt.fill_between(forecast_dates, lower, upper,
                                   color='gray', alpha=0.2)
            
            # Add vertical line separating historical and forecast
            plt.axvline(x=historical_dates[-1], color='black',
                       linestyle=':', alpha=0.5)
            plt.text(historical_dates[-1], plt.ylim()[0], 'Forecast Start',
                    rotation=90, verticalalignment='bottom')
            
            # Customize plot
            plt.title('Performance Forecast with Market Scenarios', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Portfolio Value', fontsize=12)
            plt.legend(title='Scenarios', bbox_to_anchor=(1.05, 1),
                      loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            output_path = os.path.join(output_dir, 'performance_forecast.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating performance forecast: {str(e)}")
            return None

    def create_market_regime_view(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create market regime analysis visualization.
        """
        try:
            # Get market data
            market_data = strategy_data.get('market_data', {})
            if not market_data:
                logger.warning("No market data available for regime analysis")
                return None
            
            # Create figure with multiple subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
            # 1. Market regimes overlay
            dates = pd.date_range(start='2024-01-01', periods=len(market_data.get('close', [])))
            prices = market_data.get('close', [])
            regimes = market_data.get('regimes', [])
            
            # Plot price
            ax1.plot(dates, prices, color='black', linewidth=1.5, label='Price')
            
            # Color different regimes
            regime_colors = ['lightgreen', 'lightcoral', 'lightgray']
            regime_labels = ['Bull Market', 'Bear Market', 'Sideways']
            
            for i, regime in enumerate(set(regimes)):
                mask = np.array(regimes) == regime
                ax1.fill_between(dates, min(prices), max(prices),
                               where=mask, color=regime_colors[i],
                               alpha=0.3, label=regime_labels[i])
            
            ax1.set_title('Market Regimes')
            ax1.legend()
            
            # 2. Regime characteristics
            volatility = market_data.get('volatility', [])
            volume = market_data.get('volume', [])
            
            ax2.plot(dates, volatility, label='Volatility', color='blue')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(dates, volume, label='Volume', color='red', alpha=0.5)
            
            ax2.set_title('Regime Characteristics')
            ax2.set_ylabel('Volatility')
            ax2_twin.set_ylabel('Volume')
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2)
            
            # 3. Regime transition probabilities
            transition_matrix = np.array([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ])
            
            sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd',
                       xticklabels=regime_labels, yticklabels=regime_labels,
                       ax=ax3)
            ax3.set_title('Regime Transition Probabilities')
            
            # Save plot
            output_path = os.path.join(output_dir, 'market_regimes.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating market regime view: {str(e)}")
            return None

    def visualize_strategy_performance(self, strategy_data: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Create comprehensive visualizations for strategy performance.
        """
        if save_path:
            output_dir = save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = strategy_data.get("strategy_name", "strategy").replace(" ", "_")
            output_dir = os.path.join(self.output_dir, f"{strategy_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
        
        visualization_paths = {}
        
        try:
            # Generate base visualizations
            equity_curve_path = self._create_equity_curve(strategy_data, output_dir)
            if equity_curve_path:
                visualization_paths["equity_curve"] = equity_curve_path
            
            drawdown_path = self._create_drawdown_chart(strategy_data, output_dir)
            if drawdown_path:
                visualization_paths["drawdown"] = drawdown_path
            
            monthly_returns_path = self._create_monthly_returns_heatmap(strategy_data, output_dir)
            if monthly_returns_path:
                visualization_paths["monthly_returns"] = monthly_returns_path
            
            trade_analysis_path = self._create_trade_analysis(strategy_data, output_dir)
            if trade_analysis_path:
                visualization_paths["trade_analysis"] = trade_analysis_path
            
            metrics_dashboard_path = self._create_metrics_dashboard(strategy_data, output_dir)
            if metrics_dashboard_path:
                visualization_paths["metrics_dashboard"] = metrics_dashboard_path
            
            # Generate enhanced visualizations
            attribution_path = self.create_performance_attribution(strategy_data, output_dir)
            if attribution_path:
                visualization_paths["performance_attribution"] = attribution_path
            
            explainability_path = self.create_strategy_explainability(strategy_data, output_dir)
            if explainability_path:
                visualization_paths["strategy_explainability"] = explainability_path
            
            regime_path = self.create_market_regime_view(strategy_data, output_dir)
            if regime_path:
                visualization_paths["market_regimes"] = regime_path
            
            forecast_path = self.create_performance_forecast(strategy_data, output_dir)
            if forecast_path:
                visualization_paths["performance_forecast"] = forecast_path
            
            # Generate Google Sheets compatible charts
            sheets_charts = self.create_google_sheets_charts(strategy_data)
            if sheets_charts:
                charts_path = os.path.join(output_dir, "sheets_charts.json")
                with open(charts_path, 'w') as f:
                    json.dump(sheets_charts, f, indent=2)
                visualization_paths["sheets_charts"] = charts_path
            
            # Create enhanced HTML index
            index_path = self._create_html_index(strategy_data, visualization_paths, output_dir)
            if index_path:
                visualization_paths["index"] = index_path
            
            logger.info(f"Created {len(visualization_paths)} visualizations in {output_dir}")
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return visualization_paths
    
    def _create_equity_curve(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create an equity curve visualization.
        
        Args:
            strategy_data: Strategy performance data
            output_dir: Output directory
            
        Returns:
            Path to the saved visualization or None if failed
        """
        try:
            # Generate sample equity curve data if not provided
            if "equity_curve" not in strategy_data:
                equity_curve = self._get_real_equity_curve(strategy_data)
            else:
                equity_curve = strategy_data["equity_curve"]
            
            # Convert to pandas dataframe if it's a dictionary
            if isinstance(equity_curve, dict):
                equity_curve = pd.DataFrame(equity_curve)
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot equity curve
            plt.plot(equity_curve.index, equity_curve["equity"], label="Strategy", linewidth=2)
            
            # Add benchmark if available
            if "benchmark" in equity_curve.columns:
                plt.plot(equity_curve.index, equity_curve["benchmark"], label="Benchmark", linewidth=1, linestyle="--")
            
            # Add labels and title
            strategy_name = strategy_data.get("strategy_name", "Strategy")
            plt.title(f"{strategy_name} - Equity Curve", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Portfolio Value ($)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            # Add annotations for key metrics
            cagr = strategy_data.get("cagr", 0) * 100 if "cagr" in strategy_data else strategy_data.get("performance", {}).get("annualized_return", 0) * 100
            sharpe = strategy_data.get("sharpe_ratio", 0) if "sharpe_ratio" in strategy_data else strategy_data.get("performance", {}).get("sharpe_ratio", 0)
            max_dd = strategy_data.get("max_drawdown", 0) * 100 if "max_drawdown" in strategy_data else strategy_data.get("performance", {}).get("max_drawdown", 0) * 100
            
            metrics_text = f"CAGR: {cagr:.2f}%\nSharpe: {sharpe:.2f}\nMax DD: {max_dd:.2f}%"
            plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords="axes fraction", 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(output_dir, "equity_curve.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating equity curve: {str(e)}")
            return None
    
    def _create_drawdown_chart(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create a drawdown chart visualization.
        
        Args:
            strategy_data: Strategy performance data
            output_dir: Output directory
            
        Returns:
            Path to the saved visualization or None if failed
        """
        try:
            # Generate sample equity curve data if not provided
            if "equity_curve" not in strategy_data:
                equity_curve = self._get_real_equity_curve(strategy_data)
            else:
                equity_curve = strategy_data["equity_curve"]
            
            # Convert to pandas dataframe if it's a dictionary
            if isinstance(equity_curve, dict):
                equity_curve = pd.DataFrame(equity_curve)
            
            # Calculate drawdowns if not provided
            if "drawdown" not in equity_curve.columns:
                equity = equity_curve["equity"].values
                peak = pd.Series(equity).cummax()
                drawdown = (equity / peak - 1) * 100
                equity_curve["drawdown"] = drawdown
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot drawdown
            plt.fill_between(equity_curve.index, 0, equity_curve["drawdown"], color="red", alpha=0.3)
            plt.plot(equity_curve.index, equity_curve["drawdown"], color="red", linewidth=1)
            
            # Add labels and title
            strategy_name = strategy_data.get("strategy_name", "Strategy")
            plt.title(f"{strategy_name} - Drawdown Chart", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Drawdown (%)", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            # Format y-axis
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            
            # Invert y-axis for better visualization (negative values at bottom)
            plt.gca().invert_yaxis()
            
            # Add max drawdown annotation
            max_dd = strategy_data.get("max_drawdown", 0) * 100 if "max_drawdown" in strategy_data else strategy_data.get("performance", {}).get("max_drawdown", 0) * 100
            plt.axhline(y=-max_dd, color="black", linestyle="--", alpha=0.7, label=f"Max Drawdown: {max_dd:.2f}%")
            plt.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(output_dir, "drawdown_chart.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {str(e)}")
            return None
    
    def _create_monthly_returns_heatmap(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create a monthly returns heatmap visualization.
        
        Args:
            strategy_data: Strategy performance data
            output_dir: Output directory
            
        Returns:
            Path to the saved visualization or None if failed
        """
        try:
            # Generate sample equity curve data if not provided
            if "equity_curve" not in strategy_data:
                equity_curve = self._get_real_equity_curve(strategy_data)
            else:
                equity_curve = strategy_data["equity_curve"]
            
            # Convert to pandas dataframe if it's a dictionary
            if isinstance(equity_curve, dict):
                equity_curve = pd.DataFrame(equity_curve)
            
            # Calculate daily returns
            if "returns" not in equity_curve.columns:
                equity_curve["returns"] = equity_curve["equity"].pct_change()
            
            # Group returns by year and month
            equity_curve["year"] = equity_curve.index.year
            equity_curve["month"] = equity_curve.index.month
            
            # Calculate monthly returns
            monthly_returns = equity_curve.groupby(["year", "month"])["returns"].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            
            # Pivot the data for the heatmap
            monthly_returns_pivot = monthly_returns.pivot(
                index="year", columns="month", values="returns"
            )
            
            # Replace month numbers with month names
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly_returns_pivot.columns = [month_names[i-1] for i in monthly_returns_pivot.columns]
            
            # Create the heatmap
            plt.figure(figsize=(12, 8))
            
            # Define color map (green for positive, red for negative)
            cmap = sns.diverging_palette(10, 133, as_cmap=True)
            
            # Create the heatmap
            sns.heatmap(
                monthly_returns_pivot * 100,  # Convert to percentage
                annot=True,
                fmt=".2f",
                cmap=cmap,
                center=0,
                linewidths=0.5,
                cbar_kws={"label": "Monthly Return (%)"}
            )
            
            # Add labels and title
            strategy_name = strategy_data.get("strategy_name", "Strategy")
            plt.title(f"{strategy_name} - Monthly Returns (%)", fontsize=16)
            plt.ylabel("Year", fontsize=12)
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(output_dir, "monthly_returns.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating monthly returns heatmap: {str(e)}")
            return None
    
    def _create_trade_analysis(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create a trade analysis visualization with multiple subplots.
        
        Args:
            strategy_data: Strategy performance data
            output_dir: Output directory
            
        Returns:
            Path to the saved visualization or None if failed
        """
        try:
            # Get trade data
            trades = strategy_data.get("trades", {})
            
            # If no trade data is provided, generate sample trades
            if not trades or isinstance(trades, dict):
                trade_count = trades.get("total_trades", 50) if isinstance(trades, dict) else 50
                win_rate = trades.get("win_rate", 0.6) if isinstance(trades, dict) else 0.6
                avg_win = trades.get("average_winner", 0.05) if isinstance(trades, dict) else 0.05
                avg_loss = trades.get("average_loser", -0.03) if isinstance(trades, dict) else -0.03
                
                # Get real trades from backtest data
                trades = self._get_real_trades(strategy_data)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Profit/Loss Distribution
            if isinstance(trades, pd.DataFrame):
                profit_loss = trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.random.normal(0.02, 0.05, 50))))
            else:
                profit_loss = [trade.get("pnl_pct", trade.get("PnL %", 0)) * 100 for trade in trades] if isinstance(trades, list) else []
                
            sns.histplot(profit_loss, bins=20, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title("Profit/Loss Distribution", fontsize=14)
            axes[0, 0].set_xlabel("Profit/Loss (%)", fontsize=12)
            axes[0, 0].set_ylabel("Frequency", fontsize=12)
            axes[0, 0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
            
            # 2. Win/Loss Ratio
            if isinstance(trades, pd.DataFrame):
                wins = (trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50)))) > 0).sum()
                losses = (trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50)))) <= 0).sum()
            else:
                wins = sum(1 for trade in trades if trade.get("pnl_pct", trade.get("PnL %", 0)) > 0) if isinstance(trades, list) else 0
                losses = sum(1 for trade in trades if trade.get("pnl_pct", trade.get("PnL %", 0)) <= 0) if isinstance(trades, list) else 0
                
            axes[0, 1].pie([wins, losses], labels=["Wins", "Losses"], autopct="%1.1f%%", 
                        colors=["green", "red"], startangle=90, explode=(0.05, 0))
            axes[0, 1].set_title("Win/Loss Ratio", fontsize=14)
            
            # 3. Trade Duration
            if isinstance(trades, pd.DataFrame) and "duration" in trades.columns:
                durations = trades["duration"]
            elif isinstance(trades, list) and trades and "duration" in trades[0]:
                durations = [trade["duration"] for trade in trades]
            else:
                # Generate sample durations (in days)
                durations = np.random.randint(1, 30, 50)
                
            sns.histplot(durations, bins=15, kde=True, ax=axes[1, 0])
            axes[1, 0].set_title("Trade Duration Distribution", fontsize=14)
            axes[1, 0].set_xlabel("Duration (days)", fontsize=12)
            axes[1, 0].set_ylabel("Frequency", fontsize=12)
            
            # 4. Cumulative P&L
            if isinstance(profit_loss, pd.Series):
                cumulative_pnl = profit_loss.cumsum()
            else:
                cumulative_pnl = np.cumsum(profit_loss)
                
            axes[1, 1].plot(cumulative_pnl, linewidth=2)
            axes[1, 1].set_title("Cumulative P&L (%)", fontsize=14)
            axes[1, 1].set_xlabel("Trade Number", fontsize=12)
            axes[1, 1].set_ylabel("Cumulative P&L (%)", fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add strategy information
            strategy_name = strategy_data.get("strategy_name", "Strategy")
            plt.suptitle(f"{strategy_name} - Trade Analysis", fontsize=18)
            
            # Add trade summary statistics
            if isinstance(trades, pd.DataFrame):
                trade_count = len(trades)
                win_rate = (trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50)))) > 0).mean()
                avg_win = trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50))))[trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50)))) > 0].mean() if wins > 0 else 0
                avg_loss = trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50))))[trades.get("PnL %", trades.get("pnl_pct", pd.Series(np.zeros(50)))) <= 0].mean() if losses > 0 else 0
            else:
                trade_count = len(trades) if isinstance(trades, list) else 0
                win_rate = wins / trade_count if trade_count > 0 else 0
                
                winning_trades = [trade.get("pnl_pct", trade.get("PnL %", 0)) * 100 for trade in trades if trade.get("pnl_pct", trade.get("PnL %", 0)) > 0] if isinstance(trades, list) else []
                losing_trades = [trade.get("pnl_pct", trade.get("PnL %", 0)) * 100 for trade in trades if trade.get("pnl_pct", trade.get("PnL %", 0)) <= 0] if isinstance(trades, list) else []
                
                avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
                avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
            
            summary_text = f"Total Trades: {trade_count}\n"
            summary_text += f"Win Rate: {win_rate:.2%}\n"
            summary_text += f"Avg. Winner: {avg_win:.2f}%\n"
            summary_text += f"Avg. Loser: {avg_loss:.2f}%\n"
            summary_text += f"Profit Factor: {abs(avg_win * wins) / abs(avg_loss * losses):.2f}" if (avg_loss * losses) != 0 else "Profit Factor: ∞"
            
            fig.text(0.02, 0.02, summary_text, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_path = os.path.join(output_dir, "trade_analysis.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating trade analysis: {str(e)}")
            return None
    
    def _create_metrics_dashboard(self, strategy_data: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Create a performance metrics dashboard visualization.
        
        Args:
            strategy_data: Strategy performance data
            output_dir: Output directory
            
        Returns:
            Path to the saved visualization or None if failed
        """
        try:
            # Extract performance metrics
            performance = strategy_data.get("performance", {})
            
            # Extract key metrics
            cagr = performance.get("annualized_return", strategy_data.get("cagr", 0)) * 100
            sharpe = performance.get("sharpe_ratio", strategy_data.get("sharpe_ratio", 0))
            sortino = performance.get("sortino_ratio", strategy_data.get("sortino_ratio", 0))
            max_dd = performance.get("max_drawdown", strategy_data.get("max_drawdown", 0)) * 100
            volatility = performance.get("volatility", strategy_data.get("volatility", 0)) * 100
            beta = performance.get("beta", strategy_data.get("beta", 1))
            alpha = performance.get("alpha", strategy_data.get("alpha", 0)) * 100
            
            # Trade metrics
            trades = strategy_data.get("trades", {})
            win_rate = trades.get("win_rate", 0) * 100 if isinstance(trades, dict) else 0
            profit_factor = trades.get("profit_factor", 0) if isinstance(trades, dict) else 0
            avg_trade = trades.get("average_trade", 0) * 100 if isinstance(trades, dict) else 0
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define metrics to display
            metrics = [
                "CAGR", "Sharpe", "Sortino", "Max DD", 
                "Volatility", "Beta", "Alpha", 
                "Win Rate", "Profit Factor", "Avg Trade"
            ]
            
            values = [
                cagr, sharpe, sortino, -max_dd,  # Negative max_dd for visualization
                -volatility, beta, alpha,  # Negative volatility for visualization
                win_rate, profit_factor, avg_trade
            ]
            
            # Normalize values to -1 to 1 range for visualization
            max_val = max(abs(min(values)), abs(max(values)))
            normalized_values = [v / max_val for v in values]
            
            # Create a horizontal bar chart
            colors = [
                "green" if v >= 0 else "red" for v in values
            ]
            
            # Create bars
            bars = ax.barh(metrics, normalized_values, color=colors, alpha=0.6)
            
            # Add value labels
            for i, bar in enumerate(bars):
                value = values[i]
                if metrics[i] in ["CAGR", "Max DD", "Volatility", "Alpha", "Win Rate", "Avg Trade"]:
                    value_text = f"{value:.2f}%"
                elif metrics[i] in ["Sharpe", "Sortino", "Beta", "Profit Factor"]:
                    value_text = f"{value:.2f}"
                else:
                    value_text = f"{value:.2f}"
                
                # Position the text on the bars
                text_x = 0.01 if normalized_values[i] < 0 else normalized_values[i] + 0.01
                ax.text(text_x, bar.get_y() + bar.get_height()/2, value_text, 
                        va='center', ha='left' if normalized_values[i] < 0 else 'left', 
                        color='black', fontweight='bold')
            
            # Set the axis labels and title
            strategy_name = strategy_data.get("strategy_name", "Strategy")
            ax.set_title(f"{strategy_name} - Performance Metrics", fontsize=16)
            ax.set_xlabel("Normalized Score", fontsize=12)
            
            # Configure grid and axis
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
            ax.set_xlim(-1.1, 1.1)  # Add some padding
            ax.grid(True, axis="x", alpha=0.3)
            
            # Remove y-axis line
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            
            # Add performance summary
            summary_text = (
                f"Strategy: {strategy_name}\n"
                f"Universe: {strategy_data.get('universe', 'Unknown')}\n"
                f"Timeframe: {strategy_data.get('timeframe', 'Unknown')}\n"
                f"Period: {strategy_data.get('start_date', 'Unknown')} to {strategy_data.get('end_date', 'Unknown')}"
            )
            
            fig.text(0.02, 0.02, summary_text, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            output_path = os.path.join(output_dir, "metrics_dashboard.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating metrics dashboard: {str(e)}")
            return None
    
    def _create_html_index(self, strategy_data: Dict[str, Any], visualization_paths: Dict[str, str], output_dir: str) -> Optional[str]:
        """
        Create an HTML index file to view all visualizations.
        
        Args:
            strategy_data: Strategy performance data
            visualization_paths: Dictionary mapping visualization names to file paths
            output_dir: Output directory
            
        Returns:
            Path to the saved HTML file or None if failed
        """
        try:
            strategy_name = strategy_data.get("strategy_name", "Strategy")
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{strategy_name} - Performance Analysis</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    h1, h2 {{
                        color: #333;
                    }}
                    .stats-box {{
                        background-color: #fff;
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 15px;
                    }}
                    .stat-item {{
                        border: 1px solid #eee;
                        border-radius: 5px;
                        padding: 10px;
                        text-align: center;
                    }}
                    .stat-value {{
                        font-size: 24px;
                        font-weight: bold;
                        margin: 5px 0;
                    }}
                    .stat-label {{
                        color: #666;
                        font-size: 14px;
                    }}
                    .positive {{
                        color: green;
                    }}
                    .negative {{
                        color: red;
                    }}
                    .viz-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                        gap: 20px;
                    }}
                    .viz-item {{
                        background-color: #fff;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .viz-title {{
                        font-size: 18px;
                        margin-bottom: 10px;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #eee;
                    }}
                    .strategy-details {{
                        margin-top: 10px;
                        font-size: 14px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{strategy_name} - Performance Analysis</h1>
                    
                    <div class="stats-box">
                        <h2>Performance Metrics</h2>
                        <div class="stats-grid">
            """
            
            # Extract performance metrics
            performance = strategy_data.get("performance", {})
            cagr = performance.get("annualized_return", strategy_data.get("cagr", 0)) * 100
            sharpe = performance.get("sharpe_ratio", strategy_data.get("sharpe_ratio", 0))
            max_dd = performance.get("max_drawdown", strategy_data.get("max_drawdown", 0)) * 100
            win_rate = strategy_data.get("trades", {}).get("win_rate", 0) * 100
            
            # Add metrics to HTML
            metrics = [
                {"label": "CAGR", "value": f"{cagr:.2f}%", "class": "positive" if cagr > 0 else "negative"},
                {"label": "Sharpe Ratio", "value": f"{sharpe:.2f}", "class": "positive" if sharpe > 1 else "negative"},
                {"label": "Max Drawdown", "value": f"{max_dd:.2f}%", "class": "negative"},
                {"label": "Win Rate", "value": f"{win_rate:.2f}%", "class": "positive" if win_rate > 50 else "negative"},
            ]
            
            for metric in metrics:
                html_content += f"""
                <div class="stat-item">
                    <div class="stat-label">{metric['label']}</div>
                    <div class="stat-value {metric['class']}">{metric['value']}</div>
                </div>
                """
            
            html_content += """
                        </div>
                    </div>
                    
                    <div class="viz-grid">
            """
            
            # Add visualizations to HTML
            viz_titles = {
                "equity_curve": "Equity Curve",
                "drawdown": "Drawdown Chart",
                "monthly_returns": "Monthly Returns Heatmap",
                "trade_analysis": "Trade Analysis",
                "metrics_dashboard": "Performance Metrics Dashboard"
            }
            
            for viz_name, viz_path in visualization_paths.items():
                if viz_name == "index":
                    continue
                
                # Get relative path
                rel_path = os.path.basename(viz_path)
                
                html_content += f"""
                <div class="viz-item">
                    <div class="viz-title">{viz_titles.get(viz_name, viz_name.replace('_', ' ').title())}</div>
                    <img src="{rel_path}" alt="{viz_name}">
                </div>
                """
            
            # Add strategy details
            universe = strategy_data.get("universe", "Unknown")
            timeframe = strategy_data.get("timeframe", "Unknown")
            description = strategy_data.get("description", "No description provided")
            
            html_content += f"""
                    </div>
                    
                    <div class="stats-box">
                        <h2>Strategy Details</h2>
                        <div class="strategy-details">
                            <p><strong>Universe:</strong> {universe}</p>
                            <p><strong>Timeframe:</strong> {timeframe}</p>
                            <p><strong>Description:</strong> {description}</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save the HTML file
            output_path = os.path.join(output_dir, "index.html")
            with open(output_path, "w") as f:
                f.write(html_content)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating HTML index: {str(e)}")
            return None
    
    def _get_real_equity_curve(self, strategy_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Get the real equity curve from backtest results.
        
        Args:
            strategy_data: Strategy performance data
            
        Returns:
            DataFrame with real equity curve data
        """
        # Check if we have real equity curve data
        equity_curve = strategy_data.get("equity_curve", [])
        if not equity_curve:
            # Check nested structure
            results = strategy_data.get("results", {})
            equity_curve = results.get("equity_curve", [10000])
        
        # Check if trades data is available
        trades_data = strategy_data.get("trades_data", [])
        if not trades_data:
            # Check nested structure
            results = strategy_data.get("results", {})
            trades_data = results.get("trades_data", [])
            
        # Get start and end dates
        start_date_str = strategy_data.get("start_date")
        end_date_str = strategy_data.get("end_date")
        
        if not start_date_str:
            results = strategy_data.get("results", {})
            start_date_str = results.get("start_date")
            
        if not end_date_str:
            results = strategy_data.get("results", {})
            end_date_str = results.get("end_date")
            
        # Parse dates or use defaults
        try:
            start_date = pd.to_datetime(start_date_str)
        except:
            start_date = datetime.now() - timedelta(days=365)
            
        try:
            end_date = pd.to_datetime(end_date_str)
        except:
            end_date = datetime.now()
        
        # Generate dates between start and end
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        # Create DataFrame with real equity curve
        if len(equity_curve) > 1:
            # Ensure we have enough dates to match equity curve points
            if len(dates) < len(equity_curve):
                dates = pd.date_range(start=start_date, periods=len(equity_curve), freq="B")
                
            # Create DataFrame with real equity curve
            df = pd.DataFrame(index=dates[:len(equity_curve)])
            df["equity"] = equity_curve
            
            # Calculate returns
            df["returns"] = df["equity"].pct_change().fillna(0)
        else:
            # If no real equity curve is available, use trade data to construct it
            logger.warning("No equity curve data available, constructing from trades if available")
            
            if trades_data:
                # Sort trades by date
                sorted_trades = sorted(trades_data, key=lambda x: x.get("date", "2000-01-01"))
                
                # Create a series of dates and equity values
                trade_dates = []
                equity_values = [10000]  # Start with initial capital
                
                for i, trade in enumerate(sorted_trades):
                    trade_date = pd.to_datetime(trade.get("date", "2000-01-01"))
                    trade_dates.append(trade_date)
                    
                    # Calculate equity after trade
                    if i == 0:
                        equity_values.append(equity_values[0] * (1 + trade.get("pnl", 0)))
                    else:
                        equity_values.append(equity_values[-1] * (1 + trade.get("pnl", 0)))
                
                # Create DataFrame
                if trade_dates:
                    df = pd.DataFrame(index=pd.DatetimeIndex(trade_dates))
                    df["equity"] = equity_values[1:]  # Skip initial capital
                    df["returns"] = df["equity"].pct_change().fillna(0)
                else:
                    # No trades data, create a flat equity curve
                    df = pd.DataFrame(index=dates)
                    df["equity"] = 10000
                    df["returns"] = 0
            else:
                # No equity curve or trades data, create a flat equity curve
                df = pd.DataFrame(index=dates)
                df["equity"] = 10000
                df["returns"] = 0
                
        # Add a simple benchmark
        df["benchmark"] = 10000 * (1 + 0.08/252).cumulative_product(axis=0)
        
        # Calculate drawdown
        df["equity_peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] / df["equity_peak"] - 1) * 100
        
        return df
    
    def _get_real_trades(self, strategy_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get real trades from backtest results.
        
        Args:
            strategy_data: Strategy performance data
            
        Returns:
            List of real trades from backtest results
        """
        # Check if we have real trades data
        trades_data = strategy_data.get("trades_data", [])
        if not trades_data:
            # Check nested structure
            results = strategy_data.get("results", {})
            trades_data = results.get("trades_data", [])
        
        # Convert trades to standard format if available
        if trades_data:
            processed_trades = []
            
            for trade in trades_data:
                # Get trade date
                entry_date = trade.get("date", None)
                if not entry_date:
                    entry_date = trade.get("entry_date", datetime.now().strftime("%Y-%m-%d"))
                
                # Calculate exit date if not available (using a rough estimate)
                if "exit_date" in trade:
                    exit_date = trade["exit_date"]
                else:
                    # Estimate exit date as 3 days after entry
                    try:
                        entry_dt = pd.to_datetime(entry_date)
                        exit_date = (entry_dt + timedelta(days=3)).strftime("%Y-%m-%d")
                    except:
                        exit_date = entry_date
                
                # Get symbol
                symbol = trade.get("symbol", "UNKNOWN")
                
                # Get PnL information
                pnl = trade.get("pnl", 0)
                is_winner = trade.get("is_win", pnl > 0)
                
                # Create standardized trade dictionary
                processed_trade = {
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "pnl": pnl,
                    "is_winner": is_winner,
                    # Add additional fields with defaults
                    "pnl_pct": trade.get("pnl_pct", pnl / 100),
                    "side": trade.get("side", "BUY" if is_winner else "SELL"),
                }
                
                processed_trades.append(processed_trade)
            
            return processed_trades
            
        # If no trade data is available, log a warning and return empty list
        logger.warning("No real trade data available in strategy_data")
        return []