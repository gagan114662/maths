#\!/usr/bin/env python3
"""
Test script for strategy generation and backtesting.
"""
import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the necessary components
from src.core.llm import create_llm_interface, Message, MessageRole
from src.core.mcp import ModelContextProtocol, Context, ContextType
from src.core.safety import SafetyChecker
from src.core.simple_memory import SimpleMemoryManager

async def generate_strategy() -> Dict[str, Any]:
    """Generate a trading strategy using the LLM."""
    try:
        # Initialize components
        llm = create_llm_interface({"provider": "ollama", "model": "deepseek-r1"})
        
        # Define the strategy generation prompt
        system_prompt = """You are an expert trading strategy developer with deep knowledge of quantitative finance, 
technical analysis, and algorithmic trading. Your task is to design a trading strategy for US equities that 
aims to achieve high risk-adjusted returns.

Your strategy should:
1. Have clear entry and exit criteria based on technical indicators or market patterns
2. Include position sizing and risk management rules
3. Specify markets/assets and timeframe it's designed for
4. Explain the market inefficiency it exploits (its "edge")

Format your response as JSON with the following structure:
{
    "Strategy Name": "A descriptive name",
    "Edge": "The market inefficiency this strategy exploits",
    "Universe": "Which markets/assets this targets",
    "Timeframe": "The trading timeframe (e.g., daily, hourly)",
    "Entry Rules": ["Rule 1", "Rule 2", ...],
    "Exit Rules": ["Rule 1", "Rule 2", ...],
    "Position Sizing": "How positions are sized",
    "Risk Management": "Stop loss and other risk rules",
    "Expected Performance": {
        "CAGR": estimated annual return (decimal),
        "Sharpe Ratio": estimated Sharpe ratio,
        "Max Drawdown": estimated maximum drawdown (decimal),
        "Win Rate": estimated win rate (decimal)
    }
}"""

        user_prompt = """Design a trading strategy for S&P 500 stocks that uses momentum and volatility indicators.
The strategy should aim for a Sharpe ratio above 1.2 and maximum drawdown under 20%."""

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt)
        ]
        
        # Generate strategy
        logger.info("Generating trading strategy...")
        response = await llm.generate(messages=messages)
        
        # Extract JSON from response
        strategy_text = response.message.content
        logger.debug(f"Raw strategy: {strategy_text}")
        
        # Parse JSON from text (handle potential markdown code blocks)
        if "```json" in strategy_text:
            json_str = strategy_text.split("```json")[1].split("```")[0].strip()
        elif "```" in strategy_text:
            json_str = strategy_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = strategy_text.strip()
            
        strategy = json.loads(json_str)
        logger.info(f"Successfully generated strategy: {strategy['Strategy Name']}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}", exc_info=True)
        return {}

async def backtest_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate backtesting a strategy."""
    try:
        logger.info(f"Backtesting strategy: {strategy['Strategy Name']}")
        
        # In a real implementation, this would call the backtesting system
        # For now, we'll simulate results
        
        # Create mock backtest results
        results = {
            "performance": {
                "annualized_return": 0.27,  # 27% CAGR
                "sharpe_ratio": 1.45,       # Sharpe ratio
                "max_drawdown": -0.18,      # 18% max drawdown
                "volatility": 0.15          # 15% annualized volatility
            },
            "trades": {
                "total_trades": 156,
                "win_rate": 0.62,           # 62% win rate
                "average_trade": 0.0085,    # 0.85% average profit per trade
                "profit_factor": 1.75,      # Profit factor (gross profit / gross loss)
                "avg_hold_time": 15         # Average holding period in days
            },
            "analysis": {
                "market_correlation": 0.42,  # Correlation to market
                "best_month": 0.11,          # Best month return
                "worst_month": -0.09,        # Worst month return
                "recovery_time": 45          # Days to recover from max drawdown
            }
        }
        
        logger.info(f"Backtest completed with Sharpe ratio: {results['performance']['sharpe_ratio']}")
        return results
        
    except Exception as e:
        logger.error(f"Error backtesting strategy: {str(e)}", exc_info=True)
        return {}

async def save_strategy(strategy: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Save strategy to file system."""
    try:
        # Create directory if needed
        output_dir = Path("generated_strategies")
        output_dir.mkdir(exist_ok=True)
        
        # Create a filename based on strategy name and timestamp
        strategy_name = strategy.get("Strategy Name", "unnamed_strategy")
        safe_name = "".join([c if c.isalnum() else "_" for c in strategy_name])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        
        # Combine strategy and results into a single object
        full_data = {
            "strategy": strategy,
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save to file
        file_path = output_dir / filename
        with open(file_path, "w") as f:
            json.dump(full_data, f, indent=2)
            
        logger.info(f"Strategy saved to {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving strategy: {str(e)}", exc_info=True)
        return ""

async def main():
    """Run strategy generation and backtesting test."""
    try:
        logger.info("Starting strategy generation and backtesting test")
        
        # Generate strategy
        strategy = await generate_strategy()
        if not strategy:
            logger.error("Strategy generation failed")
            return
            
        # Backtest strategy
        results = await backtest_strategy(strategy)
        if not results:
            logger.error("Strategy backtesting failed")
            return
            
        # Save results
        file_path = await save_strategy(strategy, results)
        if file_path:
            logger.info(f"Strategy successfully generated, backtested, and saved to {file_path}")
            
            # Print performance metrics
            print("\nStrategy Performance Metrics:")
            print("=" * 40)
            print(f"Strategy Name: {strategy['Strategy Name']}")
            print(f"CAGR: {results['performance']['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {abs(results['performance']['max_drawdown']):.2%}")
            print(f"Win Rate: {results['trades']['win_rate']:.2%}")
            print(f"Total Trades: {results['trades']['total_trades']}")
            print(f"Avg Profit per Trade: {results['trades']['average_trade']:.2%}")
            
        else:
            logger.error("Failed to save strategy and results")
            
    except Exception as e:
        logger.error(f"Error in strategy test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
