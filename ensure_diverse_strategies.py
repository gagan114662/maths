#!/usr/bin/env python3
"""
Script to ensure the generation of diverse strategy types with proper naming convention.
"""
import json
import os
from pathlib import Path
import logging
import asyncio
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
from src.utils.google_sheet_integration import GoogleSheetIntegration
from src.agents.generation.generation_agent import GenerationAgent
from src.core.llm import LLMInterface
from src.core.mcp import ModelContextProtocol
from src.core.safety import SafetyChecker
from src.core.memory import MemoryManager

async def main():
    """Run the strategy diversification script."""
    logger.info("Starting strategy diversification script")
    
    # Initialize agents and components (simplified)
    llm = LLMInterface()
    await llm.initialize()
    
    # Create generation agent with updated strategy templates
    agent = GenerationAgent(name="generation_diversifier")
    await agent.initialize()
    
    # Update strategy templates to include more diverse types
    additional_templates = {
        "mean_reversion": {
            "name": "Mean Reversion Strategy",
            "description": "A strategy that trades assets when they deviate from their historical mean.",
            "parameters": {
                "lookback_period": [5, 10, 20, 30],  # days
                "deviation_threshold": [1.0, 1.5, 2.0, 2.5],  # standard deviations
                "stop_loss": [0.01, 0.02, 0.03],  # percent
                "take_profit": [0.01, 0.02, 0.03]  # percent
            },
            "implementation": "Calculate historical mean and standard deviation, enter when price deviates by threshold, exit when price reverts to mean or hits stop/profit."
        },
        "relative_strength": {
            "name": "Relative Strength Strategy",
            "description": "A strategy that identifies and trades assets with strong momentum relative to peers.",
            "parameters": {
                "lookback_period": [20, 50, 100],  # days
                "universe_size": [10, 20, 50],  # number of assets
                "position_count": [3, 5, 10],  # number of positions
                "rebalance_frequency": [5, 10, 20]  # days
            },
            "implementation": "Rank assets by relative strength, take long positions in top performers and short positions in bottom performers."
        },
        "contrarian": {
            "name": "Contrarian Strategy",
            "description": "A strategy that trades against current market sentiment or trends.",
            "parameters": {
                "oversold_threshold": [20, 30],  # RSI or similar
                "overbought_threshold": [70, 80],  # RSI or similar
                "confirmation_period": [1, 2, 3],  # days
                "exit_threshold": [40, 50, 60]  # RSI or similar
            },
            "implementation": "Identify overbought/oversold conditions, enter positions opposite to current trend, exit when conditions normalize."
        },
        "pattern_recognition": {
            "name": "Pattern Recognition Strategy",
            "description": "A strategy that trades based on chart patterns.",
            "parameters": {
                "pattern_types": ["double_top", "double_bottom", "head_shoulders", "triangle"],
                "confirmation_volume": [1.5, 2.0, 2.5],  # volume multiplier
                "stop_loss": [0.02, 0.03, 0.05],  # percent
                "target": [0.05, 0.1, 0.15]  # percent
            },
            "implementation": "Detect chart patterns using price action analysis, enter when pattern confirms, exit at target or stop loss."
        },
        "adaptive_indicators": {
            "name": "Adaptive Indicators Strategy",
            "description": "A strategy that uses indicators that adapt to changing market conditions.",
            "parameters": {
                "sensitivity": [0.1, 0.2, 0.3],  # adjustment factor
                "volatility_lookback": [10, 20, 30],  # days
                "min_signal_strength": [0.5, 0.6, 0.7],  # minimum signal threshold
                "adjustment_frequency": [5, 10, 15]  # days
            },
            "implementation": "Adjust indicator parameters based on recent market volatility, generate signals when adjusted indicators align."
        }
    }
    
    # Update the agent's strategy templates
    agent.strategy_templates.update(additional_templates)
    
    # Generate diverse strategies
    strategies_to_generate = 3
    generated_strategies = []
    
    # Find existing strategy numbers
    existing_numbers = []
    vault_dir = Path("mathematricks/vault")
    for file in vault_dir.glob("strategy_*.py"):
        try:
            number = int(file.stem.split("_")[1])
            existing_numbers.append(number)
        except (IndexError, ValueError):
            continue
    
    # Determine next strategy number
    next_number = max(existing_numbers) + 1 if existing_numbers else 4
    
    # Generate strategies
    for i in range(strategies_to_generate):
        # Randomly select a strategy type
        strategy_type = random.choice(list(agent.strategy_templates.keys()))
        
        # Generate strategy name with proper numbering
        strategy_name = f"strategy_{next_number}"
        next_number += 1
        
        # Generate the strategy
        template = agent.strategy_templates[strategy_type]
        strategy = {
            "name": f"{template['name']} {next_number-1}",
            "strategy_id": strategy_name,
            "description": template['description'],
            "type": strategy_type,
            "parameters": {},
            "entry_logic": f"Entry logic for {strategy_name} based on {strategy_type}",
            "exit_logic": f"Exit logic for {strategy_name} based on {strategy_type}",
            "position_sizing": "Dynamic position sizing based on volatility",
            "risk_management": "Adaptive stop-loss and take-profit levels",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add random parameters from the template
        for param_name, param_values in template['parameters'].items():
            strategy['parameters'][param_name] = random.choice(param_values)
            
        generated_strategies.append(strategy)
        logger.info(f"Generated {strategy['name']} ({strategy_name}) of type {strategy_type}")
    
    # Log the generated strategies
    output_dir = Path("generated_strategies")
    output_dir.mkdir(exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"diverse_strategies_{current_time}.json"
    
    with open(output_file, "w") as f:
        json.dump(generated_strategies, f, indent=2)
    
    logger.info(f"Saved {len(generated_strategies)} diverse strategies to {output_file}")
    logger.info("Strategy diversification completed")

if __name__ == "__main__":
    asyncio.run(main())