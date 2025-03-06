#!/usr/bin/env python3
"""
Test script for Model Context Protocol (MCP) functionality and safety verification.
"""
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import MCP classes
from src.core.mcp.protocol import ModelContextProtocol
from src.core.mcp.context import Context, ContextType
from src.core.safety_checker import SafetyChecker

def test_mcp_basic_functionality():
    """Test basic MCP functionality."""
    logger.info("Testing basic MCP functionality...")
    
    # Initialize MCP
    mcp = ModelContextProtocol()
    
    # Create test contexts
    system_context = Context(
        type=ContextType.SYSTEM,
        data={"name": "AI Co-Scientist", "version": "1.0"},
        metadata={"test": True}
    )
    
    market_context = Context(
        type=ContextType.MARKET_DATA,
        data={"market": "US Equities", "volatility": "high"},
        metadata={"source": "test"}
    )
    
    # Update contexts
    mcp.update_context(system_context)
    mcp.update_context(market_context)
    
    # Verify contexts were stored
    all_contexts = mcp.get_all_contexts()
    assert all_contexts["system"]["data"]["name"] == "AI Co-Scientist", "System context not stored correctly"
    assert all_contexts["market_data"]["data"]["market"] == "US Equities", "Market context not stored correctly"
    
    # Test context retrieval
    retrieved_system_context = mcp.get_context(ContextType.SYSTEM)
    assert retrieved_system_context and retrieved_system_context.data["name"] == "AI Co-Scientist", "Failed to retrieve system context"
    
    # Test context update
    updated_system_context = Context(
        type=ContextType.SYSTEM,
        data={"name": "AI Co-Scientist", "version": "2.0"},
        metadata={"test": True}
    )
    mcp.update_context(updated_system_context)
    
    # Verify update
    retrieved_system_context = mcp.get_context(ContextType.SYSTEM)
    assert retrieved_system_context and retrieved_system_context.data["version"] == "2.0", "Failed to update context"
    
    logger.info("Basic MCP functionality test passed")
    return True

def test_scientific_context_management():
    """Test scientific context management with merging contexts."""
    logger.info("Testing scientific context management...")
    
    # Initialize MCP
    mcp = ModelContextProtocol()
    
    # Create strategy context
    strategy_context = Context(
        type=ContextType.STRATEGY,
        data={
            "id": "strategy-001",
            "name": "Momentum Strategy",
            "description": "Uses momentum indicators to predict future price movements",
            "parameters": {"lookback": 20, "threshold": 0.05}
        }
    )
    
    # Update context
    mcp.update_context(strategy_context)
    
    # Verify context was added
    strategy = mcp.get_context(ContextType.STRATEGY)
    assert strategy and strategy.data["id"] == "strategy-001", "Strategy not added correctly"
    
    # Create updated strategy with additional data
    updated_strategy = Context(
        type=ContextType.STRATEGY,
        data={
            "id": "strategy-001",
            "name": "Enhanced Momentum Strategy",
            "metrics": {"sharpe": 1.2, "sortino": 1.5},
            "validated": True
        }
    )
    
    # Merge context
    mcp.merge_context(updated_strategy)
    
    # Verify merged context
    merged_strategy = mcp.get_context(ContextType.STRATEGY)
    assert merged_strategy.data["name"] == "Enhanced Momentum Strategy", "Name not updated in merge"
    assert "parameters" in merged_strategy.data, "Original data lost in merge"
    assert merged_strategy.data["metrics"]["sharpe"] == 1.2, "New data not added in merge"
    
    # Test context history and rollback
    mcp.update_context(Context(
        type=ContextType.STRATEGY,
        data={"id": "strategy-001", "name": "Final Strategy"}
    ))
    
    # Verify current state
    current = mcp.get_context(ContextType.STRATEGY)
    assert current.data["name"] == "Final Strategy", "Update failed"
    
    # Rollback to previous state
    success = mcp.rollback_context()
    assert success, "Rollback failed"
    
    # Verify rolled back state
    rolled_back = mcp.get_context(ContextType.STRATEGY)
    assert rolled_back.data["name"] == "Enhanced Momentum Strategy", "Rollback incorrect"
    
    logger.info("Scientific context management test passed")
    return True

def test_context_clear_and_history():
    """Test context clearing and history management."""
    logger.info("Testing context clear and history...")
    
    # Initialize MCP
    mcp = ModelContextProtocol()
    
    # Create test contexts
    contexts = [
        Context(type=ContextType.SYSTEM, data={"name": "System 1"}),
        Context(type=ContextType.MARKET_DATA, data={"market": "US Equities"}),
        Context(type=ContextType.STRATEGY, data={"id": "strategy-001"})
    ]
    
    # Add all contexts
    for context in contexts:
        mcp.update_context(context)
    
    # Verify all contexts were added
    all_contexts = mcp.get_all_contexts()
    assert len(all_contexts) == 3, "Not all contexts were added"
    
    # Clear specific context
    mcp.clear_context(ContextType.MARKET_DATA)
    
    # Verify specific context was cleared
    all_contexts = mcp.get_all_contexts()
    assert len(all_contexts) == 2, "Context not cleared properly"
    assert "market_data" not in all_contexts, "Market data context not cleared"
    
    # Update system context to create another history entry
    mcp.update_context(Context(type=ContextType.SYSTEM, data={"name": "System 2"}))
    
    # Verify update
    system = mcp.get_context(ContextType.SYSTEM)
    assert system.data["name"] == "System 2", "Update failed"
    
    # Test history by rolling back twice
    mcp.rollback_context()  # Back to System 1 and Strategy
    
    # Verify first rollback
    system = mcp.get_context(ContextType.SYSTEM)
    assert system.data["name"] == "System 1", "First rollback failed"
    
    # Clear all contexts
    mcp.clear_context()
    
    # Verify all contexts cleared
    all_contexts = mcp.get_all_contexts()
    assert len(all_contexts) == 0, "Not all contexts were cleared"
    
    logger.info("Context clear and history test passed")
    return True

def test_safety_verification():
    """Test safety verification functionality."""
    logger.info("Testing safety verification...")
    
    # Initialize safety checker
    safety_checker = SafetyChecker()
    
    # Test dangerous pattern detection
    dangerous_prompt = "Let's exec(rm -rf /)"
    assert not safety_checker.verify_prompt(dangerous_prompt, {}), "Dangerous pattern not detected"
    
    # Test sensitive info detection
    sensitive_prompt = "My API_KEY is 12345"
    assert not safety_checker.verify_prompt(sensitive_prompt, {}), "Sensitive info not detected"
    
    # Test safe prompt
    safe_prompt = "What is the market trend for AAPL?"
    safe_context = {
        "system": {},
        "memory": {},
        "market": {},
        "tools": {}
    }
    assert safety_checker.verify_prompt(safe_prompt, safe_context), "Safe prompt incorrectly flagged"
    
    # Test trading constraints
    trading_action = {
        "size": 0.5,  # Above default max of 0.1
        "asset": "AAPL"
    }
    assert not safety_checker.verify_trading_action(trading_action), "Position size violation not detected"
    
    logger.info("Safety verification test passed")
    return True
    
def main():
    """Run all MCP tests."""
    logger.info("Starting Model Context Protocol (MCP) tests")
    
    tests = [
        test_mcp_basic_functionality,
        test_scientific_context_management,
        test_context_clear_and_history,
        test_safety_verification
    ]
    
    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with error: {str(e)}")
            results.append((test_func.__name__, False))
    
    # Print summary
    logger.info("\nTest Results:")
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"  {name}: {status}")
    
    # Overall result
    success_count = sum(1 for _, success in results if success)
    logger.info(f"\nOverall: {success_count}/{len(results)} tests passed")
    
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())