#!/usr/bin/env python
"""
Test script for QuantConnect integration.

This script tests the basic functionality of the QuantConnect integration:
1. Generates a QuantConnect algorithm from a strategy JSON
2. Tests the API connection
3. Verifies the autopilot components

Usage:
    python test_qc_integration.py --strategy path/to/strategy.json
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_adapter(strategy_path=None):
    """Test the QuantConnect adapter."""
    try:
        from quant_connect_adapter import QuantConnectAdapter
        
        logger.info("Testing QuantConnect adapter...")
        
        # Use sample strategy if none provided
        if not strategy_path:
            strategy_dir = os.path.join(os.getcwd(), 'generated_strategies')
            json_files = [f for f in os.listdir(strategy_dir) if f.endswith('.json')]
            if json_files:
                strategy_path = os.path.join(strategy_dir, json_files[0])
            else:
                logger.error("No strategy JSON files found. Cannot test adapter.")
                return False
        
        # Create adapter
        adapter = QuantConnectAdapter(strategy_path)
        
        # Load strategy
        strategy_data = adapter.load_strategy()
        logger.info(f"Loaded strategy: {strategy_data['strategy']['Strategy Name']}")
        
        # Generate algorithm
        output_dir = os.path.join(os.getcwd(), 'qc_test')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        algorithm_code = adapter.generate_algorithm(output_path=output_dir)
        logger.info("Generated algorithm code")
        
        return True
    except Exception as e:
        logger.error(f"Error testing adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_api():
    """Test the QuantConnect API client."""
    try:
        from quantconnect_api import QuantConnectAPIClient
        
        logger.info("Testing QuantConnect API client...")
        
        # Create client
        config_path = os.path.join(os.getcwd(), 'qc_config.json')
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}")
            logger.warning("Creating a temporary config file for testing")
            
            # Create a temporary config file
            with open(config_path, 'w') as f:
                json.dump({
                    "user_id": "vandan@getfoolish.com",
                    "token": "JungleW1z@rd!"
                }, f)
        
        client = QuantConnectAPIClient(config_path=config_path)
        
        # Test API connection
        try:
            # Try a simple API call
            projects = client.api.list_projects()
            if projects.get('success'):
                logger.info("Successfully connected to QuantConnect API")
                logger.info(f"Found {len(projects.get('projects', []))} existing projects")
                return True
            else:
                logger.error(f"Failed to connect to QuantConnect API: {projects.get('errors', ['Unknown error'])}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to QuantConnect API: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logger.error(f"Error testing API client: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_generator():
    """Test the QuantConnect generator."""
    try:
        from generate_qc_algorithm import QuantConnectGenerator
        
        logger.info("Testing QuantConnect generator...")
        
        # Create generator
        output_dir = os.path.join(os.getcwd(), 'qc_test')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        generator = QuantConnectGenerator(output_dir)
        
        # Try to find a strategy JSON
        strategy_dir = os.path.join(os.getcwd(), 'generated_strategies')
        json_files = [f for f in os.listdir(strategy_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.warning("No strategy JSON files found. Cannot test generator.")
            return False
        
        # Generate algorithm
        strategy_path = os.path.join(strategy_dir, json_files[0])
        qc_file = generator.generate_from_json(strategy_path)
        
        logger.info(f"Generated algorithm file: {qc_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error testing generator: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test QuantConnect integration")
    parser.add_argument("--strategy", help="Path to strategy JSON file")
    
    args = parser.parse_args()
    
    # Print test header
    print("\n" + "="*50)
    print("QuantConnect Integration Test")
    print("="*50)
    
    # Test adapter
    adapter_result = test_adapter(args.strategy)
    print(f"\nAdapter test: {'PASSED' if adapter_result else 'FAILED'}")
    
    # Test API client
    api_result = test_api()
    print(f"\nAPI client test: {'PASSED' if api_result else 'FAILED'}")
    
    # Test generator
    generator_result = test_generator()
    print(f"\nGenerator test: {'PASSED' if generator_result else 'FAILED'}")
    
    # Print test summary
    print("\n" + "="*50)
    print(f"Test result: {'PASSED' if (adapter_result and api_result and generator_result) else 'FAILED'}")
    print("="*50 + "\n")
    
    return 0 if (adapter_result and api_result and generator_result) else 1

if __name__ == "__main__":
    sys.exit(main())