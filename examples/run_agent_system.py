#!/usr/bin/env python3
"""
Example script demonstrating the usage of the agent system.
"""
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any

from src.agents import factory, DEFAULT_PIPELINE
from src.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

async def load_market_data() -> pd.DataFrame:
    """Load sample market data."""
    # In practice, replace with actual market data loading
    return pd.DataFrame({
        'timestamp': pd.date_range(
            start='2025-01-01',
            end='2025-02-03',
            freq='1D'
        ),
        'open': [100] * 34,
        'high': [105] * 34,
        'low': [95] * 34,
        'close': [101] * 34,
        'volume': [1000000] * 34
    })

async def run_strategy_development() -> Dict[str, Any]:
    """Run complete strategy development cycle."""
    try:
        logger.info("Starting strategy development cycle")
        
        # Load market data
        market_data = await load_market_data()
        logger.info(f"Loaded market data: {len(market_data)} records")
        
        # Create agent pipeline
        pipeline_agents = factory.create_agent_pipeline(DEFAULT_PIPELINE)
        logger.info(f"Created agent pipeline with {len(pipeline_agents)} agents")
        
        # Prepare initial input
        input_data = {
            'market_data': market_data.to_dict('records'),
            'parameters': {
                'lookback_period': 20,
                'initial_capital': 100000,
                'position_limit': 0.1,
                'risk_limit': 0.2
            }
        }
        
        # Process through pipeline
        logger.info("Starting pipeline processing")
        results = await factory.process_pipeline(pipeline_agents, input_data)
        
        if results['pipeline_status'] == 'success':
            logger.info("Pipeline processing completed successfully")
            
            # Extract best strategies
            strategies = _extract_best_strategies(results)
            logger.info(f"Found {len(strategies)} promising strategies")
            
            # Analyze results
            analysis = _analyze_results(results)
            logger.info("Completed result analysis")
            
            return {
                'status': 'success',
                'strategies': strategies,
                'analysis': analysis,
                'pipeline_results': results
            }
            
        else:
            logger.error("Pipeline processing failed")
            logger.error(f"Errors: {results['errors']}")
            return {
                'status': 'error',
                'errors': results['errors']
            }
            
    except Exception as e:
        logger.error(f"Error in strategy development: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }
        
def _extract_best_strategies(results: Dict[str, Any]) -> list:
    """Extract best performing strategies from results."""
    try:
        # Get ranking results
        ranking_results = results['agent_results'].get('strategy_ranker', {})
        if not ranking_results:
            return []
            
        # Get top strategies
        rankings = ranking_results.get('rankings', {})
        strategies = [
            {'id': strategy_id, **data}
            for strategy_id, data in rankings.items()
            if data['rank'] <= 3  # Top 3 strategies
        ]
        
        return strategies
        
    except Exception as e:
        logger.error(f"Error extracting strategies: {str(e)}")
        return []
        
def _analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze pipeline results."""
    try:
        # Get meta-review results
        meta_results = results['agent_results'].get('system_analyzer', {})
        if not meta_results:
            return {}
            
        # Extract key metrics
        return {
            'performance_summary': meta_results.get('analysis', {}).get('system_performance', {}),
            'risk_assessment': meta_results.get('analysis', {}).get('market_influence', {}),
            'recommendations': meta_results.get('recommendations', [])
        }
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        return {}

async def main():
    """Main execution function."""
    try:
        logger.info("Starting agent system example")
        
        # Run strategy development
        results = await run_strategy_development()
        
        if results['status'] == 'success':
            # Print results
            print("\nStrategy Development Results:")
            print("=" * 50)
            
            print("\nTop Strategies:")
            print("-" * 30)
            for strategy in results['strategies']:
                print(f"Strategy {strategy['id']}:")
                print(f"  Rank: {strategy['rank']}")
                print(f"  Score: {strategy['score']:.4f}")
                print()
                
            print("\nSystem Analysis:")
            print("-" * 30)
            analysis = results['analysis']
            
            if 'performance_summary' in analysis:
                print("\nPerformance Summary:")
                for metric, value in analysis['performance_summary'].items():
                    print(f"  {metric}: {value}")
                    
            if 'recommendations' in analysis:
                print("\nRecommendations:")
                for rec in analysis['recommendations']:
                    print(f"- {rec.get('action')}: {rec.get('details', '')}")
                    
        else:
            print("\nStrategy Development Failed:")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        
    finally:
        # Cleanup
        factory.cleanup()
        logger.info("Agent system example completed")

if __name__ == '__main__':
    # Run example
    asyncio.run(main())