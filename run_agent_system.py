#!/usr/bin/env python3
"""
Main entry point for running the AI co-scientist trading strategy development system.
"""
import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

from src.pipeline.agent_pipeline import AgentPipeline

logger = logging.getLogger(__name__)


def setup_logging(log_level: str, log_file: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level string
        log_file: Path to log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging_level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Set matplotlib to non-interactive mode for server environments
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        # Suppress verbose logs from libraries
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
    except ImportError:
        pass
        
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


async def run_experimental_plan(pipeline: AgentPipeline, args: argparse.Namespace) -> None:
    """
    Run a scientific experimental plan.
    
    Args:
        pipeline: Initialized agent pipeline
        args: Command-line arguments
    """
    # Parse constraints
    constraints = {
        "market": args.market or "us_equities",
        "time_horizon": args.time_horizon or "daily",
        "max_lookback_period": int(args.lookback) if args.lookback else 252,
        "min_sharpe": float(args.min_sharpe) if args.min_sharpe else 0.5,
        "max_drawdown": float(args.max_drawdown) if args.max_drawdown else 0.25,
        "min_win_rate": float(args.min_win_rate) if args.min_win_rate else 0.5
    }
    
    # Add any additional constraints
    if args.constraints:
        for constraint in args.constraints:
            key, value = constraint.split("=")
            # Try to convert value to appropriate type
            try:
                # Try to convert to int
                constraints[key] = int(value)
            except ValueError:
                try:
                    # Try to convert to float
                    constraints[key] = float(value)
                except ValueError:
                    # Keep as string
                    constraints[key] = value
    
    # Parse deadline if provided
    deadline = None
    if args.deadline:
        try:
            deadline = datetime.fromisoformat(args.deadline)
        except ValueError:
            logger.error(f"Invalid deadline format: {args.deadline}, expected YYYY-MM-DD[THH:MM:SS]")
            await pipeline.stop()
            return
    
    # Create research plan
    logger.info(f"Creating research plan: {args.plan_name}")
    try:
        logger.info("Creating research plan...")
        response = await pipeline.create_research_plan(
            plan_name=args.plan_name,
            goal=args.goal,
            constraints=constraints,
            deadline=deadline
        )
        
        logger.debug(f"Research plan response: {response}")
        
        plan_id = response.get("data", {}).get("plan_id")
        if not plan_id:
            error_msg = response.get("error", "Unknown error")
            logger.error(f"Failed to create research plan: {error_msg}")
            
            # If GOD MODE is enabled, use the bridge to create a research plan
            if args.god_mode:
                logger.info("Using Mathematricks bridge to create research plan in GOD MODE")
                
                try:
                    # Import the bridge
                    from src.integration.mathematricks_bridge import get_bridge
                    bridge = get_bridge()
                    
                    # Create research plan using bridge
                    response = bridge.create_research_plan(
                        plan_name=args.plan_name,
                        goal=args.goal,
                        constraints=constraints
                    )
                    
                    logger.info(f"Created research plan with bridge: {response.get('data', {}).get('plan_id')}")
                    plan_id = response.get("data", {}).get("plan_id")
                    
                    if plan_id:
                        logger.info(f"Successfully created research plan with bridge: {plan_id}")
                        
                        # If strategy file is provided, skip the rest and proceed to Google Sheets update
                        if args.strategy_file:
                            logger.info(f"Using pre-generated strategy file: {args.strategy_file}")
                            logger.info("Skipping research plan execution in GOD MODE")
                            await pipeline.stop()
                            return
                    else:
                        logger.error("Failed to create research plan with bridge")
                        await pipeline.stop()
                        return
                        
                except Exception as bridge_error:
                    logger.error(f"Error creating research plan with bridge: {str(bridge_error)}", exc_info=True)
                    await pipeline.stop()
                    return
            else:
                # Standard mode - just stop if we can't create a plan
                await pipeline.stop()
                return
                
    except Exception as e:
        logger.error(f"Error creating research plan: {str(e)}", exc_info=True)
        
        # If GOD MODE is enabled, try using the bridge as fallback
        if args.god_mode:
            logger.info("Using Mathematricks bridge as fallback in GOD MODE")
            
            try:
                # Import the bridge
                from src.integration.mathematricks_bridge import get_bridge
                bridge = get_bridge()
                
                # Create research plan using bridge
                response = bridge.create_research_plan(
                    plan_name=args.plan_name,
                    goal=args.goal,
                    constraints=constraints
                )
                
                logger.info(f"Created research plan with bridge: {response.get('data', {}).get('plan_id')}")
                plan_id = response.get("data", {}).get("plan_id")
                
                if plan_id:
                    logger.info(f"Successfully created research plan with bridge: {plan_id}")
                    
                    # If strategy file is provided, skip the rest and proceed to Google Sheets update
                    if args.strategy_file:
                        logger.info(f"Using pre-generated strategy file: {args.strategy_file}")
                        logger.info("Skipping research plan execution in GOD MODE")
                        await pipeline.stop()
                        return
                else:
                    logger.error("Failed to create research plan with bridge")
                    await pipeline.stop()
                    return
            except Exception as bridge_error:
                logger.error(f"Error creating research plan with bridge: {str(bridge_error)}", exc_info=True)
                await pipeline.stop()
                return
        else:
            # Standard mode - just stop if we can't create a plan
            await pipeline.stop()
            return
    
    logger.info(f"Created research plan with ID: {plan_id}")
    logger.info(f"Running plan: {args.plan_name}")
    
    # If not interactive, let the pipeline run indefinitely
    if not args.interactive:
        await pipeline.run_forever()
        return
    
    # Interactive mode - poll for status and display updates
    while True:
        status = await pipeline.get_plan_status(plan_id)
        
        if status.get("status") == "success":
            plan_status = status.get("data", {}).get("status", "unknown")
            progress = status.get("data", {}).get("progress", 0)
            
            logger.info(f"Plan status: {plan_status} ({progress:.1f}%)")
            
            if plan_status == "completed":
                logger.info("Research plan completed. Results:")
                results = status.get("data", {}).get("results", {})
                for strategy in results.get("strategies", []):
                    name = strategy.get("Strategy Name", "Unnamed")
                    sharpe = strategy.get("performance", {}).get("sharpe_ratio", 0)
                    logger.info(f"  - {name}: Sharpe Ratio = {sharpe:.2f}")
                break
                
            elif plan_status == "failed":
                logger.error("Research plan failed")
                logger.error(status.get("data", {}).get("error", "Unknown error"))
                break
        
        # Wait before polling again
        await asyncio.sleep(10)


async def main(args: argparse.Namespace) -> None:
    """
    Main function to run the agent system.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Log startup
    logger.info("Starting AI co-scientist trading strategy system")
    logger.info(f"Python version: {sys.version}")
    start_time = datetime.now()
    
    # Log pipeline progress to summary sheet - simple progress update only
    try:
        from src.utils.google_sheet_integration import GoogleSheetIntegration
        sheets = GoogleSheetIntegration()
        if sheets.initialize():
            action = "System Progress" if args.god_mode else "Agent System"
            component = "Pipeline"
            sheets.update_summary(
                action=action,
                component=component,
                status="Initializing",
                details="System components are being prepared",
                next_steps="Creating agent pipeline",
                notes=""
            )
            logger.info("Updated summary sheet with initialization status")
    except Exception as e:
        logger.warning(f"Could not log to summary sheet: {str(e)}")
    
    try:
        # Check for required directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Initialize pipeline with configuration
        logger.info("Initializing agent pipeline")
        pipeline_config = {
            "config_path": args.config,
            "use_mathematricks": args.use_mathematricks,
            "use_fintsb": args.use_fintsb,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "use_simple_memory": args.use_simple_memory,
        }
        
        # Add LLM parameters
        if args.temperature:
            pipeline_config["temperature"] = float(args.temperature)
            
        # Add GOD MODE configuration if enabled
        if args.god_mode:
            # Check if running with DeepSeek R1
            if args.llm_model == "deepseek-r1" and args.llm_provider == "ollama":
                logger.info("⚡ Enabling DeepSeek R1 GOD MODE ⚡")
                pipeline_config["god_mode"] = True
                pipeline_config["god_mode_config"] = {
                    "enabled": True,
                    # Default to all enhancements enabled
                    "enhancements": [
                        "AdvancedReasoningFramework",
                        "ChainOfThoughtValidator",
                        "SelfCritiqueRefinement",
                        "ParallelHypothesisTesting",
                        "AdvancedFeatureEngineering",
                        "MarketRegimeDetection",
                        "AdaptiveHyperparameterOptimization",
                        "ExplainableAIComponents",
                        "CrossMarketCorrelationAnalysis",
                        "SentimentAnalysisIntegration",
                        "ModelEnsembleArchitecture"
                    ]
                }
            else:
                logger.warning("GOD MODE is only available with DeepSeek R1 via Ollama")
                logger.warning("Using standard mode instead")
        pipeline = AgentPipeline(pipeline_config)
        
        # Initialize pipeline components
        if not await pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return
        
        # Start all agents
        if not await pipeline.start():
            logger.error("Failed to start pipeline")
            return
            
        logger.info("Pipeline successfully started")
        
        # Log to summary sheet that pipeline is ready
        try:
            from src.utils.google_sheet_integration import GoogleSheetIntegration
            sheets = GoogleSheetIntegration()
            if sheets.initialize():
                action = "System Progress" 
                component = "Pipeline"
                sheets.update_summary(
                    action=action,
                    component=component,
                    status="Ready",
                    details="All components initialized successfully",
                    next_steps="Beginning strategy research",
                    notes=""
                )
        except Exception as e:
            logger.warning(f"Could not log to summary sheet: {str(e)}")
            
        # Run with experimental plan if plan name is provided
        if args.plan_name or args.goal:
            await run_experimental_plan(pipeline, args)
        else:
            # Run indefinitely in auto-pilot mode until interrupted
            logger.info("Starting continuous auto-pilot mode for strategy development")
            await pipeline.run_forever()
            
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        
    except Exception as e:
        logger.error(f"System error: {str(e)}", exc_info=True)
        
    finally:
        # Log shutdown
        runtime = (datetime.now() - start_time).total_seconds()
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        logger.info(f"System shut down after running for {runtime_str}")
        
        # Log shutdown to summary sheet
        try:
            from src.utils.google_sheet_integration import GoogleSheetIntegration
            sheets = GoogleSheetIntegration()
            if sheets.initialize():
                action = "System Progress"
                component = "Pipeline"
                status = "Completed"
                if 'KeyboardInterrupt' in str(sys.exc_info()[0]):
                    status = "Interrupted"
                elif 'Exception' in str(sys.exc_info()[0]):
                    status = "Error"
                
                sheets.update_summary(
                    action=action,
                    component=component,
                    status=status,
                    details=f"Process finished in {runtime_str}",
                    performance="",
                    next_steps="View generated strategies in Backtest Results",
                    notes=""
                )
        except Exception as e:
            logger.warning(f"Could not log to summary sheet: {str(e)}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the AI co-scientist trading strategy system")
    
    # Basic arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Plan creation arguments
    parser.add_argument("--plan-name", default="Auto-Generated Plan", 
                       help="Name of the research plan")
    parser.add_argument("--goal", default="Develop trading strategies with CAGR > 25%, Sharpe ratio > 1.0, maximum drawdown < 20%, and average profit > 0.75%",
                       help="Goal of the research plan")
    parser.add_argument("--market", type=str, default="us_equities",
                       help="Target market (us_equities, crypto, forex, etc.)")
    parser.add_argument("--time-horizon", type=str, default="daily",
                       help="Time horizon (intraday, daily, weekly, monthly)")
    parser.add_argument("--lookback", type=str, default="252",
                       help="Max lookback period in trading days")
    parser.add_argument("--min-sharpe", type=str, default="0.5",
                       help="Minimum acceptable Sharpe ratio")
    parser.add_argument("--max-drawdown", type=str, default="0.25",
                       help="Maximum acceptable drawdown")
    parser.add_argument("--min-win-rate", type=str, default="0.5",
                       help="Minimum acceptable win rate")
    parser.add_argument("--constraints", nargs="*", 
                        help="Additional constraints (key=value pairs)")
    parser.add_argument("--deadline", help="Deadline for the research plan (YYYY-MM-DD[THH:MM:SS])")
    
    # Mode arguments
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode with status updates")
    
    # Integration arguments
    parser.add_argument("--use-mathematricks", action="store_true", default=True,
                       help="Use mathematricks for backtesting")
    parser.add_argument("--use-fintsb", action="store_true", default=True,
                       help="Use FinTSB for model training")
    parser.add_argument("--strategy-file", type=str,
                       help="Path to pre-generated strategy file")
    
    # LLM and memory arguments
    parser.add_argument("--llm-provider", type=str, default="ollama",
                       help="LLM provider to use (ollama)")
    parser.add_argument("--llm-model", type=str, default="deepseek-r1",
                       help="Model name for the LLM provider")
    parser.add_argument("--temperature", type=str,
                       help="Temperature for LLM sampling (0.0-1.0)")
    parser.add_argument("--god-mode", action="store_true", default=False,
                       help="Enable GOD MODE for unleashed DeepSeek R1 capabilities")
    parser.add_argument("--use-simple-memory", action="store_true", default=True,
                       help="Use simplified JSON file-based memory")
                       
    # Logging arguments
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-file", default="logs/agent_system.log", 
                        help="Log file path")
    
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))