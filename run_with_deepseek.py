#!/usr/bin/env python3
"""
Enhanced AI Co-Scientist system that uses DeepSeek R1 via Ollama.
This preserves all original system functionality while adding local LLM capability.
"""
import os
import sys
import json
import logging
import asyncio
import argparse
import subprocess
import aiohttp
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("deepseek_run.log")
    ]
)

logger = logging.getLogger(__name__)

async def check_ollama():
    """
    Check if Ollama is running and DeepSeek R1 is available.
    
    This function verifies that:
    1. Ollama is installed and running
    2. The DeepSeek R1 model is available locally
    3. Ollama API is responsive
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    try:
        logger.info("Checking Ollama setup...")
        
        # Step 1: Check if Ollama is installed (command exists)
        process = await asyncio.create_subprocess_exec(
            "which", "ollama",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        if not stdout:
            logger.error("Ollama is not installed. Please install Ollama first.")
            return False
            
        logger.info(f"Found Ollama installation: {stdout.decode().strip()}")
        
        # Step 2: Check if Ollama is running (process exists)
        process = await asyncio.create_subprocess_exec(
            "pgrep", "-x", "ollama",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        if not stdout:
            logger.warning("Ollama is not running. Starting Ollama...")
            
            # Start Ollama in background
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # Wait for Ollama to start (with timeout)
            max_wait = 30  # seconds
            started = False
            
            for i in range(max_wait):
                logger.info(f"Waiting for Ollama to start... ({i+1}/{max_wait}s)")
                
                try:
                    # Check if Ollama API is responsive
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:11434/api/tags", timeout=1) as response:
                            if response.status == 200:
                                started = True
                                break
                except:
                    pass
                
                await asyncio.sleep(1)
            
            if not started:
                logger.error(f"Timed out waiting for Ollama to start after {max_wait} seconds")
                return False
                
            logger.info("Ollama started successfully")
        else:
            logger.info("Ollama is already running")
        
        # Step 3: Check if Ollama API is responsive
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                    if response.status != 200:
                        logger.error(f"Ollama API returned status {response.status}")
                        return False
                    logger.info("Ollama API is responsive")
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {str(e)}")
            return False
        
        # Step 4: Check if DeepSeek R1 model is available
        process = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        stdout_str = stdout.decode().strip()
        
        if "deepseek-r1" not in stdout_str:
            logger.warning("DeepSeek R1 model not found locally")
            
            # Check internet connectivity before trying to pull
            if not await check_internet_connection():
                logger.error("No internet connection detected and DeepSeek R1 is not available locally")
                return False
            
            logger.info("Pulling DeepSeek R1 model (this may take several minutes)...")
            
            # Pull the model with progress tracking
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", "deepseek-r1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Process output in real-time to show progress
            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    logger.info(line.decode().strip())
            
            # Create tasks for stdout and stderr
            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr)
            )
            
            # Wait for completion
            await process.wait()
            
            if process.returncode != 0:
                logger.error("Failed to pull DeepSeek R1 model")
                return False
            
            logger.info("DeepSeek R1 model pulled successfully")
        else:
            logger.info("DeepSeek R1 model is already available locally")
        
        # All checks passed
        logger.info("Ollama setup complete: DeepSeek R1 is ready to use")
        return True
        
    except Exception as e:
        logger.error(f"Error checking Ollama setup: {str(e)}")
        return False

async def check_internet_connection():
    """Check if internet connection is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.google.com", timeout=5) as response:
                return response.status == 200
    except:
        return False

async def run_agent_system(args):
    """
    Run the full agent system with DeepSeek R1.
    
    This function configures and executes the main AI Co-Scientist system with
    DeepSeek R1 via Ollama, ensuring it works in offline mode.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if GOD MODE is enabled
        god_mode_enabled = args.god_mode
        
        if god_mode_enabled:
            logger.info("⚡ Running AI Co-Scientist with DeepSeek R1 in GOD MODE ⚡")
            # Create a timestamp for the GOD MODE log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            god_mode_log = f"logs/deepseek_run_{timestamp}.log"
            
            # Add file handler for GOD MODE logging
            file_handler = logging.FileHandler(god_mode_log)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
            
            # Copy the current log to this specific log file
            logger.info(f"GOD MODE session logs will be saved to {god_mode_log}")
        else:
            logger.info("Running AI Co-Scientist with DeepSeek R1 in standard mode")
        
        # Record start time for performance tracking
        start_time = datetime.now()
        
        # Create command arguments
        cmd = [
            sys.executable,
            "run_agent_system.py"
        ]
        
        # Determine working directory
        working_dir = Path.cwd()
        logger.info(f"Working directory: {working_dir}")
        
        # Add all passed arguments to preserve full system functionality
        if args.plan_name:
            cmd.extend(["--plan-name", args.plan_name])
        if args.goal:
            cmd.extend(["--goal", args.goal])
        if args.market:
            cmd.extend(["--market", args.market])
        if args.time_horizon:
            cmd.extend(["--time-horizon", args.time_horizon])
        if args.min_sharpe:
            cmd.extend(["--min-sharpe", args.min_sharpe])
        if args.max_drawdown:
            cmd.extend(["--max-drawdown", args.max_drawdown])
        if args.min_win_rate:
            cmd.extend(["--min-win-rate", args.min_win_rate])
        if args.lookback:
            cmd.extend(["--lookback", args.lookback])
        if args.constraints:
            for constraint in args.constraints:
                cmd.extend(["--constraints", constraint])
        if args.deadline:
            cmd.extend(["--deadline", args.deadline])
        if args.interactive:
            cmd.append("--interactive")
            
        if args.offline:
            cmd.append("--offline")
            logger.info("Running in fully offline mode (no internet access required)")
        
        # Add memory configuration for offline mode
        cmd.append("--use-simple-memory")
        
        # Add DeepSeek R1 via Ollama configuration
        cmd.extend([
            "--llm-provider", "ollama",
            "--llm-model", "deepseek-r1"
        ])
        
        # Add GOD MODE configuration if enabled
        if god_mode_enabled:
            cmd.append("--god-mode")
            logger.info("Applying GOD MODE enhancements to all LLM interactions")
            
            # GOD MODE uses more aggressive parameters
            if not args.temperature:
                cmd.extend(["--temperature", "0.4"])  # Higher temperature for GOD MODE
        elif args.temperature:
            # Use user-specified temperature in standard mode
            cmd.extend(["--temperature", args.temperature])
        
        # Create output directory if needed
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            cmd.extend(["--output-dir", args.output_dir])
            
        # In GOD MODE, always create an output directory if not specified
        elif god_mode_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"output_{timestamp}")
            output_dir.mkdir(exist_ok=True, parents=True)
            cmd.extend(["--output-dir", str(output_dir)])
            logger.info(f"Created GOD MODE output directory: {output_dir}")
        
        # Execute the command
        logger.info(f"Running command: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Process output in real-time for user feedback
        async def stream_output(stream, log_level=logging.INFO):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode().strip()
                
                # Log at appropriate level based on content
                if "ERROR" in decoded:
                    logger.error(decoded)
                elif "WARNING" in decoded:
                    logger.warning(decoded)
                else:
                    logger.log(log_level, decoded)
        
        # Create tasks for stdout and stderr (stderr at warning level)
        await asyncio.gather(
            stream_output(process.stdout, logging.INFO),
            stream_output(process.stderr, logging.WARNING)
        )
        
        # Wait for completion
        await process.wait()
        
        # Calculate runtime
        runtime = datetime.now() - start_time
        runtime_str = f"{runtime.total_seconds():.2f} seconds"
        if runtime.total_seconds() > 60:
            minutes, seconds = divmod(runtime.total_seconds(), 60)
            runtime_str = f"{int(minutes)}m {int(seconds)}s"
        if runtime.total_seconds() > 3600:
            hours, remainder = divmod(runtime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        if process.returncode == 0:
            logger.info(f"AI Co-Scientist completed successfully with DeepSeek R1 in {runtime_str}")
            return True
        else:
            logger.error(f"AI Co-Scientist failed with exit code {process.returncode} after {runtime_str}")
            return False
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Error running AI Co-Scientist: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main function for running the AI Co-Scientist with DeepSeek R1"""
    # Create parser with same arguments as run_agent_system.py plus DeepSeek R1 specific options
    parser = argparse.ArgumentParser(
        description="AI Co-Scientist with DeepSeek R1 for Trading Strategy Development", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Plan creation arguments - same as original system
    planning_group = parser.add_argument_group('Planning Options')
    planning_group.add_argument("--plan-name", type=str, 
                       help="Name of the research plan")
    planning_group.add_argument("--goal", type=str,
                       help="Goal of the research plan")
    planning_group.add_argument("--market", type=str,
                       help="Target market (us_equities, crypto, forex, etc.)")
    planning_group.add_argument("--time-horizon", type=str,
                       help="Time horizon (intraday, daily, weekly, monthly)")
    planning_group.add_argument("--lookback", type=str,
                       help="Max lookback period in trading days")
    planning_group.add_argument("--min-sharpe", type=str,
                       help="Minimum acceptable Sharpe ratio")
    planning_group.add_argument("--max-drawdown", type=str,
                       help="Maximum acceptable drawdown")
    planning_group.add_argument("--min-win-rate", type=str,
                       help="Minimum acceptable win rate")
    planning_group.add_argument("--constraints", nargs="*", 
                        help="Additional constraints (key=value pairs)")
    planning_group.add_argument("--deadline", 
                        help="Deadline for the research plan (YYYY-MM-DD[THH:MM:SS])")
    
    # Mode arguments
    mode_group = parser.add_argument_group('Execution Mode')
    mode_group.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode with status updates")
    mode_group.add_argument("--offline", action="store_true",
                       help="Run in fully offline mode (no internet access required)")
    
    # DeepSeek R1 specific arguments
    deepseek_group = parser.add_argument_group('DeepSeek R1 Options')
    deepseek_group.add_argument("--temperature", type=str, default="0.2",
                       help="Temperature for DeepSeek R1 (0.0-1.0)")
    deepseek_group.add_argument("--output-dir", type=str,
                       help="Custom output directory for results")
    deepseek_group.add_argument("--god-mode", action="store_true",
                       help="Enable GOD MODE for unleashed capabilities with DeepSeek R1")
    
    # GOD MODE enhancement options
    god_mode_group = parser.add_argument_group('GOD MODE Enhancements (only used with --god-mode)')
    god_mode_group.add_argument("--disable-enhancements", nargs="*",
                       help="List of enhancements to disable in GOD MODE")
    god_mode_group.add_argument("--enable-enhancements", nargs="*",
                       help="List of enhancements to specifically enable in GOD MODE")
    god_mode_group.add_argument("--reasoning-depth", choices=["standard", "deep", "maximum"], default="standard",
                       help="Depth of reasoning in GOD MODE")
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    if args.log_level:
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {args.log_level}")
        logger.setLevel(numeric_level)
    
    # Log startup with system info
    logger.info("Starting AI Co-Scientist with DeepSeek R1")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Ensure Ollama is running and DeepSeek R1 is available
        if not await check_ollama():
            logger.error("Failed to set up Ollama and DeepSeek R1")
            return 1
        
        # Run the full agent system with DeepSeek R1
        if not await run_agent_system(args):
            logger.error("Failed to run agent system with DeepSeek R1")
            return 1
            
        logger.info("DeepSeek R1 integration completed successfully")
        return 0
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1
        
    finally:
        # Log shutdown
        runtime = (datetime.now() - start_time).total_seconds()
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        logger.info(f"Process completed in {runtime_str}")

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))