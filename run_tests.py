#!/usr/bin/env python3
"""
Test runner script for Enhanced Trading Strategy System.
"""
import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run trading system tests')
    
    parser.add_argument(
        '--type',
        choices=['unit', 'integration', 'all'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--component',
        choices=['agents', 'core', 'data', 'web', 'monitoring', 'strategies', 'all'],
        default='all',
        help='Component to test'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--report',
        choices=['term', 'html', 'xml', 'all'],
        default='term',
        help='Coverage report format'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Verbosity level'
    )
    
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first failure'
    )
    
    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='Skip slow tests'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of worker processes (0 for auto)'
    )
    
    return parser.parse_args()

def setup_test_environment():
    """Setup test environment."""
    # Set environment variables
    os.environ['TEST_MODE'] = 'true'
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    # Create necessary directories
    dirs = ['logs', 'coverage', 'reports']
    for dir in dirs:
        Path(dir).mkdir(exist_ok=True)

def build_pytest_command(args):
    """Build pytest command from arguments."""
    cmd = ['pytest']
    
    # Test selection
    if args.type != 'all':
        cmd.append(f'-m {args.type}')
    
    if args.component != 'all':
        cmd.append(f'tests/{args.component}')
    
    # Coverage options
    if args.coverage:
        cmd.append('--cov=src')
        if args.report == 'all':
            cmd.extend([
                '--cov-report=term-missing',
                '--cov-report=html:coverage/html',
                '--cov-report=xml:coverage/coverage.xml'
            ])
        else:
            if args.report == 'html':
                cmd.append('--cov-report=html:coverage/html')
            elif args.report == 'xml':
                cmd.append('--cov-report=xml:coverage/coverage.xml')
            else:
                cmd.append('--cov-report=term-missing')
    
    # Verbosity
    cmd.append('-' + 'v' * args.verbose)
    
    # Other options
    if args.failfast:
        cmd.append('-x')
    
    if args.skip_slow:
        cmd.append('-m "not slow"')
    
    if args.workers:
        cmd.append(f'-n {args.workers}')
    
    return ' '.join(cmd)

def run_tests(command):
    """Run tests with given command."""
    logger.info(f"Running tests with command: {command}")
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
            
        duration = datetime.now() - start_time
        logger.info(f"Tests completed in {duration}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code {e.returncode}")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

def save_test_report(success: bool, duration: str, command: str):
    """Save test execution report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'duration': duration,
        'command': command
    }
    
    report_path = Path('reports') / f"test_run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(report_path, 'w') as f:
        for key, value in report.items():
            f.write(f"{key}: {value}\n")

def main():
    """Main entry point."""
    args = parse_args()
    setup_test_environment()
    
    # Build and run test command
    command = build_pytest_command(args)
    start_time = datetime.now()
    success = run_tests(command)
    duration = datetime.now() - start_time
    
    # Save report
    save_test_report(success, str(duration), command)
    
    # Set exit code
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())