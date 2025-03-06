#!/usr/bin/env python3
"""
Validate project setup and dependencies.
"""
import sys
import subprocess
import pkg_resources
import importlib
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check Python version meets requirements."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        logger.info(f"Python version {'.'.join(map(str, current_version))} OK")
        return True
    else:
        logger.error(
            f"Python version {'.'.join(map(str, current_version))} not supported. "
            f"Please use Python {'.'.join(map(str, required_version))} or higher."
        )
        return False

def check_dependencies() -> bool:
    """Check if all required packages are installed."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
        
    try:
        # Read requirements
        with open(requirements_file) as f:
            requirements = [
                line.strip() for line in f
                if line.strip() and not line.startswith("#") and not line.startswith("-r")
            ]
        
        # Check each requirement
        missing = []
        for req in requirements:
            try:
                pkg_resources.require(req)
            except pkg_resources.DistributionNotFound:
                missing.append(req)
                
        if missing:
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            logger.info("Install missing dependencies with: pip install -r requirements.txt")
            return False
            
        logger.info("All dependencies installed")
        return True
        
    except Exception as e:
        logger.error(f"Error checking dependencies: {str(e)}")
        return False

def check_frameworks() -> Tuple[bool, bool]:
    """Check if FinTSB and mathematricks frameworks are available."""
    fintsb_ok = Path("FinTSB").exists() and Path("FinTSB/src").exists()
    mathematricks_ok = Path("mathematricks").exists() and Path("mathematricks/src").exists()
    
    if fintsb_ok:
        logger.info("FinTSB framework found")
    else:
        logger.error("FinTSB framework not found")
        
    if mathematricks_ok:
        logger.info("mathematricks framework found")
    else:
        logger.error("mathematricks framework not found")
        
    return fintsb_ok, mathematricks_ok

def check_data_directories() -> bool:
    """Check if required data directories exist."""
    required_dirs = [
        "data/ibkr/1d",
        "data/kraken/1d",
        "data/stocksymbolslists"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            
    if missing_dirs:
        logger.error(f"Missing data directories: {', '.join(missing_dirs)}")
        logger.info("Create missing directories with: mkdir -p " + " ".join(missing_dirs))
        return False
        
    logger.info("All required data directories exist")
    return True

def check_configurations() -> bool:
    """Check if configuration files are valid."""
    config_files = [
        "FinTSB/data/eastmoney_config.yaml",
        "FinTSB/configs/fintsb_lstm.yaml",
        "src/utils/config.py"
    ]
    
    all_valid = True
    for config_file in config_files:
        path = Path(config_file)
        if not path.exists():
            logger.error(f"Configuration file not found: {config_file}")
            all_valid = False
            continue
            
        if config_file.endswith('.yaml'):
            try:
                with open(path) as f:
                    yaml.safe_load(f)
                logger.info(f"Configuration file valid: {config_file}")
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML in {config_file}: {str(e)}")
                all_valid = False
                
    return all_valid

def check_credentials() -> bool:
    """Check if API credentials are configured."""
    try:
        from src.utils.credentials import CredentialManager
        manager = CredentialManager()
        if manager.check_credentials():
            logger.info("API credentials configured")
            return True
        else:
            logger.warning(
                "API credentials not configured. "
                "Run './download_eastmoney_data.py --interactive' to set up credentials."
            )
            return False
    except Exception as e:
        logger.error(f"Error checking credentials: {str(e)}")
        return False

def run_tests() -> bool:
    """Run project tests."""
    try:
        result = subprocess.run(
            ['python', 'run_tests.py', '--fast'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("All tests passed")
            return True
        else:
            logger.error("Some tests failed")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

def main():
    """Main validation function."""
    logger.info("Starting project validation...")
    
    # Track validation results
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies()
    }
    
    # Check frameworks
    fintsb_ok, mathematricks_ok = check_frameworks()
    results["FinTSB Framework"] = fintsb_ok
    results["mathematricks Framework"] = mathematricks_ok
    
    # Additional checks
    results.update({
        "Data Directories": check_data_directories(),
        "Configurations": check_configurations(),
        "API Credentials": check_credentials(),
        "Tests": run_tests()
    })
    
    # Print summary
    logger.info("\nValidation Summary:")
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{check:.<30}{status}")
        
    # Exit with appropriate status
    if all(results.values()):
        logger.info("\nAll checks passed!")
        sys.exit(0)
    else:
        logger.error("\nSome checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()