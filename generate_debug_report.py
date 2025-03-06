#!/usr/bin/env python3
"""
Generate a debug report for troubleshooting.
"""
import os
import sys
import platform
import subprocess
import pkg_resources
from pathlib import Path
import json
from datetime import datetime
import shutil
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_path": sys.executable,
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
        "environment": {k: v for k, v in os.environ.items() if "PATH" in k or "PYTHON" in k},
        "timestamp": datetime.now().isoformat()
    }

def get_pip_packages() -> Dict[str, str]:
    """Get installed pip packages and versions."""
    return {
        pkg.key: pkg.version
        for pkg in pkg_resources.working_set
    }

def check_directories() -> Dict[str, bool]:
    """Check if required directories exist."""
    required_dirs = [
        "data/ibkr/1d",
        "data/kraken/1d",
        "FinTSB/data",
        "src/agents",
        "src/strategies",
        "src/training",
        "tests"
    ]
    return {
        dir_path: Path(dir_path).exists()
        for dir_path in required_dirs
    }

def check_executables() -> Dict[str, bool]:
    """Check if required executables are available."""
    executables = [
        "python",
        "pip",
        "git",
        "make"
    ]
    return {
        exe: bool(shutil.which(exe))
        for exe in executables
    }

def check_git_status() -> Dict[str, Any]:
    """Get git repository status."""
    try:
        return {
            "branch": subprocess.getoutput("git rev-parse --abbrev-ref HEAD"),
            "commit": subprocess.getoutput("git rev-parse HEAD"),
            "status": subprocess.getoutput("git status --porcelain"),
            "remote": subprocess.getoutput("git remote -v")
        }
    except Exception as e:
        return {"error": str(e)}

def check_config_files() -> Dict[str, bool]:
    """Check if configuration files exist and are valid."""
    config_files = {
        "eastmoney_config": "FinTSB/data/eastmoney_config.yaml",
        "lstm_config": "FinTSB/configs/fintsb_lstm.yaml",
        "pytest_config": "pytest.ini",
        "setup_config": "setup.cfg",
        "manifest": "MANIFEST.in",
        "makefile": "Makefile"
    }
    
    return {
        name: Path(path).exists()
        for name, path in config_files.items()
    }

def check_permissions() -> Dict[str, Dict[str, str]]:
    """Check permissions of important files."""
    important_files = [
        "*.py",
        "*.sh",
        "data",
        "FinTSB/data"
    ]
    
    results = {}
    for pattern in important_files:
        try:
            output = subprocess.getoutput(f"ls -l {pattern}")
            results[pattern] = {"permissions": output}
        except Exception as e:
            results[pattern] = {"error": str(e)}
    
    return results

def generate_report() -> Dict[str, Any]:
    """Generate complete debug report."""
    report = {
        "system_info": get_system_info(),
        "pip_packages": get_pip_packages(),
        "directories": check_directories(),
        "executables": check_executables(),
        "git_status": check_git_status(),
        "config_files": check_config_files(),
        "permissions": check_permissions()
    }
    
    # Add validation results if available
    try:
        import validate_setup
        report["validation"] = validate_setup.main()
    except Exception as e:
        report["validation"] = {"error": str(e)}
    
    return report

def save_report(report: Dict[str, Any], output_file: str = "debug_report.json") -> None:
    """Save report to file."""
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Debug report saved to {output_file}")

def main():
    """Main execution function."""
    try:
        logger.info("Generating debug report...")
        report = generate_report()
        
        # Save full report
        save_report(report)
        
        # Print summary
        print("\nDebug Report Summary:")
        print(f"Python: {report['system_info']['python_version'].split()[0]}")
        print(f"Platform: {report['system_info']['platform']}")
        print("\nDirectory Status:")
        for dir_path, exists in report['directories'].items():
            status = "✓" if exists else "✗"
            print(f"{status} {dir_path}")
        
        print("\nConfiguration Files:")
        for name, exists in report['config_files'].items():
            status = "✓" if exists else "✗"
            print(f"{status} {name}")
        
        print("\nValidation Status:")
        if isinstance(report.get('validation'), dict):
            if 'error' in report['validation']:
                print(f"✗ Validation failed: {report['validation']['error']}")
            else:
                print("✓ Validation passed")
        else:
            print("? Validation status unknown")
        
        print(f"\nFull report saved to {os.path.abspath('debug_report.json')}")
        
    except Exception as e:
        logger.error(f"Error generating debug report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()