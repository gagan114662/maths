"""
Utilities for managing API credentials securely.
"""

def load_credentials(keys=None):
    """Simple helper to load credentials (returns empty dict when using Ollama)."""
    return {}
import os
from pathlib import Path
from typing import Optional, Dict
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class CredentialManager:
    """Manage API credentials securely."""
    
    ENV_FILE = ".env"
    REQUIRED_VARS = {
        # No required API keys when using Ollama
    }
    
    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize credential manager.
        
        Args:
            env_file: Optional path to .env file
        """
        self.env_file = env_file or Path.cwd() / self.ENV_FILE
        self._load_env()

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        if self.env_file.exists():
            load_dotenv(self.env_file)
        else:
            logger.warning(f"No .env file found at {self.env_file}")

    def get_credentials(self) -> Dict[str, str]:
        """
        Get all required credentials.
        
        Returns:
            Dictionary of credential key-value pairs
        """
        credentials = {}
        missing = []
        
        for var, description in self.REQUIRED_VARS.items():
            value = os.getenv(var)
            if value:
                credentials[var] = value
            else:
                missing.append(var)
                
        if missing and self.REQUIRED_VARS:
            logger.error(f"Missing required credentials: {missing}")
            raise ValueError(f"Missing required credentials: {missing}")
            
        return credentials

    def set_credential(self, key: str, value: str) -> None:
        """
        Set a credential in the .env file.
        
        Args:
            key: Credential key
            value: Credential value
        """
        if key not in self.REQUIRED_VARS:
            raise ValueError(f"Invalid credential key: {key}")
            
        # Read existing contents
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
            
        # Find and replace or append
        key_found = False
        new_lines = []
        
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                new_lines.append(line)
                
        if not key_found:
            new_lines.append(f"{key}={value}\n")
            
        # Write back to file
        with open(self.env_file, 'w') as f:
            f.writelines(new_lines)
            
        logger.info(f"Updated credential: {key}")

    def check_credentials(self) -> bool:
        """
        Check if all required credentials are available.
        
        Returns:
            Boolean indicating if all credentials are available
        """
        try:
            self.get_credentials()
            return True
        except ValueError:
            return False

    def create_example_env(self) -> None:
        """Create example .env file with placeholders."""
        example_file = self.env_file.with_suffix('.env.example')
        
        with open(example_file, 'w') as f:
            f.write("# EastMoney API Credentials\n")
            for var, description in self.REQUIRED_VARS.items():
                f.write(f"\n# {description}\n")
                f.write(f"{var}=your_{var.lower()}_here\n")
                
        logger.info(f"Created example environment file: {example_file}")

    def validate_credentials(self) -> Dict[str, bool]:
        """
        Validate format of credentials.
        
        Returns:
            Dictionary of credential validation results
        """
        validation = {}
        credentials = self.get_credentials()
        
        for key, value in credentials.items():
            if key == 'EASTMONEY_API_KEY':
                # Check API key format (example: should be 32 chars)
                validation[key] = len(value) == 32
            elif key == 'EASTMONEY_API_SECRET':
                # Check API secret format (example: should be 64 chars)
                validation[key] = len(value) == 64
                
        return validation

def setup_credentials(interactive: bool = True) -> CredentialManager:
    """
    Set up credentials interactively or from environment.
    
    Args:
        interactive: Whether to prompt for missing credentials
        
    Returns:
        Configured CredentialManager instance
    """
    manager = CredentialManager()
    
    if not manager.check_credentials():
        if interactive:
            print("\nSetting up EastMoney API credentials:")
            for var in manager.REQUIRED_VARS:
                if not os.getenv(var):
                    value = input(f"Enter {var}: ")
                    manager.set_credential(var, value)
        else:
            manager.create_example_env()
            raise ValueError(
                "Missing credentials. Please set them in .env file. "
                "See .env.example for required variables."
            )
            
    return manager