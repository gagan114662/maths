"""
Training module for integrating FinTSB with enhanced evaluation metrics.
"""
from pathlib import Path

# Add FinTSB path to environment
FINTSB_PATH = Path(__file__).parent.parent.parent / "FinTSB"
MATHEMATRICKS_PATH = Path(__file__).parent.parent.parent / "mathematricks"

__all__ = ['train_fintsb']