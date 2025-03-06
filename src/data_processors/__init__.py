"""
Data processors module for handling different data sources and preprocessing.
"""
from .base_connector import BaseDataConnector
from .ibkr_connector import IBKRDataConnector

__all__ = [
    'BaseDataConnector',
    'IBKRDataConnector',
]

# Version control
__version__ = '0.1.0'

# Data source configurations
DATA_SOURCES = {
    'ibkr': {
        'timeframes': ['1d', '1m'],
        'required_columns': ['open', 'high', 'low', 'close', 'volume', 'average', 'barCount']
    },
    'fintsb': {
        'categories': ['extreme', 'fall', 'fluctuation', 'rise'],
        'datasets_per_category': 5
    }
}