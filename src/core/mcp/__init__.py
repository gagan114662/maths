"""
Model Context Protocol (MCP) package.

This package implements the MCP pattern for standardizing context between LLMs and tools.
"""

from .protocol import ModelContextProtocol
from .context import Context, ContextType, ContextProvider

__all__ = [
    'ModelContextProtocol',
    'Context',
    'ContextType',
    'ContextProvider',
]