"""
Helper functions for coverage report templates.
"""
from typing import Dict, Any
from . import COVERAGE_THRESHOLDS, BADGE_COLORS

def get_threshold_class(coverage: float) -> str:
    """
    Get CSS class based on coverage threshold.
    
    Args:
        coverage: Coverage percentage
        
    Returns:
        CSS class name for styling
    """
    if coverage >= COVERAGE_THRESHOLDS['excellent']:
        return 'excellent'
    elif coverage >= COVERAGE_THRESHOLDS['good']:
        return 'good'
    elif coverage >= COVERAGE_THRESHOLDS['acceptable']:
        return 'acceptable'
    elif coverage >= COVERAGE_THRESHOLDS['poor']:
        return 'poor'
    return 'critical'

def get_threshold_icon(coverage: float) -> str:
    """
    Get icon based on coverage threshold.
    
    Args:
        coverage: Coverage percentage
        
    Returns:
        Icon/emoji representing coverage level
    """
    if coverage >= COVERAGE_THRESHOLDS['excellent']:
        return '🟢'  # Green circle
    elif coverage >= COVERAGE_THRESHOLDS['good']:
        return '🔵'  # Blue circle
    elif coverage >= COVERAGE_THRESHOLDS['acceptable']:
        return '🟡'  # Yellow circle
    elif coverage >= COVERAGE_THRESHOLDS['poor']:
        return '🔴'  # Red circle
    return '⚫'  # Black circle

def get_threshold_emoji(coverage: float) -> str:
    """
    Get emoji based on coverage threshold for Slack.
    
    Args:
        coverage: Coverage percentage
        
    Returns:
        Slack emoji representing coverage level
    """
    if coverage >= COVERAGE_THRESHOLDS['excellent']:
        return ':large_green_circle:'
    elif coverage >= COVERAGE_THRESHOLDS['good']:
        return ':large_blue_circle:'
    elif coverage >= COVERAGE_THRESHOLDS['acceptable']:
        return ':warning:'
    elif coverage >= COVERAGE_THRESHOLDS['poor']:
        return ':x:'
    return ':skull:'

def get_coverage_trend(current: float, history: list) -> str:
    """
    Generate ASCII art trend for coverage.
    
    Args:
        current: Current coverage percentage
        history: List of historical coverage values
        
    Returns:
        ASCII art representation of trend
    """
    trend = ''
    if history:
        last = history[-1]
        if current > last:
            trend = '📈'  # Trending up
        elif current < last:
            trend = '📉'  # Trending down
        else:
            trend = '➡️'  # No change
    return trend

def format_change(value: float) -> str:
    """
    Format coverage change value.
    
    Args:
        value: Coverage change value
        
    Returns:
        Formatted string with sign and color indicators
    """
    if value > 0:
        return f"+{value:.1f}% 📈"
    elif value < 0:
        return f"{value:.1f}% 📉"
    return f"{value:.1f}% ➡️"

def generate_coverage_graph(value: float, width: int = 20) -> str:
    """
    Generate ASCII progress bar for coverage.
    
    Args:
        value: Coverage percentage
        width: Width of progress bar
        
    Returns:
        ASCII progress bar
    """
    filled = int(value * width / 100)
    empty = width - filled
    return f"[{'=' * filled}{' ' * empty}] {value:.1f}%"

def get_badge_color(coverage: float) -> str:
    """
    Get badge color based on coverage threshold.
    
    Args:
        coverage: Coverage percentage
        
    Returns:
        Color code for badge
    """
    return BADGE_COLORS[get_threshold_class(coverage)]

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def get_template_context(
    coverage_data: Dict[str, Any],
    stats: Dict[str, float],
    comparison: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Get complete template context with helper functions.
    
    Args:
        coverage_data: Coverage data dictionary
        stats: Coverage statistics
        comparison: Optional comparison data
        
    Returns:
        Template context dictionary with helper functions
    """
    return {
        'coverage_data': coverage_data,
        'stats': stats,
        'comparison': comparison,
        'thresholds': COVERAGE_THRESHOLDS,
        'get_threshold_class': get_threshold_class,
        'get_threshold_icon': get_threshold_icon,
        'get_threshold_emoji': get_threshold_emoji,
        'get_coverage_trend': get_coverage_trend,
        'format_change': format_change,
        'generate_coverage_graph': generate_coverage_graph,
        'get_badge_color': get_badge_color,
        'format_duration': format_duration
    }