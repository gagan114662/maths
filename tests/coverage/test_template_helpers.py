"""
Tests for coverage report template helpers.
"""
import pytest
from datetime import datetime, timedelta
from src.coverage.template_helpers import (
    get_threshold_class,
    get_threshold_icon,
    get_threshold_emoji,
    get_coverage_trend,
    format_change,
    generate_coverage_graph,
    get_badge_color,
    format_duration,
    get_template_context
)
from src.coverage import COVERAGE_THRESHOLDS, BADGE_COLORS

@pytest.mark.parametrize("coverage,expected_class", [
    (95.0, "excellent"),
    (85.0, "good"),
    (75.0, "acceptable"),
    (65.0, "poor"),
    (45.0, "critical"),
])
def test_get_threshold_class(coverage, expected_class):
    """Test threshold class determination."""
    assert get_threshold_class(coverage) == expected_class

@pytest.mark.parametrize("coverage,expected_icon", [
    (95.0, "🟢"),
    (85.0, "🔵"),
    (75.0, "🟡"),
    (65.0, "🔴"),
    (45.0, "⚫"),
])
def test_get_threshold_icon(coverage, expected_icon):
    """Test threshold icon determination."""
    assert get_threshold_icon(coverage) == expected_icon

@pytest.mark.parametrize("coverage,expected_emoji", [
    (95.0, ":large_green_circle:"),
    (85.0, ":large_blue_circle:"),
    (75.0, ":warning:"),
    (65.0, ":x:"),
    (45.0, ":skull:"),
])
def test_get_threshold_emoji(coverage, expected_emoji):
    """Test threshold emoji determination."""
    assert get_threshold_emoji(coverage) == expected_emoji

@pytest.mark.parametrize("current,history,expected", [
    (80.0, [75.0], "📈"),
    (70.0, [75.0], "📉"),
    (75.0, [75.0], "➡️"),
    (80.0, [], ""),
])
def test_get_coverage_trend(current, history, expected):
    """Test coverage trend determination."""
    assert get_coverage_trend(current, history) == expected

@pytest.mark.parametrize("value,expected", [
    (5.0, "+5.0% 📈"),
    (-3.0, "-3.0% 📉"),
    (0.0, "0.0% ➡️"),
])
def test_format_change(value, expected):
    """Test change value formatting."""
    assert format_change(value) == expected

@pytest.mark.parametrize("value,width,expected", [
    (100.0, 10, "[==========] 100.0%"),
    (50.0, 10, "[=====     ] 50.0%"),
    (0.0, 10, "[          ] 0.0%"),
])
def test_generate_coverage_graph(value, width, expected):
    """Test ASCII progress bar generation."""
    assert generate_coverage_graph(value, width) == expected

@pytest.mark.parametrize("coverage,expected_color", [
    (95.0, BADGE_COLORS["excellent"]),
    (85.0, BADGE_COLORS["good"]),
    (75.0, BADGE_COLORS["acceptable"]),
    (65.0, BADGE_COLORS["poor"]),
    (45.0, BADGE_COLORS["critical"]),
])
def test_get_badge_color(coverage, expected_color):
    """Test badge color determination."""
    assert get_badge_color(coverage) == expected_color

@pytest.mark.parametrize("seconds,expected", [
    (30, "30.0s"),
    (90, "1.5m"),
    (3600, "1.0h"),
])
def test_format_duration(seconds, expected):
    """Test duration formatting."""
    assert format_duration(seconds) == expected

def test_get_template_context():
    """Test template context generation."""
    coverage_data = {
        "lines": {"total": 100, "covered": 75, "missed": 25},
        "files": []
    }
    stats = {
        "line_coverage": 75.0,
        "branch_coverage": 80.0,
        "total_coverage": 77.5
    }
    comparison = {
        "line_coverage_change": 5.0,
        "branch_coverage_change": -2.0
    }
    
    context = get_template_context(coverage_data, stats, comparison)
    
    # Check required keys
    assert "coverage_data" in context
    assert "stats" in context
    assert "comparison" in context
    assert "thresholds" in context
    
    # Check helper functions
    assert callable(context["get_threshold_class"])
    assert callable(context["get_threshold_icon"])
    assert callable(context["get_threshold_emoji"])
    assert callable(context["get_coverage_trend"])
    assert callable(context["format_change"])
    assert callable(context["generate_coverage_graph"])
    assert callable(context["get_badge_color"])
    assert callable(context["format_duration"])

def test_template_context_without_comparison():
    """Test template context generation without comparison data."""
    context = get_template_context({"files": []}, {"total_coverage": 75.0})
    assert context["comparison"] is None

@pytest.mark.parametrize("coverage", [
    -10.0,  # Below valid range
    110.0,  # Above valid range
])
def test_invalid_coverage_values(coverage):
    """Test handling of invalid coverage values."""
    # Should not raise exceptions
    get_threshold_class(coverage)
    get_threshold_icon(coverage)
    get_threshold_emoji(coverage)
    get_badge_color(coverage)
    generate_coverage_graph(coverage)

def test_zero_width_graph():
    """Test graph generation with zero width."""
    assert generate_coverage_graph(75.0, 0) == "[] 75.0%"

def test_negative_duration():
    """Test duration formatting with negative value."""
    assert format_duration(-10) == "0.0s"