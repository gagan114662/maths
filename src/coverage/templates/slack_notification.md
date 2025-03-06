:chart_with_upwards_trend: *Coverage Report Update*

*Generated on {{ timestamp }}*

*Overall Coverage: {{ "%.1f"|format(stats.total_coverage) }}%*
{{ get_coverage_emoji(stats.total_coverage) }}

## Coverage Metrics
• Line Coverage: {{ "%.1f"|format(stats.line_coverage) }}% {{ get_threshold_emoji(stats.line_coverage) }}
• Branch Coverage: {{ "%.1f"|format(stats.branch_coverage) }}% {{ get_threshold_emoji(stats.branch_coverage) }}
• Total Coverage: {{ "%.1f"|format(stats.total_coverage) }}% {{ get_threshold_emoji(stats.total_coverage) }}

{% if comparison %}
## Changes from Previous Report
{% if comparison.line_coverage_change >= 0 %}
:arrow_up: Line Coverage: +{{ "%.1f"|format(comparison.line_coverage_change) }}%
{% else %}
:arrow_down: Line Coverage: {{ "%.1f"|format(comparison.line_coverage_change) }}%
{% endif %}

{% if comparison.branch_coverage_change >= 0 %}
:arrow_up: Branch Coverage: +{{ "%.1f"|format(comparison.branch_coverage_change) }}%
{% else %}
:arrow_down: Branch Coverage: {{ "%.1f"|format(comparison.branch_coverage_change) }}%
{% endif %}

{% if comparison.new_files %}
*New Files:*
{% for file in comparison.new_files %}
• {{ file }}
{% endfor %}
{% endif %}

{% if comparison.removed_files %}
*Removed Files:*
{% for file in comparison.removed_files %}
• {{ file }}
{% endfor %}
{% endif %}
{% endif %}

## Status
{% set status_class = get_threshold_class(stats.total_coverage) %}
{% if status_class == "excellent" %}
:white_check_mark: *Coverage is excellent*
{% elif status_class == "good" %}
:large_blue_circle: *Coverage is good*
{% elif status_class == "acceptable" %}
:warning: *Coverage needs improvement*
{% elif status_class == "poor" %}
:x: *Coverage is poor*
{% else %}
:rotating_light: *Coverage is critical*
{% endif %}

## Thresholds
{% for level, value in thresholds.items() %}
• {{ level|title }}: {{ value }}%
{% endfor %}

---
_<{{ dashboard_url }}|View detailed report>_