"""
Utility functions for coverage operations.
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import yaml

logger = logging.getLogger(__name__)

def parse_coverage_xml(xml_path: str) -> Dict[str, Any]:
    """
    Parse coverage data from XML report.
    
    Args:
        xml_path: Path to coverage XML file
        
    Returns:
        Dictionary containing coverage data
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        coverage_data = {
            'timestamp': datetime.now().isoformat(),
            'lines': {
                'total': int(root.get('lines-valid', 0)),
                'covered': int(root.get('lines-covered', 0)),
                'missed': 0
            },
            'branches': {
                'total': int(root.get('branches-valid', 0)),
                'covered': int(root.get('branches-covered', 0)),
                'missed': 0
            },
            'files': []
        }
        
        # Calculate missed lines/branches
        coverage_data['lines']['missed'] = (
            coverage_data['lines']['total'] - coverage_data['lines']['covered']
        )
        coverage_data['branches']['missed'] = (
            coverage_data['branches']['total'] - coverage_data['branches']['covered']
        )
        
        # Process individual files
        for class_elem in root.findall('.//class'):
            file_data = {
                'name': class_elem.get('name', ''),
                'lines': {
                    'total': len(class_elem.findall('.//line')),
                    'covered': len([
                        l for l in class_elem.findall('.//line')
                        if int(l.get('hits', 0)) > 0
                    ])
                }
            }
            file_data['lines']['missed'] = (
                file_data['lines']['total'] - file_data['lines']['covered']
            )
            coverage_data['files'].append(file_data)
            
        return coverage_data
        
    except Exception as e:
        logger.error(f"Error parsing coverage XML: {str(e)}")
        return {}

def calculate_coverage_stats(coverage_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate coverage statistics.
    
    Args:
        coverage_data: Coverage data dictionary
        
    Returns:
        Dictionary of coverage statistics
    """
    stats = {
        'line_coverage': 0.0,
        'branch_coverage': 0.0,
        'total_coverage': 0.0
    }
    
    try:
        # Line coverage
        if coverage_data['lines']['total'] > 0:
            stats['line_coverage'] = (
                coverage_data['lines']['covered'] /
                coverage_data['lines']['total'] * 100
            )
            
        # Branch coverage
        if coverage_data['branches']['total'] > 0:
            stats['branch_coverage'] = (
                coverage_data['branches']['covered'] /
                coverage_data['branches']['total'] * 100
            )
            
        # Total coverage (weighted average)
        total_elements = (
            coverage_data['lines']['total'] +
            coverage_data['branches']['total']
        )
        if total_elements > 0:
            covered_elements = (
                coverage_data['lines']['covered'] +
                coverage_data['branches']['covered']
            )
            stats['total_coverage'] = (
                covered_elements / total_elements * 100
            )
            
        return {k: round(v, 2) for k, v in stats.items()}
        
    except Exception as e:
        logger.error(f"Error calculating coverage stats: {str(e)}")
        return stats

def find_uncovered_lines(coverage_data: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Find uncovered lines in each file.
    
    Args:
        coverage_data: Coverage data dictionary
        
    Returns:
        Dictionary mapping filenames to lists of uncovered line numbers
    """
    uncovered = {}
    
    try:
        tree = ET.parse(coverage_data['xml_file'])
        root = tree.getroot()
        
        for class_elem in root.findall('.//class'):
            filename = class_elem.get('name', '')
            uncovered_lines = [
                int(line.get('number'))
                for line in class_elem.findall('.//line')
                if int(line.get('hits', 0)) == 0
            ]
            if uncovered_lines:
                uncovered[filename] = sorted(uncovered_lines)
                
        return uncovered
        
    except Exception as e:
        logger.error(f"Error finding uncovered lines: {str(e)}")
        return {}

def merge_coverage_data(data_files: List[str]) -> Dict[str, Any]:
    """
    Merge multiple coverage data files.
    
    Args:
        data_files: List of coverage data file paths
        
    Returns:
        Merged coverage data dictionary
    """
    merged = {
        'lines': {'total': 0, 'covered': 0, 'missed': 0},
        'branches': {'total': 0, 'covered': 0, 'missed': 0},
        'files': []
    }
    
    try:
        for data_file in data_files:
            data = parse_coverage_xml(data_file)
            if not data:
                continue
                
            # Merge line and branch counts
            for key in ['lines', 'branches']:
                for metric in ['total', 'covered', 'missed']:
                    merged[key][metric] += data[key][metric]
                    
            # Merge file data
            merged['files'].extend(data['files'])
            
        return merged
        
    except Exception as e:
        logger.error(f"Error merging coverage data: {str(e)}")
        return merged

def generate_coverage_badge(
    coverage: float,
    thresholds: Dict[str, float],
    output_path: str
) -> None:
    """
    Generate coverage badge.
    
    Args:
        coverage: Coverage percentage
        thresholds: Coverage threshold levels
        output_path: Path to save badge
    """
    try:
        from . import BADGE_COLORS
        
        # Determine color based on thresholds
        color = 'red'
        for level, threshold in sorted(
            thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if coverage >= threshold:
                color = BADGE_COLORS[level]
                break
                
        # Generate badge using shields.io
        import requests
        badge_url = (
            f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}"
        )
        response = requests.get(badge_url)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
                
    except Exception as e:
        logger.error(f"Error generating coverage badge: {str(e)}")

def cleanup_old_reports(
    reports_dir: str,
    max_age_days: int = 30
) -> None:
    """
    Clean up old coverage reports.
    
    Args:
        reports_dir: Reports directory path
        max_age_days: Maximum age of reports to keep
    """
    try:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        reports_path = Path(reports_dir)
        
        for item in reports_path.glob('**/*'):
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    item.unlink()
                    
    except Exception as e:
        logger.error(f"Error cleaning up reports: {str(e)}")

def archive_coverage_data(
    coverage_data: Dict[str, Any],
    archive_dir: str
) -> None:
    """
    Archive coverage data.
    
    Args:
        coverage_data: Coverage data dictionary
        archive_dir: Archive directory path
    """
    try:
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_file = archive_path / f"coverage_{timestamp}.json"
        
        with open(archive_file, 'w') as f:
            json.dump(coverage_data, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error archiving coverage data: {str(e)}")

def compare_coverage(
    current: Dict[str, Any],
    previous: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare current and previous coverage data.
    
    Args:
        current: Current coverage data
        previous: Previous coverage data
        
    Returns:
        Comparison results dictionary
    """
    try:
        return {
            'line_coverage_change': (
                current['lines']['covered'] / current['lines']['total'] * 100 -
                previous['lines']['covered'] / previous['lines']['total'] * 100
            ),
            'branch_coverage_change': (
                current['branches']['covered'] / current['branches']['total'] * 100 -
                previous['branches']['covered'] / previous['branches']['total'] * 100
            ),
            'new_files': [
                f['name'] for f in current['files']
                if f['name'] not in [pf['name'] for pf in previous['files']]
            ],
            'removed_files': [
                f['name'] for f in previous['files']
                if f['name'] not in [cf['name'] for cf in current['files']]
            ]
        }
    except Exception as e:
        logger.error(f"Error comparing coverage: {str(e)}")
        return {}