"""
Generate and distribute coverage reports.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import jinja2
import markdown

from .utils import calculate_coverage_stats, compare_coverage
from . import COVERAGE_THRESHOLDS, BADGE_COLORS

logger = logging.getLogger(__name__)

class CoverageReporter:
    """Generates and distributes coverage reports."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reporter with configuration."""
        self.config = config
        self.env = jinja2.Environment(
            loader=jinja2.PackageLoader('src.coverage', 'templates')
        )
        
    def generate_reports(
        self,
        coverage_data: Dict[str, Any],
        output_dir: str,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate coverage reports in specified formats.
        
        Args:
            coverage_data: Coverage data dictionary
            output_dir: Output directory path
            formats: List of report formats to generate
            
        Returns:
            Dictionary mapping format to report file path
        """
        if formats is None:
            formats = ['html', 'markdown', 'json']
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        try:
            stats = calculate_coverage_stats(coverage_data)
            
            for fmt in formats:
                if fmt == 'html':
                    reports['html'] = self._generate_html_report(
                        coverage_data,
                        stats,
                        output_path / 'coverage.html'
                    )
                elif fmt == 'markdown':
                    reports['markdown'] = self._generate_markdown_report(
                        coverage_data,
                        stats,
                        output_path / 'coverage.md'
                    )
                elif fmt == 'json':
                    reports['json'] = self._generate_json_report(
                        coverage_data,
                        stats,
                        output_path / 'coverage.json'
                    )
                    
            # Generate badge if configured
            if self.config.get('badges', {}).get('generate', False):
                self._generate_badge(
                    stats['total_coverage'],
                    output_path / 'coverage-badge.svg'
                )
                
            return reports
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return {}
            
    def _generate_html_report(
        self,
        coverage_data: Dict[str, Any],
        stats: Dict[str, float],
        output_path: Path
    ) -> str:
        """Generate HTML coverage report."""
        try:
            template = self.env.get_template('coverage_report.html')
            html = template.render(
                coverage_data=coverage_data,
                stats=stats,
                thresholds=COVERAGE_THRESHOLDS,
                timestamp=datetime.now().isoformat()
            )
            
            output_path.write_text(html)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return ""
            
    def _generate_markdown_report(
        self,
        coverage_data: Dict[str, Any],
        stats: Dict[str, float],
        output_path: Path
    ) -> str:
        """Generate Markdown coverage report."""
        try:
            template = self.env.get_template('coverage_report.md')
            md = template.render(
                coverage_data=coverage_data,
                stats=stats,
                thresholds=COVERAGE_THRESHOLDS,
                timestamp=datetime.now().isoformat()
            )
            
            output_path.write_text(md)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating Markdown report: {str(e)}")
            return ""
            
    def _generate_json_report(
        self,
        coverage_data: Dict[str, Any],
        stats: Dict[str, float],
        output_path: Path
    ) -> str:
        """Generate JSON coverage report."""
        try:
            report_data = {
                'coverage': coverage_data,
                'statistics': stats,
                'thresholds': COVERAGE_THRESHOLDS,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            return ""
            
    def _generate_badge(self, coverage: float, output_path: Path) -> None:
        """Generate coverage badge."""
        try:
            # Determine badge color
            color = BADGE_COLORS['critical']
            for level, threshold in sorted(
                COVERAGE_THRESHOLDS.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if coverage >= threshold:
                    color = BADGE_COLORS[level]
                    break
                    
            # Generate badge using shields.io
            badge_url = (
                f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}"
            )
            response = requests.get(badge_url)
            
            if response.status_code == 200:
                output_path.write_bytes(response.content)
                
        except Exception as e:
            logger.error(f"Error generating badge: {str(e)}")
            
    def notify(
        self,
        current_data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send notifications about coverage changes."""
        try:
            stats = calculate_coverage_stats(current_data)
            comparison = None
            
            if previous_data:
                comparison = compare_coverage(current_data, previous_data)
                
            notification_config = self.config.get('notifications', {})
            
            # Email notification
            if notification_config.get('email', {}).get('enabled', False):
                self._send_email_notification(stats, comparison)
                
            # Slack notification
            if notification_config.get('slack', {}).get('enabled', False):
                self._send_slack_notification(stats, comparison)
                
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
            
    def _send_email_notification(
        self,
        stats: Dict[str, float],
        comparison: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send email notification."""
        try:
            email_config = self.config['notifications']['email']
            
            msg = MIMEMultipart()
            msg['Subject'] = 'Coverage Report Update'
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['recipients'])
            
            template = self.env.get_template('email_notification.html')
            body = template.render(
                stats=stats,
                comparison=comparison,
                thresholds=COVERAGE_THRESHOLDS
            )
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(email_config['smtp_server']) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                if 'username' in email_config:
                    server.login(
                        email_config['username'],
                        email_config['password']
                    )
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            
    def _send_slack_notification(
        self,
        stats: Dict[str, float],
        comparison: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send Slack notification."""
        try:
            slack_config = self.config['notifications']['slack']
            
            template = self.env.get_template('slack_notification.md')
            message = template.render(
                stats=stats,
                comparison=comparison,
                thresholds=COVERAGE_THRESHOLDS
            )
            
            payload = {
                'text': markdown.markdown(message),
                'username': slack_config.get('username', 'Coverage Bot'),
                'icon_emoji': slack_config.get('icon_emoji', ':chart_with_upwards_trend:')
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(
                    f"Error sending Slack notification: {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            
    def archive_report(
        self,
        report_path: str,
        archive_dir: str
    ) -> None:
        """Archive a coverage report."""
        try:
            source_path = Path(report_path)
            archive_path = Path(archive_dir)
            archive_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = source_path.suffix
            archived_file = archive_path / f"coverage_{timestamp}{extension}"
            
            import shutil
            shutil.copy2(source_path, archived_file)
            
        except Exception as e:
            logger.error(f"Error archiving report: {str(e)}")