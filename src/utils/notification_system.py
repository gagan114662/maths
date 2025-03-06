"""
Notification system for strategy completion and system events.
"""
import logging
import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import requests

logger = logging.getLogger(__name__)

class NotificationSystem:
    """
    Notification system for strategy completion and system events.
    Supports email, Slack, and console notifications.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the notification system.
        
        Args:
            config_path: Path to the notification config JSON file
        """
        self.config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "",
                "to_addresses": []
            },
            "slack": {
                "enabled": False,
                "webhook_url": ""
            },
            "console": {
                "enabled": True,
                "color": True
            }
        }
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Update config with user settings
                    for key, value in user_config.items():
                        if key in self.config:
                            self.config[key].update(value)
            except Exception as e:
                logger.error(f"Error loading notification config: {str(e)}")
    
    def notify_strategy_completion(self, strategy_data: Dict[str, Any]) -> bool:
        """
        Send a notification for strategy completion.
        
        Args:
            strategy_data: Dictionary containing strategy information
                Should contain:
                - strategy_name: Name of the strategy
                - performance: Performance metrics dictionary
                
        Returns:
            True if notification was successful, False otherwise
        """
        # Create the notification message
        subject = f"Strategy Completed: {strategy_data.get('strategy_name', 'Unknown')}"
        
        performance = strategy_data.get('performance', {})
        cagr = performance.get('annualized_return', 0) * 100
        sharpe = performance.get('sharpe_ratio', 0)
        drawdown = performance.get('max_drawdown', 0) * 100
        
        # Generate message with performance details
        message = f"Strategy '{strategy_data.get('strategy_name', 'Unknown')}' has completed processing.\n\n"
        message += f"Performance Metrics:\n"
        message += f"- CAGR: {cagr:.2f}%\n"
        message += f"- Sharpe Ratio: {sharpe:.2f}\n"
        message += f"- Max Drawdown: {drawdown:.2f}%\n"
        
        # Add trade metrics if available
        trades = strategy_data.get('trades', {})
        if trades:
            message += f"\nTrade Metrics:\n"
            message += f"- Total Trades: {trades.get('total_trades', 0)}\n"
            message += f"- Win Rate: {trades.get('win_rate', 0) * 100:.2f}%\n"
            message += f"- Average Trade: {trades.get('average_trade', 0) * 100:.2f}%\n"
        
        # Add strategy details
        message += f"\nStrategy Details:\n"
        message += f"- Universe: {strategy_data.get('universe', 'Unknown')}\n"
        message += f"- Timeframe: {strategy_data.get('timeframe', 'Unknown')}\n"
        message += f"- Description: {strategy_data.get('description', 'No description provided')}\n"
        
        # Send notifications through all enabled channels
        success = True
        
        # Console notification (always enabled)
        self._notify_console(subject, message)
        
        # Email notification
        if self.config["email"]["enabled"]:
            email_success = self._notify_email(subject, message)
            success = success and email_success
        
        # Slack notification
        if self.config["slack"]["enabled"]:
            slack_success = self._notify_slack(subject, message)
            success = success and slack_success
        
        return success
    
    def notify_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a notification for a system event.
        
        Args:
            event_type: Type of event (e.g., "ERROR", "WARNING", "INFO", "SUCCESS")
            message: Main message for the notification
            details: Additional details to include in the notification
            
        Returns:
            True if notification was successful, False otherwise
        """
        # Create the notification message
        subject = f"System Event: {event_type}"
        
        # Format the message
        full_message = f"{message}\n"
        
        # Add details if provided
        if details:
            full_message += f"\nDetails:\n"
            for key, value in details.items():
                full_message += f"- {key}: {value}\n"
        
        # Add timestamp
        full_message += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send notifications through all enabled channels
        success = True
        
        # Console notification (always enabled)
        self._notify_console(subject, full_message, event_type)
        
        # Only send emails and Slack notifications for ERROR and WARNING events
        if event_type in ["ERROR", "WARNING"]:
            # Email notification
            if self.config["email"]["enabled"]:
                email_success = self._notify_email(subject, full_message)
                success = success and email_success
            
            # Slack notification
            if self.config["slack"]["enabled"]:
                slack_success = self._notify_slack(subject, full_message)
                success = success and slack_success
        
        return success
    
    def _notify_console(self, subject: str, message: str, event_type: str = "INFO") -> None:
        """
        Print a notification to the console.
        
        Args:
            subject: Notification subject
            message: Notification message
            event_type: Type of event for color coding
        """
        if not self.config["console"]["enabled"]:
            return
        
        # Use colors if enabled
        if self.config["console"]["color"]:
            # ANSI color codes
            colors = {
                "ERROR": "\033[91m",  # Red
                "WARNING": "\033[93m",  # Yellow
                "INFO": "\033[94m",    # Blue
                "SUCCESS": "\033[92m",  # Green
                "RESET": "\033[0m"     # Reset
            }
            
            color = colors.get(event_type, colors["INFO"])
            reset = colors["RESET"]
            
            print(f"\n{color}{'=' * 80}{reset}")
            print(f"{color}{subject}{reset}")
            print(f"{color}{'-' * 80}{reset}")
            print(f"{message}")
            print(f"{color}{'=' * 80}{reset}\n")
        else:
            # No colors
            print(f"\n{'=' * 80}")
            print(f"{subject}")
            print(f"{'-' * 80}")
            print(f"{message}")
            print(f"{'=' * 80}\n")
    
    def _notify_email(self, subject: str, message: str) -> bool:
        """
        Send an email notification.
        
        Args:
            subject: Email subject
            message: Email message
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        try:
            # Check if email is configured
            if not self.config["email"]["username"] or not self.config["email"]["password"]:
                logger.warning("Email notification not sent: Missing username or password")
                return False
            
            if not self.config["email"]["to_addresses"]:
                logger.warning("Email notification not sent: No recipients specified")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.config["email"]["from_address"] or self.config["email"]["username"]
            msg["To"] = ", ".join(self.config["email"]["to_addresses"])
            msg["Subject"] = subject
            
            # Add message body
            msg.attach(MIMEText(message, "plain"))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"]) as server:
                server.starttls()
                server.login(self.config["email"]["username"], self.config["email"]["password"])
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {len(self.config['email']['to_addresses'])} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False
    
    def _notify_slack(self, subject: str, message: str) -> bool:
        """
        Send a Slack notification.
        
        Args:
            subject: Notification subject
            message: Notification message
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            # Check if Slack is configured
            if not self.config["slack"]["webhook_url"]:
                logger.warning("Slack notification not sent: Missing webhook URL")
                return False
            
            # Format the message for Slack
            slack_message = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": subject,
                            "emoji": True
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message.replace("\n", "\n> ")
                        }
                    }
                ]
            }
            
            # Send the message to Slack
            response = requests.post(
                self.config["slack"]["webhook_url"],
                json=slack_message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
                return True
            else:
                logger.error(f"Error sending Slack notification: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                        from_address: str, to_addresses: List[str]) -> None:
        """
        Configure email notifications.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: Email username
            password: Email password
            from_address: Sender email address
            to_addresses: List of recipient email addresses
        """
        self.config["email"]["enabled"] = True
        self.config["email"]["smtp_server"] = smtp_server
        self.config["email"]["smtp_port"] = smtp_port
        self.config["email"]["username"] = username
        self.config["email"]["password"] = password
        self.config["email"]["from_address"] = from_address
        self.config["email"]["to_addresses"] = to_addresses
        logger.info("Email notifications configured")
    
    def configure_slack(self, webhook_url: str) -> None:
        """
        Configure Slack notifications.
        
        Args:
            webhook_url: Slack webhook URL
        """
        self.config["slack"]["enabled"] = True
        self.config["slack"]["webhook_url"] = webhook_url
        logger.info("Slack notifications configured")
    
    def disable_email(self) -> None:
        """Disable email notifications."""
        self.config["email"]["enabled"] = False
        logger.info("Email notifications disabled")
    
    def disable_slack(self) -> None:
        """Disable Slack notifications."""
        self.config["slack"]["enabled"] = False
        logger.info("Slack notifications disabled")
    
    def disable_console(self) -> None:
        """Disable console notifications."""
        self.config["console"]["enabled"] = False
        logger.info("Console notifications disabled")