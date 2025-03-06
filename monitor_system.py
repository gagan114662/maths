#!/usr/bin/env python3
"""
Command-line script to manage system monitoring.
"""
import os
import sys
import argparse
import logging
import threading
from pathlib import Path
import webbrowser
from typing import Optional

from src.utils.monitor import SystemMonitor
from src.utils.monitor_web import start_web_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringManager:
    """Manages system monitoring components."""
    
    def __init__(self):
        """Initialize manager."""
        self.monitor = None
        self.web_thread = None
        self.running = False
        
    def start_monitoring(
        self,
        web: bool = True,
        host: str = '0.0.0.0',
        port: int = 5000,
        open_browser: bool = True
    ) -> None:
        """
        Start system monitoring.
        
        Args:
            web: Whether to start web interface
            host: Web interface host
            port: Web interface port
            open_browser: Whether to open browser automatically
        """
        try:
            # Create logs directory
            Path('logs').mkdir(exist_ok=True)
            
            # Start system monitor
            self.monitor = SystemMonitor()
            self.monitor.start()
            logger.info("System monitoring started")
            
            # Start web interface if requested
            if web:
                self.web_thread = threading.Thread(
                    target=self._run_web_interface,
                    args=(host, port)
                )
                self.web_thread.daemon = True
                self.web_thread.start()
                logger.info(f"Web interface started at http://{host}:{port}")
                
                if open_browser:
                    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
                    webbrowser.open(url)
            
            self.running = True
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            self.stop_monitoring()
            sys.exit(1)
            
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if self.monitor:
            self.monitor.stop()
            logger.info("System monitoring stopped")
            
        self.running = False
        
    def _run_web_interface(self, host: str, port: int) -> None:
        """Run web interface in separate thread."""
        try:
            start_web_monitor(host=host, port=port, debug=False)
        except Exception as e:
            logger.error(f"Error in web interface: {str(e)}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="System Monitoring Management"
    )
    
    parser.add_argument(
        '--no-web',
        action='store_true',
        help='Run without web interface'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Web interface host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web interface port (default: 5000)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    args = parser.parse_args()
    
    try:
        manager = MonitoringManager()
        
        # Register signal handlers
        import signal
        def signal_handler(signum, frame):
            logger.info("Stopping monitoring...")
            manager.stop_monitoring()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start monitoring
        manager.start_monitoring(
            web=not args.no_web,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )
        
        # Keep main thread alive
        while manager.running:
            signal.pause()
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        manager.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()