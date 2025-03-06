"""
Real-time monitoring dashboard for system performance.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
import queue

import psutil
import pandas as pd
import plotly.graph_objs as go
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..utils.config import load_config
from .dashboard_updates import DashboardUpdater

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """
    Real-time monitoring dashboard for system performance.
    
    Attributes:
        config: Configuration dictionary
        data_queue: Queue for real-time data updates
        metrics_history: Historical metrics data
        updater: Dashboard updater instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dashboard."""
        self.config = config
        self.data_queue = queue.Queue()
        self.metrics_history = {
            'system': [],
            'strategies': [],
            'execution': [],
            'performance': []
        }
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="System Monitor",
            description="Trading System Monitoring Dashboard"
        )
        
        # Initialize updater
        self.updater = DashboardUpdater(config)
        
        # Initialize routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up API routes."""
        @self.app.get("/api/metrics/system")
        async def get_system_metrics():
            """Get current system metrics."""
            return JSONResponse(content=self.get_system_metrics())
            
        @self.app.get("/api/metrics/performance")
        async def get_performance_metrics():
            """Get current performance metrics."""
            return JSONResponse(content=self.get_performance_metrics())
            
        @self.app.get("/api/metrics/risk")
        async def get_risk_metrics():
            """Get current risk metrics."""
            return JSONResponse(content=self.get_risk_metrics())
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            if self.metrics_history['performance']:
                return self.metrics_history['performance'][-1]
            return {}
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
            
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        try:
            if self.metrics_history['risk']:
                return self.metrics_history['risk'][-1]
            return {}
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {}
            
    def start(self):
        """Start monitoring dashboard."""
        try:
            # Start updater
            self.updater.start()
            
            # Start FastAPI server
            import uvicorn
            uvicorn.run(
                self.app,
                host=self.config.get('monitoring', {}).get('host', '0.0.0.0'),
                port=self.config.get('monitoring', {}).get('port', 8000)
            )
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {str(e)}")
            self.stop()
            
    def stop(self):
        """Stop monitoring dashboard."""
        try:
            # Stop updater
            self.updater.stop()
            
        except Exception as e:
            logger.error(f"Error stopping dashboard: {str(e)}")
            
    def update_metrics(self, category: str, data: Dict[str, Any]):
        """
        Update metrics from external sources.
        
        Args:
            category: Metric category
            data: Metric data
        """
        try:
            data['timestamp'] = data.get('timestamp', datetime.now().isoformat())
            self.data_queue.put({'category': category, 'data': data})
            
            # Update history
            if category in self.metrics_history:
                self.metrics_history[category].append(data)
                
                # Trim history
                max_history = self.config.get('monitoring', {}).get('max_history_size', 1000)
                if len(self.metrics_history[category]) > max_history:
                    self.metrics_history[category] = self.metrics_history[category][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            
    def get_metrics_history(
        self,
        category: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics.
        
        Args:
            category: Metric category
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of metric data points
        """
        try:
            if category not in self.metrics_history:
                return []
                
            metrics = self.metrics_history[category]
            
            if start_time:
                metrics = [
                    m for m in metrics
                    if datetime.fromisoformat(m['timestamp']) >= start_time
                ]
                
            if end_time:
                metrics = [
                    m for m in metrics
                    if datetime.fromisoformat(m['timestamp']) <= end_time
                ]
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {str(e)}")
            return []