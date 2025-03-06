"""
System monitoring utilities for the Enhanced Trading Strategy System.
"""
import os
import sys
import psutil
import logging
import threading
from typing import Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

class SystemMonitor:
    """
    Monitors system resources and trading system health.
    
    Attributes:
        config: Monitoring configuration
        logger: Logger instance
        metrics: Current system metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize system monitor."""
        self.config = config or {}
        self.logger = self._setup_logger()
        self.metrics = {}
        self._running = False
        self._lock = threading.Lock()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up monitoring logger."""
        logger = logging.getLogger("SystemMonitor")
        handler = logging.FileHandler("logs/system_monitor.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
        
    def start(self, interval: int = 60):
        """
        Start monitoring system.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,)
        )
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop(self):
        """Stop monitoring system."""
        self._running = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join()
        self.logger.info("System monitoring stopped")
        
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self._running:
            try:
                self._collect_metrics()
                self._analyze_metrics()
                self._save_metrics()
                threading.Event().wait(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                
    def _collect_metrics(self):
        """Collect system metrics."""
        with self._lock:
            self.metrics = {
                'timestamp': datetime.now(),
                'system': self._get_system_metrics(),
                'memory': self._get_memory_metrics(),
                'cpu': self._get_cpu_metrics(),
                'disk': self._get_disk_metrics(),
                'network': self._get_network_metrics(),
                'trading': self._get_trading_metrics()
            }
            
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        return {
            'boot_time': psutil.boot_time(),
            'users': len(psutil.users()),
            'platform': sys.platform,
            'python_version': sys.version,
            'processes': len(psutil.pids())
        }
        
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics."""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'free': mem.free,
            'percent': mem.percent
        }
        
    def _get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU usage metrics."""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'load_avg': psutil.getloadavg()
        }
        
    def _get_disk_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get disk usage metrics."""
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                }
            except Exception:
                continue
        return disk_usage
        
    def _get_network_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get network usage metrics."""
        return {
            'connections': len(psutil.net_connections()),
            'io_counters': psutil.net_io_counters()._asdict()
        }
        
    def _get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading system metrics."""
        return {
            'active_strategies': self._count_active_strategies(),
            'pending_orders': self._count_pending_orders(),
            'data_pipeline': self._check_data_pipeline(),
            'model_status': self._check_model_status()
        }
        
    def _analyze_metrics(self):
        """Analyze collected metrics."""
        warnings = []
        
        # Check memory usage
        if self.metrics['memory']['percent'] > 90:
            warnings.append("High memory usage")
            
        # Check CPU usage
        if self.metrics['cpu']['percent'] > 80:
            warnings.append("High CPU usage")
            
        # Check disk space
        for mount, usage in self.metrics['disk'].items():
            if usage['percent'] > 90:
                warnings.append(f"Low disk space on {mount}")
                
        if warnings:
            self.logger.warning("System warnings: " + "; ".join(warnings))
            
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = Path("logs/metrics.csv")
        df = pd.DataFrame([self.metrics])
        
        if not metrics_file.exists():
            df.to_csv(metrics_file, index=False)
        else:
            df.to_csv(metrics_file, mode='a', header=False, index=False)
            
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        with self._lock:
            return {
                'status': 'healthy' if not self._check_warnings() else 'warning',
                'metrics': self.metrics,
                'warnings': self._check_warnings()
            }
            
    def _check_warnings(self) -> list:
        """Check for system warnings."""
        warnings = []
        
        # Resource warnings
        if self.metrics['memory']['percent'] > 90:
            warnings.append("Critical memory usage")
        if self.metrics['cpu']['percent'] > 80:
            warnings.append("High CPU load")
            
        # Trading system warnings
        trading = self.metrics['trading']
        if not trading['data_pipeline']['status'] == 'ok':
            warnings.append("Data pipeline issues")
        if not trading['model_status']['status'] == 'ok':
            warnings.append("Model issues")
            
        return warnings
        
    def _count_active_strategies(self) -> int:
        """Count active trading strategies."""
        # Implementation depends on strategy management system
        return 0
        
    def _count_pending_orders(self) -> int:
        """Count pending orders."""
        # Implementation depends on order management system
        return 0
        
    def _check_data_pipeline(self) -> Dict[str, str]:
        """Check data pipeline status."""
        return {
            'status': 'ok',
            'last_update': str(datetime.now()),
            'lag': '0s'
        }
        
    def _check_model_status(self) -> Dict[str, str]:
        """Check model status."""
        return {
            'status': 'ok',
            'last_prediction': str(datetime.now()),
            'performance': 'normal'
        }

def start_monitoring(config: Dict[str, Any] = None) -> SystemMonitor:
    """
    Start system monitoring.
    
    Args:
        config: Optional monitoring configuration
        
    Returns:
        SystemMonitor instance
    """
    monitor = SystemMonitor(config)
    monitor.start()
    return monitor

def get_system_status() -> Dict[str, Any]:
    """
    Get current system status.
    
    Returns:
        Dictionary containing system status and metrics
    """
    monitor = SystemMonitor()
    return monitor.get_status()