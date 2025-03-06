"""
Real-time dashboard updates interface.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import psutil
import numpy as np

from ..web.app import interface
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class DashboardUpdater:
    """
    Handles real-time updates for the dashboard.
    
    Attributes:
        config: Configuration dictionary
        update_interval: Update interval in seconds
        metrics_history: Historical metrics data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dashboard updater."""
        self.config = config or load_config()
        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 5)
        self.metrics_history = {
            'system': [],
            'performance': [],
            'risk': [],
            'activities': []
        }
        self.running = False
        
    async def start(self):
        """Start sending dashboard updates."""
        self.running = True
        try:
            while self.running:
                await self._send_updates()
                await asyncio.sleep(self.update_interval)
        except Exception as e:
            logger.error(f"Error in dashboard updater: {str(e)}")
            self.running = False
            
    def stop(self):
        """Stop sending dashboard updates."""
        self.running = False
        
    async def _send_updates(self):
        """Send all dashboard updates."""
        try:
            # Collect metrics
            system_metrics = self._collect_system_metrics()
            performance_metrics = await self._collect_performance_metrics()
            risk_metrics = await self._collect_risk_metrics()
            activities = await self._collect_activities()
            
            # Update history
            self.metrics_history['system'].append(system_metrics)
            self.metrics_history['performance'].append(performance_metrics)
            self.metrics_history['risk'].append(risk_metrics)
            self.metrics_history['activities'].extend(activities)
            
            # Trim history
            max_history = self.config.get('monitoring', {}).get('max_history_size', 1000)
            for category in self.metrics_history:
                if len(self.metrics_history[category]) > max_history:
                    self.metrics_history[category] = (
                        self.metrics_history[category][-max_history:]
                    )
            
            # Send updates
            await self._broadcast_system_metrics(system_metrics)
            await self._broadcast_performance_metrics(performance_metrics)
            await self._broadcast_risk_metrics(risk_metrics)
            await self._broadcast_activities(activities)
            
        except Exception as e:
            logger.error(f"Error sending updates: {str(e)}")
            
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            active_strategies = len(interface.agent_factory.get_agents_by_type('generation'))
            pending_actions = len(interface.agent_factory.get_agents_by_type('backtesting'))
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': cpu_percent,
                'memory': memory.percent,
                'disk': disk.percent,
                'active_strategies': active_strategies,
                'pending_actions': pending_actions
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {}
            
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            # Get performance from backtesting agents
            metrics = []
            for agent in interface.agent_factory.get_agents_by_type('backtesting'):
                metrics.extend(await agent.get_all_performance())
                
            if not metrics:
                return {}
                
            # Calculate aggregate metrics
            returns = [m.get('total_return', 0) for m in metrics]
            sharpe_ratios = [m.get('sharpe_ratio', 0) for m in metrics]
            drawdowns = [m.get('max_drawdown', 0) for m in metrics]
            win_rates = [m.get('win_rate', 0) for m in metrics]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_return': np.mean(returns),
                'sharpe_ratio': np.mean(sharpe_ratios),
                'max_drawdown': min(drawdowns),
                'win_rate': np.mean(win_rates)
            }
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
            return {}
            
    async def _collect_risk_metrics(self) -> Dict[str, Any]:
        """Collect risk metrics."""
        try:
            # Get risk metrics from risk agents
            metrics = []
            alerts = []
            for agent in interface.agent_factory.get_agents_by_type('risk'):
                agent_metrics = await agent.get_all_risk_metrics()
                metrics.extend(agent_metrics)
                alerts.extend(await agent.get_active_alerts())
                
            if not metrics:
                return {}
                
            # Determine alert level
            alert_level = 'success'
            if any(a['severity'] == 'high' for a in alerts):
                alert_level = 'danger'
            elif any(a['severity'] == 'medium' for a in alerts):
                alert_level = 'warning'
                
            # Format alert message
            alert_message = (
                alerts[0]['message'] if alerts else "No active risk alerts"
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'alert_level': alert_level,
                'alert_message': alert_message
            }
            
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {str(e)}")
            return {}
            
    async def _collect_activities(self) -> List[Dict[str, Any]]:
        """Collect recent activities."""
        try:
            activities = []
            for agent in interface.agent_factory.agents.values():
                if hasattr(agent, 'get_recent_activities'):
                    activities.extend(await agent.get_recent_activities())
                    
            return sorted(
                activities,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:10]  # Last 10 activities
            
        except Exception as e:
            logger.error(f"Error collecting activities: {str(e)}")
            return []
            
    async def _broadcast_system_metrics(self, metrics: Dict[str, Any]):
        """Broadcast system metrics update."""
        if metrics:
            await interface.broadcast_update({
                'type': 'system_metrics',
                'data': metrics
            })
            
    async def _broadcast_performance_metrics(self, metrics: Dict[str, Any]):
        """Broadcast performance metrics update."""
        if metrics:
            await interface.broadcast_update({
                'type': 'performance',
                'data': metrics
            })
            
    async def _broadcast_risk_metrics(self, metrics: Dict[str, Any]):
        """Broadcast risk metrics update."""
        if metrics:
            await interface.broadcast_update({
                'type': 'risk',
                'data': metrics
            })
            
    async def _broadcast_activities(self, activities: List[Dict[str, Any]]):
        """Broadcast activities update."""
        if activities:
            for activity in activities:
                await interface.broadcast_update({
                    'type': 'activity',
                    'data': activity
                })