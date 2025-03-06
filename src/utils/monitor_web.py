"""
Web interface for system monitoring.
"""
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, jsonify
from .monitor import SystemMonitor

app = Flask(__name__)
monitor = SystemMonitor()

# Ensure templates directory exists
template_dir = Path(__file__).parent / 'templates'
template_dir.mkdir(exist_ok=True)

# Create HTML template
template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>System Monitor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card {
            margin-bottom: 20px;
        }
        .warning {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1>System Monitor</h1>
        <div class="row">
            <div class="col-md-6">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">System Status</h5>
                        <p class="card-text" id="system-status"></p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Warnings</h5>
                        <ul id="warnings-list"></ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Memory Usage</h5>
                        <canvas id="memory-chart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">CPU Usage</h5>
                        <canvas id="cpu-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Trading System Status</h5>
                        <div id="trading-status"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update system status
                    document.getElementById('system-status').textContent = 
                        `Status: ${data.status} (Updated: ${new Date().toLocaleTimeString()})`;
                    
                    // Update warnings
                    const warningsList = document.getElementById('warnings-list');
                    warningsList.innerHTML = '';
                    data.warnings.forEach(warning => {
                        const li = document.createElement('li');
                        li.textContent = warning;
                        li.className = 'warning';
                        warningsList.appendChild(li);
                    });
                    
                    // Update trading status
                    const tradingStatus = document.getElementById('trading-status');
                    tradingStatus.innerHTML = `
                        <p>Active Strategies: ${data.metrics.trading.active_strategies}</p>
                        <p>Pending Orders: ${data.metrics.trading.pending_orders}</p>
                        <p>Data Pipeline: ${data.metrics.trading.data_pipeline.status}</p>
                        <p>Model Status: ${data.metrics.trading.model_status.status}</p>
                    `;
                    
                    // Update charts
                    updateCharts(data.metrics);
                });
        }
        
        function updateCharts(metrics) {
            // Memory usage chart
            const memoryCtx = document.getElementById('memory-chart').getContext('2d');
            new Chart(memoryCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Used', 'Free'],
                    datasets: [{
                        data: [metrics.memory.used, metrics.memory.free],
                        backgroundColor: ['#dc3545', '#28a745']
                    }]
                }
            });
            
            // CPU usage chart
            const cpuCtx = document.getElementById('cpu-chart').getContext('2d');
            new Chart(cpuCtx, {
                type: 'line',
                data: {
                    labels: Array(10).fill(''),
                    datasets: [{
                        label: 'CPU Usage',
                        data: [metrics.cpu.percent],
                        borderColor: '#007bff'
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
        
        // Update status every 5 seconds
        setInterval(updateStatus, 5000);
        updateStatus();  // Initial update
    </script>
</body>
</html>
"""

# Write template file
template_path = template_dir / 'monitor.html'
template_path.write_text(template_content)

@app.route('/')
def index():
    """Render monitoring dashboard."""
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify(monitor.get_status())

@app.route('/api/metrics/history')
def get_metrics_history():
    """Get historical metrics."""
    metrics_file = Path("logs/metrics.csv")
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent = df[df['timestamp'] > datetime.now() - timedelta(hours=1)]
        return jsonify(recent.to_dict(orient='records'))
    return jsonify([])

def start_web_monitor(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start the monitoring web interface."""
    monitor.start()  # Start system monitoring
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_web_monitor()