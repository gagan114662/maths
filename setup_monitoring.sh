#!/bin/bash
# Setup script for monitoring system

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${YELLOW}==>${NC} $1"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}==>${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}==>${NC} $1"
}

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_error "Virtual environment not activated. Please activate it first."
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p logs/monitoring
mkdir -p logs/metrics
mkdir -p logs/alerts

# Set up permissions
print_status "Setting permissions..."
chmod 755 monitor_system.py
chmod 755 logs/monitoring
chmod 755 logs/metrics
chmod 755 logs/alerts

# Install monitoring dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Create monitoring configuration
print_status "Creating monitoring configuration..."
cat > config/monitoring.yaml << EOL
monitoring:
  intervals:
    system: 60
    trading: 30
  thresholds:
    memory: 90
    cpu: 80
    disk: 90
  retention:
    metrics: 7
    logs: 30
  web:
    host: "0.0.0.0"
    port: 5000
    debug: false
  alerts:
    email: false
    slack: false
    telegram: false
EOL

# Create log rotation configuration
print_status "Setting up log rotation..."
cat > config/monitoring_logrotate.conf << EOL
logs/monitoring/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}

logs/metrics/*.csv {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}

logs/alerts/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
EOL

# Verify monitoring system
print_status "Verifying monitoring system..."
if timeout 5s ./monitor_system.py --test-mode; then
    print_success "Monitoring system verified"
else
    print_error "Monitoring system verification failed"
    exit 1
fi

# Create systemd service file (if running on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/trading-monitor.service << EOL
[Unit]
Description=Trading System Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PYTHONPATH=$(pwd)
ExecStart=$(which python) monitor_system.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOL

    sudo systemctl daemon-reload
    print_success "Systemd service created"
fi

# Final instructions
print_success "Monitoring system setup complete!"
echo ""
echo "To start monitoring:"
echo "1. Start the monitor:"
echo "   ./monitor_system.py"
echo ""
echo "2. Or use systemd (on Linux):"
echo "   sudo systemctl start trading-monitor"
echo ""
echo "3. Access web interface:"
echo "   http://localhost:5000"
echo ""
echo "4. View logs:"
echo "   tail -f logs/monitoring/system.log"
echo ""
echo "Configuration files:"
echo "- config/monitoring.yaml"
echo "- config/monitoring_logrotate.conf"