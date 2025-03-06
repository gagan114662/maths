#!/bin/bash
# Installation script for test cleanup service

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run as root"
    exit 1
fi

# Get absolute path to project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create trading system user if it doesn't exist
if ! id -u trading >/dev/null 2>&1; then
    log "Creating trading system user..."
    useradd -r -s /bin/false trading
fi

# Create necessary directories
log "Creating necessary directories..."
mkdir -p /var/log/trading-system
chown trading:trading /var/log/trading-system
chmod 755 /var/log/trading-system

# Copy service files
log "Installing systemd service files..."
cp "$PROJECT_DIR/config/systemd/test-cleanup.service" /etc/systemd/system/
cp "$PROJECT_DIR/config/systemd/test-cleanup.timer" /etc/systemd/system/

# Update file permissions
chmod 644 /etc/systemd/system/test-cleanup.service
chmod 644 /etc/systemd/system/test-cleanup.timer

# Create log rotation configuration
log "Setting up log rotation..."
cat > /etc/logrotate.d/trading-test-cleanup << EOF
/var/log/trading-system/test-cleanup.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 trading trading
}
EOF

# Reload systemd
log "Reloading systemd..."
systemctl daemon-reload

# Enable and start timer
log "Enabling and starting test cleanup timer..."
systemctl enable test-cleanup.timer
systemctl start test-cleanup.timer

# Check status
log "Checking service status..."
systemctl status test-cleanup.timer

log "Installation completed successfully"
log "You can check the timer status with: systemctl status test-cleanup.timer"
log "You can check the next run time with: systemctl list-timers test-cleanup.timer"
log "Logs will be available in /var/log/trading-system/test-cleanup.log"

# Verify installation
if systemctl is-active --quiet test-cleanup.timer; then
    log "Service is running successfully"
else
    error "Service failed to start. Please check the logs"
    exit 1
fi