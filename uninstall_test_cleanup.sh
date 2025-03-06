#!/bin/bash
# Uninstallation script for test cleanup service

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

# Stop and disable services
log "Stopping and disabling services..."
systemctl stop test-cleanup.timer
systemctl stop test-cleanup.service
systemctl disable test-cleanup.timer
systemctl disable test-cleanup.service

# Remove service files
log "Removing systemd service files..."
rm -f /etc/systemd/system/test-cleanup.service
rm -f /etc/systemd/system/test-cleanup.timer

# Remove log rotation configuration
log "Removing log rotation configuration..."
rm -f /etc/logrotate.d/trading-test-cleanup

# Reload systemd
log "Reloading systemd..."
systemctl daemon-reload

# Backup and remove logs
BACKUP_DIR="/var/backup/trading-system/cleanup-logs-$(date +%Y%m%d_%H%M%S)"
if [ -d "/var/log/trading-system" ]; then
    log "Backing up logs to $BACKUP_DIR..."
    mkdir -p "$BACKUP_DIR"
    cp -r /var/log/trading-system/* "$BACKUP_DIR/"
    
    log "Removing log directory..."
    rm -rf /var/log/trading-system
fi

# Remove archived reports
if [ -d "tests/reports/archive" ]; then
    log "Backing up archived reports..."
    cp -r tests/reports/archive "$BACKUP_DIR/archived_reports"
    
    log "Removing archived reports..."
    rm -rf tests/reports/archive
fi

# Optional: Remove trading user if no other services need it
read -p "Do you want to remove the trading user? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if id -u trading >/dev/null 2>&1; then
        log "Removing trading user..."
        userdel trading
        warning "Note: Any files owned by the trading user will need to be manually managed."
    else
        warning "Trading user does not exist."
    fi
fi

log "Uninstallation completed successfully"
log "Backups are available in $BACKUP_DIR"
log "The following manual cleanup steps may be needed:"
echo "  1. Review any remaining files in tests/reports/"
echo "  2. Check for any remaining cron jobs"
echo "  3. Verify permissions on project directories"
echo "  4. Remove any related environment variables"

# Final verification
if systemctl is-active --quiet test-cleanup.timer; then
    error "Service is still running! Please check systemctl status"
    exit 1
else
    log "Service successfully removed"
fi