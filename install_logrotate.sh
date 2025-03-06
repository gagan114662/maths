#!/bin/bash
# Install and configure log rotation for trading system monitoring

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

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root"
    echo "Usage: sudo $0"
    exit 1
fi

# Get username from argument or prompt
USERNAME=${1:-$(logname)}
if [ -z "$USERNAME" ]; then
    read -p "Enter username for the trading system: " USERNAME
fi

# Verify user exists
if ! id "$USERNAME" >/dev/null 2>&1; then
    print_error "User $USERNAME does not exist"
    exit 1
fi

# Get installation directory
INSTALL_DIR=$(eval echo ~$USERNAME)/trading-system
if [ ! -d "$INSTALL_DIR" ]; then
    print_error "Trading system directory not found at $INSTALL_DIR"
    exit 1
fi

# Create log directories
print_status "Creating log directories..."
directories=(
    "monitoring"
    "metrics"
    "alerts"
    "web"
    "debug"
    "performance"
    "api"
    "errors"
    "trading"
    "audit"
)

for dir in "${directories[@]}"; do
    mkdir -p "$INSTALL_DIR/logs/$dir"
    chown -R $USERNAME:$USERNAME "$INSTALL_DIR/logs/$dir"
    chmod -R 755 "$INSTALL_DIR/logs/$dir"
done

# Install logrotate configuration
print_status "Installing logrotate configuration..."

# Update paths in configuration
sed "s|/opt/trading-system|$INSTALL_DIR|g" config/logrotate/monitoring.conf > /etc/logrotate.d/trading-monitor

# Set correct permissions
chmod 644 /etc/logrotate.d/trading-monitor

# Create log files if they don't exist
print_status "Creating initial log files..."
for dir in "${directories[@]}"; do
    touch "$INSTALL_DIR/logs/$dir/system.log"
    chown $USERNAME:$USERNAME "$INSTALL_DIR/logs/$dir/system.log"
    chmod 644 "$INSTALL_DIR/logs/$dir/system.log"
done

# Test logrotate configuration
print_status "Testing logrotate configuration..."
if logrotate -d /etc/logrotate.d/trading-monitor; then
    print_success "Logrotate configuration test passed"
else
    print_error "Logrotate configuration test failed"
    exit 1
fi

# Add daily cron job for log cleanup
print_status "Setting up log cleanup cron job..."
echo "0 0 * * * find $INSTALL_DIR/logs -type f -name \"*.gz\" -mtime +90 -delete" | crontab -u $USERNAME -

print_success "Log rotation setup complete!"
echo ""
echo "Log directories created at:"
echo "$INSTALL_DIR/logs/"
echo ""
echo "Logrotate configuration installed at:"
echo "/etc/logrotate.d/trading-monitor"
echo ""
echo "To manually rotate logs:"
echo "sudo logrotate -f /etc/logrotate.d/trading-monitor"
echo ""
echo "To view current log sizes:"
echo "du -sh $INSTALL_DIR/logs/*"
echo ""
echo "To monitor log rotation:"
echo "tail -f /var/log/syslog | grep logrotate"

# Create log viewing script
print_status "Creating log viewer script..."
cat > "$INSTALL_DIR/view_logs.sh" << 'EOL'
#!/bin/bash
# Script to view and manage logs

case "$1" in
    "monitor")
        tail -f logs/monitoring/system.log
        ;;
    "errors")
        tail -f logs/errors/system.log
        ;;
    "trading")
        tail -f logs/trading/system.log
        ;;
    "all")
        tail -f logs/*/system.log
        ;;
    "sizes")
        du -sh logs/*
        ;;
    *)
        echo "Usage: $0 [monitor|errors|trading|all|sizes]"
        exit 1
        ;;
esac
EOL

chmod +x "$INSTALL_DIR/view_logs.sh"
chown $USERNAME:$USERNAME "$INSTALL_DIR/view_logs.sh"

print_success "Log viewer script created at: $INSTALL_DIR/view_logs.sh"

exit 0