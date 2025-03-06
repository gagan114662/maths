#!/bin/bash
# Setup script for test logging system

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

# Base directory for all logs
BASE_LOG_DIR="/var/log/trading-system"

# Log directory structure
LOG_DIRS=(
    "test_reports"
    "performance"
    "debug"
    "coverage"
    "error"
    "dev"
    "metrics"
)

# Create trading user if it doesn't exist
if ! id -u trading >/dev/null 2>&1; then
    log "Creating trading user..."
    useradd -r -s /bin/false trading
fi

# Create base log directory
log "Creating base log directory..."
mkdir -p "$BASE_LOG_DIR"

# Create log subdirectories
log "Creating log subdirectories..."
for dir in "${LOG_DIRS[@]}"; do
    dir_path="$BASE_LOG_DIR/$dir"
    mkdir -p "$dir_path"
    log "Created $dir_path"
done

# Set ownership and permissions
log "Setting ownership and permissions..."
chown -R trading:trading "$BASE_LOG_DIR"
chmod -R 750 "$BASE_LOG_DIR"

# Install logrotate configuration
log "Installing logrotate configuration..."
cp config/logrotate/test-logs.conf /etc/logrotate.d/trading-test-logs
chmod 644 /etc/logrotate.d/trading-test-logs

# Create log files with correct permissions
log "Creating initial log files..."
for dir in "${LOG_DIRS[@]}"; do
    touch "$BASE_LOG_DIR/$dir/current.log"
    chown trading:trading "$BASE_LOG_DIR/$dir/current.log"
    chmod 640 "$BASE_LOG_DIR/$dir/current.log"
done

# Test logrotate configuration
log "Testing logrotate configuration..."
logrotate -d /etc/logrotate.d/trading-test-logs

# Create symlinks for convenient access
log "Creating symlinks for convenient access..."
ln -sf "$BASE_LOG_DIR/error/current.log" "$BASE_LOG_DIR/error.log"
ln -sf "$BASE_LOG_DIR/debug/current.log" "$BASE_LOG_DIR/debug.log"

# Add log directories to system directory configuration
if [ ! -f "/etc/tmpfiles.d/trading-test.conf" ]; then
    log "Creating tmpfiles configuration..."
    cat > /etc/tmpfiles.d/trading-test.conf << EOF
d $BASE_LOG_DIR 0750 trading trading -
d $BASE_LOG_DIR/test_reports 0750 trading trading -
d $BASE_LOG_DIR/performance 0750 trading trading -
d $BASE_LOG_DIR/debug 0750 trading trading -
d $BASE_LOG_DIR/coverage 0750 trading trading -
d $BASE_LOG_DIR/error 0750 trading trading -
d $BASE_LOG_DIR/dev 0750 trading trading -
d $BASE_LOG_DIR/metrics 0750 trading trading -
EOF
fi

# Create log cleanup cron job
log "Setting up log cleanup cron job..."
cat > /etc/cron.daily/trading-test-cleanup << EOF
#!/bin/sh
find $BASE_LOG_DIR -type f -name "*.log" -mtime +90 -delete
EOF
chmod +x /etc/cron.daily/trading-test-cleanup

# Verify setup
log "Verifying setup..."
VERIFY_ERRORS=0

# Check directory permissions
for dir in "${LOG_DIRS[@]}"; do
    dir_path="$BASE_LOG_DIR/$dir"
    if [ ! -d "$dir_path" ]; then
        error "Directory $dir_path does not exist"
        VERIFY_ERRORS=$((VERIFY_ERRORS + 1))
    fi
    
    if [ "$(stat -c '%U:%G' "$dir_path")" != "trading:trading" ]; then
        error "Incorrect ownership on $dir_path"
        VERIFY_ERRORS=$((VERIFY_ERRORS + 1))
    fi
done

# Check logrotate configuration
if [ ! -f "/etc/logrotate.d/trading-test-logs" ]; then
    error "Logrotate configuration not installed"
    VERIFY_ERRORS=$((VERIFY_ERRORS + 1))
fi

# Final status
if [ $VERIFY_ERRORS -eq 0 ]; then
    log "Setup completed successfully"
    log "Log files will be stored in $BASE_LOG_DIR"
    log "Logrotate configuration is active"
else
    error "Setup completed with $VERIFY_ERRORS errors"
    exit 1
fi