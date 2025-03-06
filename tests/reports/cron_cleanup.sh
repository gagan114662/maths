#!/bin/bash
# Cron script for automated test report cleanup

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Set up environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
source "$PROJECT_ROOT/venv/bin/activate"

# Log file for cron output
LOG_FILE="$SCRIPT_DIR/logs/cron_cleanup.log"
ARCHIVE_DIR="$SCRIPT_DIR/archive"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$ARCHIVE_DIR"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Start cleanup process
log_message "Starting automated report cleanup"

# Archive old reports first
log_message "Archiving old reports"
python3 "$SCRIPT_DIR/clean_reports.py" --archive "$ARCHIVE_DIR/$(date +%Y%m)" >> "$LOG_FILE" 2>&1

# Clean old reports
log_message "Cleaning old reports"
python3 "$SCRIPT_DIR/clean_reports.py" >> "$LOG_FILE" 2>&1

# Generate statistics
log_message "Generating report statistics"
python3 "$SCRIPT_DIR/clean_reports.py" --stats >> "$LOG_FILE" 2>&1

# Cleanup old archives (keep last 6 months)
find "$ARCHIVE_DIR" -type f -mtime +180 -name "*.tar.gz" -delete

# Rotate log file if it gets too large (>1MB)
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE") -gt 1048576 ]; then
    mv "$LOG_FILE" "$LOG_FILE.old"
    log_message "Log file rotated"
fi

log_message "Cleanup completed"

# Example crontab entry (run daily at 2 AM):
# 0 2 * * * /path/to/tests/reports/cron_cleanup.sh