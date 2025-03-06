#!/bin/bash
# Install and manage systemd service for trading system monitor

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
    read -p "Enter username for the service: " USERNAME
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

# Function to install service
install_service() {
    print_status "Installing trading monitor service..."
    
    # Copy service file
    cp config/systemd/trading-monitor.service /etc/systemd/system/trading-monitor@.service
    
    # Update directory paths
    sed -i "s|/opt/trading-system|$INSTALL_DIR|g" /etc/systemd/system/trading-monitor@.service
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable trading-monitor@$USERNAME
    
    print_success "Service installed successfully"
}

# Function to remove service
remove_service() {
    print_status "Removing trading monitor service..."
    
    # Stop service
    systemctl stop trading-monitor@$USERNAME
    
    # Disable service
    systemctl disable trading-monitor@$USERNAME
    
    # Remove service file
    rm -f /etc/systemd/system/trading-monitor@.service
    
    # Reload systemd
    systemctl daemon-reload
    
    print_success "Service removed successfully"
}

# Function to show status
show_status() {
    systemctl status trading-monitor@$USERNAME
}

# Function to show logs
show_logs() {
    journalctl -u trading-monitor@$USERNAME -n 50 -f
}

# Parse command
case "${2:-install}" in
    "install")
        install_service
        ;;
    "remove")
        remove_service
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    *)
        echo "Usage: sudo $0 [username] [install|remove|status|logs]"
        exit 1
        ;;
esac

# Print final instructions
if [ "${2:-install}" == "install" ]; then
    echo ""
    echo "Service installed successfully!"
    echo ""
    echo "Commands:"
    echo "1. Start the service:"
    echo "   sudo systemctl start trading-monitor@$USERNAME"
    echo ""
    echo "2. Check status:"
    echo "   sudo systemctl status trading-monitor@$USERNAME"
    echo ""
    echo "3. View logs:"
    echo "   sudo journalctl -u trading-monitor@$USERNAME -f"
    echo ""
    echo "4. Stop service:"
    echo "   sudo systemctl stop trading-monitor@$USERNAME"
    echo ""
    echo "5. Remove service:"
    echo "   sudo $0 $USERNAME remove"
fi

exit 0