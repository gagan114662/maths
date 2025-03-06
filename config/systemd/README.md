# Test Cleanup Service

Systemd service and timer for automated test report cleanup and management.

## Components

```
systemd/
├── README.md
├── test-cleanup.service    # Service definition
└── test-cleanup.timer     # Timer configuration
```

## Setup

### Prerequisites
- Systemd-based Linux system
- Root or sudo access
- Python 3.8 or higher
- Trading system installed in `/opt/trading-system`

### Installation
```bash
# Install the service
sudo ./install_test_cleanup.sh

# Verify installation
systemctl status test-cleanup.timer
```

## Service Configuration

### Service Settings (`test-cleanup.service`)
```ini
[Unit]
Description=Test Report Cleanup Service
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/env bash -c 'cd /opt/trading-system && tests/reports/cron_cleanup.sh'
User=trading
Group=trading
```

### Timer Settings (`test-cleanup.timer`)
```ini
[Timer]
# Run daily at 2:00 AM
OnCalendar=*-*-* 02:00:00
# Add randomization to prevent system load spikes
RandomizedDelaySec=900
```

## Usage

### Managing the Service

```bash
# Start service manually
sudo systemctl start test-cleanup.service

# Check timer status
sudo systemctl status test-cleanup.timer

# View next scheduled run
sudo systemctl list-timers test-cleanup.timer

# View logs
sudo journalctl -u test-cleanup.service
```

### Log Files

- Service logs: `/var/log/trading-system/test-cleanup.log`
- System journal: `journalctl -u test-cleanup.service`
- Clean script logs: `/var/log/trading-system/cron_cleanup.log`

## Maintenance

### Log Rotation
Logs are automatically rotated:
- Daily rotation
- 7 days retention
- Compressed archives
- Managed by logrotate

### Monitoring
Monitor service health:
```bash
# Check service status
systemctl status test-cleanup.service

# View recent logs
journalctl -u test-cleanup.service -n 50

# Monitor in real-time
journalctl -u test-cleanup.service -f
```

### Troubleshooting

1. Service Failures
   ```bash
   # Check service status
   systemctl status test-cleanup.service
   
   # View error logs
   journalctl -u test-cleanup.service -p err
   ```

2. Timer Issues
   ```bash
   # Verify timer is active
   systemctl is-active test-cleanup.timer
   
   # Check timer configuration
   systemctl cat test-cleanup.timer
   ```

3. Permission Problems
   ```bash
   # Check file ownership
   ls -l /opt/trading-system/tests/reports/
   
   # Verify user permissions
   sudo -u trading test -w /var/log/trading-system/
   ```

## Security

The service runs with these security measures:
- Dedicated system user (`trading`)
- No new privileges
- Protected system directories
- Private tmp directory
- Read-only home access

## Uninstallation

```bash
# Remove the service
sudo ./uninstall_test_cleanup.sh

# Verify removal
systemctl status test-cleanup.timer
ls -l /etc/systemd/system/test-cleanup.*
```

## Support

For issues:
1. Check service logs
2. Review system journal
3. Verify file permissions
4. Check configuration
5. Contact system administrator