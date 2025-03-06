# Test Logging System

Configuration and management for the test system logging infrastructure.

## Directory Structure

```
/var/log/trading-system/
├── test_reports/     # Test execution reports
├── performance/      # Performance test results
├── debug/           # Debug level logs
├── coverage/        # Coverage report logs
├── error/          # Error logs
├── dev/            # Development environment logs
└── metrics/        # System metrics logs
```

## Configuration Files

- `test-logs.conf`: Logrotate configuration for all test logs
- `/etc/tmpfiles.d/trading-test.conf`: System directory configuration
- `/etc/cron.daily/trading-test-cleanup`: Daily cleanup script

## Setup

1. Run the setup script:
```bash
sudo ./setup_test_logs.sh
```

2. Verify installation:
```bash
ls -l /var/log/trading-system
logrotate -d /etc/logrotate.d/trading-test-logs
```

## Log Rotation Policies

### Test Reports
- Rotated: Daily
- Retention: 14 days
- Maximum size: 100MB
- Compressed: Yes

### Performance Logs
- Rotated: Daily
- Retention: 30 days
- Maximum size: 500MB
- Timestamp format: YYYYMMDD-timestamp

### Debug Logs
- Rotated: By size (50MB)
- Retention: 5 versions
- Compressed: Yes

### Coverage Reports
- Rotated: Weekly
- Retention: 12 weeks
- Compressed: Yes

### Error Logs
- Rotated: Daily
- Retention: 30 days
- Service reload after rotation

### Development Logs
- Rotated: By size (100MB)
- Retention: 3 versions
- Compressed: Yes

### Metrics Logs
- Rotated: Daily
- Retention: 90 days
- Maximum size: 1GB
- Timestamp format: YYYYMMDD

## Monitoring

### Check Log Sizes
```bash
du -sh /var/log/trading-system/*
```

### View Recent Errors
```bash
tail -f /var/log/trading-system/error/current.log
```

### Check Rotation Status
```bash
logrotate -d /etc/logrotate.d/trading-test-logs
```

## Maintenance

### Manual Rotation
```bash
sudo logrotate -f /etc/logrotate.d/trading-test-logs
```

### Clean Old Logs
```bash
sudo find /var/log/trading-system -type f -name "*.gz" -mtime +90 -delete
```

### Verify Permissions
```bash
sudo find /var/log/trading-system -type f -exec ls -l {} \;
```

## Security

### File Permissions
- Directories: 750 (trading:trading)
- Log files: 640 (trading:trading)
- Configuration: 644 (root:root)

### Access Control
- Only trading user can write logs
- Only trading group can read logs
- Root access required for configuration

## Troubleshooting

### Common Issues

1. Permission Denied
```bash
sudo chown -R trading:trading /var/log/trading-system
sudo chmod -R u=rwX,g=rX,o= /var/log/trading-system
```

2. Disk Space Issues
```bash
# Check disk usage
df -h /var/log
# Force rotation
sudo logrotate -f /etc/logrotate.d/trading-test-logs
```

3. Missing Log Files
```bash
# Recreate directory structure
sudo ./setup_test_logs.sh --repair
```

### Log Analysis

1. Find Error Patterns
```bash
grep -r "ERROR" /var/log/trading-system/error/
```

2. Check System Impact
```bash
journalctl --since "1 hour ago" | grep trading-system
```

## Integration

### Syslog Configuration
Add to `/etc/rsyslog.d/trading-test.conf`:
```
# Test system logs
local0.*                        /var/log/trading-system/test_reports/current.log
local0.error                    /var/log/trading-system/error/current.log
```

### Log Forwarding
Configure in `/etc/rsyslog.d/forward-test.conf`:
```
# Forward to log server
*.* @logserver:514
```

## Support

For issues:
1. Check permissions
2. Verify logrotate configuration
3. Review system logs
4. Contact system administrator