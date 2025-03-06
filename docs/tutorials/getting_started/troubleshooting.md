# Troubleshooting Guide

This guide helps diagnose and resolve common issues encountered while using the Enhanced Trading Strategy System.

## Quick Diagnosis

First, generate a debug report:
```bash
./generate_debug_report.py
```

## Common Issues

### 1. Installation Problems

#### Python Version Mismatch
```
Error: Python version must be >= 3.8
```

**Solution:**
```bash
# Check Python version
python --version

# Install correct version
# On Ubuntu:
sudo apt install python3.8
# On macOS:
brew install python@3.8
```

#### Dependency Conflicts
```
Error: Conflicting dependencies detected
```

**Solution:**
```bash
# Clean environment
rm -rf venv
python -m venv venv
source venv/bin/activate

# Install dependencies in order
pip install -r requirements.txt --no-cache-dir
```

### 2. Data Access Issues

#### API Authentication Fails
```
Error: Invalid API credentials
```

**Solution:**
1. Verify credentials:
```bash
./download_eastmoney_data.py --interactive
```

2. Check environment variables:
```bash
echo $EASTMONEY_API_KEY
```

3. Reset credentials:
```bash
rm .env
./download_eastmoney_data.py --interactive
```

#### Data Download Fails
```
Error: Failed to download dataset
```

**Solution:**
1. Check network:
```bash
ping quantapi.eastmoney.com
```

2. Verify rate limits:
```bash
# Check API usage
./generate_debug_report.py
```

3. Use offline data:
```bash
cp data/backup/* data/current/
```

### 3. Model Training Issues

#### Out of Memory
```
Error: CUDA out of memory
```

**Solution:**
1. Reduce batch size in config:
```yaml
training:
  batch_size: 32  # Reduce this value
```

2. Free GPU memory:
```python
import torch
torch.cuda.empty_cache()
```

#### Training Not Converging
```
Warning: Loss not decreasing
```

**Solution:**
1. Check learning rate:
```yaml
training:
  learning_rate: 0.001  # Adjust this value
```

2. Verify data:
```python
from src.utils.validation import validate_data
validate_data(training_data)
```

### 4. Runtime Errors

#### Permission Denied
```
Error: Permission denied executing script
```

**Solution:**
```bash
# Make scripts executable
chmod +x *.py *.sh

# Fix directory permissions
chmod -R u+rw data/
```

#### Import Errors
```
Error: No module named 'src'
```

**Solution:**
1. Add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

2. Verify installation:
```bash
pip install -e .
```

### 5. Performance Issues

#### Slow Processing
```
Warning: Processing taking longer than expected
```

**Solution:**
1. Check resource usage:
```bash
top
nvidia-smi
```

2. Optimize configuration:
```yaml
processing:
  batch_size: 64
  num_workers: 4
  use_cache: true
```

3. Clear cache:
```bash
rm -rf .cache/*
```

### 6. Integration Issues

#### FinTSB Integration Fails
```
Error: Cannot import FinTSB modules
```

**Solution:**
1. Update submodules:
```bash
git submodule update --init --recursive
```

2. Install FinTSB:
```bash
cd FinTSB
pip install -e .
```

#### Mathematricks Integration Issues
```
Error: Mathematricks component not found
```

**Solution:**
1. Check installation:
```bash
cd mathematricks
pip install -e .
```

2. Verify configuration:
```bash
./validate_setup.py
```

## Advanced Troubleshooting

### System Verification

Run complete system check:
```bash
# Verify setup
./validate_setup.py

# Run tests
./run_tests.py --coverage

# Check documentation
./verify_docs.py
```

### Log Analysis

Check various log files:
```bash
# System logs
tail -f logs/system.log

# Training logs
tail -f logs/training.log

# Data pipeline logs
tail -f logs/pipeline.log
```

### Database Issues

Verify database connection:
```bash
# Check connection
python -c "from src.utils.db import verify_connection; verify_connection()"

# Reset database
python scripts/reset_db.py
```

### Configuration Issues

Validate configurations:
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Verify against schema
./validate_config.py configs/*.yaml
```

## Getting Help

If issues persist:

1. Check documentation:
   - [API Reference](../../api/index.md)
   - [Examples](../../examples/index.md)
   - [Architecture Guide](../../architecture/index.md)

2. Generate debug info:
   ```bash
   ./generate_debug_report.py > debug_info.txt
   ```

3. Contact support:
   - Open GitHub issue (include debug_info.txt)
   - Join Discord community
   - Email support team

## Prevention

To prevent future issues:

1. Regular maintenance:
   ```bash
   # Update dependencies
   make update-deps

   # Run tests
   make test

   # Validate setup
   make validate
   ```

2. Monitor system:
   ```bash
   # Check resource usage
   ./monitor_system.py

   # Watch logs
   tail -f logs/*.log
   ```

3. Backup data:
   ```bash
   # Create backup
   ./backup_data.sh

   # Verify backup
   ./verify_backup.sh