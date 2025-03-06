# Test Reports Directory

This directory contains test execution reports, coverage data, and logs generated during test runs.

## Directory Structure

```
reports/
├── README.md          # This file
├── runs/             # Test execution reports
│   └── .gitkeep
├── coverage/         # Test coverage reports
│   └── .gitkeep
└── logs/            # Test execution logs
    └── .gitkeep
```

## Contents

### Runs Directory (`runs/`)
- Contains individual test run reports
- Each run creates a timestamped report file
- Reports include:
  - Test execution time
  - Pass/fail status
  - Error messages
  - Test suite statistics

### Coverage Directory (`coverage/`)
- Contains test coverage reports
- Formats include:
  - HTML reports
  - XML reports
  - Console summary
- Coverage data includes:
  - Line coverage
  - Branch coverage
  - Missing lines
  - Coverage trends

### Logs Directory (`logs/`)
- Contains detailed test execution logs
- Includes:
  - Debug output
  - Warning messages
  - Error traces
  - Performance metrics

## Usage

Reports are automatically generated when running tests using:
```bash
./run_tests.py --coverage --report all
```

View the HTML coverage report:
```bash
open tests/reports/coverage/html/index.html
```

View recent test runs:
```bash
ls -lt tests/reports/runs/
```

## Retention Policy

- Run reports are kept for 30 days
- Coverage reports are kept for 90 days
- Logs are rotated weekly
- Historical trends are maintained in the database

## Notes

- This directory is ignored by git (except for .gitkeep files)
- Clean old reports using `./run_tests.py --clean-reports`
- Export reports using `./run_tests.py --export-reports`