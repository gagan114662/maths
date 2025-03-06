#!/bin/bash
# Run all code quality checks and tests before committing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize error flag
error=0

# Function to run a command and check its exit status
run_check() {
    echo -e "${YELLOW}Running $1...${NC}"
    if $2; then
        echo -e "${GREEN}✓ $1 passed${NC}\n"
    else
        echo -e "${RED}✗ $1 failed${NC}\n"
        error=1
    fi
}

# Check if changes include documentation
doc_changes=$(git diff --cached --name-only | grep -E "\.md$|\.rst$|docs/|\.yaml$")
if [ -n "$doc_changes" ]; then
    echo -e "${YELLOW}Documentation changes detected. Running documentation checks...${NC}"
    run_check "documentation verification" "./verify_docs.py"
fi

# Format code
run_check "black" "black src tests"
run_check "isort" "isort src tests"

# Run static type checking
run_check "mypy" "mypy src"

# Run style checks
run_check "flake8" "flake8 src tests"

# Run monitoring system tests
echo -e "${YELLOW}Running monitoring system tests...${NC}"
if pytest tests/utils/test_monitor*.py -v; then
    echo -e "${GREEN}✓ Monitoring tests passed${NC}\n"
else
    echo -e "${RED}✗ Monitoring tests failed${NC}\n"
    error=1
fi

# Run all tests
echo -e "${YELLOW}Running all tests...${NC}"
if pytest tests/ -v; then
    echo -e "${GREEN}✓ All tests passed${NC}\n"
else
    echo -e "${RED}✗ Some tests failed${NC}\n"
    error=1
fi

# Run validation
echo -e "${YELLOW}Running setup validation...${NC}"
if ./validate_setup.py; then
    echo -e "${GREEN}✓ Validation passed${NC}\n"
else
    echo -e "${RED}✗ Validation failed${NC}\n"
    error=1
fi

# Check for sensitive data
echo -e "${YELLOW}Checking for sensitive data...${NC}"
if git diff --cached -S"API_KEY|SECRET|PASSWORD" --pickaxe-all; then
    echo -e "${RED}✗ Possible sensitive data detected in changes${NC}\n"
    echo -e "Please review the above changes for sensitive information"
    error=1
else
    echo -e "${GREEN}✓ No sensitive data detected${NC}\n"
fi

# Check monitoring system
echo -e "${YELLOW}Verifying monitoring system...${NC}"
if timeout 5s ./monitor_system.py --no-web --test-mode; then
    echo -e "${GREEN}✓ Monitoring system check passed${NC}\n"
else
    echo -e "${RED}✗ Monitoring system check failed${NC}\n"
    error=1
fi

# Check web interface
echo -e "${YELLOW}Verifying web interface...${NC}"
if timeout 5s python -c "from src.utils.monitor_web import app; app.test_client().get('/')"; then
    echo -e "${GREEN}✓ Web interface check passed${NC}\n"
else
    echo -e "${RED}✗ Web interface check failed${NC}\n"
    error=1
fi

# Final status
if [ $error -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please fix the issues before committing.${NC}"
    exit 1
fi