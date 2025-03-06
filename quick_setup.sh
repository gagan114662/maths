#!/bin/bash
# Quick setup script for Enhanced Trading Strategy System

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Enhanced Trading Strategy System...${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}Python version $python_version is compatible${NC}"
else 
    echo -e "${RED}Error: Python version $required_version or higher is required${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Found existing virtual environment"
else
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "\n${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt

# Install test requirements
echo -e "\n${YELLOW}Installing test requirements...${NC}"
pip install -r tests/requirements-test.txt

# Install package in development mode
echo -e "\n${YELLOW}Installing package in development mode...${NC}"
pip install -e .

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
directories=(
    "data"
    "data/cache"
    "logs"
    "config"
    "results"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

# Copy example config if not exists
echo -e "\n${YELLOW}Setting up configuration...${NC}"
if [ ! -f "config/config.yaml" ]; then
    cp examples/config/example_config.yaml config/config.yaml
    echo "Created default configuration file"
fi

# Setup logging
echo -e "\n${YELLOW}Setting up logging...${NC}"
if [ ! -f "config/logging.yaml" ]; then
    cat > config/logging.yaml << EOL
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: default
    filename: logs/trading_system.log
    maxBytes: 10485760
    backupCount: 5
root:
  level: INFO
  handlers: [console, file]
EOL
    echo "Created default logging configuration"
fi

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}"
pytest tests/

# Create pre-commit hook
echo -e "\n${YELLOW}Setting up pre-commit hook...${NC}"
if [ ! -f ".git/hooks/pre-commit" ]; then
    cat > .git/hooks/pre-commit << EOL
#!/bin/bash
# Pre-commit checks

# Run tests
echo "Running tests..."
pytest tests/

# Check code formatting
echo "Checking code formatting..."
black --check src/ tests/
isort --check-only src/ tests/

# Run type checking
echo "Running type checking..."
mypy src/

# Run linting
echo "Running linting..."
flake8 src/ tests/
EOL
    chmod +x .git/hooks/pre-commit
    echo "Created pre-commit hook"
fi

# Make scripts executable
echo -e "\n${YELLOW}Making scripts executable...${NC}"
chmod +x *.sh
chmod +x run_tests.py

echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo -e "\nTo get started:"
echo -e "1. Update config/config.yaml with your settings"
echo -e "2. Run the system: python -m src.web.app"
echo -e "3. Visit http://localhost:8000 in your browser"
echo -e "\nFor more information, see the README.md file"

# Deactivate virtual environment
deactivate