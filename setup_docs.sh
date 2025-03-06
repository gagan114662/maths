#!/bin/bash
# Set up documentation tools and git hooks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_error "Virtual environment not activated. Please activate it first."
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Install documentation dependencies
print_status "Installing documentation tools..."
pip install -r requirements.txt

# Verify installation
print_status "Verifying installations..."
for cmd in markdown-cli linkchecker doc8 rstcheck; do
    if ! command -v $cmd &> /dev/null; then
        print_error "Failed to install $cmd"
        exit 1
    fi
done

# Make scripts executable
print_status "Making scripts executable..."
chmod +x verify_docs.py
chmod +x pre-commit-check.sh
chmod +x quick_setup.sh

# Set up git hooks
print_status "Setting up git hooks..."
if [ -d ".git" ]; then
    # Backup existing pre-commit hook if it exists
    if [ -f ".git/hooks/pre-commit" ]; then
        mv .git/hooks/pre-commit .git/hooks/pre-commit.backup
        print_status "Backed up existing pre-commit hook"
    fi
    
    # Install new pre-commit hook
    cp pre-commit-check.sh .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    print_success "Installed pre-commit hook"
else
    print_error "Not a git repository"
    exit 1
fi

# Create documentation directories if they don't exist
print_status "Setting up documentation directories..."
mkdir -p docs/{api,examples,tutorials,architecture}

# Create initial documentation files if they don't exist
print_status "Creating initial documentation files..."
for dir in api examples tutorials architecture; do
    if [ ! -f "docs/$dir/index.md" ]; then
        echo "# $dir Documentation" > "docs/$dir/index.md"
        echo "Documentation for $dir" >> "docs/$dir/index.md"
    fi
done

# Verify documentation
print_status "Verifying documentation setup..."
if ./verify_docs.py; then
    print_success "Documentation verification passed"
else
    print_error "Documentation verification failed"
    print_status "Please review and fix documentation issues"
fi

# Final status
print_success "Documentation setup complete!"
echo ""
echo "Next steps:"
echo "1. Review documentation in docs/ directory"
echo "2. Run './verify_docs.py' to check documentation"
echo "3. Commit changes to trigger pre-commit hooks"
echo ""
echo "For more information, see:"
echo "- docs/README.md"
echo "- CONTRIBUTING.md"