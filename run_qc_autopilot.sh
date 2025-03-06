#!/bin/bash
# Script to set up QuantConnect dependencies and run the autopilot

# Print header
echo "======================================================"
echo "QuantConnect Autopilot Setup and Execution"
echo "======================================================"

# Check for Python environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated existing Python virtual environment"
else
    echo "Creating new Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Activated new Python virtual environment"
fi

# Install dependencies
echo "Installing required packages..."
pip install -r qc_requirements.txt
pip install -r requirements.txt

# Check if integration works
echo "Testing QuantConnect integration..."
python test_qc_integration.py
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "Integration test failed. Please check the logs and fix any issues before running the autopilot."
    exit 1
fi

# Parse command line arguments
STRATEGIES=5
ITERATIONS=3
MODE="auto"
REFINE=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --refine)
            REFINE="$2"
            shift 2
            ;;
        --start)
            START_DATE="$2"
            shift 2
            ;;
        --end)
            END_DATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Construct command to run autopilot
CMD="python run_quantconnect_autopilot.py --strategies $STRATEGIES --iterations $ITERATIONS --mode $MODE --refine $REFINE"

if [ ! -z "$START_DATE" ]; then
    CMD="$CMD --start $START_DATE"
fi

if [ ! -z "$END_DATE" ]; then
    CMD="$CMD --end $END_DATE"
fi

# Run autopilot
echo "Running QuantConnect autopilot with the following parameters:"
echo "  Strategies per iteration: $STRATEGIES"
echo "  Number of iterations: $ITERATIONS"
echo "  Generation mode: $MODE"
echo "  Strategies to refine: $REFINE"
if [ ! -z "$START_DATE" ]; then
    echo "  Start date: $START_DATE"
fi
if [ ! -z "$END_DATE" ]; then
    echo "  End date: $END_DATE"
fi
echo ""

# Execute command
echo "Executing: $CMD"
echo "======================================================"
eval $CMD