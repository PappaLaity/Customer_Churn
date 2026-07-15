#!/bin/bash
# Test runner: runs the full automated pytest suite under tests/.
set -u

export ENV=test

echo "Running full test suite (tests/)"
echo "======================================"
echo ""

# Run the entire tests/ package (unit tests, API tests, DAG tests, monitoring).
python3 -m pytest tests/ -v --tb=short
UNIT_RESULT=$?

echo ""
echo "======================================"

if [ $UNIT_RESULT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed"
    exit 1
fi
