#!/bin/bash
# Quick test runner script for drift-based retraining tests

echo "🧪 Running Drift-Based Retraining Tests"
echo "======================================"
echo ""

# Unit tests
echo "📝 Running unit tests (tests/test_retrain.py)..."
python3 -m pytest tests/test_retrain.py -v --tb=short
UNIT_RESULT=$?

echo ""

# Integration tests  
echo "🔗 Running integration tests (test_drift_retrain_logic.py)..."
python3 test_drift_retrain_logic.py
INTEGRATION_RESULT=$?

echo ""
echo "======================================"

if [ $UNIT_RESULT -eq 0 ] && [ $INTEGRATION_RESULT -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi
