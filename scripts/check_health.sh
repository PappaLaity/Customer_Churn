#!/bin/bash
# API Health Check Script
# Usage: ./check_health.sh [api_url] [api_key]

API_URL="${1:-http://localhost:8000}"
API_KEY="${2:-${API_KEY_SECRET}}"

echo "========================================"
echo " API Health Check"
echo "========================================"
echo "URL: $API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check root endpoint
printf "Checking root endpoint... "
if curl -s -f "$API_URL/" > /dev/null 2>&1; then
  printf "${GREEN}✓${NC} OK\n"
else
  printf "${RED}✗${NC} FAILED\n"
  exit 1
fi

# Check health endpoint
echo "Checking API health..."
if curl -s -f "$API_URL/health" -H "X-API-Key: $API_KEY" > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is not healthy"
    exit 1
fi

# Check model version
echo "Checking model version..."
RESPONSE=$(curl -s "$API_URL/model/version" -H "X-API-Key: $API_KEY")
if [[ $RESPONSE == *"production_model_version"* ]]; then
    echo "✅ Model version endpoint working"
    echo "   Response: $RESPONSE"
else
    echo "❌ Model version endpoint failed"
    echo "   Response: $RESPONSE"
    exit 1
fi

# Check public models endpoint
echo "Checking public models endpoint..."
if curl -s -f "$API_URL/models" > /dev/null 2>&1; then
    echo "✅ Public models endpoint working"
else
    echo "❌ Public models endpoint failed"
    exit 1
fi
echo ""
echo "========================================"
echo " ${GREEN}All critical checks passed!${NC}"
echo "========================================"
