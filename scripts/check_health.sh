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
printf "Checking health endpoint... "
if curl -s -f "$API_URL/api/v1/health" -H "X-API-Key: $API_KEY" > /dev/null 2>&1; then
  printf "${GREEN}✓${NC} OK\n"
else
  printf "${RED}✗${NC} FAILED\n"
  exit 1
fi

# Check model version endpoint
printf "Checking model version... "
RESPONSE=$(curl -s "$API_URL/api/v1/model/version" -H "X-API-Key: $API_KEY")
if [ $? -eq 0 ]; then
  printf "${GREEN}✓${NC} OK\n"
  echo "  Response: $RESPONSE"
else
  printf "${RED}✗${NC} FAILED\n"
  exit 1
fi

# Check models endpoint
printf "Checking models endpoint... "
if curl -s -f "$API_URL/api/v1/models" > /dev/null 2>&1; then
  printf "${GREEN}✓${NC} OK\n"
else
  printf "${RED}✗${NC} FAILED\n"
fi

echo ""
echo "========================================"
echo " ${GREEN}All critical checks passed!${NC}"
echo "========================================"
