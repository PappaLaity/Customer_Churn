# Quick Reference: API Enhancements

This guide provides quick commands for using the new API features.

## API Versioning

All endpoints now use `/api/v1/` prefix for future compatibility.

### Updated Endpoints

| Old Endpoint | New Endpoint | Status |
|--------------|--------------|--------|
| `/auth/login` | `/api/v1/auth/login` | ✅ Active |
| `/survey/submit` | `/api/v1/survey/submit` | ✅ Active |
| `/model/version` | `/api/v1/model/version` | ✅ Active |
| `/health` | `/api/v1/health` | ✅ Active |
| `/ab/config` | `/api/v1/ab/config` | ✅ Active |

### Testing New Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health \
  -H "X-API-Key: your-api-key"

# Get model version
curl http://localhost:8000/api/v1/model/version \
  -H "X-API-Key: your-api-key"

# Submit prediction
curl -X POST http://localhost:8000/api/v1/survey/submit \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12.0,
    "InternetService_Fiber_optic": true,
    "Contract_Two_year": false,
    "PaymentMethod_Electronic_check": true,
    "No_internet_service": 0,
    "TotalCharges": 1200.50,
    "MonthlyCharges": 85.25,
    "PaperlessBilling": 1
  }'
```

---

## Load Testing

### Quick Start

```bash
# Install locust
pip install locust

# Run with web UI
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Then open: http://localhost:8089
```

### Common Commands

```bash
# Smoke test (5 users, 1 minute)
SCENARIO=smoke locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --users 5 --spawn-rate 1 --run-time 1m --headless

# Normal load (50 users, 5 minutes)
SCENARIO=normal locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 5m --headless

# Stress test (200 users, 10 minutes)
SCENARIO=stress locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --users 200 --spawn-rate 20 --run-time 10m --headless

# Generate HTML report
locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 5m \
  --headless --html=load_report.html
```

### Test Rate Limiting

```bash
# Rapid requests to trigger rate limit
SCENARIO=spike locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --users 10 --spawn-rate 10 --run-time 2m --headless
```

---

## CI/CD Workflows

### Manual Triggers

#### Run Load Tests

```bash
# Via GitHub Actions UI:
# 1. Go to Actions → Load Tests
# 2. Click "Run workflow"
# 3. Select scenario: smoke | normal | stress | spike
# 4. Set duration (e.g., 5m)
# 5. Click "Run workflow"
```

#### Deploy to Staging

```bash
# Automatic on merge to main, or manual:
# 1. Go to Actions → Deploy to Staging
# 2. Click "Run workflow"
# 3. Click "Run workflow" (confirm)
```

#### Deploy to Production

```bash
# Option 1: Create Release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
# Then create GitHub release from tag

# Option 2: Manual Trigger
# 1. Go to Actions → Deploy to Production
# 2. Click "Run workflow"
# 3. Enter version (e.g., v1.0.0)
# 4. Click "Run workflow"
```

### Check Deployment Status

```bash
# View workflow runs
# Go to: https://github.com/USERNAME/REPO/actions

# Check staging deployment
curl https://staging-api.your-domain.com/api/v1/health \
  -H "X-API-Key: $STAGING_API_KEY"

# Check production deployment
curl https://api.your-domain.com/api/v1/health \
  -H "X-API-Key: $PROD_API_KEY"
```

---

## Performance Targets

### Response Times
- **P50**: < 200ms
- **P95**: < 500ms
- **P99**: < 1000ms

### Throughput
- **Normal**: 100-500 requests/sec
- **Peak**: 1000+ requests/sec

### Error Rates
- **Target**: < 0.1%
- **Acceptable**: < 1%
- **Critical**: > 5%

---

## Troubleshooting

### API Version Migration

If you have existing clients:

```python
# Update your API client code
# Old
api_url = "http://localhost:8000/health"

# New
api_url = "http://localhost:8000/api/v1/health"
```

### Load Test Issues

```bash
# Check if API is running
curl http://localhost:8000/

# View API logs
docker compose logs fastapi

# Check MLflow is ready
curl http://localhost:5000/
```

### CI/CD Issues

```bash
# View workflow logs
# GitHub → Actions → Click on workflow run

# Check secrets are configured
# GitHub → Settings → Secrets and variables → Actions
```

---

## Environment Variables

```bash
# For local testing
export API_URL="http://localhost:8000"
export API_KEY_SECRET="your-api-key"

# For load testing
export SCENARIO="normal"  # smoke | normal | stress | spike
```

---

## Documentation Links

- **Load Testing Guide**: [tests/load/README.md](../tests/load/README.md)
- **CI/CD Setup**: [docs/CICD_SETUP.md](CICD_SETUP.md)
- **API Documentation**: http://localhost:8000/docs (when running)
- **Locust Docs**: https://docs.locust.io/

---

## Quick Health Check

```bash
#!/bin/bash
# Save as: check_health.sh

API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY_SECRET:-your-api-key}"

echo "Checking API health..."

# Check root
if curl -s "$API_URL/" > /dev/null; then
  echo "✅ Root endpoint OK"
else
  echo "❌ Root endpoint FAILED"
fi

# Check health
if curl -s "$API_URL/api/v1/health" -H "X-API-Key: $API_KEY" > /dev/null; then
  echo "✅ Health endpoint OK"
else
  echo "❌ Health endpoint FAILED"
fi

# Check model version
if curl -s "$API_URL/api/v1/model/version" -H "X-API-Key: $API_KEY" > /dev/null; then
  echo "✅ Model endpoint OK"
else
  echo "❌ Model endpoint FAILED"
fi

echo "Health check complete!"
```

Usage:
```bash
chmod +x check_health.sh
./check_health.sh
```
