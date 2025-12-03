# Load Testing

This directory contains load testing infrastructure for the Customer Churn API.

## Overview

We use [Locust](https://locust.io/) for load testing because:
- ✅ Python-based (easy to maintain)
- ✅ Distributed testing support
- ✅ Real-time web UI
- ✅ Comprehensive metrics

## Installation

```bash
pip install locust
```

## Quick Start

### 1. Basic Web UI Mode
```bash
cd /Users/mahamatabakarassouna/Customer_Churn
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 and configure:
- Number of users: 50
- Spawn rate: 5 users/sec
- Host: http://localhost:8000

### 2. Headless Mode (CI/CD)
```bash
locust -f tests/load/locustfile.py \
       --host=http://localhost:8000 \
       --users 50 \
       --spawn-rate 5 \
       --run-time 5m \
       --headless \
       --html=reports/load_test_report.html
```

## Test Scenarios

### Smoke Test (Quick validation)
```bash
SCENARIO=smoke locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 5 --spawn-rate 1 --run-time 1m --headless
```

### Normal Load (Typical traffic)
```bash
SCENARIO=normal locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 50 --spawn-rate 5 --run-time 5m --headless
```

### Stress Test (Heavy load)
```bash
SCENARIO=stress locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 200 --spawn-rate 20 --run-time 10m --headless
```

### Spike Test (Sudden traffic burst)
```bash
SCENARIO=spike locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 500 --spawn-rate 100 --run-time 2m --headless
```

## Testing Rate Limiting

To specifically test rate limiting:

```bash
# This will rapidly send requests to trigger rate limits
SCENARIO=spike locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 10 --spawn-rate 10 --run-time 2m --headless
```

Expected behavior:
- ✅ API responds with 429 (Too Many Requests) after 100 requests/hour per IP
- ✅ Rate limited requests don't cause cascading failures
- ✅ API continues serving requests after rate limit window

## Environment Variables

```bash
# API Configuration
export API_URL="http://localhost:8000"
export API_KEY_SECRET="your-api-key"

# Test Scenario
export SCENARIO="normal"  # smoke, normal, stress, spike
```

## Metrics to Monitor

### Response Time
- **P50**: Should be < 200ms
- **P95**: Should be < 500ms
- **P99**: Should be < 1000ms

### Throughput
- **Typical**: 100-500 requests/sec
- **Peak**: 1000+ requests/sec

### Error Rate
- **Target**: < 1% under normal load
- **Acceptable**: < 5% under stress

### Rate Limiting
- **429 Errors**: Expected when limits hit
- **Recovery**: Immediate after window expires

## Distributed Testing

For very high loads, run distributed tests:

### Master Node
```bash
locust -f tests/load/locustfile.py --master \
       --host=http://localhost:8000 \
       --expect-workers=4
```

### Worker Nodes (run 4 terminals)
```bash
locust -f tests/load/locustfile.py --worker --master-host=127.0.0.1
```

## Interpreting Results

### Good Performance
```
Response time (ms)
 Name                          # reqs      50%     95%     99%    Avg
 /api/v1/survey/submit         10000       150     350     600    200
 /api/v1/model/version          5000        50     150     300     80
 
Total RPS: 450
Failure Rate: 0.2%
```

### Poor Performance
```
Response time (ms)
 Name                          # reqs      50%     95%     99%    Avg
 /api/v1/survey/submit         10000       800    2500    5000   1200
 
Total RPS: 50
Failure Rate: 5.0%
```

## Troubleshooting

### High Response Times
- Check Docker container resources
- Monitor MLflow model loading
- Review database connection pooling

### High Error Rate
- Check API logs: `docker compose logs fastapi`
- Verify MLflow is running: `curl http://localhost:5000`
- Check database connectivity

### Rate Limiting Not Working
- Verify slowapi is installed
- Check rate limiter configuration in `main.py`
- Review X-Forwarded-For headers if behind proxy

## CI/CD Integration

See [.github/workflows/load-tests.yml](../../.github/workflows/load-tests.yml) for automated load testing in CI/CD pipeline.
