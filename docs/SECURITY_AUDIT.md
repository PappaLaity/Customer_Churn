# Security Audit: Public API Endpoints

## ✅ **FIXED: `/models` Endpoint**

**Before:** ❌ PUBLIC (no auth)
**After:** ✅ PROTECTED (requires API key)

```bash
# Now requires authentication
curl http://localhost:8000/models
# Returns: {"detail": "Invalid or missing API Key"}

# With API key
curl http://localhost:8000/models \
  -H "X-API-Key: YOUR_API_KEY_HERE"
# Returns: Model information
```

---

## ⚠️ **NEEDS ATTENTION: `/metrics` Endpoint**

**Status:** 🟡 Currently PUBLIC (Prometheus scraping)

**Location:** `GET /metrics` (exposed by Prometheus Instrumentator)

**What's exposed:**
```
prediction_requests_total{model_version="4"} 1523.0
model_accuracy 0.8567
feature_drift_stat{feature="tenure"} 0.213
prediction_latency_seconds_bucket{...} ...
```

### **Options:**

#### Option 1: Keep Public for Grafana (Recommended for internal deployments)
- Use **firewall rules** to restrict `/metrics` to Grafana IP only
- **Docker network isolation** (Grafana in same network)
- **Kubernetes NetworkPolicy** (if using K8s)

#### Option 2: Add Authentication
```python
# In main.py
from fastapi import Depends
from src.api.core.security import verify_api_key

# Protected metrics endpoint
@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def custom_metrics():
    # Return Prometheus metrics with auth
    pass
```

**Trade-off:** Grafana would need API key configuration

#### Option 3: Move to Internal Endpoint
```python
# Expose only on internal port (8001)
# External: 8000 (user-facing, no /metrics)
# Internal: 8001 (admin/monitoring, has /metrics)
```

---

## 📊 **Complete Security Matrix**

### Public Endpoints (No Auth Required)

| Endpoint | Purpose | Security Risk | Status |
|----------|---------|---------------|--------|
| `GET /` | API info | ✅ Low (generic) | OK |
| `POST /survey/submit` | User predictions | ✅ Low (rate limited) | ✅ **SECURED** |
| `GET /metrics` | Prometheus metrics | 🟡 Medium (internal data) | ⚠️ **NEEDS REVIEW** |

### Protected Endpoints (API Key Required)

| Endpoint | Purpose | Data Exposed |
|----------|---------|--------------|
| `GET /model/version` | Current model versions | Version numbers |
| `GET /models` | All model metadata | ✅ **NOW PROTECTED** |
| `POST /predict` | Batch predictions | Predictions only |
| `GET /ab/config` | A/B test config | Experiment settings |
| `POST /ab/config` | Update A/B config | N/A (write) |
| `GET /ab/results` | A/B analysis | Statistical results |
| `GET /health` | System health | Service status |
| `GET /customers/infos` | Production data | Customer records |
| `POST /monitoring/baseline` | Set baseline | N/A (write) |
| `GET /monitoring/baseline` | Get baseline data | Training data stats |

---

## 🔒 **Recommendations**

### Immediate (Critical)
- [x] ✅ **DONE:** Protect `/models` endpoint
- [ ] 🟡 **REVIEW:** Decide on `/metrics` access strategy

### Short-term (Important)
- [ ] Add **IP whitelisting** for `/metrics`
- [ ] Configure **Grafana service account** with API key
- [ ] **Audit logs** for all authenticated endpoints
- [ ] **Rate limiting** for authenticated endpoints (prevent API key abuse)

### Long-term (Best Practices)
- [ ] Implement **OAuth2** for admin endpoints
- [ ] **Role-based access control** (RBAC)
  - `viewer`: Read-only access to metrics
  - `analyst`: A/B test results
  - `admin`: Full control
- [ ] **mTLS** for service-to-service communication
- [ ] **Secrets rotation** for API keys

---

## 🧪 **Testing Security**

### Test 1: Models Endpoint (Should Fail)
```bash
curl http://localhost:8000/models
# Expected: {"detail": "Invalid or missing API Key"}
```

### Test 2: Metrics Endpoint (Currently Public)
```bash
curl http://localhost:8000/metrics
# Expected: Prometheus metrics (still public)
```

### Test 3: Survey Submit (Rate Limited)
```bash
# Should succeed (10 times/minute)
for i in {1..11}; do
  curl -X POST http://localhost:8000/survey/submit -H "Content-Type: application/json" -d '{...}'
done
# 11th request should fail with 429 Too Many Requests
```

---

## 📝 **Security Checklist**

### Sensitive Data Protection
- [x] Model versions - Protected
- [x] Model accuracy - Protected  
- [x] Training metadata - Protected
- [x] A/B test configuration - Protected
- [x] Customer data - Protected
- [x] User predictions - Rate limited
- [ ] Prometheus metrics - **Needs review**

### Access Controls
- [x] API key authentication - Implemented
- [x] Rate limiting - Implemented (10/min)
- [x] CORS restrictions - Configured
- [x] Input validation - Implemented
- [x] Error message sanitization - Implemented
- [ ] IP whitelisting - **TODO**
- [ ] Role-based access - **TODO**

---

## 🚀 **Next Steps**

1. **Restart API** to apply `/models` protection:
   ```bash
   docker-compose restart fastapi
   ```

2. **Test protected endpoint**:
   ```bash
   curl http://localhost:8000/models  # Should fail
   curl http://localhost:8000/models -H "X-API-Key: YOUR_API_KEY_HERE"  # Should work
   ```

3. **Decide on `/metrics` strategy** based on your deployment:
   - Internal network only? Keep public (use firewall)
   - Public internet? Add authentication or IP whitelist

---

**Security Status:** 🟢 **IMPROVED** - Critical leaks fixed, monitoring endpoint needs deployment-specific configuration.
