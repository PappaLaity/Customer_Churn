# API Security Model

## Overview

The Customer Churn API implements a **two-tier security model** to support both guest users and authenticated administrators.

## Access Levels

### 🌐 Guest Access (No Authentication Required)

Guest users can access public endpoints without any API key or login:

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `/` | API root | None |
| `/` | API root | None |
| `/auth/login` | Admin login | None |
| `/survey/submit` | Submit survey for prediction | 10/minute per IP |
| `/health` | Health check | None |
| `/models` | View available models | None |

### 🔐 Admin Access (API Key Required)

Admin users must authenticate to access protected endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/model/version` | Get current model versions |
| `/predict` | Batch predictions |
| `/ab/config` | A/B test configuration |
| `/ab/results` | A/B test results |
| `/monitoring/baseline` | Drift detection baseline |
| `/customers/infos` | Customer data (dashboard) |
| `/users/*` | User management |

---

## Guest User Flow

### 1. Access Public Endpoints

No authentication needed:

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# View available models
curl http://localhost:8000/api/v1/models

# Submit survey for prediction
curl -X POST http://localhost:8000/api/v1/survey/submit \
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

**Response:**
```json
{
  "success": "Thank you for your submission"
}
```

### 2. Rate Limiting

Public endpoints have rate limits to prevent abuse:
- `/survey/submit`: 10 requests per minute per IP
- Other public endpoints: 100 requests per hour per IP

If you exceed the limit, you'll receive:
```json
{
  "detail": "Rate limit exceeded"
}
```
**HTTP Status**: 429 Too Many Requests

---

## Admin User Flow

### 1. Login to Get API Key

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "admin"
  }'
```

**Response:**
```json
{
  "user": {
    "id": 1,
    "email": "admin@example.com",
    "name": "Admin",
    "role": "admin"
  },
  "api_key": "your-api-key-here"
}
```

**Default Credentials:**
- Email: `admin@example.com`
- Password: `admin`

> ⚠️ **IMPORTANT**: Change the default password in production!

### 2. Use API Key for Protected Endpoints

Include the API key in the `X-API-Key` header:

```bash
# Get model versions
curl http://localhost:8000/api/v1/model/version \
  -H "X-API-Key: your-api-key-here"

# Get A/B test configuration
curl http://localhost:8000/api/v1/ab/config \
  -H "X-API-Key: your-api-key-here"

# Get customer data
curl http://localhost:8000/api/v1/customers/infos \
  -H "X-API-Key: your-api-key-here"

# Batch predictions
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"tenure": 12.0, "TotalCharges": 1200.50, ...}
    ]
  }'
```

### 3. Error Handling

**Missing API Key:**
```json
{
  "detail": "Invalid or missing API Key"
}
```
**HTTP Status**: 403 Forbidden

**Invalid API Key:**
```json
{
  "detail": "Invalid or missing API Key"
}
```
**HTTP Status**: 403 Forbidden

---

## Frontend Integration

### Guest User (Survey Page)

```javascript
// No authentication needed
const submitSurvey = async (customerData) => {
  const response = await fetch('http://localhost:8000/api/v1/survey/submit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(customerData)
  });
  
  return await response.json();
};
```

### Admin User (Dashboard)

```javascript
// Store API key after login
const login = async (email, password) => {
  const response = await fetch('http://localhost:8000/api/v1/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password })
  });
  
  const data = await response.json();
  localStorage.setItem('api_key', data.api_key);
  return data;
};

// Use API key for protected requests
const getModelVersion = async () => {
  const apiKey = localStorage.getItem('api_key');
  
  const response = await fetch('http://localhost:8000/api/v1/model/version', {
    headers: {
      'X-API-Key': apiKey
    }
  });
  
  return await response.json();
};
```

---

## Security Best Practices

### For API Administrators

1. **Change Default Credentials**
   ```bash
   # Generate new API key
   python scripts/generate_api_key.py --update-env
   
   # Restart API
   docker compose restart fastapi
   ```

2. **Rotate API Keys Regularly**
   - Rotate keys every 90 days
   - Immediately rotate if compromised

3. **Monitor Rate Limits**
   - Check `/metrics` endpoint for rate limit hits
   - Adjust limits if needed in `main.py`

4. **Use HTTPS in Production**
   - Never send API keys over HTTP
   - Configure TLS certificates

### For Frontend Developers

1. **Never Expose API Keys in Frontend Code**
   - Store API keys securely (localStorage, cookies with httpOnly)
   - Don't commit API keys to git

2. **Handle Authentication Errors**
   ```javascript
   if (response.status === 403) {
     // Redirect to login
     window.location.href = '/login';
   }
   ```

3. **Implement Token Refresh**
   - Re-authenticate when API key expires
   - Handle 403 responses gracefully

---

## API Documentation

### Interactive Documentation

Visit these URLs when the API is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both interfaces show:
- All available endpoints
- Required authentication
- Request/response schemas
-Try-it-out functionality

### Security Scheme

The API uses **API Key** authentication with the following scheme:

- **Type**: API Key
- **Header Name**: `X-API-Key`
- **Location**: Header

---

## Summary

✅ **Public Endpoints**: Anyone can submit surveys and view basic info  
✅ **Protected Endpoints**: Admins only, requires API key  
✅ **Rate Limiting**: Prevents abuse of public endpoints  
✅ **Simple Authentication**: Login once, get API key  
✅ **Frontend Friendly**: Easy integration without barriers for guests
