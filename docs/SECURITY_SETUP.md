# Security Setup Guide

## Generating a Secure API Key

### Quick Start

1. **Generate a new API key:**
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Update your `.env` file:**
   ```bash
   # Copy the generated key
   nano .env  # or your preferred editor
   # Update the line: API_KEY_SECRET = "your-generated-key-here"
   ```

3. **Restart your API:**
   ```bash
   docker-compose restart fastapi
   ```

4. **Test the new key:**
   ```bash
   curl http://localhost:8000/models \
     -H "X-API-Key: your-new-key-here"
   ```

---

## API Key Rotation

### Why Rotate?

- **Security breach**: Key may have been exposed
- **Personnel changes**: Team member leaving
- **Regular maintenance**: Scheduled rotation (every 90 days recommended)

### How to Rotate

1. Generate new key (see above)
2. Update `.env` file  
3. Restart API services
4. **Important**: All users must log in again with new key

---

## Best Practices

### ✅ DO:
- Store API keys in `.env` file (gitignored)
- Use strong random keys (32+ characters)
- Rotate keys regularly
- Use different keys for dev/staging/production
- Limit key sharing to necessary personnel

### ❌ DON'T:
- Commit keys to Git
- Share keys in plain text (email, Slack, etc.)
- Use simple/predictable keys
- Use same key across environments
- Hardcode keys in source code

---

## Environment-Specific Keys

### Development
```bash
# .env (local)
API_KEY_SECRET = "dev-key-abc123..."
```

### Staging
```bash
# .env.staging  
API_KEY_SECRET = "staging-key-xyz789..."
```

### Production
```bash
# .env.production (managed by deployment system)
API_KEY_SECRET = "prod-key-secure-random..."
```

---

## Checking for Exposed Keys

### Search your repository:
```bash
# Check for any remaining hardcoded keys
grep -r "secret-api-key" . --exclude-dir=.git

# Check Git history
git log -p | grep -i "api.key"
```

### If key was committed to Git:

**Option 1: Rotate immediately** (recommended)
- Generate new key
- Update `.env`
- Consider the old key compromised

**Option 2: Remove from history** (advanced)
```bash
# Use git-filter-repo or BFG Repo-Cleaner
# WARNING: Rewrites history, requires force push
```

---

## Troubleshooting

### "Invalid or missing API Key" error

**Cause**: API key mismatch between client and server

**Solution**:
1. Check `.env` file has correct key
2. Restart API: `docker-compose restart fastapi`
3. Clear frontend localStorage and log in again
4. Verify key in request headers

### Frontend login fails

**Cause**: Old API key cached in browser

**Solution**:
```javascript
// Open browser console (F12)
localStorage.clear()
// Refresh page and log in again
```

---

## Security Audit Checklist

- [ ] API key is not in `.env.example`
- [ ] API key is not in test files
- [ ] API key is not in documentation
- [ ] `.env` is in `.gitignore`
- [ ] Using strong random key (32+ chars)
- [ ] Different keys for dev/prod
- [ ] Keys rotated in last 90 days

---

## Getting Help

If you suspect your API key has been compromised:
1. **Rotate immediately** (see above)
2. Review API access logs
3. Check for unusual activity
4. Document the incident

**For production systems, consider:**
- Implementing per-user API keys
- Adding API usage monitoring
- Setting up automated alerts
- Using secrets management systems (AWS Secrets Manager, HashiCorp Vault, etc.)
