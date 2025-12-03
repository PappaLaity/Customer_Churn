# CI/CD Pipeline Setup Guide

This document explains the CI/CD pipeline configuration for the Customer Churn API.

## Overview

Our CI/CD pipeline provides:
- ✅ **Automated Testing**: Run tests on every push/PR
- ✅ **Load Testing**: Weekly performance validation
- ✅ **Staging Deployment**: Automatic deployment on merge to main
- ✅ **Production Deployment**: Controlled releases with blue-green strategy
- ✅ **Automatic Rollback**: Rollback on health check failures

## Architecture

```
┌─────────────┐
│   GitHub    │
│   Actions   │
└──────┬──────┘
       │
       ├──► API Tests (on push/PR)
       │    └── Run pytest
       │
       ├──► Load Tests (weekly/manual)
       │    └── Run Locust tests
       │    └── Validate performance
       │
       ├──► Deploy Staging (on merge to main)
       │    └── Build Docker image
       │    └── Deploy to staging
       │    └── Run smoke tests
       │
       └──► Deploy Production (on release tag)
            └── Build Docker image
            └── Deploy to GREEN environment
            └── Health checks
            └── Switch traffic (Blue → Green)
            └── Monitor for errors
            └── Rollback on failure
```

## GitHub Actions Workflows

### 1. API Tests (`api-tests.yml`)

**Trigger**: Every push/PR  
**Duration**: ~2-3 minutes

```yaml
on:
  push:
    branches: [ main, monitoring-observability ]
  pull_request:
    branches: [ main ]
```

**What it does**:
- Runs all API unit tests
- Validates code quality
- Reports test results

### 2. Load Tests (`load-tests.yml`)

**Trigger**: Weekly (Sundays 2 AM UTC) or manual  
**Duration**: 5-10 minutes

```yaml
on:
  schedule:
    - cron: '0 2 * * 0'
  workflow_dispatch:
```

**What it does**:
- Spins up full environment with Docker Compose
- Runs Locust load tests
- Validates performance thresholds:
  - P95 latency < 1000ms
  - Failure rate < 5%
- Uploads results as artifacts

**Manual trigger**:
```bash
# Go to Actions tab → Load Tests → Run workflow
# Choose scenario: smoke, normal, stress, or spike
```

### 3. Deploy to Staging (`deploy-staging.yml`)

**Trigger**: Merge to `main` branch  
**Duration**: ~5-7 minutes

```yaml
on:
  push:
    branches:
      - main
```

**What it does**:
1. Build Docker image
2. Push to GitHub Container Registry
3. Deploy to staging environment
4. Run smoke tests
5. Notify team

**Environment URL**: `https://staging-api.your-domain.com`

### 4. Deploy to Production (`deploy-production.yml`)

**Trigger**: Release tag or manual  
**Duration**: ~10-15 minutes

```yaml
on:
  release:
    types: [published]
```

**What it does**:
1. **Build**: Create production Docker image
2. **Deploy GREEN**: Deploy to new environment
3. **Health Check**: Validate GREEN is healthy
4. **Smoke Tests**: Test critical endpoints
5. **Traffic Switch**: Route traffic to GREEN
6. **Monitor**: Watch for errors (2 minutes)
7. **Rollback**: Automatic if errors detected
8. **Cleanup**: Decommission old BLUE environment

**Deployment Strategy**: Blue-Green with automatic rollback

## Required GitHub Secrets

Configure these secrets in your repository settings:

### Repository Secrets

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `API_KEY_SECRET` | API authentication key | `your-api-key` |
| `STAGING_API_KEY` | Staging environment API key | `staging-key` |
| `PROD_API_KEY` | Production environment API key | `prod-key` |

### Optional Secrets (for deployment)

| Secret Name | Description | Used For |
|-------------|-------------|----------|
| `AWS_ACCESS_KEY_ID` | AWS credentials | AWS ECS/EKS deployment |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | AWS ECS/EKS deployment |
| `GCP_PROJECT_ID` | Google Cloud project | Cloud Run deployment |
| `GCP_SA_KEY` | Service account key | Cloud Run deployment |
| `KUBE_CONFIG` | Kubernetes config | K8s deployment |
| `SLACK_WEBHOOK` | Slack notifications | Deployment notifications |

### How to add secrets

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret with its value

## Environment Configuration

### Staging Environment

**Purpose**: Test changes before production

- **URL**: `https://staging-api.your-domain.com`
- **Auto-deploy**: Yes (on merge to main)
- **Data**: Test/synthetic data
- **Monitoring**: Same as production

### Production Environment

**Purpose**: Serve real users

- **URL**: `https://api.your-domain.com`
- **Auto-deploy**: No (requires release tag)
- **Data**: Real customer data
- **Monitoring**: Critical alerts enabled
- **Rollback**: Automatic on failure

## Deployment Instructions

### Deploy to Staging

Automatic on merge to `main`:

```bash
git checkout main
git pull origin main
git merge feature-branch
git push origin main
```

Or manually trigger:
1. Go to Actions → Deploy to Staging
2. Click "Run workflow"

### Deploy to Production

#### Option 1: GitHub Release (Recommended)

```bash
# Create and push a tag
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# Create GitHub release
# Go to Releases → Draft a new release → Create release
```

#### Option 2: Manual Workflow

1. Go to Actions → Deploy to Production
2. Click "Run workflow"
3. Enter version tag (e.g., `v1.2.0`)
4. Confirm deployment

### Rollback Production

If you need to manually rollback:

```bash
# Option 1: Trigger previous version deployment
# Actions → Deploy to Production → Run workflow → Enter previous version

# Option 2: Emergency rollback (if configured)
# Your infrastructure should keep the BLUE environment for quick rollback
```

## Monitoring Deployments

### GitHub Actions UI

1. Go to your repository → Actions tab
2. Click on the workflow run
3. View detailed logs for each step

### Check Deployment Status

```bash
# Staging
curl https://staging-api.your-domain.com/api/v1/health \
  -H "X-API-Key: YOUR_KEY"

# Production
curl https://api.your-domain.com/api/v1/health \
  -H "X-API-Key: YOUR_KEY"
```

## Performance Benchmarks

Target metrics from load tests:

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| **P50 Latency** | < 200ms | < 300ms | > 500ms |
| **P95 Latency** | < 500ms | < 1000ms | > 2000ms |
| **P99 Latency** | < 1000ms | < 2000ms | > 5000ms |
| **Throughput** | 500 RPS | 300 RPS | < 100 RPS |
| **Failure Rate** | < 0.1% | < 1% | > 5% |
| **CPU Usage** | < 50% | < 70% | > 90% |
| **Memory Usage** | < 60% | < 80% | > 95% |

## Troubleshooting

### Tests Failing in CI

```bash
# Run tests locally first
pytest tests/api/ -v

# Check if environment variables are set
export ENV=test
pytest tests/api/ -v
```

### Load Tests Timing Out

```bash
# Check if services started
docker compose ps

# Check API health
curl http://localhost:8000/api/v1/health -H "X-API-Key: $API_KEY_SECRET"

# View API logs
docker compose logs fastapi
```

### Deployment Failed

1. Check workflow logs in GitHub Actions
2. Verify Docker image was pushed: `https://github.com/USER/REPO/pkgs/container/REPO`
3. Check environment configuration
4. Verify secrets are set correctly

### Blue-Green Swap Failed

1. Check GREEN environment health
2. Review smoke test results
3. Check load balancer configuration
4. Manually rollback if needed

## Best Practices

### Before Deploying

- ✅ All tests pass locally
- ✅ Code reviewed and approved
- ✅ Feature tested in development
- ✅ Database migrations prepared (if any)
- ✅ Changelog updated

### During Deployment

- ✅ Monitor workflow progress
- ✅ Watch for errors in logs
- ✅ Check metrics dashboard
- ✅ Validate critical endpoints

### After Deployment

- ✅ Verify deployment in environment
- ✅ Run manual smoke tests
- ✅ Monitor error rates
- ✅ Check user feedback
- ✅ Document any issues

## Notifications

### Configure Slack Notifications

Add to your workflow:

```yaml
- name: Notify Slack
  uses: slackapi/slack-github-action@v1
  with:
    webhook: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "Deployment to production completed! 🚀"
      }
```

### Configure Email Notifications

GitHub automatically sends emails to:
- Workflow initiator
- Repository watchers (if configured)

## Advanced Configuration

### Custom Deployment Targets

Edit `deploy-staging.yml` or `deploy-production.yml`:

```yaml
# For AWS ECS
- name: Deploy to ECS
  run: |
    aws ecs update-service \\
      --cluster production \\
      --service api \\
      --force-new-deployment

# For Kubernetes
- name: Deploy to K8s
  run: |
    kubectl set image deployment/api \\
      api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.version.outputs.VERSION }}

# For Cloud Run
- name: Deploy to Cloud Run
  run: |
    gcloud run deploy api \\
      --image ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.version.outputs.VERSION }}
```

### Customize Health Checks

Modify the health check endpoints in workflows:

```yaml
- name: Health check
  run: |
    # Add your custom health checks
    curl -f $API_URL/api/v1/health || exit 1
    curl -f $API_URL/api/v1/model/version || exit 1
```

## Support

For issues with CI/CD:
1. Check workflow logs
2. Review this documentation
3. Open an issue in the repository
4. Contact DevOps team

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Locust Documentation](https://docs.locust.io/)
- [Blue-Green Deployment Pattern](https://martinfowler.com/bliki/BlueGreenDeployment.html)
