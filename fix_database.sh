#!/bin/bash

# Quick Fix Script for Customer Churn MLOps Project
# This script fixes database authentication issues

echo "🔧 Fixing Customer Churn Project Configuration..."
echo ""

# Step 1: Stop all containers
echo "1️⃣ Stopping all running containers..."
docker-compose down -v  # -v removes volumes to reset databases
echo "✅ Containers stopped and volumes removed"
echo ""

# Step 2: Create proper .env file
echo "2️⃣ Creating .env file with correct credentials..."
cat > .env << 'EOF'
# Database Configuration (MUST match docker-compose.yml)
DATABASE_URL=postgresql+psycopg2://user:password@churn_db:5432/churn_db
DB_USER=user
DB_PASSWORD=password
DB_NAME=churn_db
DB_HOST=churn_db
DB_PORT=5432

# MLflow Database Configuration
MLFLOW_DB_URL=postgresql+psycopg2://user:password@mlflow_db:5432/mlflow_db
MLFLOW_DB_USER=user
MLFLOW_DB_PASSWORD=password
MLFLOW_DB_NAME=mlflow_db
MLFLOW_URI=http://mlflow:5000

# Grafana Configuration
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin

# API Configuration
IP_ADDRESS=http://localhost
API_KEY_SECRET=secret-api-key-change-in-production
ENV=dev

# Additional Security (for future use)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
EOF

echo "✅ .env file created successfully"
echo ""

# Step 3: Show the .env content
echo "3️⃣ Environment file content:"
cat .env
echo ""

# Step 4: Start services
echo "4️⃣ Starting all services with fresh databases..."
docker-compose up -d
echo ""

# Step 5: Wait for services to be ready
echo "5️⃣ Waiting for services to start (30 seconds)..."
sleep 30
echo ""

# Step 6: Check container status
echo "6️⃣ Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Step 7: Test API
echo "7️⃣ Testing API Health..."
sleep 5
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is responding!"
    curl http://localhost:8000/health
else
    echo "❌ API not responding yet. Checking logs..."
    echo ""
    echo "API Logs (last 20 lines):"
    docker logs churn_api --tail 20
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Fix Applied!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Access your services:"
echo "  • API:        http://localhost:8000/docs"
echo "  • Airflow:    http://localhost:8080 (admin/admin)"
echo "  • MLflow:     http://localhost:5001"
echo "  • Grafana:    http://localhost:3000 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo ""
echo "🔍 Useful commands:"
echo "  • Check API logs:  docker logs -f churn_api"
echo "  • Check all logs:  docker-compose logs -f"
echo "  • Restart all:     docker-compose restart"
echo "  • Stop all:        docker-compose down"
echo ""
