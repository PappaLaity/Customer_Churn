#!/bin/bash

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 DÉPLOIEMENT CUSTOMER CHURN APPLICATION                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Couleurs pour le terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier les fichiers nécessaires
echo ""
log_info "Vérification des fichiers de configuration..."

if [ ! -f "docker-compose.yml" ]; then
    log_error "docker-compose.yml non trouvé"
    exit 1
fi

if [ ! -f "nginx.conf" ]; then
    log_error "nginx.conf non trouvé"
    exit 1
fi

if [ ! -f ".env" ]; then
    log_error ".env non trouvé"
    exit 1
fi

log_success "Tous les fichiers de configuration trouvés"

# Créer les répertoires nécessaires
echo ""
log_info "Création des répertoires pour Certbot..."
mkdir -p certbot/conf certbot/www
chmod 755 certbot/conf certbot/www
log_success "Répertoires créés"

# Copier la configuration Nginx HTTP (temporaire)
echo ""
log_info "Configuration de Nginx en mode HTTP (temporaire)..."
cp nginx.conf nginx.conf.bak
cp nginx-http.conf nginx.conf
log_success "Nginx configuré en mode HTTP"

# Arrêter les services en cours
echo ""
log_info "Arrêt des services existants..."
docker compose down || true
log_success "Services arrêtés"

# Construire les images
echo ""
log_info "Construction des images Docker..."
docker compose build
log_success "Images construites"

# Démarrer les bases de données d'abord
echo ""
log_info "Démarrage des bases de données..."
docker compose up -d db mlflow_db
sleep 10
log_success "Bases de données démarrées"

# Démarrer MLflow
echo ""
log_info "Démarrage de MLflow..."
docker compose up -d mlflow
sleep 5
log_success "MLflow démarré"

# Démarrer les services d'application
echo ""
log_info "Démarrage de FastAPI, Airflow et services de monitoring..."
docker compose up -d fastapi airflow prometheus grafana
sleep 15
log_success "Services d'application démarrés"

# Démarrer Nginx et Certbot ensemble
echo ""
log_info "Démarrage de Nginx et Certbot..."
docker compose up -d nginx certbot

log_warning "Attente de la génération du certificat (cela peut prendre 2-3 minutes)..."

# Vérifier le certificat avec plus d'attente
CERT_PATH="certbot/conf/live/customer-churn.francecentral.cloudapp.azure.com/fullchain.pem"
MAX_ATTEMPTS=36  # 3 minutes
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if [ -f "$CERT_PATH" ]; then
        log_success "Certificat SSL généré avec succès !"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        REMAINING=$((MAX_ATTEMPTS - ATTEMPT))
        echo -e "${BLUE}Tentative $ATTEMPT/$MAX_ATTEMPTS - Encore $((REMAINING * 5)) secondes d'attente...${NC}"
        sleep 5
    fi
done

if [ ! -f "$CERT_PATH" ]; then
    log_warning "Certificat non généré - Vérification des logs Certbot..."
    docker compose logs certbot | tail -40
    
    # Vérifier si Nginx est actif
    if docker compose ps nginx | grep -q "Up"; then
        log_warning "Nginx est actif mais Certbot n'a pas généré de certificat"
        log_info "Tentative manuelle de génération..."
        docker compose exec -T certbot certbot certonly --webroot -w /var/www/certbot -d customer-churn.francecentral.cloudapp.azure.com --email admin@example.com --agree-tos --non-interactive --keep-until-expiring
        sleep 10
        
        if [ -f "$CERT_PATH" ]; then
            log_success "Certificat généré avec succès (tentative manuelle) !"
        else
            log_error "La génération manuelle a aussi échoué. Vérifiez les logs."
            exit 1
        fi
    else
        log_error "Nginx n'est pas actif. Vérifiez ses logs."
        docker compose logs nginx | tail -40
        exit 1
    fi
fi

# Afficher l'état final
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  📊 ÉTAT DES SERVICES                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
docker compose ps

# Afficher les logs importants
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  📋 LOGS NGINX (dernières 10 lignes)                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
docker compose logs nginx | tail -10

# Vérifier la santé des services
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🏥 VÉRIFICATION DE LA SANTÉ DES SERVICES                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

for service in nginx fastapi grafana prometheus mlflow airflow; do
    if docker compose ps | grep "$service" | grep -q "Up"; then
        log_success "$service est actif"
    else
        log_error "$service n'est pas actif"
    fi
done

# Afficher les informations finales
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎉 DÉPLOIEMENT TERMINÉ AVEC SUCCÈS !                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Services disponibles :${NC}"
echo ""
echo "  🌐 Application (page d'accueil)"
echo "     ${BLUE}https://customer-churn.francecentral.cloudapp.azure.com${NC}"
echo ""
echo "  📊 Grafana (Dashboards)"
echo "     ${BLUE}https://customer-churn.francecentral.cloudapp.azure.com/grafana${NC}"
echo ""
echo "  🔧 API FastAPI"
echo "     ${BLUE}https://customer-churn.francecentral.cloudapp.azure.com/api/docs${NC}"
echo ""
echo "  🔄 Airflow (Orchestration)"
echo "     ${BLUE}https://customer-churn.francecentral.cloudapp.azure.com/airflow${NC}"
echo ""
echo "  🧪 MLflow (Model Registry)"
echo "     ${BLUE}https://customer-churn.francecentral.cloudapp.azure.com/mlflow${NC}"
echo ""
echo "  📈 Prometheus (Métriques)"
echo "     ${BLUE}https://customer-churn.francecentral.cloudapp.azure.com/prometheus${NC}"
echo ""
echo -e "${YELLOW}Certbot renouvellera automatiquement le certificat tous les 90 jours${NC}"
echo ""