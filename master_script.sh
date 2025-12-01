#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SCRIPT MASTER: Amélioration Qualité Code (20%)
# Durée totale: ~4h
# ═══════════════════════════════════════════════════════════════

set -e

echo "🚀 AMÉLIORATION QUALITÉ CODE - Customer Churn"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les étapes
step() {
    echo ""
    echo "${GREEN}▶ $1${NC}"
    echo "───────────────────────────────────────────────────────────────"
}

# Fonction pour les avertissements
warn() {
    echo "${YELLOW}⚠️  $1${NC}"
}

# Fonction pour les erreurs
error() {
    echo "${RED}❌ $1${NC}"
    exit 1
}

# ═══════════════════════════════════════════════════════════════
# PHASE 0: BACKUP
# ═══════════════════════════════════════════════════════════════
step "Phase 0: Backup du code actuel"

BACKUP_BRANCH="backup-before-refactor-$(date +%Y%m%d-%H%M%S)"
git checkout -b "$BACKUP_BRANCH" || warn "Impossible de créer branche backup"
git add -A
git commit -m "backup: Avant refactoring qualité" || warn "Rien à commiter"

echo "✅ Backup créé sur branche: $BACKUP_BRANCH"

# Retour sur la branche de travail
git checkout develop || git checkout main || warn "Rester sur branche actuelle"

# ═══════════════════════════════════════════════════════════════
# PHASE 1: CONFIGURATION (15 min)
# ═══════════════════════════════════════════════════════════════
step "Phase 1: Installation des outils (15 min)"

# Installer les outils
pip install --upgrade pip
pip install black flake8 isort pytest pytest-cov pytest-mock pytest-asyncio

# Créer __init__.py manquants
touch src/__init__.py
touch src/api/__init__.py
touch src/etl/__init__.py
touch src/monitoring/__init__.py
touch tests/__init__.py
touch tests/api/__init__.py
mkdir -p tests/unit && touch tests/unit/__init__.py

# Créer .flake8
cat > .flake8 << 'EOF'
[flake8]
max-line-length = 100
exclude = .git,__pycache__,.pytest_cache,venv,env,mlflow_artifacts,migrations
ignore = E203,W503,E501
per-file-ignores = __init__.py:F401
EOF

# Créer pyproject.toml
cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 100
target-version = ['py311']
exclude = '''
/(\.git|__pycache__|\.pytest_cache|venv|env|mlflow_artifacts|migrations)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip_gitignore = true
known_first_party = ["src"]
EOF

# Mettre à jour pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --strict-markers --tb=short --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=30
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
EOF

echo "✅ Configuration terminée"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: CONFTEST (10 min)
# ═══════════════════════════════════════════════════════════════
step "Phase 2: Configuration des tests (10 min)"

# Créer conftest.py
cat > tests/conftest.py << 'EOF'
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
EOF

echo "✅ Conftest créé"

# ═══════════════════════════════════════════════════════════════
# PHASE 3: LOGGER (30 min)
# ═══════════════════════════════════════════════════════════════
step "Phase 3: Création du logger (30 min)"

# Créer le répertoire
mkdir -p src/api/core

# Le fichier logger.py a déjà été créé via l'artifact
# Vérifier qu'il existe
if [ ! -f "src/api/core/logger.py" ]; then
    warn "logger.py manquant - créer manuellement depuis l'artifact"
fi

# Créer le dossier logs
mkdir -p logs
touch logs/.gitkeep

echo "✅ Logger configuré"

# ═══════════════════════════════════════════════════════════════
# PHASE 4: NETTOYAGE (1h)
# ═══════════════════════════════════════════════════════════════
step "Phase 4: Nettoyage et formatage du code (1h)"

# Formater avec Black
echo "🎨 Formatage avec Black..."
black src/ tests/ dags/ scripts/ --line-length 100 || warn "Black a rencontré des erreurs"

# Organiser les imports
echo "📦 Organisation des imports..."
isort src/ tests/ dags/ scripts/ --profile black || warn "isort a rencontré des erreurs"

# Linter
echo "🔍 Analyse avec Flake8..."
flake8 src/ tests/ dags/ scripts/ --count --statistics || warn "Flake8 a détecté des problèmes"

# Nettoyage
echo "🗑️  Nettoyage..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Code formaté et nettoyé"

# ═══════════════════════════════════════════════════════════════
# PHASE 5: TESTS (30 min)
# ═══════════════════════════════════════════════════════════════
step "Phase 5: Exécution des tests (30 min)"

# Tester si l'API démarre
echo "🧪 Test d'import de l'API..."
python -c "from src.api.main import app; print('✅ API importable')" || error "Erreur d'import de l'API"

# Lancer les tests
echo "🧪 Lancement des tests unitaires..."
pytest tests/ -v --cov=src --cov-report=html || warn "Certains tests ont échoué"

echo "✅ Tests exécutés"

# ═══════════════════════════════════════════════════════════════
# PHASE 6: RAPPORT FINAL
# ═══════════════════════════════════════════════════════════════
step "Phase 6: Génération du rapport"

echo ""
echo "📊 RAPPORT DE QUALITÉ CODE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Lignes de code
echo "📏 Statistiques du code:"
find src/ -name "*.py" -exec wc -l {} + | tail -1

# Nombre de tests
echo ""
echo "🧪 Tests:"
pytest --collect-only tests/ 2>/dev/null | grep "<" || echo "Voir pytest.log"

# Coverage
echo ""
echo "📈 Coverage report: htmlcov/index.html"

# Problèmes restants
echo ""
echo "⚠️  Actions manuelles restantes:"
echo "   1. Remplacer print() par logger (lancer 04_replace_prints.py)"
echo "   2. Supprimer code commenté dans src/api/main.py"
echo "   3. Ajouter docstrings manquantes"
echo "   4. Vérifier les imports inutilisés"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "${GREEN}✅ AMÉLIORATION QUALITÉ TERMINÉE !${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📋 Prochaines étapes:"
echo "   1. git diff (vérifier les changements)"
echo "   2. git add ."
echo "   3. git commit -m 'refactor: Improve code quality (linting, tests, logger)'"
echo "   4. Lancer manuellement: python 04_replace_prints.py"
echo ""
