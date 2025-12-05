# """
# Configuration centralisée pour l'API Customer Churn
# Gère les variables d'environnement et les constantes
# """

# import os
# from typing import List


# class Settings:
#     """Configuration de l'application"""

#     # Environment
#     ENV: str = os.getenv("ENV", "dev")
#     DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
#     LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

#     # MLflow
#     MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
#     MODEL_NAME: str = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")
#     MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")

#     # CORS
#     CORS_ORIGINS: List[str] = [
#         "http://localhost:8081",
#         "http://127.0.0.1:8081",
#         "https://customer-churn-dusky.vercel.app",
#     ]

#     # Paths
#     DATA_PATH: str = "data/production/production.csv"

#     # Model reloading
#     MODEL_RELOAD_INTERVAL: int = 300  # 5 minutes

#     # A/B Testing
#     AB_TESTING_RATIO: float = 0.8  # 80% Production, 20% Staging


# # Instance globale
# settings = Settings()
"""
Configuration centralisée pour l'API Customer Churn
Gère les variables d'environnement et les constantes
"""
#src/api/core/config.py
import os
from typing import List


class Settings:
    """Configuration de l'application"""

    # Environment
    ENV: str = os.getenv("ENV", "dev")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- NOUVEAU: Configuration de la base de données principale (churn_db) ---
    # Valeur par défaut 'localhost' pour le développement local
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_USER: str = os.getenv("DB_USER", "user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
    DB_NAME: str = os.getenv("DB_NAME", "churn_db")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    
    @property
    def DATABASE_URL(self) -> str:
        """Construit l'URL de connexion complète pour SQLAlchemy"""
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


    # MLflow
    # Note: La valeur par défaut était 'http://mlflow:5000' dans votre version, 
    # ce qui causera une erreur 'connection refused' en local. Changeons-la.
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MODEL_NAME: str = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")
    MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "https://customer-churn-dusky.vercel.app",
    ]

    # Paths
    DATA_PATH: str = "data/production/production.csv"

    # Model reloading
    MODEL_RELOAD_INTERVAL: int = 300  # 5 minutes

    # A/B Testing
    AB_TESTING_RATIO: float = 0.8  # 80% Production, 20% Staging


# Instance globale
settings = Settings()
