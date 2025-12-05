# """
# Modèle SQLAlchemy pour stocker les prédictions de production
# """
# from datetime import datetime

# from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean
# from sqlalchemy.ext.declarative import declarative_base

# Base = declarative_base()


# class Prediction(Base):
#     """
#     Table pour stocker TOUTES les prédictions faites en production.
    
#     ⚠️ IMPORTANT pour drift detection:
#     - Stocke les features utilisées pour la prédiction
#     - Permet d'analyser l'évolution dans le temps
#     - Remplace le fichier CSV production.csv
#     """
#     __tablename__ = "predictions"
    
#     # Identifiant unique
#     id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
#     # Identifiant client (optionnel)
#     customer_id = Column(String(100), nullable=True, index=True)
    
#     # ═══════════════════════════════════════════════════════════════
#     # FEATURES utilisées pour la prédiction
#     # ⚠️ Ces colonnes DOIVENT correspondre à vos features d'entraînement
#     # ═══════════════════════════════════════════════════════════════
    
#     # Features numériques principales
#     tenure = Column(Integer, nullable=False)
#     monthly_charges = Column(Float, nullable=False)
#     total_charges = Column(Float, nullable=False)
    
#     # Features catégorielles encodées (one-hot encoding)
#     # Basé sur votre CSV: InternetService_Fiber_optic, Contract_Two_year, etc.
#     internet_service_fiber_optic = Column(Boolean, default=False)
#     contract_two_year = Column(Boolean, default=False)
#     payment_method_electronic_check = Column(Boolean, default=False)
#     no_internet_service = Column(Boolean, default=False)
#     paperless_billing = Column(Boolean, default=False)
    
#     # ⚠️ Ajouter TOUTES les autres features que votre modèle utilise
#     # Vous pouvez les trouver dans:
#     # - /opt/airflow/data/features/features.csv (colonnes)
#     # - Ou dans votre InputCustomer schema
    
#     # ═══════════════════════════════════════════════════════════════
#     # RÉSULTATS de la prédiction
#     # ═══════════════════════════════════════════════════════════════
#     prediction = Column(Integer, nullable=False)  # 0 = no churn, 1 = churn
#     probability = Column(Float, nullable=True)  # Probabilité de churn (0-1)
    
#     # Modèle utilisé (A/B testing)
#     model_version = Column(String(50), nullable=True)  # "Production" ou "Staging"
#     model_stage = Column(String(20), nullable=True)  # "production", "staging"
    
#     # ═══════════════════════════════════════════════════════════════
#     # MÉTADONNÉES
#     # ═══════════════════════════════════════════════════════════════
#     latency = Column(Float, nullable=True)  # Temps de prédiction (secondes)
#     created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
#     # Optionnel: Label réel (si fourni plus tard pour calculer accuracy)
#     actual_churn = Column(Integer, nullable=True)  # Vrai label (si connu)
#     feedback_received_at = Column(DateTime, nullable=True)  # Quand le feedback a été reçu
    
#     def __repr__(self):
#         return (
#             f"<Prediction(id={self.id}, customer={self.customer_id}, "
#             f"prediction={self.prediction}, prob={self.probability:.3f}, "
#             f"created_at={self.created_at})>"
#         )
    
#     def to_dict(self):
#         """Convertir en dictionnaire pour JSON responses"""
#         return {
#             "id": self.id,
#             "customer_id": self.customer_id,
#             "tenure": self.tenure,
#             "monthly_charges": self.monthly_charges,
#             "total_charges": self.total_charges,
#             "prediction": self.prediction,
#             "probability": self.probability,
#             "model_version": self.model_version,
#             "created_at": self.created_at.isoformat() if self.created_at else None,
#         }

"""
Modèle SQLModel pour stocker les prédictions de production
"""
from datetime import datetime
from sqlmodel import SQLModel, Field, Column, Integer, String, Float, Boolean, DateTime
# Note: Nous n'avons plus besoin de declarative_base()

class Prediction(SQLModel, table=True):
    """
    Table pour stocker TOUTES les prédictions faites en production.
    """
    __tablename__ = "predictions"
    
    # Identifiant unique
    id: int = Field(default=None, primary_key=True)
    
    # Identifiant client (optionnel)
    customer_id: str = Field(nullable=True, index=True, max_length=100)
    
    # ═══════════════════════════════════════════════════════════════
    # FEATURES utilisées pour la prédiction
    # ═══════════════════════════════════════════════════════════════
    
    # Features numériques principales
    tenure: int = Field(nullable=False)
    monthly_charges: float = Field(nullable=False)
    total_charges: float = Field(nullable=False)
    
    # Features catégorielles encodées (one-hot encoding)
    internet_service_fiber_optic: bool = Field(default=False)
    contract_two_year: bool = Field(default=False)
    payment_method_electronic_check: bool = Field(default=False)
    no_internet_service: bool = Field(default=False)
    paperless_billing: bool = Field(default=False)
    
    # ═══════════════════════════════════════════════════════════════
    # RÉSULTATS de la prédiction
    # ═══════════════════════════════════════════════════════════════
    prediction: int = Field(nullable=False)  # 0 = no churn, 1 = churn
    probability: float = Field(nullable=True)  # Probabilité de churn (0-1)
    
    # Modèle utilisé (A/B testing)
    model_version: str = Field(nullable=True, max_length=50)
    model_stage: str = Field(nullable=True, max_length=20)
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTADONNÉES
    # ═══════════════════════════════════════════════════════════════
    latency: float = Field(nullable=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False, index=True)
    
    # Optionnel: Label réel
    actual_churn: int = Field(nullable=True)
    feedback_received_at: datetime = Field(nullable=True)
    
    # Note: Les méthodes __repr__ et to_dict() ne sont plus nécessaires avec SQLModel
    # car Pydantic s'en charge automatiquement.
