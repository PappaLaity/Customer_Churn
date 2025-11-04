from pydantic import BaseModel
from sqlmodel import Column, Field, String


# class InputCustomer(BaseModel):
#     tenure: float = Field()
#     totalCharges: str = Field()
class InputCustomer(BaseModel):
    Contract: int = Field(..., description="Type de contrat (0, 1, 2, etc.)")
    tenure: float = Field(..., description="Durée d’abonnement en mois")
    OnlineSecurity: int = Field(..., description="Statut de sécurité en ligne (0=Non, 1=Oui)")
    TechSupport: int = Field(..., description="Assistance technique (0=Non, 1=Oui)")
    TotalCharges: float = Field(..., description="Total des frais facturés au client")
    OnlineBackup: int = Field(..., description="Sauvegarde en ligne (0=Non, 1=Oui, 2=Autre)")
    MonthlyCharges: float = Field(..., description="Montant des frais mensuels")
    PaperlessBilling: int = Field(..., description="Facturation sans papier (0=Non, 1=Oui)")