from pydantic import BaseModel
from sqlmodel import Column, Field, String


# class InputCustomer(BaseModel):
#     tenure: float = Field()
#     totalCharges: str = Field()
# class InputCustomer(BaseModel):
#     Contract: int = Field(..., description="Type de contrat (0, 1, 2, etc.)")
#     tenure: float = Field(..., description="Durée d’abonnement en mois")
#     OnlineSecurity: int = Field(..., description="Statut de sécurité en ligne (0=Non, 1=Oui)")
#     TechSupport: int = Field(..., description="Assistance technique (0=Non, 1=Oui)")
#     TotalCharges: float = Field(..., description="Total des frais facturés au client")
#     OnlineBackup: int = Field(..., description="Sauvegarde en ligne (0=Non, 1=Oui, 2=Autre)")
#     MonthlyCharges: float = Field(..., description="Montant des frais mensuels")
#     PaperlessBilling: int = Field(..., description="Facturation sans papier (0=Non, 1=Oui)")

class InputCustomer(BaseModel):
    tenure: float = Field(..., description="Durée d'abonnement en mois")
    InternetService_Fiber_optic: bool = Field(..., description="Client avec fibre optique",alias="InternetService_Fiber optic")
    Contract_Two_year: bool = Field(..., description="Contrat sur deux ans",alias="Contract_Two year")
    PaymentMethod_Electronic_check: bool = Field(..., description="Paiement par chèque électronique",alias="PaymentMethod_Electronic check")
    No_internet_service: int = Field(..., description="Pas de service internet (0/1)",alias="No_internet_service")
    TotalCharges: float = Field(..., description="Total facturé au client")
    MonthlyCharges: float = Field(..., description="Montant mensuel facturé")
    PaperlessBilling: int = Field(..., description="Facturation sans papier (0/1)")