from pydantic import BaseModel, Field, field_validator
from sqlmodel import Column, String


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
    """Customer survey input with security validations."""
    
    # Numeric fields with reasonable business constraints
    tenure: float = Field(
        ..., 
        ge=0, 
        le=120,
        description="Subscription duration in months (0-120)"
    )
    TotalCharges: float = Field(
        ..., 
        ge=0, 
        le=100000,
        description="Total charges ($0-$100,000)"
    )
    MonthlyCharges: float = Field(
        ..., 
        ge=0, 
        le=500,
        description="Monthly charges ($0-$500)"
    )
    
    # Boolean/binary fields
    InternetService_Fiber_optic: bool = Field(
        ..., 
        description="Customer has fiber optic internet"
    )
    Contract_Two_year: bool = Field(
        ..., 
        description="Has two-year contract"
    )
    PaymentMethod_Electronic_check: bool = Field(
        ..., 
        description="Pays by electronic check"
    )
    No_internet_service: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="No internet service (0/1)"
    )
    PaperlessBilling: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="Paperless billing (0/1)"
    )
    
    @field_validator('TotalCharges', 'MonthlyCharges', 'tenure')
    @classmethod
    def check_positive_values(cls, v, info):
        """Ensure financial and tenure values are non-negative."""
        if v < 0:
            raise ValueError(f'{info.field_name} must be non-negative')
        return v
    
    @field_validator('TotalCharges')
    @classmethod
    def validate_total_charges(cls, v, info):
        """Business logic: TotalCharges should be reasonable."""
        if v > 100000:  # $100k seems unrealistic for telecom
            raise ValueError('TotalCharges exceeds reasonable limit')
        return v