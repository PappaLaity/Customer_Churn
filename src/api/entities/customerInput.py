from pydantic import BaseModel
from sqlmodel import Column, Field, String


class InputCustomer(BaseModel):
    tenure: float = Field()
    totalCharges: str = Field()
