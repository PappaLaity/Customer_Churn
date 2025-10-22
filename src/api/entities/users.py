from typing import Optional
from pydantic import EmailStr
from sqlmodel import Column, Field, SQLModel, String
from src.api.utils.enum.UserRole import UserRole
from pydantic import BaseModel

#  Telecom Company User's

E164_REGEX = r"^\+?[1-9]\d{7,14}$"


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String(50), nullable=False))
    email: EmailStr = Field(sa_column=Column("email", String(30), unique=True))
    phone: str = Field(regex=E164_REGEX, unique=True)
    password: str = Field(sa_column=Column(String(255), nullable=False))
    role: UserRole = Field(sa_column=Column(String(50), nullable=False))


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    phone: str
    password: str
    role: UserRole


class UserRead(BaseModel):
    id: int
    username: str
    email: EmailStr
    phone: str
    role: UserRole

    class Config:
        from_attributes = True  # important pour compatibilité SQLModel


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
