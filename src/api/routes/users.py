from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from src.api.core.database import engine
from src.api.core.security import hash_password, verify_api_key
from src.api.entities.users import User, UserCreate, UserRead, UserUpdate
from fastapi import Depends, HTTPException, status

from pwdlib import PasswordHash

pwd_hash = PasswordHash.recommended()
router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserRead, dependencies=[Depends(verify_api_key)])
def create_user(user_data: UserCreate):
    with Session(engine) as session:
        existing = session.exec(
            select(User).where(User.email == user_data.email)
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        existing = session.exec(
            select(User).where(User.phone == user_data.phone)
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Phone Number already registered"
            )

        user_data.password = hash_password(user_data.password)

        user = User(**user_data.model_dump())
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


@router.get("/", response_model=list[UserRead], dependencies=[Depends(verify_api_key)])
def list_users():
    with Session(engine) as session:
        users = session.exec(select(User)).all()
        return users


@router.get("/{id}", response_model=UserRead, dependencies=[Depends(verify_api_key)])
def details_user(id: int):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.id == id)).first()
        return user


@router.put("/{id}", response_model=UserRead, dependencies=[Depends(verify_api_key)])
def update_user(id: int, user_data: UserUpdate):
    with Session(engine) as session:
        user = session.get(User, id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Vérifie si email déjà pris par un autre utilisateur
        if user_data.email:
            existing = session.exec(
                select(User).where(User.email == user_data.email, User.id != id)
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")

        # Vérifie si numéro déjà pris
        if user_data.phone:
            existing = session.exec(
                select(User).where(User.phone == user_data.phone, User.id != id)
            ).first()
            if existing:
                raise HTTPException(
                    status_code=400, detail="Phone number already registered"
                )

        # Met à jour les champs
        for key, value in user_data.model_dump(exclude_unset=True).items():
            if key == "password":
                value = hash_password(value)
            setattr(user, key, value)

        session.add(user)
        session.commit()
        session.refresh(user)
        return user


@router.delete("/{id}", dependencies=[Depends(verify_api_key)])
def delete_user(id: int):
    with Session(engine) as session:
        user = session.get(User, id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        session.delete(user)
        session.commit()
        return {"message": f"User with id={id} successfully deleted"}
