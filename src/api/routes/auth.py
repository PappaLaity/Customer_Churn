from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session, select
from src.api.entities.users import User, UserRead,UserCreate
from src.api.core.database import engine
# from src.api.core.database import get_session
from src.api.core.security import API_KEY_SECRET, hash_password, verify_password
# from src.api.core.jwt import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])

# ------------- REGISTER -------------
@router.post("/register", response_model=UserRead)
def register_user(user_data: UserCreate):
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.email == user_data.email)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        existing = session.exec(select(User).where(User.phone == user_data.phone)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Phone number already registered")

        user_data.password = hash_password(user_data.password)

        user = User(**user_data.model_dump())
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

# ------------- LOGIN -------------
@router.post("/login")
def login_user(email: str, password: str):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(password, user.password):
            raise HTTPException(status_code=401, detail="Invalid password")

        # retourne la clé API globale stockée dans .env
        return {"api_key": API_KEY_SECRET}