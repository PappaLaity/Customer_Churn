from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session, select
from src.api.entities.users import User
from src.api.core.database import get_session
from src.api.core.security import verify_password
from src.api.core.jwt import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/login")
def login(email: str, password: str, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Génération du token JWT
    token = create_access_token({"sub": user.email, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}
