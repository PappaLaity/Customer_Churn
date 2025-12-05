#src/api/routes/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from src.api.core.database import engine
# from src.api.core.database import get_session
from src.api.core.security import API_KEY_SECRET, verify_password
from src.api.entities.users import User, UserRead

# from src.api.core.jwt import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginInput(BaseModel):
    email: str
    password: str


@router.post("/login")
def login_user(login_data: LoginInput):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == login_data.email)).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(login_data.password, user.password):
            raise HTTPException(status_code=401, detail="Invalid password")

        # retourne la clé API globale stockée dans .env
        return {"user": UserRead.model_validate(user), "api_key": API_KEY_SECRET}
