# reset_admin_password.py
from sqlmodel import Session, select
from src.api.core.database import engine
from src.api.core.security import hash_password
from src.api.entities.users import User

new_password = "admin"  # mot de passe en clair que tu veux mettre
hashed = hash_password(new_password)

with Session(engine) as session:
    user = session.exec(select(User).where(User.email == "admin@example.com")).first()
    if not user:
        print("❌ User not found")
    else:
        user.password = hashed
        session.add(user)
        session.commit()
        print("✅ Admin password updated successfully")
