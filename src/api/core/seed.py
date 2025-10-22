from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from pwdlib import PasswordHash

from src.api.utils.enum.UserRole import UserRole
from src.api.core.database import get_session
from src.api.entities.users import User

# Initialisation de l'instance Argon2
pwd_context = PasswordHash.recommended()  # par défaut, utilise Argon2id


def seed_admin():
    # Use next() to get the session from the generator
    db = next(get_session())
    try:
        existing_admin = db.exec(select(User).where(User.email == "admin@example.com")).first()
        
        if not existing_admin:
            admin = User(
                username="Admin",
                phone="+221773423567",
                email="admin@example.com",
                password=pwd_context.hash("admin"),  # <— hachage Argon2
                role=UserRole.ADMIN,
            )
            db.add(admin)
            db.commit()
            print("✅ Admin par défaut créé (Argon2 utilisé) !")
        else:
            print("ℹ️  Admin déjà existant.")
    except IntegrityError as e:
        db.rollback()
        print("❌ Erreur lors de la création de l'admin :", e)
    except Exception as e:
        db.rollback()
        print(f"❌ Erreur inattendue: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin()