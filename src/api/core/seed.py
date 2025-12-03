from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

# Importez get_db au lieu de get_session
from src.api.core.database import get_db 
from src.api.core.logger import api_logger as logger
from src.api.entities.users import User
from src.api.utils.enum.UserRole import UserRole

# Initialisation de l'instance Argon2
pwd_context = PasswordHash.recommended()  # par défaut, utilise Argon2id


def seed_admin():
    # Use next() to get the session from the generator (maintenant get_db)
    db = next(get_db())
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
            logger.info("Admin par défaut créé (Argon2 utilisé) !")
        else:
            logger.info("Admin déjà existant.")
    except IntegrityError as e:
        # Assurez-vous que logger.info accepte e comme argument positionnel ou utilisez f-string
        logger.info(f"Erreur lors de la création de l'admin : {e}")
        db.rollback()
    except Exception as e:
        logger.info(f"Erreur inattendue: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin()

# from sqlalchemy import select
# from sqlalchemy.exc import IntegrityError
# from pwdlib import PasswordHash
# from sqlmodel import Session

# from src.api.utils.enum.UserRole import UserRole
# from src.api.core.database import get_session
# from src.api.entities.users import User

# # Initialisation de l'instance Argon2
# pwd_context = PasswordHash.recommended()  # par défaut, utilise Argon2id


# def seed_admin(session: Session = None):
#     """Crée un admin par défaut. Si session est None, utilise get_session()"""

#     # Si pas de session fournie, utiliser le générateur
#     if session is None:
#         db = next(get_session())
#         should_close = True
#     else:
#         db = session
#         should_close = False

#     try:
#         existing_admin = db.exec(select(User).where(User.email == "admin@example.com")).first()

#         if not existing_admin:
#             admin = User(
#                 username="Admin",
#                 phone="+221773423567",
#                 email="admin@example.com",
#                 password=pwd_context.hash("admin"),
#                 role=UserRole.ADMIN,
#             )
#             db.add(admin)
#             db.commit()
#             logger.info("Admin par défaut créé (Argon2 utilisé) !")
#         else:
#             logger.info("Admin déjà existant.")
#     except IntegrityError as e:
#         db.rollback()
#         logger.info("Erreur lors de la création de l'admin :", e)
#     except Exception as e:
#         db.rollback()
#         logger.info(f"Erreur inattendue: {e}")
#     finally:
#         if should_close:
#             db.close()


# if __name__ == "__main__":
#     seed_admin()
