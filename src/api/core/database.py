import os
from sqlmodel import Session, create_engine, SQLModel

ENV = os.getenv("ENV", "dev")


if ENV == "test":
    DATABASE_URL = "sqlite:///./test.db"
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/churn_db")

# DATABASE_URL = "postgresql+psycopg2://user:password@db:5432/churn_db"
# DATABASE_URL = "mysql+pymysql://root@localhost/churn_test"
engine = create_engine(DATABASE_URL, echo=True)

# engine = create_engine()


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
