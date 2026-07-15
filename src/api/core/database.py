import os
from sqlmodel import Session, create_engine, SQLModel

ENV = os.getenv("ENV", "dev")


if ENV == "test":
    DATABASE_URL = "sqlite:///./test.db"
else:
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        if ENV == "dev":
            DATABASE_URL = "postgresql://user:password@db:5432/churn_db"
        else:
            raise RuntimeError(
                "DATABASE_URL environment variable must be set in non-dev environments"
            )

# Echo SQL to logs only in local development to avoid leaking queries in prod.
engine = create_engine(DATABASE_URL, echo=(ENV == "dev"))


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
