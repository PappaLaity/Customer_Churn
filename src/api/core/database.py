from sqlmodel import Session, create_engine, SQLModel

DATABASE_URL = "sqlite:///database.db"
# DATABASE_URL = "mysql+pymysql://root@localhost/churn_test"
engine = create_engine(DATABASE_URL, echo=True)

# engine = create_engine()


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
