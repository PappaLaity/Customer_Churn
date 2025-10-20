from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from sqlalchemy import create_engine, select
from sqlmodel import SQLModel, Session

from src.api.utils.enum.UserRole import UserRole
from src.api.entities.users import User


app = FastAPI(title="Customer Churn Prediction")

from pydantic import BaseModel

class UserRead(BaseModel):
    id: int
    username: str
    email: str
    phone: str
    role: str

    class Config:
        orm_mode = True

# engine = create_engine("sqlite:///database.db")
DATABASE_URL = "mysql+pymysql://root@localhost/churn_test"
engine = create_engine(DATABASE_URL, echo=True)

SQLModel.metadata.create_all(engine)


@app.get("/",response_model=UserRead)
async def home():

    # with Session(engine) as session:
    #     statement = select(User).where(User.username == "Username")
    #     user = session.exec(statement).first()
    #     print(type(user))
    
    return {"results": "Welcome"}


@app.get("/health")
async def check_healh():
    return {"check": "I'm ok! No worry"}


        # user = user.model_dump()
        # print(user)
        # user = User(
        #     username="Username",
        #     email="username1@example.com",
        #     phone="+221772343321",
        #     password="secret",
        #     role=UserRole.ADMIN
        # )
        # session.add(user)
        # session.commit()