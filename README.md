# Customer_Churn
Mlops Project for Real Time Customer Churn Prediction in the Telecom Domain


### Process
Assume you have docker already installed in your computer:
- Go to the project folder using this command:

    `cd Customer_churn`

- Copy [Env Example](.env.example) manually
    or by cmd `cp .env.example .env`
- Setup Environment file according to different Variables 
- Run `docker compose up --build -d`

- FastAPI [API](http://localhost:8000)

    - A default User Admin is created when launch api
    - You can manipulate endpoint by using api given when you logged to api
    - email: admin@example.com
    - password:admin

- Airflow [Airflow](http://localhost:8080)
    - Login: admin
    - Password: admin
      
- Mlflow [Mlflow](http://localhost:5001)
  
- Grafana [Grafana](http://localhost:3000)
    - login: admin
    - password: admin
