# Customer_Churn
Mlops Project for Real Time Customer Churn Prediction in the Telecom Domain

## Production 

We Containerize the project using docker and deploy it automatically using [Deploy workflow](.github/workflows/deploy.yml) (For each Commit and Pull request in main and develop branch we Test it)

- The Frond End is deployed using Vercel and is accessible via this link [Customer Churn](https://customer-churn-dusky.vercel.app)

    -  [Homepage](https://customer-churn-dusky.vercel.app)
    -  [Survey Form](https://customer-churn-dusky.vercel.app/survey)
    -  [Dashboard] (https://customer-churn-dusky.vercel.app/dashboard)
        - This Url require the user to be connected
    -   [Login](https://customer-churn-dusky.vercel.app/login)
        - email: admin@example.com
        - password:admin

    - [Customer Infos](https://customer-churn-dusky.vercel.app/customers-dashboard)
        - This Url require the user to be connected

    - [USER](https://customer-churn-dusky.vercel.app/users)
        - This Url require the user to be connected


- FastAPI [API](https://customer-churn.francecentral.cloudapp.azure.com/docs)

    - A default User Admin is created when launch api
    - You can manipulate endpoint by using api given when you logged to api
    - email: admin@example.com
    - password:admin

- Airflow [Airflow](http://customer-churn.francecentral.cloudapp.azure.com/:8080)
    - Login: admin
    - Password: admin
      
- Mlflow [Mlflow](http://customer-churn.francecentral.cloudapp.azure.com/:5001)
  
- Grafana [Grafana](http://customer-churn.francecentral.cloudapp.azure.com/:3000)
    - login: admin
    - password: admin

## Local Run
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
- To build locally the front End you should Go to the frontend folder using cmd

    `cd frontend`

    - Install dependacy by running `npm install`
    - Copy [.env.example](./frontend/.env.example) to .env and setup You environment
    - Run CMD `npm run serve`
    - URL 
        -  [Homepage](http://localhost:8081)
        -  [Survey Form](http://localhost:8081/survey)
        -  [Dashboard] (http://localhost:8081/dashboard)
            - This Url require the user to be connected
        -   [Login](http://localhost:8081/login)
            - email: admin@example.com
            - password:admin

        - [Customer Infos](http://localhost:8081/customers-dashboard)
            - This Url require the user to be connected

        - [USER](http://localhost:8081/users)
            - This Url require the user to be connected

