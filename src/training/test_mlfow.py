import mlflow
import mlflow.sklearn
import os

# from sklearn.base import accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from dotenv import load_dotenv

load_dotenv()
IP_ADDRESS = os.getenv("IP_ADDRESS")
mlflow_uri = IP_ADDRESS + ":5001"
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("MLflow Autologging Demo")
mlflow.autolog()

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# # 👉 le tracking URI doit pointer vers le container du serveur
# mlflow.set_tracking_uri("http://mlflow:5001")  # si tu es dans le même docker-compose réseau
# # ou mlflow.set_tracking_uri("http://<IP_VM>:5000") si tu es sur une autre machine

# mlflow.set_experiment("churn-prediction")

# with mlflow.start_run(run_name="baseline_randomforest"):
#     model = RandomForestClassifier().fit(X_train, y_train)
#     mlflow.sklearn.log_model(model, "model")
#     mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
