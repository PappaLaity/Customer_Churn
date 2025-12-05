import mlflow
import os

# 1️⃣ Assurez-toi que MLflow pointe vers ton serveur
mlflow.set_tracking_uri("http://localhost:5001")  # docker-compose name et port
mlflow.set_registry_uri("http://localhot:5001")

# 2️⃣ Crée un run test
with mlflow.start_run(run_name="test_artifacts") as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # 3️⃣ Crée un petit fichier test dans les artefacts
    test_file_path = "test.txt"
    with open(test_file_path, "w") as f:
        f.write("MLflow artifact test successful!")

    mlflow.log_artifact(test_file_path, artifact_path="test_folder")
    os.remove(test_file_path)

    print("✅ Fichier artefact loggé avec succès !")

# 4️⃣ Lister les artefacts pour vérifier
artifacts = mlflow.artifacts.list_artifacts(f"runs:/{run_id}/test_folder")
print("Liste des artefacts dans le run test :")
for a in artifacts:
    print(a.path)
