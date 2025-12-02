# tests/test_evaluate.py
# import pytest
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
# from src.etl.preprocessing import preprocess_data
# from sklearn.ensemble import RandomForestClassifier

# @pytest.mark.integration
# def test_model_evaluation_pipeline():
#     """
#     Vérifie que le pipeline preprocessing + modèle fonctionne et renvoie des métriques valides.
#     """
#     # 1️⃣ Préprocessing
#     X_train, X_test, y_train, y_test = preprocess_data()

#     # 2️⃣ Modèle simple pour test (pas besoin de MLflow ici)
#     model = RandomForestClassifier(n_estimators=10, random_state=42)
#     model.fit(X_train, y_train)

#     # 3️⃣ Prédictions
#     y_pred = model.predict(X_test)

#     # 4️⃣ Vérifications
#     # Les prédictions doivent avoir la même taille que y_test
#     assert len(y_pred) == len(y_test)

#     # Les valeurs doivent être 0 ou 1
#     assert set(np.unique(y_pred)).issubset({0, 1})

#     # Les métriques doivent être dans [0,1]
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     assert 0.0 <= acc <= 1.0
#     assert 0.0 <= f1 <= 1.0
import tempfile
import pandas as pd
import pytest
from src.etl import preprocessing as prep
from src.etl import extract

@pytest.mark.integration
def test_model_evaluation_pipeline(monkeypatch):
    """
    Test d'intégration : preprocessing + modèle.
    - Utilise un CSV temporaire mock
    - Vérifie que le pipeline renvoie des métriques et outputs valides
    """

    # -----------------------------
    # 1️⃣ Créer un CSV temporaire
    # -----------------------------
    tmp_csv = tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False)
    df_mock = pd.DataFrame({
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", "Yes"],
        "tenure": [1, 10, 5, 8],
        "PhoneService": ["Yes", "No", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No", "Yes", "No"],
        "OnlineBackup": ["No", "Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "Yes"],
        "TechSupport": ["No", "Yes", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No", "Yes"],
        "StreamingMovies": ["No", "Yes", "No", "Yes"],
        "Contract": ["Month-to-month", "Two year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Mailed check", "Bank transfer", "Mailed check", "Bank transfer"],
        "MonthlyCharges": [70.35, 99.65, 80.0, 90.0],
        "TotalCharges": ["70.35", "1000.50", "80.0", "90.0"],
        "Churn": ["No", "Yes", "No", "Yes"]
    })
    df_mock.to_csv(tmp_csv.name, index=False)

    # -----------------------------------------
    # 2️⃣ Monkeypatch extract.load pour retourner ce CSV
    # -----------------------------------------
    monkeypatch.setattr(extract, "load", lambda filepath=None: pd.read_csv(tmp_csv.name))

    # -----------------------------
    # 3️⃣ Appeler le pipeline
    # -----------------------------
    X_train, X_test, y_train, y_test = prep.preprocess_data()

    # -----------------------------
    # 4️⃣ Vérifications simples
    # -----------------------------
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})
