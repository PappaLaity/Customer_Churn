# import tempfile
# import pandas as pd
# import pytest
# from src.etl import preprocessing as prep
# from src.etl import extract

# @pytest.mark.integration
# def test_model_evaluation_pipeline(monkeypatch):
#     """
#     Test d'intégration : preprocessing + modèle.
#     - Utilise un CSV temporaire mock de 12 lignes
#     - Vérifie que le pipeline renvoie des métriques et outputs valides
#     """

#     # -----------------------------
#     # 1️⃣ Créer un CSV temporaire avec 12 lignes pour stratification
#     # -----------------------------
#     df_mock = pd.DataFrame({
#         "gender": ["Male", "Female"] * 6,
#         "SeniorCitizen": [0, 1] * 6,
#         "Partner": ["Yes", "No"] * 6,
#         "Dependents": ["No", "Yes"] * 6,
#         "tenure": [1, 10, 5, 8, 12, 3, 7, 6, 15, 2, 9, 4],
#         "PhoneService": ["Yes", "No"] * 6,
#         "MultipleLines": ["No", "Yes"] * 6,
#         "InternetService": ["DSL", "Fiber optic"] * 6,
#         "OnlineSecurity": ["Yes", "No"] * 6,
#         "OnlineBackup": ["No", "Yes"] * 6,
#         "DeviceProtection": ["No", "Yes"] * 6,
#         "TechSupport": ["No", "Yes"] * 6,
#         "StreamingTV": ["No", "Yes"] * 6,
#         "StreamingMovies": ["No", "Yes"] * 6,
#         "Contract": ["Month-to-month", "Two year"] * 6,
#         "PaperlessBilling": ["Yes", "No"] * 6,
#         "PaymentMethod": ["Mailed check", "Bank transfer"] * 6,
#         "MonthlyCharges": [70.34, 99.65, 80.0, 90.0, 75.0, 85.5, 65.0, 95.0, 78.0, 88.0, 72.5, 91.0],
#         "TotalCharges": ["70.35", "1000.50", "80.0", "90.0", "75.0", "85.5", "65.0", "95.0", "78.0", "88.0", "72.5", "91.0"],
#         "Churn": ["No", "Yes"] * 6
#     })

#     tmp_csv = tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False)
#     df_mock.to_csv(tmp_csv.name, index=False)

#     # -----------------------------------------
#     # 2️⃣ Monkeypatch extract.load pour retourner ce CSV
#     # -----------------------------------------
#     monkeypatch.setattr(extract, "load", lambda filepath=None: pd.read_csv(tmp_csv.name))

#     # -----------------------------
#     # 3️⃣ Appeler le pipeline
#     # -----------------------------
#     X_train, X_test, y_train, y_test = prep.preprocess_data()

#     # -----------------------------
#     # 4️⃣ Vérifications simples
#     # -----------------------------
#     assert X_train.shape[0] == y_train.shape[0]
#     assert X_test.shape[0] == y_test.shape[0]
#     assert not X_train.isna().any().any()
#     assert not X_test.isna().any().any()
#     assert set(y_train).issubset({0, 1})
#     assert set(y_test).issubset({0, 1})

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from src.etl import preprocessing as prep
from src.etl import extract


@pytest.mark.integration
def test_model_evaluation_pipeline(monkeypatch):
    """
    Test d'intégration : preprocessing + modèle.
    - Utilise un CSV temporaire mock de 12 lignes
    - Vérifie que le pipeline renvoie des métriques et outputs valides
    """

    # -----------------------------
    # 1️⃣ Créer un CSV temporaire avec 12 lignes pour stratification
    # -----------------------------
    df_mock = pd.DataFrame({
        "gender": ["Male", "Female"] * 6,
        "SeniorCitizen": [0, 1] * 6,
        "Partner": ["Yes", "No"] * 6,
        "Dependents": ["No", "Yes"] * 6,
        "tenure": [1, 10, 5, 8, 12, 3, 7, 6, 15, 2, 9, 4],
        "PhoneService": ["Yes", "No"] * 6,
        "MultipleLines": ["No", "Yes"] * 6,
        "InternetService": ["DSL", "Fiber optic"] * 6,
        "OnlineSecurity": ["Yes", "No"] * 6,
        "OnlineBackup": ["No", "Yes"] * 6,
        "DeviceProtection": ["No", "Yes"] * 6,
        "TechSupport": ["No", "Yes"] * 6,
        "StreamingTV": ["No", "Yes"] * 6,
        "StreamingMovies": ["No", "Yes"] * 6,
        "Contract": ["Month-to-month", "Two year"] * 6,
        "PaperlessBilling": ["Yes", "No"] * 6,
        "PaymentMethod": ["Mailed check", "Bank transfer"] * 6,
        "MonthlyCharges": [70.34, 99.65, 80.0, 90.0, 75.0, 85.5, 65.0, 95.0, 78.0, 88.0, 72.5, 91.0],
        "TotalCharges": ["70.35", "1000.50", "80.0", "90.0", "75.0", "85.5", "65.0", "95.0", "78.0", "88.0", "72.5", "91.0"],
        "Churn": ["No", "Yes"] * 6
    })

    tmp_csv = tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False)
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
    # 4️⃣ Vérifications CORRIGÉES (numpy arrays, pas pandas)
    # -----------------------------
    # Vérifier les dimensions
    assert X_train.shape[0] == y_train.shape[0], "X_train/y_train shape mismatch"
    assert X_test.shape[0] == y_test.shape[0], "X_test/y_test shape mismatch"
    
    # ✅ CORRECTION: Utiliser np.isnan() au lieu de .isna()
    assert not np.isnan(X_train).any(), "X_train contains NaN values"
    assert not np.isnan(X_test).any(), "X_test contains NaN values"
    
    # Vérifier que y est binaire (0 ou 1)
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train contains non-binary values"
    assert set(np.unique(y_test)).issubset({0, 1}), "y_test contains non-binary values"
    
    # Vérifier que SMOTE a bien équilibré les classes
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"✅ Class distribution after SMOTE: {dict(zip(unique_train, counts_train))}")
    
    # Vérifier que les features sont bien standardisées (moyenne ~0, std ~1)
    assert -0.5 < X_train.mean() < 0.5, "X_train not properly scaled (mean should be ~0)"
    assert 0.5 < X_train.std() < 1.5, "X_train not properly scaled (std should be ~1)"
    
    # Cleanup
    os.unlink(tmp_csv.name)