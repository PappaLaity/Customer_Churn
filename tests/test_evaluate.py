# tests/test_evaluate.py
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.etl.preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier

@pytest.mark.integration
def test_model_evaluation_pipeline():
    """
    Vérifie que le pipeline preprocessing + modèle fonctionne et renvoie des métriques valides.
    """
    # 1️⃣ Préprocessing
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2️⃣ Modèle simple pour test (pas besoin de MLflow ici)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # 3️⃣ Prédictions
    y_pred = model.predict(X_test)

    # 4️⃣ Vérifications
    # Les prédictions doivent avoir la même taille que y_test
    assert len(y_pred) == len(y_test)

    # Les valeurs doivent être 0 ou 1
    assert set(np.unique(y_pred)).issubset({0, 1})

    # Les métriques doivent être dans [0,1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
