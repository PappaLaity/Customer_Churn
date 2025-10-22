from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.etl.preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

def train_and_select_best_model(cv_folds=5):
    X_train, X_test, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=1000, random_state=42
        )
    }

    trained_models = {}
    test_accuracies = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining and cross-validating: {name}")
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_accuracies[name] = test_acc

        print(f"\nTest set performance for {name}:")
        print(f"Accuracy: {test_acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        trained_models[name] = model

    # Select best model based on test accuracy
    best_model_name = max(test_accuracies, key=test_accuracies.get)
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name} with test accuracy: {test_accuracies[best_model_name]:.4f}")

    return best_model, trained_models
