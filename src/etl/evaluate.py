from src.training.train import train_and_select_best_model
from src.etl.preprocessing import preprocess_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_best_model(cv_folds=5):
    # Train all models and pick the best one
    best_model, _ = train_and_select_best_model(cv_folds=cv_folds)
    # Reload data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Predict on the held‐out test set
    y_pred = best_model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }

def main():
    results = evaluate_best_model()
    print(f"Best model test accuracy: {results['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("Classification Report:")
    print(results['classification_report'])

if __name__ == "__main__":
    main()