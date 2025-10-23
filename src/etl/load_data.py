import os
import pandas as pd

# Déterminer le chemin absolu du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Définir le chemin du fichier brut
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")

def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load dataset from an Excel or CSV file.
    
    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    # Charger automatiquement selon l’extension
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xls") or path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    
    print(f"✅ Dataset successfully loaded from: {path}")
    print(f"📊 Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Test manuel si tu exécutes le script directement
    df = load_data()
    print(df.head())
