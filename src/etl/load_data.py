import os
import pandas as pd

# Déterminer le chemin absolu du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Définir le chemin du fichier brut
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load dataset from a CSV or Excel file, automatically selecting the engine.
    
    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    # Détection automatique du type de fichier
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xls"):
        try:
            import xlrd
        except ImportError:
            raise ImportError("Please install xlrd to read .xls files: pip install xlrd")
        df = pd.read_excel(path, engine="xlrd")
    elif path.endswith(".xlsx"):
        try:
            import openpyxl
        except ImportError:
            raise ImportError("Please install openpyxl to read .xlsx files: pip install openpyxl")
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    
    print(f"✅ Dataset successfully loaded from: {path}")
    print(f"📊 Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Test manuel si tu exécutes le script directement
    df = load_data()
    print(df.head())
