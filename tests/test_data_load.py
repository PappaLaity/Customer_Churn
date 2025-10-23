# tests/test_load_data.py

from src.data.load_data import load_data

def test_load_data():
    """
    Test that the dataset can be loaded and has expected columns.
    """
    df = load_data("/Users/elhadjimamadou/Documents/Customer_Churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")
    assert df is not None, "Dataframe is None"
    assert not df.empty, "Dataframe is empty"
    # Replace with one or more known columns in your dataset
    assert "customerID" in df.columns, "'customerID' column missing"

if __name__ == "__main__":
    test_load_data()
    print("Load data test passed!")

