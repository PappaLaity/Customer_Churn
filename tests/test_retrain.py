import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.training.retrain import (
    train_features_only,
    train_combined,
    _align_and_concat,
    _split_scale_smote,
    _ensure_label,
)


@pytest.fixture
def sample_features_df():
    """Create a sample features dataframe with proper structure."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 80, 100),
        'Tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(100, 8000, 100),
        'Churn': np.random.randint(0, 2, 100),
    })
    return df


@pytest.fixture
def sample_production_df():
    """Create a sample production dataframe."""
    np.random.seed(43)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 80, 50),
        'Tenure': np.random.randint(1, 72, 50),
        'MonthlyCharges': np.random.uniform(20, 120, 50),
        'TotalCharges': np.random.uniform(100, 8000, 50),
        'Churn': np.random.randint(0, 2, 50),
    })
    return df


@pytest.fixture
def temp_csv_files(sample_features_df, sample_production_df):
    """Create temporary CSV files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, "features.csv")
        production_path = os.path.join(tmpdir, "production.csv")
        empty_production_path = os.path.join(tmpdir, "empty_production.csv")
        
        sample_features_df.to_csv(features_path, index=False)
        sample_production_df.to_csv(production_path, index=False)
        
        # Create empty CSV
        pd.DataFrame().to_csv(empty_production_path, index=False)
        
        yield {
            'features': features_path,
            'production': production_path,
            'empty_production': empty_production_path,
            'tmpdir': tmpdir,
        }


class TestEnsureLabel:
    """Tests for _ensure_label function."""
    
    def test_ensure_label_with_churn_column(self, sample_features_df):
        """Test that label validation passes when Churn column exists."""
        result = _ensure_label(sample_features_df, label="Churn")
        assert result is not None
        assert "Churn" in result.columns
    
    def test_ensure_label_missing_column(self, sample_features_df):
        """Test that KeyError is raised when label column is missing."""
        with pytest.raises(KeyError):
            _ensure_label(sample_features_df, label="NonExistent")


class TestAlignAndConcat:
    """Tests for _align_and_concat function."""
    
    def test_align_concat_both_have_churn(self, sample_features_df, sample_production_df):
        """Test alignment and concatenation when both datasets have Churn label."""
        result = _align_and_concat(sample_features_df, sample_production_df, label="Churn")
        
        assert isinstance(result, pd.DataFrame)
        assert "Churn" in result.columns
        assert len(result) > 0
        assert len(result) <= len(sample_features_df) + len(sample_production_df)
    
    def test_align_concat_production_missing_churn(self, sample_features_df, sample_production_df):
        """Test fallback when production data lacks Churn label."""
        prod_df = sample_production_df.drop(columns=["Churn"])
        result = _align_and_concat(sample_features_df, prod_df, label="Churn")
        
        # Should fall back to features only
        assert result.equals(sample_features_df)
    
    def test_align_concat_features_missing_churn(self, sample_features_df, sample_production_df):
        """Test that error is raised when features data lacks Churn label."""
        features_df = sample_features_df.drop(columns=["Churn"])
        with pytest.raises(KeyError):
            _align_and_concat(features_df, sample_production_df, label="Churn")
    
    def test_align_concat_removes_nans(self):
        """Test that NaN values in label column are removed."""
        df1 = pd.DataFrame({
            'Age': [25, 30, 35],
            'Churn': [0, 1, 0]
        })
        df2 = pd.DataFrame({
            'Age': [40, 45],
            'Churn': [np.nan, 1]
        })
        result = _align_and_concat(df1, df2, label="Churn")
        
        # Should exclude the row with NaN in Churn
        assert result['Churn'].isna().sum() == 0


class TestSplitScaleSmote:
    """Tests for _split_scale_smote function."""
    
    def test_split_scale_smote_returns_arrays(self, sample_features_df):
        """Test that function returns properly shaped arrays."""
        X_train, X_test, y_train, y_test = _split_scale_smote(sample_features_df, label="Churn")
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        # SMOTE may return numpy arrays or pandas Series
        assert isinstance(y_train, (np.ndarray, pd.Series))
        assert isinstance(y_test, (np.ndarray, pd.Series, pd.core.series.Series))
        
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
    
    def test_split_scale_smote_train_test_split(self, sample_features_df):
        """Test that train-test split is approximately 80-20."""
        X_train, X_test, y_train, y_test = _split_scale_smote(sample_features_df, label="Churn")
        
        total_samples = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total_samples
        
        # Allow some tolerance due to SMOTE
        assert 0.75 < train_ratio < 0.9


class TestTrainCombined:
    """Tests for train_combined function."""
    
    @patch('src.training.retrain._train_and_log_from_arrays')
    @patch('src.training.retrain.register_best_model')
    def test_train_combined_with_production_data(
        self, mock_register, mock_train, temp_csv_files, sample_features_df, sample_production_df
    ):
        """Test combined training when both features and production data exist."""
        mock_train.return_value = {}
        mock_register.return_value = 1
        
        result = train_combined(
            temp_csv_files['features'],
            temp_csv_files['production']
        )
        
        assert mock_train.called
        assert mock_register.called
        assert result == 1
    
    def test_train_combined_empty_production_returns_minus_one(
        self, temp_csv_files, sample_features_df
    ):
        """Test that -1 is returned when production data is empty."""
        result = train_combined(
            temp_csv_files['features'],
            temp_csv_files['empty_production']
        )
        
        assert result == -1
    
    def test_train_combined_missing_production_returns_minus_one(
        self, temp_csv_files
    ):
        """Test that -1 is returned when production file doesn't exist."""
        missing_path = os.path.join(temp_csv_files['tmpdir'], "nonexistent.csv")
        
        result = train_combined(
            temp_csv_files['features'],
            missing_path
        )
        
        assert result == -1
    
    def test_train_combined_missing_features_raises_error(
        self, temp_csv_files, sample_production_df
    ):
        """Test that error is raised when features file is missing."""
        missing_path = os.path.join(temp_csv_files['tmpdir'], "nonexistent_features.csv")
        
        with pytest.raises(FileNotFoundError):
            train_combined(
                missing_path,
                temp_csv_files['production']
            )
    
    def test_train_combined_empty_features_raises_error(
        self, temp_csv_files
    ):
        """Test that error is raised when features file is empty."""
        with pytest.raises(Exception):  # Will raise EmptyDataError
            train_combined(
                temp_csv_files['empty_production'],  # This is empty
                temp_csv_files['production']
            )


class TestTrainFeaturesOnly:
    """Tests for train_features_only function."""
    
    @patch('src.training.retrain.preprocess_data')
    @patch('src.training.retrain._train_and_log_from_arrays')
    @patch('src.training.retrain.register_best_model')
    def test_train_features_only(
        self, mock_register, mock_train, mock_preprocess
    ):
        """Test feature-only training flow."""
        X_train = np.random.rand(80, 4)
        X_test = np.random.rand(20, 4)
        y_train = np.random.randint(0, 2, 80)
        y_test = np.random.randint(0, 2, 20)
        
        mock_preprocess.return_value = (X_train, X_test, y_train, y_test)
        mock_train.return_value = {}
        mock_register.return_value = 1
        
        result = train_features_only()
        
        assert mock_preprocess.called
        assert mock_train.called
        assert mock_register.called
        assert result == 1


class TestMainFunction:
    """Tests for main function behavior."""
    
    @patch('src.training.retrain.train_combined')
    @patch('sys.argv', ['retrain.py', '--mode', 'combined', '--features-path', '/tmp/f.csv', '--production-path', '/tmp/p.csv'])
    def test_main_combined_mode_no_drift(self, mock_train_combined):
        """Test main function with combined mode and no drift (returns -1)."""
        mock_train_combined.return_value = -1
        
        from src.training.retrain import main
        
        # Should not raise an exception
        main()
        assert mock_train_combined.called
    
    @patch('src.training.retrain.train_combined')
    @patch('sys.argv', ['retrain.py', '--mode', 'combined', '--features-path', '/tmp/f.csv', '--production-path', '/tmp/p.csv'])
    def test_main_combined_mode_with_drift(self, mock_train_combined):
        """Test main function with combined mode and drift detected."""
        mock_train_combined.return_value = 1
        
        from src.training.retrain import main
        
        # Should not raise an exception
        main()
        assert mock_train_combined.called


class TestDriftSkipScenario:
    """Integration tests for drift skip scenario."""
    
    def test_skip_retraining_when_no_production_data(self, temp_csv_files):
        """Integration test: skip retraining when production data is empty/missing."""
        # Scenario: Production file is empty
        result = train_combined(
            temp_csv_files['features'],
            temp_csv_files['empty_production']
        )
        
        assert result == -1, "Should return -1 to indicate no retraining"
    
    def test_retrain_when_production_data_exists(
        self, temp_csv_files, sample_features_df, sample_production_df
    ):
        """Integration test: attempt retraining when production data exists."""
        # This test verifies the function attempts to process data
        # We mock the training functions to avoid needing MLflow setup
        with patch('src.training.retrain._train_and_log_from_arrays') as mock_train, \
             patch('src.training.retrain.register_best_model') as mock_register:
            
            mock_train.return_value = {}
            mock_register.return_value = 1
            
            result = train_combined(
                temp_csv_files['features'],
                temp_csv_files['production']
            )
            
            assert result == 1, "Should return version number when retraining succeeds"
            assert mock_train.called, "Training should be called"
