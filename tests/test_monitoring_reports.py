import pytest
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.monitoring.reports import (
    generate_drift_report,
    generate_data_quality_report,
    generate_summary_report,
)


@pytest.fixture
def sample_baseline_data():
    """Create sample baseline dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 100, 100),
        'TotalCharges': np.random.uniform(100, 5000, 100),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
        'Churn': np.random.choice([0, 1], 100),
    })


@pytest.fixture
def sample_production_data():
    """Create sample production dataset with slight drift."""
    np.random.seed(43)
    return pd.DataFrame({
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(25, 110, 100),  # Slight shift
        'TotalCharges': np.random.uniform(150, 5500, 100),  # Slight shift
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
        'Churn': np.random.choice([0, 1], 100),
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_generate_drift_report_success(sample_baseline_data, sample_production_data, temp_dir):
    """Test drift report generation with valid data."""
    baseline_path = os.path.join(temp_dir, "baseline.csv")
    production_path = os.path.join(temp_dir, "production.csv")
    
    sample_baseline_data.to_csv(baseline_path, index=False)
    sample_production_data.to_csv(production_path, index=False)
    
    report = generate_drift_report(
        baseline_path=baseline_path,
        production_path=production_path,
        output_dir=temp_dir,
        target_column="Churn",
    )
    
    assert report["status"] == "completed"
    assert "html_report" in report
    assert "json_report" in report
    assert "drift_detected" in report
    assert "drift_share" in report
    assert os.path.exists(report["html_report"])
    assert os.path.exists(report["json_report"])


def test_generate_drift_report_missing_production(sample_baseline_data, temp_dir):
    """Test drift report when production data is missing."""
    baseline_path = os.path.join(temp_dir, "baseline.csv")
    production_path = os.path.join(temp_dir, "nonexistent.csv")
    
    sample_baseline_data.to_csv(baseline_path, index=False)
    
    report = generate_drift_report(
        baseline_path=baseline_path,
        production_path=production_path,
        output_dir=temp_dir,
        target_column="Churn",
    )
    
    assert report["status"] == "skipped"
    assert "reason" in report
    assert "missing" in report["reason"].lower()


def test_generate_data_quality_report_success(sample_production_data, temp_dir):
    """Test data quality report generation."""
    data_path = os.path.join(temp_dir, "production.csv")
    sample_production_data.to_csv(data_path, index=False)
    
    report = generate_data_quality_report(
        data_path=data_path,
        output_dir=temp_dir,
        report_name="quality_test",
    )
    
    assert report["status"] == "completed"
    assert "html_report" in report
    assert "json_report" in report
    assert report["num_rows"] == 100
    assert report["num_columns"] == 5
    assert os.path.exists(report["html_report"])


def test_generate_data_quality_report_missing_file(temp_dir):
    """Test data quality report with missing file."""
    data_path = os.path.join(temp_dir, "nonexistent.csv")
    
    report = generate_data_quality_report(
        data_path=data_path,
        output_dir=temp_dir,
        report_name="quality_test",
    )
    
    assert report["status"] == "skipped"
    assert report["reason"] == "data file missing"


def test_generate_summary_report(temp_dir):
    """Test summary report generation."""
    drift_report = {
        "status": "completed",
        "drift_detected": True,
        "drift_share": 0.25,
        "html_report": "/path/to/drift.html",
    }
    
    quality_report = {
        "status": "completed",
        "num_rows": 100,
        "num_columns": 5,
    }
    
    summary_path = os.path.join(temp_dir, "summary.json")
    
    summary = generate_summary_report(
        drift_report=drift_report,
        quality_report=quality_report,
        output_path=summary_path,
    )
    
    assert "timestamp" in summary
    assert "drift_analysis" in summary
    assert "data_quality" in summary
    assert "alerts" in summary
    assert len(summary["alerts"]) > 0
    assert os.path.exists(summary_path)
    
    # Verify alert for drift
    drift_alert = [a for a in summary["alerts"] if a["type"] == "drift"]
    assert len(drift_alert) > 0
    assert drift_alert[0]["severity"] == "warning"


def test_generate_summary_report_with_skipped_drift(temp_dir):
    """Test summary report when drift analysis is skipped."""
    drift_report = {
        "status": "skipped",
        "reason": "production data missing",
    }
    
    quality_report = {
        "status": "completed",
        "num_rows": 100,
        "num_columns": 5,
    }
    
    summary_path = os.path.join(temp_dir, "summary.json")
    
    summary = generate_summary_report(
        drift_report=drift_report,
        quality_report=quality_report,
        output_path=summary_path,
    )
    
    # Should have info alert about skipped analysis
    info_alerts = [a for a in summary["alerts"] if a["severity"] == "info"]
    assert len(info_alerts) > 0
    assert "skipped" in info_alerts[0]["message"].lower()


def test_drift_report_html_content(sample_baseline_data, sample_production_data, temp_dir):
    """Test that HTML report contains expected content."""
    baseline_path = os.path.join(temp_dir, "baseline.csv")
    production_path = os.path.join(temp_dir, "production.csv")
    
    sample_baseline_data.to_csv(baseline_path, index=False)
    sample_production_data.to_csv(production_path, index=False)
    
    report = generate_drift_report(
        baseline_path=baseline_path,
        production_path=production_path,
        output_dir=temp_dir,
        target_column="Churn",
    )
    
    # Read HTML file
    with open(report["html_report"], "r") as f:
        html_content = f.read()
    
    # Check for Evidently markers
    assert len(html_content) > 0
    # HTML should be substantial (Evidently generates detailed reports)
    assert len(html_content) > 1000


def test_drift_report_json_structure(sample_baseline_data, sample_production_data, temp_dir):
    """Test that JSON report has expected structure."""
    baseline_path = os.path.join(temp_dir, "baseline.csv")
    production_path = os.path.join(temp_dir, "production.csv")
    
    sample_baseline_data.to_csv(baseline_path, index=False)
    sample_production_data.to_csv(production_path, index=False)
    
    report = generate_drift_report(
        baseline_path=baseline_path,
        production_path=production_path,
        output_dir=temp_dir,
        target_column="Churn",
    )
    
    # Read JSON file
    with open(report["json_report"], "r") as f:
        json_data = json.load(f)
    
    # Check structure
    assert "metrics" in json_data
    assert isinstance(json_data["metrics"], list)
    assert len(json_data["metrics"]) > 0
