# Model Monitoring & Reporting

This document describes the automated monitoring and reporting system for the Customer Churn prediction model.

## Overview

The monitoring system uses [Evidently](https://www.evidentlyai.com/) to generate comprehensive reports on:
- **Data Drift**: Detect distribution changes between baseline and production data
- **Data Quality**: Monitor missing values, data types, and data integrity
- **Target Drift**: Track changes in the target variable distribution
- **Model Performance**: Compare model metrics over time (when predictions are available)

## Architecture

```
┌─────────────────┐
│  Airflow DAG    │
│ drift_retrain   │
└────────┬────────┘
         │
         ├─ build_features
         │
         ├─ detect_drift (PSI-based)
         │
         ├─ generate_reports (Evidently)
         │     ├─ Drift Report (HTML + JSON)
         │     ├─ Data Quality Report (HTML + JSON)
         │     └─ Summary Report (JSON)
         │
         ├─ branch_on_drift
         │
         └─ retrain_combined / retrain_features
```

## Reports Generated

### 1. Drift Report
**Location**: `/opt/airflow/data/monitoring/reports/drift_report_YYYYMMDD_HHMMSS.html`

Monitors distribution changes between baseline (training) and production data:
- Per-feature drift detection
- Statistical tests (KS test, Chi-square)
- Visual distribution comparisons
- Drift share (% of features with drift)

**Metrics**:
- Dataset-level drift (boolean)
- Drift share (0-1)
- Per-column drift scores

### 2. Data Quality Report
**Location**: `/opt/airflow/data/monitoring/reports/production_data_quality_YYYYMMDD_HHMMSS.html`

Analyzes data quality issues in production data:
- Missing values per column
- Data type consistency
- Value ranges and outliers
- Correlation changes
- Duplicate rows

### 3. Summary Report
**Location**: `/opt/airflow/data/monitoring/summary_report.json`

JSON file combining key metrics and alerts:
```json
{
  "timestamp": "2025-11-05T16:00:00",
  "drift_analysis": {
    "status": "completed",
    "drift_detected": true,
    "drift_share": 0.23,
    "html_report": "/path/to/drift_report.html"
  },
  "data_quality": {
    "status": "completed",
    "num_rows": 1000,
    "num_columns": 15
  },
  "alerts": [
    {
      "severity": "warning",
      "type": "drift",
      "message": "Data drift detected in 23.0% of features"
    }
  ]
}
```

## Integration with DAG

The monitoring system is integrated into `dags/drift_retrain_dag.py`:

### Workflow
1. **build_features**: Generate feature engineering
2. **detect_drift**: Run PSI-based drift detection (existing)
3. **generate_reports**: Generate Evidently reports *(NEW)*
4. **branch_on_drift**: Decide retraining strategy
5. **retrain_combined/retrain_features**: Train models
6. **done**: Complete

### Configuration

Environment variables (set in `.env` or Airflow Variables):

```bash
# Data paths
FEATURES_PATH=/opt/airflow/data/features/features.csv
PRODUCTION_DATA_PATH=/opt/airflow/data/production/production.csv

# Monitoring
REPORTS_DIR=/opt/airflow/data/monitoring/reports
DRIFT_REPORT_PATH=/opt/airflow/data/monitoring/drift_report.json
PSI_THRESHOLD=0.2

# MLflow
MLFLOW_URI=http://mlflow:5000
```

## Manual Report Generation

Use the standalone script for on-demand reporting:

### Full Drift + Quality Analysis
```bash
python scripts/generate_reports.py \
  --baseline data/features/features.csv \
  --production data/production/production.csv \
  --output-dir data/monitoring/reports
```

### Quality-Only Mode
```bash
python scripts/generate_reports.py \
  --quality-only \
  --production data/production/production.csv \
  --output-dir data/monitoring/reports
```

### Options
- `--baseline`: Path to baseline/reference dataset
- `--production`: Path to production/current dataset
- `--output-dir`: Directory to save reports (default: `data/monitoring/reports`)
- `--target-column`: Name of target column (default: `Churn`)
- `--quality-only`: Generate only data quality report

## Viewing Reports

### HTML Reports
Open the generated HTML files in a browser:
```bash
open data/monitoring/reports/drift_report_20251105_160000.html
```

The HTML reports are interactive and include:
- Interactive plots and charts
- Expandable sections
- Drill-down capabilities
- Export options

### JSON Reports
Parse programmatically for alerting/dashboards:
```python
import json

with open('data/monitoring/reports/drift_report_20251105_160000.json') as f:
    report = json.load(f)
    
for metric in report['metrics']:
    if metric['metric'] == 'DatasetDriftMetric':
        print(f"Drift detected: {metric['result']['dataset_drift']}")
        print(f"Drift share: {metric['result']['drift_share']}")
```

## Alert Configuration

Alerts are automatically generated in the summary report based on:

| Condition | Severity | Type |
|-----------|----------|------|
| Drift detected | warning | drift |
| Production data missing | info | data_availability |
| Quality issues detected | warning | data_quality |

### Integrating with External Systems

To send alerts to Slack, Email, or PagerDuty:

```python
# In your DAG
def send_alerts(**context):
    summary = context['ti'].xcom_pull(task_ids='generate_reports', key='summary_report')
    
    for alert in summary.get('alerts', []):
        if alert['severity'] == 'warning':
            # Send to Slack
            slack_webhook(alert['message'])
            
            # Or send email
            send_email(
                to='ml-ops@company.com',
                subject=f"Model Monitoring Alert: {alert['type']}",
                body=alert['message']
            )
```

## Scheduling

The DAG runs hourly by default:
```python
schedule_interval=timedelta(hours=1)
```

To change the schedule, modify in `drift_retrain_dag.py`:
```python
schedule_interval='0 */6 * * *'  # Every 6 hours
# or
schedule_interval='@daily'  # Daily at midnight
```

## Best Practices

### 1. Report Retention
Set up cleanup for old reports:
```bash
# Add to DAG or cron
find /opt/airflow/data/monitoring/reports -type f -mtime +30 -delete
```

### 2. Baseline Updates
Periodically update the baseline dataset after retraining:
```python
# After successful retraining
shutil.copy(
    '/opt/airflow/data/features/features.csv',
    '/opt/airflow/data/monitoring/baseline.csv'
)
```

### 3. Dashboard Integration
- Serve HTML reports via web server (nginx/Apache)
- Parse JSON reports into monitoring dashboards (Grafana/Kibana)
- Store metrics in time-series DB (Prometheus/InfluxDB)

### 4. Thresholds
Tune drift thresholds based on your model sensitivity:
```python
# In drift.py
PSI_THRESHOLD = 0.2  # Lower = more sensitive
```

## Troubleshooting

### Reports Not Generated
**Symptom**: Reports show "status": "skipped"

**Solutions**:
1. Check production data exists:
   ```bash
   ls -lh /opt/airflow/data/production/production.csv
   ```

2. Verify data format:
   ```python
   import pandas as pd
   df = pd.read_csv('production.csv')
   print(df.head())
   print(df.info())
   ```

3. Check logs:
   ```bash
   airflow tasks logs drift_retrain generate_reports <execution_date>
   ```

### Memory Issues with Large Datasets
For datasets > 1M rows:
```python
# Sample data before reporting
production_sample = production_df.sample(n=100000, random_state=42)
```

### Evidently Import Errors
```bash
pip install --upgrade evidently
# or in requirements.txt
evidently>=0.4.0
```

## API Reference

### `generate_drift_report()`
```python
def generate_drift_report(
    baseline_path: str,
    production_path: str,
    output_dir: str = "/opt/airflow/data/monitoring/reports",
    target_column: str = "Churn",
) -> Dict[str, Any]
```

### `generate_data_quality_report()`
```python
def generate_data_quality_report(
    data_path: str,
    output_dir: str = "/opt/airflow/data/monitoring/reports",
    report_name: str = "data_quality",
) -> Dict[str, Any]
```

### `generate_summary_report()`
```python
def generate_summary_report(
    drift_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    output_path: str = "/opt/airflow/data/monitoring/summary_report.json",
) -> Dict[str, Any]
```

## Further Reading

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [ML Monitoring Best Practices](https://www.evidentlyai.com/blog/ml-monitoring-best-practices)
- [Data Drift Detection Methods](https://docs.evidentlyai.com/reference/data-drift-algorithm)
