# Monitoring & Reporting Setup - Quick Start

## 🎯 What's Been Added

Comprehensive automated monitoring and reporting for your Customer Churn model using **Evidently AI**.

### New Components

1. **`src/monitoring/reports.py`** - Evidently-based reporting module
   - Drift detection reports (HTML + JSON)
   - Data quality reports
   - Summary reports with alerts

2. **`scripts/generate_reports.py`** - Standalone report generator
   - Run reports on-demand
   - Quality-only mode
   - Customizable outputs

3. **Updated `dags/drift_retrain_dag.py`**
   - New `generate_reports` task
   - Integrated into workflow after drift detection
   - Generates reports before retraining decision

4. **`tests/test_monitoring_reports.py`** - Comprehensive test suite

5. **`docs/MONITORING.md`** - Complete documentation

## 🚀 Quick Start

### 1. Install Dependencies (Already Done!)

Evidently is already in your `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run Manual Report Generation

Test the reporting system immediately:

```bash
# If you have sample data
python scripts/generate_reports.py \
  --baseline data/features/features.csv \
  --production data/production/production.csv \
  --output-dir data/monitoring/reports
```

### 3. View Reports

Open the generated HTML reports in your browser:
```bash
# Find the latest report
ls -lt data/monitoring/reports/

# Open it
open data/monitoring/reports/drift_report_*.html
```

### 4. Automated Monitoring via Airflow

The DAG now automatically:
1. Builds features
2. Detects drift (PSI-based)
3. **Generates Evidently reports** *(NEW)*
4. Decides retraining strategy
5. Retrains models

#### DAG Workflow
```
build_features → detect_drift → generate_reports → branch_on_drift
                                                   ├─→ retrain_combined → done
                                                   └─→ retrain_features → done
```

## 📊 Reports Generated

### 1. Drift Report
**File**: `drift_report_YYYYMMDD_HHMMSS.html`

Interactive HTML showing:
- Dataset-level drift (Yes/No)
- Per-feature drift scores
- Distribution comparisons (histograms, KDE plots)
- Statistical test results
- Drift share (% of features with drift)

**Example Metrics**:
- Drift detected: `True`
- Drift share: `0.23` (23% of features)
- Features with drift: tenure, MonthlyCharges, Contract

### 2. Data Quality Report
**File**: `production_data_quality_YYYYMMDD_HHMMSS.html`

Monitors:
- Missing values per column
- Data type consistency
- Outliers and anomalies
- Value ranges
- Duplicate rows
- Correlation changes

### 3. Summary Report
**File**: `summary_report.json`

JSON with key metrics and alerts:
```json
{
  "timestamp": "2025-11-05T16:00:00",
  "drift_analysis": {
    "drift_detected": true,
    "drift_share": 0.23,
    "html_report": "/path/to/report.html"
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

## 🔧 Configuration

### Environment Variables

Set in `.env` or Airflow Variables:

```bash
# Required paths
FEATURES_PATH=/opt/airflow/data/features/features.csv
PRODUCTION_DATA_PATH=/opt/airflow/data/production/production.csv

# Monitoring config
REPORTS_DIR=/opt/airflow/data/monitoring/reports
PSI_THRESHOLD=0.2

# MLflow
MLFLOW_URI=http://mlflow:5000
```

### Customize Report Schedule

Edit `dags/drift_retrain_dag.py`:

```python
# Current: hourly
schedule_interval=timedelta(hours=1)

# Change to daily
schedule_interval='@daily'

# Or every 6 hours
schedule_interval='0 */6 * * *'
```

## 🧪 Testing

Run the test suite:

```bash
# Run all monitoring tests
pytest tests/test_monitoring_reports.py -v

# Run specific test
pytest tests/test_monitoring_reports.py::test_generate_drift_report_success -v
```

## 📁 File Structure

```
Customer_Churn/
├── src/
│   └── monitoring/
│       ├── drift.py                    # Existing PSI-based drift
│       └── reports.py                  # NEW: Evidently reports
├── scripts/
│   └── generate_reports.py             # NEW: Standalone script
├── dags/
│   └── drift_retrain_dag.py            # UPDATED: Added generate_reports task
├── tests/
│   └── test_monitoring_reports.py      # NEW: Test suite
├── docs/
│   └── MONITORING.md                   # NEW: Full documentation
└── data/
    └── monitoring/
        ├── reports/                    # NEW: HTML/JSON reports
        │   ├── drift_report_*.html
        │   ├── drift_report_*.json
        │   └── production_data_quality_*.html
        └── summary_report.json         # NEW: Summary with alerts
```

## 🎓 Next Steps

### 1. Set Up Alerting

Integrate with your notification system:

```python
# Example: Slack webhook
def send_slack_alert(summary):
    for alert in summary.get('alerts', []):
        if alert['severity'] == 'warning':
            requests.post(SLACK_WEBHOOK_URL, json={
                'text': f"⚠️ {alert['message']}"
            })
```

### 2. Dashboard Integration

Options:
- Serve HTML reports via nginx/Apache
- Parse JSON into Grafana/Kibana
- Store metrics in Prometheus/InfluxDB

### 3. Baseline Updates

After successful retraining:

```python
# Update baseline dataset
shutil.copy(
    'data/features/features.csv',
    'data/monitoring/baseline.csv'
)
```

### 4. Report Cleanup

Add to DAG or cron:

```bash
# Delete reports older than 30 days
find data/monitoring/reports -type f -mtime +30 -delete
```

## 🐛 Troubleshooting

### Reports Show "Status: Skipped"

**Check**:
1. Production data exists and is not empty
2. File paths are correct
3. Data has required columns

```bash
# Verify production data
ls -lh data/production/production.csv
head data/production/production.csv
```

### Import Errors

```bash
# Reinstall evidently
pip install --upgrade evidently

# Or specific version
pip install evidently==0.4.11
```

### Memory Issues (Large Datasets)

Sample data before reporting:

```python
# In reports.py, before generating report
if len(production_df) > 100000:
    production_df = production_df.sample(n=100000, random_state=42)
```

## 📚 Resources

- **Full Documentation**: See `docs/MONITORING.md`
- **Evidently Docs**: https://docs.evidentlyai.com/
- **Test Suite**: `tests/test_monitoring_reports.py`
- **Standalone Script**: `scripts/generate_reports.py --help`

## ✅ Verification Checklist

- [x] Evidently installed (`evidently` in requirements.txt)
- [x] Reports module created (`src/monitoring/reports.py`)
- [x] DAG updated with `generate_reports` task
- [x] Standalone script available (`scripts/generate_reports.py`)
- [x] Tests created (`tests/test_monitoring_reports.py`)
- [x] Documentation written (`docs/MONITORING.md`)

## 🎉 Ready to Use!

Your Customer Churn model now has:
- ✅ Automated drift detection (PSI + Evidently)
- ✅ Data quality monitoring
- ✅ Interactive HTML reports
- ✅ JSON reports for automation
- ✅ Alert generation
- ✅ On-demand report generation

**Try it now**:
```bash
python scripts/generate_reports.py --help
```
