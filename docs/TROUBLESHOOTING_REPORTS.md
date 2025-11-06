# Troubleshooting: Monitoring Reports

## Common Issues and Solutions

### Issue: "No columns to parse from file" - Production data is empty

**Symptom**: The `generate_reports` task fails with:
```
pandas.errors.EmptyDataError: No columns to parse from file
```

**Root Cause**: The production data file exists but is empty or has no columns.

**Solution**: 

#### Immediate Fix (Already Applied)
The code now handles this gracefully:
- Reports will show `status: "skipped"` with reason `"data file has no columns or is malformed"`
- DAG will continue to run (no longer fails)
- Retraining will proceed based on drift detection results

#### Long-term Solutions

1. **Verify production data is being generated**:
   ```bash
   # Check if file exists and has content
   ls -lh /opt/airflow/data/production/production.csv
   
   # View first few lines
   head /opt/airflow/data/production/production.csv
   
   # Count rows
   wc -l /opt/airflow/data/production/production.csv
   ```

2. **Check your data pipeline**:
   - Ensure your inference/prediction system is writing to production.csv
   - Verify the file permissions
   - Check for any errors in upstream data collection

3. **Initialize with sample data** (for testing):
   ```bash
   # Copy features as production data for testing
   cp /opt/airflow/data/features/features.csv \
      /opt/airflow/data/production/production.csv
   ```

4. **Update your data ingestion**:
   - Make sure your API or data collection service writes to this file
   - Consider using append mode to accumulate production data
   - Set up monitoring to alert when file is empty

---

### Issue: Reports show "status: skipped"

**Expected Behavior**: Reports gracefully skip when data is unavailable rather than failing the DAG.

**Common Reasons**:

| Reason | Solution |
|--------|----------|
| `production data missing` | Create the production CSV file |
| `production data file is empty` | Add data to the file |
| `data file has no columns` | Ensure CSV has proper headers |
| `dataframe is empty` | Add at least one row of data |
| `Evidently not properly installed` | Run `pip install evidently` |

**Verify Reports Are Working**:
```bash
# Test manually with sample data
python3 scripts/generate_reports.py \
  --baseline /opt/airflow/data/features/features.csv \
  --production /opt/airflow/data/features/features.csv \
  --output-dir /tmp/test_reports

# Check if reports were created
ls -lh /tmp/test_reports/
```

---

### Issue: "failed to read baseline" error

**Symptom**: Reports show `status: "error"` with reason about baseline file.

**Solution**:
1. Check baseline file exists:
   ```bash
   ls -lh /opt/airflow/data/features/features.csv
   ```

2. Verify file is not corrupted:
   ```bash
   head -20 /opt/airflow/data/features/features.csv
   ```

3. Re-run preprocessing if needed:
   ```bash
   python -m src.etl.preprocessing
   ```

---

### Issue: Drift reports show no drift when expected

**Possible Causes**:
1. **Same data in both files**: Production = Baseline
   - This is normal if no production data exists yet
   
2. **Insufficient data**: Need enough samples for statistical significance
   - Minimum recommended: 100+ rows
   
3. **Threshold too high**: PSI_THRESHOLD = 0.2 by default
   - Adjust in environment variables or DAG config

**Debug**:
```python
import pandas as pd

baseline = pd.read_csv('/opt/airflow/data/features/features.csv')
production = pd.read_csv('/opt/airflow/data/production/production.csv')

print(f"Baseline shape: {baseline.shape}")
print(f"Production shape: {production.shape}")
print(f"Columns match: {set(baseline.columns) == set(production.columns)}")
print(f"Are they identical: {baseline.equals(production)}")
```

---

### Issue: HTML reports not displaying properly

**Solution**:
1. Check file was created:
   ```bash
   ls -lh /opt/airflow/data/monitoring/reports/*.html
   ```

2. View in browser:
   ```bash
   # Mac
   open /opt/airflow/data/monitoring/reports/drift_report_*.html
   
   # Linux with X server
   xdg-open /opt/airflow/data/monitoring/reports/drift_report_*.html
   ```

3. If file is 0 bytes, check Evidently logs in task output

---

### Issue: Task takes too long / times out

**Cause**: Large datasets (>1M rows) can be slow.

**Solutions**:

1. **Sample data before reporting**:
   Edit `src/monitoring/reports.py`:
   ```python
   # Add after loading production_df
   if len(production_df) > 100000:
       production_df = production_df.sample(n=100000, random_state=42)
       print(f"Sampled production data to 100k rows")
   ```

2. **Increase task timeout** in DAG:
   ```python
   generate_reports = PythonOperator(
       task_id='generate_reports',
       python_callable=generate_monitoring_reports,
       provide_context=True,
       execution_timeout=timedelta(minutes=10),  # Add this
   )
   ```

---

### Issue: XCom size exceeded

**Symptom**: Warning about XCom value being too large.

**Solution**: Reports are already saved to files. You can disable XCom push:

Edit `dags/drift_retrain_dag.py`:
```python
# Comment out XCom pushes if not needed
# context['ti'].xcom_push(key='drift_report', value=drift_report)
# context['ti'].xcom_push(key='quality_report', value=quality_report)
# context['ti'].xcom_push(key='summary_report', value=summary)
```

Read from summary JSON instead:
```python
import json
with open('/opt/airflow/data/monitoring/summary_report.json') as f:
    summary = json.load(f)
```

---

## Getting Help

### Check Logs
```bash
# View task logs in Airflow UI
# Or via CLI:
airflow tasks logs customer_churn_drift_retrain generate_reports <execution_date>
```

### Run Verification
```bash
cd /Users/mahamatabakarassouna/projects/Customer_Churn
python3 scripts/verify_monitoring_setup.py
```

### Test Manually
```bash
# Test with existing data
python3 scripts/generate_reports.py \
  --baseline data/features/features.csv \
  --production data/features/features.csv \
  --output-dir /tmp/test
```

### Enable Debug Logging
Add to DAG file:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Prevention

### Set Up Data Validation

Create a pre-task to validate data:
```python
def validate_data(**context):
    import os
    import pandas as pd
    
    prod_path = os.getenv("PRODUCTION_DATA_PATH")
    
    if not os.path.exists(prod_path):
        print("WARNING: Production data file does not exist")
        return False
    
    if os.path.getsize(prod_path) == 0:
        print("WARNING: Production data file is empty")
        return False
    
    try:
        df = pd.read_csv(prod_path)
        if df.empty:
            print("WARNING: Production dataframe is empty")
            return False
        print(f"✓ Production data OK: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"ERROR: Cannot read production data: {e}")
        return False
```

### Monitor File Creation

Add an Airflow sensor before `generate_reports`:
```python
from airflow.sensors.filesystem import FileSensor

wait_for_production = FileSensor(
    task_id='wait_for_production',
    filepath='/opt/airflow/data/production/production.csv',
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,  # Wait up to 1 hour
    mode='poke',
)
```

---

## Current Behavior (After Fix)

✅ **Graceful Handling**: Task no longer fails when production data is empty  
✅ **Clear Logging**: Detailed status messages explain what happened  
✅ **DAG Continues**: Retraining proceeds based on drift detection  
✅ **Reports Available**: Summary JSON created even when detailed reports skip  
✅ **Alerts Generated**: Summary includes appropriate alerts for missing data  

The monitoring system is now **production-hardened** and handles edge cases gracefully!
