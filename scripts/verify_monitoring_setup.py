#!/usr/bin/env python3
"""
Verify that monitoring and reporting setup is complete and functional.

Usage:
    python scripts/verify_monitoring_setup.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Check that all required packages can be imported."""
    print(" Checking imports...")
    
    required_imports = {
        'evidently': 'Evidently AI',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'mlflow': 'MLflow',
    }
    
    missing = []
    for module, name in required_imports.items():
        try:
            __import__(module)
            print(f"   {name}")
        except ImportError:
            print(f"   {name} - NOT INSTALLED")
            missing.append(module)
    
    return len(missing) == 0, missing


def check_files():
    """Check that all required files exist."""
    print("\n Checking files...")
    
    required_files = {
        'src/monitoring/drift.py': 'Drift detection module',
        'src/monitoring/reports.py': 'Reports module',
        'dags/drift_retrain_dag.py': 'Drift retrain DAG',
        'scripts/generate_reports.py': 'Standalone report script',
        'tests/test_monitoring_reports.py': 'Test suite',
        'docs/MONITORING.md': 'Documentation',
        'MONITORING_SETUP.md': 'Setup guide',
    }
    
    missing = []
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   {description}: {file_path}")
        else:
            print(f"   {description}: {file_path} - NOT FOUND")
            missing.append(file_path)
    
    return len(missing) == 0, missing


def check_module_imports():
    """Check that custom modules can be imported."""
    print("\n Checking custom modules...")
    
    modules_to_check = [
        ('src.monitoring.drift', 'detect_drift'),
        ('src.monitoring.reports', 'generate_drift_report'),
        ('src.monitoring.reports', 'generate_data_quality_report'),
        ('src.monitoring.reports', 'generate_summary_report'),
    ]
    
    errors = []
    for module_name, function_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name)
            print(f"   {module_name}.{function_name}")
        except Exception as e:
            print(f"   {module_name}.{function_name} - ERROR: {e}")
            errors.append((module_name, function_name, str(e)))
    
    return len(errors) == 0, errors


def check_dag_integration():
    """Check that DAG has the generate_reports task."""
    print("\n Checking DAG integration...")
    
    dag_path = project_root / 'dags' / 'drift_retrain_dag.py'
    
    try:
        with open(dag_path, 'r') as f:
            dag_content = f.read()
        
        checks = {
            'generate_monitoring_reports function': 'def generate_monitoring_reports',
            'generate_reports task': "task_id='generate_reports'",
            'reports import': 'from src.monitoring.reports import',
            'REPORTS_DIR config': 'REPORTS_DIR',
            'task dependency': 'detect_drift_task >> generate_reports',
        }
        
        missing = []
        for check_name, search_string in checks.items():
            if search_string in dag_content:
                print(f"   {check_name}")
            else:
                print(f"   {check_name} - NOT FOUND")
                missing.append(check_name)
        
        return len(missing) == 0, missing
    
    except Exception as e:
        print(f"   Error reading DAG file: {e}")
        return False, [str(e)]


def run_simple_test():
    """Run a simple test to verify functionality."""
    print("\n Running simple functionality test...")
    
    try:
        import pandas as pd
        import numpy as np
        import tempfile
        from src.monitoring.reports import generate_summary_report
        
        # Create dummy reports
        drift_report = {
            "status": "completed",
            "drift_detected": False,
            "drift_share": 0.0,
        }
        
        quality_report = {
            "status": "completed",
            "num_rows": 100,
            "num_columns": 5,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.json")
            summary = generate_summary_report(
                drift_report=drift_report,
                quality_report=quality_report,
                output_path=summary_path,
            )
            
            if os.path.exists(summary_path):
                print("   Summary report generated successfully")
                print(f"   Alerts count: {len(summary.get('alerts', []))}")
                return True, None
            else:
                print("   Summary report file not created")
                return False, "Report file not created"
    
    except Exception as e:
        print(f"   Test failed: {e}")
        return False, str(e)


def main():
    """Run all verification checks."""
    print("=" * 60)
    print(" Customer Churn Monitoring Setup Verification")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results['imports'] = check_imports()
    results['files'] = check_files()
    results['module_imports'] = check_module_imports()
    results['dag_integration'] = check_dag_integration()
    results['functionality'] = run_simple_test()
    
    # Summary
    print("\n" + "=" * 60)
    print(" VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, (passed, details) in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"{status} - {check_name.replace('_', ' ').title()}")
        if not passed and details:
            print(f"     Issues: {details}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    
    if all_passed:
        print(" ALL CHECKS PASSED!")
        print("\n Your monitoring system is ready to use!")
        print("\nNext steps:")
        print("  1. Run manual report: python scripts/generate_reports.py --help")
        print("  2. Test DAG: airflow dags test customer_churn_drift_retrain")
        print("  3. Read docs: cat docs/MONITORING.md")
        return 0
    else:
        print(" SOME CHECKS FAILED")
        print("\nPlease review the errors above and:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check file paths and permissions")
        print("  3. Review setup guide: cat MONITORING_SETUP.md")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
