#!/usr/bin/env python
"""Test Evidently 0.7.x API to find correct save methods"""

import pandas as pd
from evidently import Report
from evidently.metrics import DatasetMissingValueCount

# Create dummy data
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1]
})

# Create report
report = Report(metrics=[DatasetMissingValueCount()])
report.run(reference_data=df, current_data=df)

# Check available methods
print("Evidently Report methods containing 'save':")
save_methods = [m for m in dir(report) if 'save' in m.lower() and not m.startswith('_')]
print(save_methods)

print("\nAll public methods:")
all_methods = [m for m in dir(report) if not m.startswith('_')]
print(all_methods[:30])

# Try to save
try:
    if hasattr(report, 'save_html'):
        print("\n save_html() method exists")
        report.save_html('/tmp/test_report.html')
        print("Saved HTML successfully")
    else:
        print("\n save_html() method NOT found")
except Exception as e:
    print(f" Error saving: {e}")

# Try as_dict
try:
    if hasattr(report, 'as_dict'):
        print("\n as_dict() method exists")
    else:
        print("\n as_dict() method NOT found")
except Exception as e:
    print(f" Error: {e}")

# Try json
try:
    if hasattr(report, 'json'):
        print("\n json() method exists")
    else:
        print("\n json() method NOT found")
except Exception as e:
    print(f" Error: {e}")
