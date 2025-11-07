#!/usr/bin/env python3
"""
Standalone script to generate monitoring reports for the customer churn model.

Usage:
    python scripts/generate_reports.py --baseline data/features/features.csv --production data/production/production.csv
    python scripts/generate_reports.py --quality-only data/production/production.csv
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.reports import (
    generate_drift_report,
    generate_data_quality_report,
    generate_summary_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate monitoring reports for customer churn model"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline/reference dataset",
    )
    
    parser.add_argument(
        "--production",
        type=str,
        help="Path to production/current dataset",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/monitoring/reports",
        help="Directory to save reports (default: data/monitoring/reports)",
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        default="Churn",
        help="Name of the target column (default: Churn)",
    )
    
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Generate only data quality report for production data",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if args.quality_only:
        if not args.production:
            print("Error: --production is required for quality-only mode")
            sys.exit(1)
        
        print(f"Generating data quality report for: {args.production}")
        quality_report = generate_data_quality_report(
            data_path=args.production,
            output_dir=output_dir,
            report_name="data_quality",
        )
        
        print(f"\nData Quality Report:")
        print(f"  Status: {quality_report.get('status')}")
        if quality_report.get('status') == 'completed':
            print(f"  HTML Report: {quality_report.get('html_report')}")
            print(f"  JSON Report: {quality_report.get('json_report')}")
            print(f"  Rows: {quality_report.get('num_rows')}")
            print(f"  Columns: {quality_report.get('num_columns')}")
        else:
            print(f"  Reason: {quality_report.get('reason')}")
        
        return
    
    # Full drift + quality analysis
    if not args.baseline or not args.production:
        print("Error: Both --baseline and --production are required")
        print("Use --quality-only flag to generate only quality report")
        sys.exit(1)
    
    print(f"Generating drift report...")
    print(f"  Baseline: {args.baseline}")
    print(f"  Production: {args.production}")
    
    drift_report = generate_drift_report(
        baseline_path=args.baseline,
        production_path=args.production,
        output_dir=output_dir,
        target_column=args.target_column,
    )
    
    print(f"\nDrift Report:")
    print(f"  Status: {drift_report.get('status')}")
    if drift_report.get('status') == 'completed':
        print(f"  HTML Report: {drift_report.get('html_report')}")
        print(f"  JSON Report: {drift_report.get('json_report')}")
        print(f"  Drift Detected: {drift_report.get('drift_detected')}")
        print(f"  Drift Share: {drift_report.get('drift_share', 0)*100:.1f}%")
        print(f"  Columns Analyzed: {drift_report.get('num_columns')}")
    else:
        print(f"  Reason: {drift_report.get('reason')}")
    
    print(f"\nGenerating data quality report...")
    quality_report = generate_data_quality_report(
        data_path=args.production,
        output_dir=output_dir,
        report_name="production_data_quality",
    )
    
    print(f"\nData Quality Report:")
    print(f"  Status: {quality_report.get('status')}")
    if quality_report.get('status') == 'completed':
        print(f"  HTML Report: {quality_report.get('html_report')}")
        print(f"  JSON Report: {quality_report.get('json_report')}")
    
    print(f"\nGenerating summary report...")
    summary = generate_summary_report(
        drift_report=drift_report,
        quality_report=quality_report,
        output_path=os.path.join(output_dir, "summary_report.json"),
    )
    
    print(f"\nSummary Report:")
    print(f"  Alerts: {len(summary.get('alerts', []))}")
    for alert in summary.get('alerts', []):
        print(f"    [{alert['severity']}] {alert['type']}: {alert['message']}")
    
    print(f"\n All reports generated successfully in: {output_dir}")


if __name__ == "__main__":
    main()
