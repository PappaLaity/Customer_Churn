"""
A/B Testing Statistical Analysis Module

This module provides statistical analysis capabilities for A/B test results.
It complements the existing traffic splitting and exposure logging in src/experiments/ab.py.

Author: Customer Churn ML Team
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ABTestResult:
    """Results from A/B test statistical analysis."""
    
    variant_a_count: int
    variant_b_count: int
    variant_a_metric: float
    variant_b_metric: float
    lift_percent: float
    p_value: float
    is_significant: bool
    confidence_level: float = 0.95
    min_detectable_effect: float = 0.05  # 5%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'variant_a': {
                'sample_size': self.variant_a_count,
                'metric_value': round(self.variant_a_metric, 4),
            },
            'variant_b': {
                'sample_size': self.variant_b_count,
                'metric_value': round(self.variant_b_metric, 4),
            },
            'analysis': {
                'lift_percent': round(self.lift_percent, 2),
                'p_value': round(self.p_value, 4),
                'is_significant': self.is_significant,
                'confidence_level': self.confidence_level,
            },
            'recommendation': self._get_recommendation(),
        }
    
    def _get_recommendation(self) -> str:
        """Generate recommendation based on results."""
        if self.variant_b_count < 100 or self.variant_a_count < 100:
            return "CONTINUE: Need more samples (minimum 100 per variant)"
        
        if not self.is_significant:
            return "CONTINUE: No statistical significance detected yet"
        
        if self.lift_percent >= self.min_detectable_effect * 100:
            return f"PROMOTE: Variant B shows {self.lift_percent:.1f}% improvement (significant at p={self.p_value:.3f})"
        elif self.lift_percent <= -self.min_detectable_effect * 100:
            return f"ROLLBACK: Variant B performs {abs(self.lift_percent):.1f}% worse (significant at p={self.p_value:.3f})"
        else:
            return f"CONTINUE: Lift of {self.lift_percent:.1f}% below minimum detectable effect ({self.min_detectable_effect*100}%)"


def analyze_latency(
    exposures_path: str,
    confidence_level: float = 0.95,
) -> ABTestResult:
    """
    Analyze latency difference between variants A and B.
    
    Uses two-sample t-test for continuous metric (latency).
    
    Args:
        exposures_path: Path to exposure CSV file
        confidence_level: Confidence level for significance testing (default: 0.95)
        
    Returns:
        ABTestResult with statistical analysis
        
    Raises:
        FileNotFoundError: If exposure file doesn't exist
        ValueError: If insufficient data
    """
    df = pd.read_csv(exposures_path)
    
    # Split by bucket
    latency_a = df[df['bucket'] == 'A']['latency_sec'].dropna()
    latency_b = df[df['bucket'] == 'B']['latency_sec'].dropna()
    
    if len(latency_a) < 1 or len(latency_b) < 1:
        raise ValueError(f"Insufficient samples: A={len(latency_a)}, B={len(latency_b)}")
    
    # Calculate means
    mean_a = latency_a.mean()
    mean_b = latency_b.mean()
    
    # Two-sample t-test (assumes unequal variances - Welch's t-test)
    t_stat, p_value = stats.ttest_ind(latency_a, latency_b, equal_var=False)
    
    # Calculate lift (negative lift is better for latency)
    lift_percent = ((mean_b - mean_a) / mean_a) * 100
    
    # Determine significance
    alpha = 1 - confidence_level
    is_significant = p_value < alpha
    
    return ABTestResult(
        variant_a_count=int(len(latency_a)),
        variant_b_count=int(len(latency_b)),
        variant_a_metric=float(mean_a),
        variant_b_metric=float(mean_b),
        lift_percent=float(lift_percent),
        p_value=float(p_value),
        is_significant=bool(is_significant),
        confidence_level=float(confidence_level),
    )



def analyze_conversion_rate(
    exposures_path: str,
    ground_truth_path: Optional[str] = None,
    confidence_level: float = 0.95,
) -> ABTestResult:
    """
    Analyze conversion rate or accuracy difference between variants.
    
    Uses proportion z-test for binary outcomes.
    
    Args:
        exposures_path: Path to exposure CSV file
        ground_truth_path: Optional path to CSV with actual outcomes/labels
        confidence_level: Confidence level for significance testing
        
    Returns:
        ABTestResult with statistical analysis
        
    Note:
        If ground_truth_path is not provided, this function cannot calculate
        actual accuracy. You need to join exposure logs with actual churn outcomes.
    """
    # Load exposures
    df = pd.read_csv(exposures_path)
    
    if ground_truth_path:
        # Join with ground truth labels
        truth_df = pd.read_csv(ground_truth_path)
        df = df.merge(truth_df, left_on='subject_id', right_on='customer_id', how='inner')
        
        # Calculate prediction correctness per variant
        df['correct'] = (df['prediction'] == df['actual_churn']).astype(int)
        
        success_a = df[df['bucket'] == 'A']['correct'].sum()
        success_b = df[df['bucket'] == 'B']['correct'].sum()
        count_a = len(df[df['bucket'] == 'A'])
        count_b = len(df[df['bucket'] == 'B'])
    else:
        # Placeholder: without ground truth, cannot calculate accuracy
        raise ValueError(
            "Ground truth data required for conversion rate analysis. "
            "Provide ground_truth_path or use analyze_latency() instead."
        )
    
    if count_a < 2 or count_b < 2:
        raise ValueError(f"Insufficient samples: A={count_a}, B={count_b}")
    
    # Calculate conversion rates
    rate_a = success_a / count_a
    rate_b = success_b / count_b
    
    # Proportion z-test
    count = np.array([success_b, success_a])
    nobs = np.array([count_b, count_a])
    
    # Use scipy's binomial test as approximation
    # For proper z-test, use statsmodels.stats.proportion.proportions_ztest
    # Here we use a simplified approach
    pooled_rate = (success_a + success_b) / (count_a + count_b)
    se_pooled = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/count_a + 1/count_b))
    
    if se_pooled > 0:
        z_stat = (rate_b - rate_a) / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
    else:
        z_stat = 0
        p_value = 1.0
    
    # Calculate lift
    lift_percent = ((rate_b - rate_a) / rate_a) * 100 if rate_a > 0 else 0
    
    # Determine significance
    alpha = 1 - confidence_level
    is_significant = p_value < alpha
    
    return ABTestResult(
        variant_a_count=int(count_a),
        variant_b_count=int(count_b),
        variant_a_metric=float(rate_a),
        variant_b_metric=float(rate_b),
        lift_percent=float(lift_percent),
        p_value=float(p_value),
        is_significant=bool(is_significant),
        confidence_level=float(confidence_level),
    )



def calculate_sample_size(
    baseline_rate: float,
    min_detectable_effect: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Calculate required sample size per variant.
    
    Args:
        baseline_rate: Expected baseline conversion rate (0-1)
        min_detectable_effect: Minimum relative effect to detect (default: 5%)
        alpha: Significance level (default: 0.05 for 95% confidence)
        power: Statistical power (default: 0.8)
        
    Returns:
        Required sample size per variant
        
    Example:
        >>> calculate_sample_size(baseline_rate=0.85, min_detectable_effect=0.05)
        3841  # Need ~3841 samples per variant
    """
    # Target rate for variant B
    target_rate = baseline_rate * (1 + min_detectable_effect)
    
    # Effect size (Cohen's h for proportions)
    effect_size = 2 * (np.arcsin(np.sqrt(target_rate)) - np.arcsin(np.sqrt(baseline_rate)))
    
    # Z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)
    
    # Sample size formula
    n = ((z_alpha + z_beta) / effect_size) ** 2
    
    return int(np.ceil(n))


def generate_report(
    exposures_path: str,
    output_path: Optional[str] = None,
    metric: str = 'latency',
) -> Dict[str, Any]:
    """
    Generate comprehensive A/B test analysis report.
    
    Args:
        exposures_path: Path to exposure CSV file
        output_path: Optional path to save JSON report
        metric: Metric to analyze ('latency' or 'conversion')
        
    Returns:
        Dictionary with analysis results
    """
    try:
        if metric == 'latency':
            result = analyze_latency(exposures_path)
        elif metric == 'conversion':
            result = analyze_conversion_rate(exposures_path)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        report = {
            'status': 'success',
            'metric_analyzed': metric,
            'results': result.to_dict(),
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
    except Exception as e:
        report = {
            'status': 'error',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat(),
        }
    
    # Save to file if specified
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


if __name__ == '__main__':
    """
    Example usage for testing.
    """
    import sys
    
    exposures_path = 'data/experiments/ab_exposures.csv'
    
    if not Path(exposures_path).exists():
        print(f"Exposure file not found: {exposures_path}")
        print("Run some A/B tests first to generate data.")
        sys.exit(1)
    
    print("=" * 60)
    print("A/B Test Analysis - Latency Comparison")
    print("=" * 60)
    
    try:
        report = generate_report(
            exposures_path=exposures_path,
            output_path='data/experiments/ab_analysis_report.json',
            metric='latency'
        )
        
        if report['status'] == 'success':
            results = report['results']
            print(f"\n Analysis completed successfully\n")
            print(f"Variant A (Production):")
            print(f"  - Sample size: {results['variant_a']['sample_size']}")
            print(f"  - Avg latency: {results['variant_a']['metric_value']:.4f}s")
            
            print(f"\nVariant B (Staging):")
            print(f"  - Sample size: {results['variant_b']['sample_size']}")
            print(f"  - Avg latency: {results['variant_b']['metric_value']:.4f}s")
            
            print(f"\nStatistical Analysis:")
            print(f"  - Lift: {results['analysis']['lift_percent']:.2f}% (negative is better for latency)")
            print(f"  - P-value: {results['analysis']['p_value']:.4f}")
            print(f"  - Significant: {results['analysis']['is_significant']}")
            
            print(f"\n Recommendation:")
            print(f"  {results['recommendation']}")
            
            print(f"\n Report saved to: data/experiments/ab_analysis_report.json")
        else:
            print(f"\n Analysis failed: {report['error']}")
            
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
