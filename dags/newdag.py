from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

from sqlalchemy import create_engine
from src.api.core.logger import api_logger as logger

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@churn_db:5432/churn_db")
FEATURES_PATH = os.getenv("FEATURES_PATH", "/opt/airflow/data/features/features.csv")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/drifts/monitoring/reports")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

# Seuils standards
PSI_THRESHOLD = 0.2  # PSI > 0.2 = drift significatif
KS_THRESHOLD = 0.05  # p-value < 0.05 = distributions différentes
CHI2_THRESHOLD = 0.05
DRIFT_PERCENTAGE_THRESHOLD = 30  # Si >30% des features → retrain


# ═══════════════════════════════════════════════════════════════
# Fonctions de Calcul de Drift
# ═══════════════════════════════════════════════════════════════

def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI)
    - PSI < 0.1 : Pas de changement significatif
    - PSI 0.1-0.2 : Changement modéré
    - PSI > 0.2 : Changement significatif (DRIFT)
    """
    try:
        breakpoints = np.linspace(0, 100, buckets + 1)
        breakpoints = np.unique(np.percentile(expected, breakpoints))
        
        if len(breakpoints) < 2:
            return 0.0
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        expected_percents = (expected_counts + 0.0001) / len(expected)
        actual_percents = (actual_counts + 0.0001) / len(actual)
        
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return float(psi)
    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return 0.0


def calculate_ks_test(expected, actual):
    """
    Kolmogorov-Smirnov Test
    - Teste si deux échantillons viennent de la même distribution
    - p-value < 0.05 → distributions différentes (DRIFT)
    """
    try:
        stat, pvalue = ks_2samp(expected, actual)
        return {'statistic': float(stat), 'p_value': float(pvalue)}
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return {'statistic': 0.0, 'p_value': 1.0}


def calculate_jensen_shannon(expected, actual, bins=20):
    """
    Jensen-Shannon Divergence
    - Mesure la similarité entre deux distributions
    - 0 = identiques, 1 = complètement différentes
    - > 0.1 = DRIFT potentiel
    """
    try:
        hist_exp, edges = np.histogram(expected, bins=bins, density=True)
        hist_act, _ = np.histogram(actual, bins=edges, density=True)
        
        # Normaliser pour que la somme = 1
        hist_exp = hist_exp / np.sum(hist_exp)
        hist_act = hist_act / np.sum(hist_act)
        
        js_div = jensenshannon(hist_exp, hist_act, base=2)
        return float(js_div)
    except Exception as e:
        logger.warning(f"JS divergence failed: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════
# Chargement des Données
# ═══════════════════════════════════════════════════════════════

def load_reference_data():
    """Charge les données de référence (CSV preprocessed)"""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Reference data not found: {FEATURES_PATH}")
    
    df_ref = pd.read_csv(FEATURES_PATH)
    logger.info(f"✅ Loaded {len(df_ref)} reference samples from {FEATURES_PATH}")
    return df_ref


def load_production_data_from_db(days_back=7):
    """
    Charge les données de production depuis PostgreSQL
    Retourne un DataFrame avec les MÊMES colonnes que la référence
    """
    engine = create_engine(DATABASE_URL)
    
    # ⚠️ IMPORTANT: Les noms de colonnes doivent MATCHER avec features.csv
    query = f"""
        SELECT 
            tenure,
            monthly_charges as "MonthlyCharges",
            total_charges as "TotalCharges",
            internet_service_fiber_optic::int as "InternetService_Fiber_optic",
            contract_two_year::int as "Contract_Two_year",
            payment_method_electronic_check::int as "PaymentMethod_Electronic_check",
            no_internet_service::int as "No_internet_service",
            paperless_billing::int as "PaperlessBilling",
            prediction as "Churn",
            created_at
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '{days_back} days'
        ORDER BY created_at DESC
    """
    
    try:
        df_prod = pd.read_sql(query, engine)
        logger.info(f"✅ Loaded {len(df_prod)} production samples from last {days_back} days")
        
        if len(df_prod) == 0:
            logger.warning(f"⚠️  No production data in last {days_back} days")
            return None
        
        # Supprimer created_at car pas dans la référence
        if 'created_at' in df_prod.columns:
            df_prod = df_prod.drop(columns=['created_at'])
        
        return df_prod
    
    except Exception as e:
        logger.error(f"❌ Failed to load production data: {e}")
        raise


# ═══════════════════════════════════════════════════════════════
# Détection de Drift
# ═══════════════════════════════════════════════════════════════

def detect_drift(**context):
    """
    Détection de drift avec méthodes statistiques standards
    """
    logger.info("🔍 Starting drift detection...")
    
    # 1. Charger les données
    df_reference = load_reference_data()
    df_production = load_production_data_from_db(days_back=7)
    
    if df_production is None or len(df_production) < 10:
        context['ti'].xcom_push(key='is_drift', value=False)
        context['ti'].xcom_push(key='drift_percentage', value=0.0)
        context['ti'].xcom_push(key='needs_retraining', value=False)
        return {"status": "insufficient_data"}
    
    # 2. Trouver les features communes
    numerical_cols = df_reference.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    
    common_cols = [col for col in numerical_cols if col in df_production.columns]
    
    if len(common_cols) == 0:
        logger.error("❌ No common features found between reference and production")
        logger.error(f"Reference features: {df_reference.columns.tolist()}")
        logger.error(f"Production features: {df_production.columns.tolist()}")
        return {"status": "error", "reason": "no_common_features"}
    
    logger.info(f"📊 Analyzing {len(common_cols)} features: {common_cols}")
    
    # 3. Analyser chaque feature
    drift_results = []
    features_with_drift = []
    
    for col in common_cols:
        expected = df_reference[col].dropna().values
        actual = df_production[col].dropna().values
        
        if len(actual) < 5:
            logger.warning(f"  ⚠️  Skipping {col}: insufficient data")
            continue
        
        # Calculs statistiques
        psi = calculate_psi(expected, actual)
        ks_result = calculate_ks_test(expected, actual)
        js_div = calculate_jensen_shannon(expected, actual)
        
        # Statistiques descriptives
        ref_mean = float(np.mean(expected))
        prod_mean = float(np.mean(actual))
        ref_std = float(np.std(expected))
        prod_std = float(np.std(actual))
        mean_change_pct = ((prod_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0.0
        
        # Décision: drift si AU MOINS UNE condition
        has_drift = (
            psi > PSI_THRESHOLD or 
            ks_result['p_value'] < KS_THRESHOLD or 
            js_div > 0.1
        )
        
        if has_drift:
            features_with_drift.append(col)
        
        # Stocker les résultats
        drift_results.append({
            'feature': col,
            'psi': round(psi, 4),
            'ks_statistic': round(ks_result['statistic'], 4),
            'ks_p_value': round(ks_result['p_value'], 4),
            'js_divergence': round(js_div, 4),
            'ref_mean': round(ref_mean, 2),
            'prod_mean': round(prod_mean, 2),
            'ref_std': round(ref_std, 2),
            'prod_std': round(prod_std, 2),
            'mean_change_pct': round(mean_change_pct, 2),
            'has_drift': has_drift,
        })
        
        status = "🔴 DRIFT" if has_drift else "✅ OK"
        logger.info(f"  {col}: {status} | PSI: {psi:.4f} | KS p: {ks_result['p_value']:.4f}")
    
    # 4. Résumé
    drift_percentage = (len(features_with_drift) / len(common_cols) * 100) if common_cols else 0.0
    needs_retraining = drift_percentage > DRIFT_PERCENTAGE_THRESHOLD
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 DRIFT DETECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Features analyzed: {len(common_cols)}")
    logger.info(f"Features with drift: {len(features_with_drift)} ({drift_percentage:.1f}%)")
    logger.info(f"Drift features: {features_with_drift}")
    logger.info(f"Needs retraining: {'YES ✅' if needs_retraining else 'NO ❌'}")
    logger.info(f"{'='*60}")
    
    # 5. Push XCom
    context['ti'].xcom_push(key='is_drift', value=len(features_with_drift) > 0)
    context['ti'].xcom_push(key='drift_percentage', value=drift_percentage)
    context['ti'].xcom_push(key='needs_retraining', value=needs_retraining)
    context['ti'].xcom_push(key='features_with_drift', value=features_with_drift)
    context['ti'].xcom_push(key='drift_results', value=drift_results)
    
    return {
        "status": "success",
        "drift_detected": len(features_with_drift) > 0,
        "drift_percentage": drift_percentage,
        "features_with_drift": features_with_drift,
    }


# ═══════════════════════════════════════════════════════════════
# Génération du Rapport HTML
# ═══════════════════════════════════════════════════════════════

def generate_html_report(**context):
    """Génère un rapport HTML interactif simple et élégant"""
    
    drift_results = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_results')
    drift_percentage = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_percentage')
    features_with_drift = context['ti'].xcom_pull(task_ids='detect_drift', key='features_with_drift')
    
    if not drift_results:
        logger.warning("No drift results to generate report")
        return {"status": "skipped"}
    
    # Créer le répertoire
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Générer le HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drift Detection Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .summary-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .summary-card:hover {{
            transform: translateY(-5px);
        }}
        .summary-card h3 {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .drift-alert {{
            background: #ff6b6b;
            color: white;
        }}
        .no-drift {{
            background: #51cf66;
            color: white;
        }}
        .features-table {{
            padding: 40px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .drift-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-drift {{
            background: #ff6b6b;
            color: white;
        }}
        .badge-ok {{
            background: #51cf66;
            color: white;
        }}
        .metric {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Drift Detection Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card {'drift-alert' if drift_percentage > DRIFT_PERCENTAGE_THRESHOLD else 'no-drift'}">
                <h3>Overall Status</h3>
                <div class="value">{'🔴 DRIFT' if drift_percentage > DRIFT_PERCENTAGE_THRESHOLD else '✅ NO DRIFT'}</div>
            </div>
            <div class="summary-card">
                <h3>Drift Percentage</h3>
                <div class="value">{drift_percentage:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Features Analyzed</h3>
                <div class="value">{len(drift_results)}</div>
            </div>
            <div class="summary-card">
                <h3>Features with Drift</h3>
                <div class="value">{len(features_with_drift)}</div>
            </div>
        </div>
        
        <div class="features-table">
            <h2 style="margin-bottom: 20px; color: #333;">Feature-Level Drift Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Status</th>
                        <th>PSI</th>
                        <th>KS p-value</th>
                        <th>JS Divergence</th>
                        <th>Mean Change</th>
                        <th>Ref Mean</th>
                        <th>Prod Mean</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Ajouter les résultats
    for result in drift_results:
        status_badge = 'badge-drift' if result['has_drift'] else 'badge-ok'
        status_text = '🔴 DRIFT' if result['has_drift'] else '✅ OK'
        
        html_content += f"""
                    <tr>
                        <td><strong>{result['feature']}</strong></td>
                        <td><span class="drift-badge {status_badge}">{status_text}</span></td>
                        <td><span class="metric">{result['psi']}</span></td>
                        <td><span class="metric">{result['ks_p_value']}</span></td>
                        <td><span class="metric">{result['js_divergence']}</span></td>
                        <td><span class="metric">{result['mean_change_pct']:+.2f}%</span></td>
                        <td>{result['ref_mean']:.2f}</td>
                        <td>{result['prod_mean']:.2f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <h3>Interpretation Guide</h3>
            <p><strong>PSI (Population Stability Index)</strong>: &lt;0.1 OK | 0.1-0.2 Moderate | &gt;0.2 Drift</p>
            <p><strong>KS p-value</strong>: &lt;0.05 means distributions are significantly different (Drift)</p>
            <p><strong>JS Divergence</strong>: 0 = identical | &gt;0.1 = potential drift</p>
            <p style="margin-top: 20px; font-size: 0.9em;">Generated by Customer Churn MLOps Pipeline</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Sauvegarder
    report_path = os.path.join(REPORTS_DIR, "drift_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✅ HTML Report generated: {report_path}")
    
    return {
        "status": "success",
        "report_path": report_path,
    }


def choose_branch(**context):
    """Décide s'il faut retrain"""
    needs_retraining = context['ti'].xcom_pull(task_ids='detect_drift', key='needs_retraining')
    return 'retrain_combined' if needs_retraining else 'skip_retrain'


# ═══════════════════════════════════════════════════════════════
# Définition du DAG
# ═══════════════════════════════════════════════════════════════

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='drift_detection_simple',
    default_args=default_args,
    description='Simple drift detection with standard statistical methods',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['drift', 'monitoring', 'simple'],
) as dag:

    detect_drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
        provide_context=True,
    )

    generate_report_task = PythonOperator(
        task_id='generate_html_report',
        python_callable=generate_html_report,
        provide_context=True,
    )

    branch_task = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=choose_branch,
        provide_context=True,
    )

    retrain_combined = BashOperator(
        task_id='retrain_combined',
        bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && python -m src.training.retrain --mode combined',
    )

    skip_retrain = DummyOperator(
        task_id='skip_retrain',
    )

    done = DummyOperator(
        task_id='done',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Flow
    detect_drift_task >> generate_report_task >> branch_task
    branch_task >> [retrain_combined, skip_retrain] >> done