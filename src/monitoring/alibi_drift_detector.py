"""
Détection de drift avancée avec Alibi Detect
Utilise des méthodes statistiques robustes
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from alibi_detect.cd import TabularDrift
from alibi_detect.cd.pytorch import ClassifierDrift
from alibi_detect.utils.saving import save_detector, load_detector

from src.api.core.logger import api_logger as logger


class AlibiDriftDetector:
    """
    Wrapper pour Alibi Detect avec persistance des détecteurs.
    """
    
    def __init__(self, detector_path: str = "/opt/airflow/models/drift_detectors"):
        self.detector_path = Path(detector_path)
        self.detector_path.mkdir(parents=True, exist_ok=True)
        self.detector = None
    
    def train_detector(
        self,
        reference_data: np.ndarray,
        feature_names: list,
        p_val: float = 0.05
    ) -> Dict:
        """
        Entraîne un détecteur de drift sur les données de référence.
        
        Args:
            reference_data: Données de référence (numpy array)
            feature_names: Noms des features
            p_val: Seuil de p-value pour détecter le drift
        
        Returns:
            Dict avec les infos du détecteur
        """
        try:
            logger.info(f"🔧 Training Alibi drift detector on {reference_data.shape[0]} samples...")
            
            # ═══════════════════════════════════════════════════════════
            # Utiliser TabularDrift (Kolmogorov-Smirnov test)
            # ═══════════════════════════════════════════════════════════
            self.detector = TabularDrift(
                x_ref=reference_data,
                p_val=p_val,
                categories_per_feature=None,  # Auto-détection
            )
            
            # Sauvegarder le détecteur
            detector_file = self.detector_path / "tabular_drift_detector.pkl"
            save_detector(self.detector, str(detector_file))
            
            logger.info(f"✅ Drift detector trained and saved to {detector_file}")
            
            return {
                'status': 'success',
                'detector_path': str(detector_file),
                'n_features': reference_data.shape[1],
                'n_samples': reference_data.shape[0],
                'p_val_threshold': p_val,
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to train drift detector: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def load_detector(self) -> bool:
        """Charge un détecteur sauvegardé."""
        try:
            detector_file = self.detector_path / "tabular_drift_detector.pkl"
            
            if not detector_file.exists():
                logger.warning(f"⚠️  Detector not found: {detector_file}")
                return False
            
            self.detector = load_detector(str(detector_file))
            logger.info(f"✅ Drift detector loaded from {detector_file}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to load detector: {e}")
            return False
    
    def detect_drift(
        self,
        current_data: np.ndarray,
        feature_names: list
    ) -> Dict:
        """
        Détecte le drift sur de nouvelles données.
        
        Args:
            current_data: Données actuelles (numpy array)
            feature_names: Noms des features
        
        Returns:
            Dict avec les résultats de détection
        """
        if self.detector is None:
            if not self.load_detector():
                return {
                    'status': 'error',
                    'error': 'No detector available. Train one first.'
                }
        
        try:
            logger.info(f"🔍 Detecting drift on {current_data.shape[0]} samples...")
            
            # Prédiction de drift
            drift_prediction = self.detector.predict(current_data)
            
            # Extraire les résultats
            is_drift = drift_prediction['data']['is_drift']
            p_vals = drift_prediction['data']['p_val']
            distance = drift_prediction['data'].get('distance', None)
            
            # Identifier les features en drift
            features_with_drift = []
            if p_vals is not None:
                for idx, (feat_name, p_val) in enumerate(zip(feature_names, p_vals)):
                    if p_val < self.detector.p_val:
                        features_with_drift.append({
                            'feature': feat_name,
                            'p_value': float(p_val),
                            'drifted': True,
                        })
            
            drift_percentage = (len(features_with_drift) / len(feature_names)) * 100 if feature_names else 0
            
            logger.info(f"{'🔴 DRIFT DETECTED' if is_drift else '✅ No drift detected'}")
            logger.info(f"   Features with drift: {len(features_with_drift)}/{len(feature_names)} ({drift_percentage:.1f}%)")
            
            return {
                'status': 'success',
                'is_drift': int(is_drift),  # Convertir en int pour JSON
                'overall_p_value': float(drift_prediction['data']['p_val'].mean()) if p_vals is not None else None,
                'features_with_drift': features_with_drift,
                'drift_percentage': drift_percentage,
                'n_features_analyzed': len(feature_names),
                'n_samples': current_data.shape[0],
            }
        
        except Exception as e:
            logger.error(f"❌ Drift detection failed: {e}")
            return {'status': 'error', 'error': str(e)}


def detect_drift_with_alibi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list,
    p_val_threshold: float = 0.05,
    retrain_detector: bool = False
) -> Dict:
    """
    Fonction helper pour détecter le drift avec Alibi.
    
    Args:
        reference_df: Données de référence
        current_df: Données actuelles
        feature_cols: Liste des colonnes à analyser
        p_val_threshold: Seuil de p-value
        retrain_detector: Si True, réentraîne le détecteur
    
    Returns:
        Dict avec les résultats
    """
    detector = AlibiDriftDetector()
    
    # Préparer les données
    X_ref = reference_df[feature_cols].values.astype(float)
    X_curr = current_df[feature_cols].values.astype(float)
    
    # Entraîner ou charger le détecteur
    if retrain_detector or not detector.load_detector():
        logger.info("🔧 Training new drift detector...")
        train_result = detector.train_detector(X_ref, feature_cols, p_val_threshold)
        if train_result['status'] != 'success':
            return train_result
    
    # Détecter le drift
    drift_result = detector.detect_drift(X_curr, feature_cols)
    
    return drift_result