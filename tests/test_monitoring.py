"""
Test monitoring components
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.drift_detection import DriftDetector


def test_drift_detector_initialization():
    """Test drift detector can be initialized"""
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.uniform(0, 10, 1000)
    })
    
    detector = DriftDetector(reference_data_path=None)
    detector.reference_data = reference_data
    detector.feature_names = ['feature1', 'feature2', 'feature3']
    
    assert detector.reference_data is not None
    assert len(detector.feature_names) == 3


def test_drift_detection_no_drift():
    """Test drift detection when distributions are similar"""
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })
    
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 500),
        'feature2': np.random.normal(5, 2, 500)
    })
    
    detector = DriftDetector(reference_data_path=None)
    detector.reference_data = reference_data
    detector.feature_names = ['feature1', 'feature2']
    
    result = detector.detect_drift(current_data, threshold=0.05, save_report=False)
    
    assert 'drift_detected' in result
    assert 'n_drifted_features' in result
    assert 'n_features_checked' in result
    assert result['n_features_checked'] == 2


def test_drift_detection_with_drift():
    """Test drift detection when distributions differ"""
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })
    
    current_data = pd.DataFrame({
        'feature1': np.random.normal(3, 1, 500),
        'feature2': np.random.normal(10, 2, 500)
    })
    
    detector = DriftDetector(reference_data_path=None)
    detector.reference_data = reference_data
    detector.feature_names = ['feature1', 'feature2']
    
    result = detector.detect_drift(current_data, threshold=0.05, save_report=False)
    
    assert result['drift_detected'] is True
    assert result['n_drifted_features'] > 0


def test_ks_statistic_calculation():
    """Test that KS statistic is calculated correctly"""
    from scipy import stats
    
    ref = np.random.normal(0, 1, 1000)
    curr_same = np.random.normal(0, 1, 500)
    curr_diff = np.random.normal(3, 1, 500)
    
    ks_same, p_same = stats.ks_2samp(ref, curr_same)
    assert ks_same < 0.1
    assert p_same > 0.05
    
    ks_diff, p_diff = stats.ks_2samp(ref, curr_diff)
    assert ks_diff > 0.3
    assert p_diff < 0.05