import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np

from src.compare_oversampling import load_data, evaluate_model

def test_load_data(tmp_path):
    """
    Test that load_data properly loads a CSV file.
    """
    # Create a dummy CSV file in the temporary directory.
    df_original = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    dummy_csv = tmp_path / "dummy.csv"
    df_original.to_csv(dummy_csv, index=False)
    
    # Patch __file__ in the module so that load_data computes the path relative to tmp_path.
    from src import compare_oversampling
    original_file = compare_oversampling.__file__
    try:
        # Set __file__ to a file in the temporary directory.
        compare_oversampling.__file__ = str(tmp_path / "dummy_module.py")
        # Since __file__ is now in tmp_path, calling load_data("dummy.csv") should load our file.
        df_loaded = load_data("dummy.csv")
        pd.testing.assert_frame_equal(df_original, df_loaded)
    finally:
        compare_oversampling.__file__ = original_file


# Define two dummy models for testing evaluate_model.

class DummyModelWithProba:
    def predict(self, X):
        # Return fixed predictions.
        return np.array([0, 1, 0])
    
    def predict_proba(self, X):
        # Return fixed probability estimates.
        return np.array([
            [0.7, 0.3],
            [0.4, 0.6],
            [0.8, 0.2]
        ])

class DummyModelWithoutProba:
    def predict(self, X):
        # Return fixed predictions.
        return np.array([1, 0, 1])
    
    def decision_function(self, X):
        # Return decision scores.
        return np.array([0.2, -0.3, 0.8])


@pytest.fixture
def dummy_X():
    """
    Fixture for dummy feature DataFrame.
    """
    return pd.DataFrame({'dummy': [1, 2, 3]})

@pytest.fixture
def dummy_y():
    """
    Fixture for dummy target Series.
    """
    return pd.Series([0, 1, 0])


def test_evaluate_model_with_proba(dummy_X, dummy_y):
    """
    Test evaluate_model with a model that implements predict_proba.
    """
    model = DummyModelWithProba()
    metrics = evaluate_model(model, dummy_X, dummy_y)
    
    # Check that all expected metric keys are present.
    for key in ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"]:
        assert key in metrics, f"Metric '{key}' not found in the output."
    
    # Optionally, you can add assertions for expected values if needed.
    # For example, accuracy should be 1.0 if predictions exactly match dummy_y.
    

def test_evaluate_model_without_proba(dummy_X, dummy_y):
    """
    Test evaluate_model with a model that uses decision_function instead of predict_proba.
    """
    model = DummyModelWithoutProba()
    metrics = evaluate_model(model, dummy_X, dummy_y)
    
    for key in ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"]:
        assert key in metrics, f"Metric '{key}' not found in the output."
