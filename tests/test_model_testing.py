import sys
import os
from time import sleep
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import joblib

from src.model_testing import load_data, reindex_features, evaluate_model, load_models

def test_load_data(tmp_path):
    """
    Test that load_data properly loads a CSV file.
    """
    # Create a dummy CSV file in the temporary directory.
    df_original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    dummy_csv = tmp_path / "dummy_test.csv"
    df_original.to_csv(dummy_csv, index=False)
    
    # Patch __file__ in the model_testing module so that the relative path is computed from tmp_path.
    from src import model_testing
    original_file = model_testing.__file__
    try:
        # Set __file__ to a dummy file in tmp_path.
        model_testing.__file__ = str(tmp_path / "dummy_module.py")
        df_loaded = load_data("dummy_test.csv")
        pd.testing.assert_frame_equal(df_loaded, df_original)
    finally:
        model_testing.__file__ = original_file


class DummyModel:
    """
    Dummy model with a feature_names_in_ attribute and simple predict methods.
    """
    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names)
        
    def predict(self, X):
        # Return zeros for all samples.
        return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X):
        # Return fixed probabilities.
        return np.column_stack((np.ones(len(X)) * 0.7, np.ones(len(X)) * 0.3))


def test_reindex_features():
    """
    Test that reindex_features returns a DataFrame with columns matching the model's feature_names_in_.
    """
    # Create a dummy DataFrame with extra columns.
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4],
        'C': [5, 6]
    })
    # Create a dummy model that expects only columns 'B' and 'C'.
    dummy_model = DummyModel(feature_names=['B', 'C'])
    df_reindexed = reindex_features(df, dummy_model)
    # Expect df_reindexed to have only columns 'B' and 'C'.
    assert list(df_reindexed.columns) == ['B', 'C']
    
    # Also test if a missing column is added with zeros.
    dummy_model2 = DummyModel(feature_names=['B', 'D'])
    df_reindexed2 = reindex_features(df, dummy_model2)
    assert list(df_reindexed2.columns) == ['B', 'D']
    # Column 'D' did not exist in the original, so its values should be 0.
    assert (df_reindexed2['D'] == 0).all()


def test_evaluate_model():
    """
    Test that evaluate_model returns a dictionary containing the expected metric keys.
    """
    # Create dummy test data.
    X = pd.DataFrame({'A': [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    
    # Define a dummy model with predictable outputs.
    class DummyModel2:
        def __init__(self):
            self.feature_names_in_ = np.array(['A'])
        def predict(self, X):
            # Return predictions that match y.
            return np.array([0, 1, 0])
        def predict_proba(self, X):
            # Return fixed probability estimates.
            return np.array([
                [0.8, 0.2],
                [0.3, 0.7],
                [0.9, 0.1]
            ])
    dummy_model = DummyModel2()
    metrics = evaluate_model(dummy_model, X, y)
    # Check that all expected metric keys are present.
    for key in ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"]:
        assert key in metrics


def test_load_models(tmp_path):
    """
    Test that load_models loads the latest model file from a temporary models directory.
    """
    # Create a temporary "models" directory.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Create a dummy model object.
    dummy_model = DummyModel(feature_names=['A', 'B'])
    
    # Create a dummy file for a RandomForest model.
    model_file1 = models_dir / "RandomForest_model_dummy.pkl"
    joblib.dump(dummy_model, model_file1)
    
    # Wait to ensure a different timestamp.
    sleep(1)
    model_file2 = models_dir / "RandomForest_model_new.pkl"
    joblib.dump(dummy_model, model_file2)
    
    # Call load_models to load from the temporary models directory.
    loaded_models = load_models(str(models_dir))
    
    # Expect the "RandomForest" key to be present.
    assert "RandomForest" in loaded_models
    # Check that the loaded model is an instance of DummyModel.
    assert isinstance(loaded_models["RandomForest"], type(dummy_model))
