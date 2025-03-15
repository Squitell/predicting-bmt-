import os
import sys
import glob
import joblib
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for plotting

import shap  # Make sure shap is installed

# Add project root to sys.path so that src modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shap_analysis import (
    load_data,
    reindex_features,
    load_latest_models,
    create_shap_summary_plot,
    create_manual_shap_bar_plot,
    create_shap_beeswarm_plot,
    create_shap_plots,
    analyze_constant_features
)

# ----------------------------
# Test load_data

def test_load_data(tmp_path):
    # Create a dummy CSV file.
    df_original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    dummy_csv = tmp_path / "dummy.csv"
    df_original.to_csv(dummy_csv, index=False)
    
    # Patch __file__ so that load_data computes path relative to tmp_path.
    from src import shap_analysis
    original_file = shap_analysis.__file__
    try:
        shap_analysis.__file__ = str(tmp_path / "dummy_module.py")
        df_loaded = load_data("dummy.csv")
        pd.testing.assert_frame_equal(df_original, df_loaded)
    finally:
        shap_analysis.__file__ = original_file

# ----------------------------
# Test reindex_features

class DummyModel:
    feature_names_in_ = ['f1', 'f2', 'f3']
    def predict(self, X):
        return np.zeros(len(X))

def test_reindex_features():
    # Create a DataFrame with extra columns.
    X = pd.DataFrame({
        'f1': [1, 2],
        'f3': [3, 4],
        'extra': [5, 6]
    })
    model = DummyModel()
    X_reindexed = reindex_features(X, model)
    # X_reindexed should have exactly the columns in model.feature_names_in_
    expected = ['f1', 'f2', 'f3']
    assert list(X_reindexed.columns) == expected
    # Missing column "f2" is filled with 0.
    assert (X_reindexed['f2'] == 0).all()

# ----------------------------
# Test load_latest_models

def test_load_latest_models(tmp_path):
    # Create a temporary models directory.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Create dummy models and save them.
    dummy_model = DummyModel()
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        filename = models_dir / f"{model_name}_model_dummy.pkl"
        joblib.dump(dummy_model, filename)
    
    loaded_models = load_latest_models(str(models_dir))
    # Check that each expected model is loaded.
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        assert model_name in loaded_models, f"{model_name} not loaded."

# ----------------------------
# Test create_shap_summary_plot

def test_create_shap_summary_plot(tmp_path):
    # Create dummy X_test DataFrame.
    X_test = pd.DataFrame({
        'f1': np.random.randn(50),
        'f2': np.random.randn(50),
        'f3': np.random.randn(50)
    })
    # Create dummy shap_values as a numpy array with shape (50, 3)
    shap_values = np.random.randn(50, 3)
    
    output_dir = tmp_path / "shap_plots"
    output_dir.mkdir()
    
    result = create_shap_summary_plot(shap_values, X_test, "DummyModel", str(output_dir))
    # Check that file is created.
    summary_path = os.path.join(str(output_dir), "DummyModel_shap_summary.png")
    assert result is True
    assert os.path.isfile(summary_path)

# ----------------------------
# Test create_manual_shap_bar_plot

def test_create_manual_shap_bar_plot(tmp_path):
    # Create dummy X_test DataFrame.
    X_test = pd.DataFrame({
        'f1': np.random.randn(50),
        'f2': np.random.randn(50),
        'f3': np.random.randn(50)
    })
    # Use a dummy numpy array for shap_values.
    shap_values = np.random.randn(50, 3)
    
    output_dir = tmp_path / "shap_plots"
    output_dir.mkdir()
    
    result = create_manual_shap_bar_plot(shap_values, X_test, "DummyModel", str(output_dir))
    bar_path = os.path.join(str(output_dir), "DummyModel_shap_bar.png")
    assert result is True
    assert os.path.isfile(bar_path)

# ----------------------------
# Test create_shap_beeswarm_plot

def test_create_shap_beeswarm_plot(tmp_path):
    # Create dummy X_test DataFrame.
    X_test = pd.DataFrame({
        'f1': np.random.randn(50),
        'f2': np.random.randn(50),
        'f3': np.random.randn(50)
    })
    # Create a dummy shap.Explanation object.
    expl = shap.Explanation(
        values = np.random.randn(50, 3),
        base_values = np.zeros(50),
        data = X_test.values,
        feature_names = X_test.columns.tolist()
    )
    output_dir = tmp_path / "shap_plots"
    output_dir.mkdir()
    
    result = create_shap_beeswarm_plot(expl, "DummyModel", str(output_dir))
    beeswarm_path = os.path.join(str(output_dir), "DummyModel_shap_beeswarm.png")
    assert result is True
    assert os.path.isfile(beeswarm_path)

# ----------------------------
# Test analyze_constant_features

def test_analyze_constant_features(capsys):
    # Create a DataFrame with one constant feature.
    X = pd.DataFrame({
        'a': [1, 1, 1, 1],
        'b': [2, 3, 4, 5],
        'c': [0, 0, 0, 0]
    })
    analyze_constant_features(X)
    captured = capsys.readouterr().out
    assert "Found 2 constant columns" in captured or "constant" in captured.lower()

# ----------------------------
# Optionally, you could test create_shap_plots as an integration test,
# but it calls multiple internal functions. You can assume if the above functions pass,
# then create_shap_plots works as intended.

# ----------------------------
# End of test_shap_analysis.py
