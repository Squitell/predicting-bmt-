import sys
import os
import glob
import joblib
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Set the matplotlib backend to "Agg" to avoid Tkinter issues during testing.
import matplotlib
matplotlib.use("Agg")

# Add project root to sys.path so that src modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import (
    load_data,
    preprocess_data,
    plot_feature_importance,
    train_models,
    save_models,
    save_performance
)

# ----------------------------
# Test load_data

def test_load_data(tmp_path):
    """
    Create a dummy CSV file and ensure load_data loads it correctly.
    """
    df_original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    dummy_csv = tmp_path / "dummy_train.csv"
    df_original.to_csv(dummy_csv, index=False)
    
    # Patch __file__ so that load_data computes the path relative to tmp_path.
    from src import model_training
    original_file = model_training.__file__
    try:
        model_training.__file__ = str(tmp_path / "dummy_module.py")
        df_loaded = load_data("dummy_train.csv")
        pd.testing.assert_frame_equal(df_original, df_loaded)
    finally:
        model_training.__file__ = original_file

# ----------------------------
# Test preprocess_data

def test_preprocess_data():
    """
    Test that preprocess_data one-hot encodes categorical variables and removes highly correlated features.
    """
    # Create a dummy DataFrame.
    # Adjusted dummy data so that the one-hot encoded column is not perfectly correlated with survival_status.
    df = pd.DataFrame({
        'survival_status': [0, 1, 0, 1],
        'cat': ['a', 'a', 'b', 'b'],   # one-hot encoding yields "cat_b": [0,0,1,1]
        'num1': [1, 2, 1, 2],
        'num2': [1.1, 2.1, 1.1, 2.1]   # Highly correlated with num1
    })
    df_processed = preprocess_data(df)
    
    # Check that one-hot encoding occurred.
    dummy_cat = [col for col in df_processed.columns if "cat_" in col]
    assert dummy_cat, "Categorical column not one-hot encoded."
    
    # Check that one of the highly correlated numeric features was dropped.
    num_cols = set(['num1', 'num2'])
    dropped = num_cols - set(df_processed.columns)
    assert len(dropped) >= 1, "Highly correlated numeric features were not dropped."

# ----------------------------
# Test plot_feature_importance

class DummyModelForImportance:
    # Provide dummy feature_importances_ attribute.
    feature_importances_ = np.array([0.2, 0.5, 0.3])
    # Dummy predict method.
    def predict(self, X):
        return np.zeros(len(X))
    
def test_plot_feature_importance(tmp_path):
    """
    Test that plot_feature_importance creates and saves a plot file.
    """
    # Create dummy feature data.
    X = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [4, 5, 6],
        'f3': [7, 8, 9]
    })
    dummy_model = DummyModelForImportance()
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    
    plot_feature_importance(dummy_model, X, "DummyModel", str(output_dir), top_n=3)
    
    # Check that the file was saved.
    expected_plot = os.path.join(str(output_dir), "DummyModel_feature_importance.png")
    assert os.path.isfile(expected_plot), "Feature importance plot was not saved."

# ----------------------------
# Test train_models

def test_train_models():
    """
    Create a larger, more separable dummy dataset and ensure train_models returns the expected keys 
    and performance metrics without triggering UndefinedMetricWarning.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Set seed for reproducibility.
    np.random.seed(42)
    
    # Generate 50 samples for class 0 and 50 for class 1 with clear separation.
    class0 = np.random.normal(loc=0.0, scale=1.0, size=(50, 2))
    class1 = np.random.normal(loc=5.0, scale=1.0, size=(50, 2))
    
    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)
    
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["survival_status"] = y
    
    # Split into train and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=["survival_status"]),
        df["survival_status"],
        test_size=0.33,
        random_state=42,
        stratify=df["survival_status"]
    )
    
    from src.model_training import train_models
    trained_models, performance_results = train_models(X_train, y_train, X_val, y_val)
    
    # Check that all expected models are trained.
    for key in ["RandomForest", "XGBoost", "LightGBM"]:
        assert key in trained_models, f"{key} model not found in trained models."
        assert key in performance_results, f"{key} performance metrics not found."
    
    # Check that performance metrics contain expected keys.
    expected_metric_keys = {"Train Accuracy", "Train ROC-AUC", "Validation Accuracy", "Validation ROC-AUC"}
    for metrics in performance_results.values():
        assert expected_metric_keys.issubset(metrics.keys()), "Missing expected performance metrics."


# ----------------------------
# Test save_models

class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))

def test_save_models(tmp_path):
    """
    Test that save_models saves model files with timestamps.
    """
    # Create dummy models dictionary.
    dummy_models = {
        "RandomForest": DummyModel(),
        "XGBoost": DummyModel(),
        "LightGBM": DummyModel()
    }
    output_dir = tmp_path / "models"
    output_dir.mkdir()
    
    save_models(dummy_models, str(output_dir))
    
    # Look for files that match the naming pattern.
    for model_name in dummy_models.keys():
        pattern = os.path.join(str(output_dir), f"{model_name}_model_*.pkl")
        files = glob.glob(pattern)
        assert len(files) > 0, f"No saved model file found for {model_name}."

# ----------------------------
# Test save_performance

def test_save_performance(tmp_path):
    """
    Test that save_performance creates a CSV file with performance metrics.
    """
    performance_results = {
        "RandomForest": {"Train Accuracy": 0.9, "Train ROC-AUC": 0.95,
                         "Validation Accuracy": 0.85, "Validation ROC-AUC": 0.90},
        "XGBoost": {"Train Accuracy": 0.88, "Train ROC-AUC": 0.93,
                    "Validation Accuracy": 0.83, "Validation ROC-AUC": 0.88}
    }
    output_dir = tmp_path / "performance"
    output_dir.mkdir()
    
    save_performance(performance_results, str(output_dir))
    
    performance_file = os.path.join(str(output_dir), "validation_performance.csv")
    assert os.path.isfile(performance_file), "Performance CSV file was not saved."
    
    df_perf = pd.read_csv(performance_file, index_col=0)
    # Check that the DataFrame contains the keys from performance_results.
    for key in performance_results.keys():
        assert key in df_perf.index, f"Performance metrics for {key} not found in CSV."

# ----------------------------
# End of test_model_training.py
