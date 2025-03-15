import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions you want to test from the src folder
from src.class_imbalance import (
    load_data,
    create_plots_dir,
    plot_class_distribution,
    balance_with_smote,
    compute_class_weights,
    split_and_save_train_test
)

@pytest.fixture
def sample_df():
    """
    Returns a small DataFrame with a binary target column 'survival_status'
    and two numeric features for testing.
    """
    data = {
        'feature1': [10, 20, 30, 40],
        'feature2': [1, 2, 3, 4],
        'survival_status': [0, 0, 1, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir(tmp_path):
    """
    Pytest's built-in fixture that provides a temporary directory unique to each test.
    """
    return tmp_path


def test_compute_class_weights(sample_df):
    """
    Test that compute_class_weights returns a dictionary with correct keys and float values.
    """
    y = sample_df['survival_status']
    weights = compute_class_weights(y)

    # We expect 2 classes: 0 and 1
    assert 0 in weights and 1 in weights, "Expected keys (0, 1) in class weights."
    # Ensure the weights are floats
    assert all(isinstance(w, float) for w in weights.values()), "Class weights should be floats."


def test_balance_with_smote(sample_df):
    """
    Test that SMOTE oversampling balances the minority and majority classes.
    """
    X = sample_df.drop(columns=['survival_status'])
    y = sample_df['survival_status']

    X_res, y_res = balance_with_smote(X, y, random_state=42)

    # Check that we still have the same columns
    assert list(X_res.columns) == list(X.columns), "Feature columns should remain the same after SMOTE."

    # The number of samples should be greater or equal to the original
    assert len(X_res) >= len(X), "After SMOTE, we expect at least as many samples as before."

    # Check class distribution is now balanced
    unique, counts = np.unique(y_res, return_counts=True)
    assert len(unique) == 2, "We should still have 2 classes after SMOTE."
    # For a 2-class problem, counts[0] should equal counts[1]
    assert counts[0] == counts[1], "SMOTE should produce equal counts for both classes."


def test_create_plots_dir(temp_dir):
    """
    Test create_plots_dir to ensure it creates (or returns) the directory path.
    We patch __file__ to point to a temporary location so we don't affect real directories.
    """
    with patch('src.class_imbalance.__file__', os.path.join(temp_dir, 'class_imbalance.py')):
        plots_path = create_plots_dir()
        # Verify that the returned path exists
        assert os.path.exists(plots_path), "Expected the plots directory to be created."


def test_plot_class_distribution(sample_df, temp_dir):
    """
    Test the plot_class_distribution function to ensure it runs without error
    and saves a plot file.
    """
    y = sample_df['survival_status']
    filename = "test_distribution.png"
    folder = str(temp_dir)  # Convert Path object to string

    # Run the function
    plot_class_distribution(y, title="Test Distribution", filename=filename, folder=folder)

    # Check the file was created
    expected_plot_path = os.path.join(folder, filename)
    assert os.path.exists(expected_plot_path), "Expected the distribution plot to be saved."



@patch('src.class_imbalance.balance_with_smote')
def test_split_and_save_train_test(mock_smote, temp_dir):
    """
    Test split_and_save_train_test to ensure train and test CSVs are created.
    We patch balance_with_smote to avoid real SMOTE processing.
    """
    import pandas as pd
    # Create imbalanced sample data so that SMOTE is triggered
    data = {
        'feature1': [10, 20, 30, 40, 50],
        'feature2': [1, 2, 3, 4, 5],
        'survival_status': [0, 0, 0, 1, 1]  # Imbalance: three 0's, two 1's
    }
    sample_df = pd.DataFrame(data)

    # Make SMOTE return the input data unchanged, for simplicity.
    mock_smote.side_effect = lambda X, y, random_state: (X, y)

    # Patch __file__ so that CSVs are written to a temp directory.
    with patch('src.class_imbalance.__file__', os.path.join(temp_dir, 'class_imbalance.py')):
        # Pre-create the directory where the CSVs will be saved.
        processed_dir = os.path.join(temp_dir, "..", "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        X = sample_df.drop(columns=['survival_status'])
        y = sample_df['survival_status']

        # Use test_size=0.4 so that the test set has at least 2 samples
        split_and_save_train_test(X, y, test_size=0.4, random_state=42)

        # Paths where the function attempts to save train/test CSVs.
        train_path = os.path.join(temp_dir, "..", "data", "processed", "bmt_train.csv")
        test_path = os.path.join(temp_dir, "..", "data", "processed", "bmt_test.csv")

        # Convert to absolute path for checking.
        train_abs = os.path.abspath(train_path)
        test_abs = os.path.abspath(test_path)

        # Check that the CSV files exist.
        assert os.path.isfile(train_abs), "Training CSV should be saved."
        assert os.path.isfile(test_abs), "Testing CSV should be saved."

        # Now, because the input training set is imbalanced, SMOTE should have been applied.
        mock_smote.assert_called()