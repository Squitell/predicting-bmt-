import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend for plotting
import pytest

# Add the notebooks directory to sys.path so that eda.py can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import eda  # Now eda.py from the notebooks folder is imported.

# Fixture to override the global plots directory to a temporary directory.
@pytest.fixture(autouse=True)
def override_plots_dir(tmp_path, monkeypatch):
    test_plots_dir = tmp_path / "plots"
    test_plots_dir.mkdir()
    monkeypatch.setattr(eda, "plots_dir", str(test_plots_dir))
    yield

# Test load_data: Create a dummy CSV file and verify the DataFrame is loaded.
def test_load_data(tmp_path):
    df_original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    dummy_csv = tmp_path / "dummy.csv"
    df_original.to_csv(dummy_csv, index=False)
    
    # Patch __file__ to ensure relative paths work in load_data.
    original_file = eda.__file__
    try:
        eda.__file__ = str(tmp_path / "dummy_module.py")
        df_loaded = eda.load_data("dummy.csv")
        pd.testing.assert_frame_equal(df_original, df_loaded)
    finally:
        eda.__file__ = original_file

# Test basic_info: Capture printed output.
def test_basic_info(capsys):
    df = pd.DataFrame({'A': [1,2,3], 'B': ['a','b','c']})
    eda.basic_info(df)
    captured = capsys.readouterr().out
    assert "HEAD" in captured
    assert "INFO" in captured
    assert "DESCRIBE" in captured
    assert "MISSING VALUES" in captured

# Test save_plot: Create a dummy plot and verify file creation.
def test_save_plot():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([1,2,3])
    filename = "test_plot.png"
    eda.save_plot(filename)
    filepath = os.path.join(eda.plots_dir, filename)
    assert os.path.isfile(filepath), f"Plot file {filepath} not found."

# Test plot_bar_chart: Provide dummy DataFrame with 'Recipientgender'.
def test_plot_bar_chart():
    df = pd.DataFrame({'Recipientgender': ['M', 'F', 'F', 'M', 'M']})
    eda.plot_bar_chart(df)
    filepath = os.path.join(eda.plots_dir, "bar_chart_recipientgender.png")
    assert os.path.isfile(filepath)

# Test plot_scatter: Provide dummy DataFrame with required columns.
def test_plot_scatter():
    df = pd.DataFrame({
        'Donorage': np.random.randint(20, 50, size=10),
        'CD34kgx10d6': np.random.rand(10)*100,
        'Recipientgender': ['M', 'F']*5
    })
    eda.plot_scatter(df)
    filepath = os.path.join(eda.plots_dir, "scatter_donorage_cd34.png")
    assert os.path.isfile(filepath)

# Test plot_boxplot.
def test_plot_boxplot():
    df = pd.DataFrame({
        'Donorage': np.random.randint(20, 50, size=10),
        'Stemcellsource': ['Bone Marrow', 'Peripheral', 'Umbilical Cord', 'Bone Marrow', 'Peripheral',
                           'Umbilical Cord', 'Bone Marrow', 'Peripheral', 'Umbilical Cord', 'Bone Marrow']
    })
    eda.plot_boxplot(df)
    filepath = os.path.join(eda.plots_dir, "boxplot_donorage_by_stemcellsource.png")
    assert os.path.isfile(filepath)

# Test plot_histogram.
def test_plot_histogram():
    df = pd.DataFrame({'Donorage': np.random.randint(20, 60, size=50)})
    eda.plot_histogram(df)
    filepath = os.path.join(eda.plots_dir, "histogram_donorage.png")
    assert os.path.isfile(filepath)

# Test plot_area.
def test_plot_area():
    df = pd.DataFrame({'Donorage': np.random.randint(20, 60, size=30)})
    eda.plot_area(df)
    filepath = os.path.join(eda.plots_dir, "area_plot_donorage.png")
    assert os.path.isfile(filepath)

# Test plot_pie_chart.
def test_plot_pie_chart():
    df = pd.DataFrame({'Disease': ['DiseaseA', 'DiseaseB', 'DiseaseA', 'DiseaseC', 'DiseaseB', 'DiseaseA']})
    eda.plot_pie_chart(df)
    filepath = os.path.join(eda.plots_dir, "pie_chart_disease.png")
    assert os.path.isfile(filepath)

# Test plot_treemap: Only if squarify is available.
@pytest.mark.skipif(not eda.HAS_SQUARIFY, reason="squarify not installed")
def test_plot_treemap():
    df = pd.DataFrame({'Disease': ['DiseaseA', 'DiseaseB', 'DiseaseA', 'DiseaseC', 'DiseaseB', 'DiseaseA']})
    eda.plot_treemap(df)
    filepath = os.path.join(eda.plots_dir, "treemap_disease.png")
    assert os.path.isfile(filepath)

# Test plot_missing_values_heatmap.
def test_plot_missing_values_heatmap():
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, 3]})
    eda.plot_missing_values_heatmap(df)
    filepath = os.path.join(eda.plots_dir, "missing_values_heatmap.png")
    assert os.path.isfile(filepath)

# Test plot_missing_percentage.
def test_plot_missing_percentage():
    df = pd.DataFrame({'A': [1, np.nan, 3, np.nan], 'B': [np.nan, 2, 3, 4]})
    eda.plot_missing_percentage(df)
    filepath = os.path.join(eda.plots_dir, "missing_percentage.png")
    assert os.path.isfile(filepath)

# Test plot_correlation_matrix.
def test_plot_correlation_matrix():
    df = pd.DataFrame({
        'A': np.random.rand(20),
        'B': np.random.rand(20),
        'C': np.random.rand(20)
    })
    eda.plot_correlation_matrix(df)
    filepath = os.path.join(eda.plots_dir, "correlation_matrix.png")
    assert os.path.isfile(filepath)
