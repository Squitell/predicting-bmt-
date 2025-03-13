import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set SHAP to display full feature names
shap.initjs()

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading test data from:", full_path)
    df = pd.read_csv(full_path)
    print(f"Loaded test data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def reindex_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure the feature set matches the trained model.
    """
    if hasattr(model, "feature_names_in_"):
        training_features = list(model.feature_names_in_)
        print(f"Reindexing test data from {X.shape[1]} to {len(training_features)} features...")
        
        # Debug info
        common_features = set(X.columns).intersection(set(training_features))
        print(f"Features in common: {len(common_features)} out of {len(training_features)}")
        
        # Print a few sample columns from both sets
        print(f"Sample test columns: {X.columns[:5].tolist()}")
        print(f"Sample model columns: {training_features[:5]}")
        
        X = X.reindex(columns=training_features, fill_value=0)
    else:
        print("Model does not have 'feature_names_in_'; using X as is.")
    return X

def load_latest_models(models_dir: str) -> dict:
    """
    Load the latest trained models dynamically from the models directory.
    """
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    loaded_models = {}

    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        latest_model_file = sorted(
            [f for f in model_files if f.startswith(model_name)], reverse=True
        )
        if latest_model_file:
            model_path = os.path.join(models_dir, latest_model_file[0])
            loaded_models[model_name] = joblib.load(model_path)
            print(f"✅ Loaded latest {model_name} model from {model_path}")
        else:
            print(f"⚠️ Warning: No trained {model_name} model found!")
    
    return loaded_models

def create_shap_summary_plot(shap_values, X_test, model_name, output_dir):
    """
    Create only the SHAP summary plot, which is more reliable than the bar plot.
    """
    try:
        plt.figure(figsize=(16, 10))
        shap.summary_plot(
            shap_values, 
            X_test, 
            show=False, 
            max_display=min(20, X_test.shape[1])
        )
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
        plt.savefig(summary_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"📊 SHAP summary plot saved for {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ SHAP summary plot failed for {model_name}: {str(e)}")
        return False

def create_manual_shap_bar_plot(shap_values, X_test, model_name, output_dir):
    """
    Create a manual SHAP bar plot that doesn't use shap.plots.bar to avoid the bug.
    """
    try:
        # Calculate mean absolute SHAP values for each feature
        feature_names = X_test.columns.tolist()
        
        # For multi-output models, focus on the first output
        if hasattr(shap_values, "shape") and len(shap_values.shape) > 2:
            mean_shap_values = np.abs(shap_values.values[:, :, 0]).mean(0)
        elif hasattr(shap_values, "values"):
            mean_shap_values = np.abs(shap_values.values).mean(0)
        else:
            mean_shap_values = np.abs(shap_values).mean(0)
        
        # Create a DataFrame for sorting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap_values
        })
        
        # Sort by importance
        shap_df = shap_df.sort_values('importance', ascending=False)
        
        # Take top 20 features
        top_features = shap_df.head(min(20, len(shap_df)))
        
        # Create the plot
        plt.figure(figsize=(16, 10))
        plt.barh(
            y=top_features['feature'],
            width=top_features['importance'],
            color='#1E88E5'
        )
        plt.title(f'Mean |SHAP| value (impact on model output) - {model_name}')
        plt.xlabel('mean |SHAP value|')
        plt.tight_layout()
        
        # Save the plot
        bar_path = os.path.join(output_dir, f"{model_name}_shap_bar.png")
        plt.savefig(bar_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"📊 Custom SHAP bar plot saved for {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ Custom SHAP bar plot failed for {model_name}: {str(e)}")
        return False

def create_shap_beeswarm_plot(shap_values, model_name, output_dir):
    """
    Create a SHAP beeswarm plot.
    """
    try:
        plt.figure(figsize=(16, 10))
        shap.plots.beeswarm(
            shap_values,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        beeswarm_path = os.path.join(output_dir, f"{model_name}_shap_beeswarm.png")
        plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"📊 SHAP beeswarm plot saved for {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ SHAP beeswarm plot failed for {model_name}: {str(e)}")
        return False

def create_shap_plots(model, X_test, model_name, output_dir):
    """
    Generate SHAP explanations for a given model using a safer approach.
    """
    print(f"Generating SHAP explanations for {model_name}...")

    try:
        # Set larger figure size to prevent truncation
        plt.rcParams['figure.figsize'] = (16, 10)
        
        # Get feature names
        feature_names = X_test.columns.tolist()
        print(f"Number of features for SHAP analysis: {len(feature_names)}")
        
        # Check for potential issues
        if len(feature_names) <= 1:
            print("⚠️ ERROR: Too few features for meaningful SHAP analysis!")
            return
            
        # For tree-based models, use the TreeExplainer which is more efficient
        if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            try:
                explainer = shap.TreeExplainer(model)
                print(f"Using TreeExplainer for {model_name}")
            except Exception as e:
                print(f"TreeExplainer failed: {e}. Falling back to Explainer.")
                explainer = shap.Explainer(model, X_test)
        else:
            explainer = shap.Explainer(model, X_test)
            
        # Calculate SHAP values
        shap_values = explainer(X_test)
        
        # Create the plots using safer methods
        create_shap_summary_plot(shap_values, X_test, model_name, output_dir)
        create_manual_shap_bar_plot(shap_values, X_test, model_name, output_dir)
        create_shap_beeswarm_plot(shap_values, model_name, output_dir)

    except Exception as e:
        print(f"⚠️ SHAP analysis failed for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_constant_features(X):
    """
    Identify and report constant features which may cause issues.
    """
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"WARNING: Found {len(constant_cols)} constant columns:")
        print(constant_cols)
        print("These constant features may cause issues with SHAP plots.")

def main():
    # Load test data
    test_data_path = os.path.join("..", "data", "processed", "bmt_test.csv")
    df_test = load_data(test_data_path)
    
    # Check target column exists
    if "survival_status" not in df_test.columns:
        raise ValueError("Target column 'survival_status' not found in test data.")
    
    # Prepare features
    X_test = df_test.drop(columns=["survival_status"])
    
    # Before one-hot encoding
    print(f"Test data shape before encoding: {X_test.shape}")
    print(f"Test data columns before encoding: {X_test.columns[:5].tolist()}...")
    
    # Identify categorical columns
    cat_columns = X_test.select_dtypes(include=['object', 'category']).columns
    print(f"Found {len(cat_columns)} categorical columns")
    
    # Apply one-hot encoding
    X_test = pd.get_dummies(X_test, drop_first=True)  # Ensure numeric format
    
    # After one-hot encoding
    print(f"Test data after encoding: {X_test.shape}")
    
    # Check for constant features
    analyze_constant_features(X_test)

    # Load latest models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    models = load_latest_models(models_dir)

    if not models:
        print("No models found. Exiting.")
        return

    # Create SHAP output directory
    shap_dir = os.path.join(script_dir, "..", "shap_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    # Generate SHAP analysis for each model
    for model_name, model in models.items():
        print(f"\n🔍 Analyzing {model_name} with SHAP...")
        X_test_reindexed = reindex_features(X_test, model)
        
        # Verify reindexed data
        print(f"Reindexed data shape: {X_test_reindexed.shape}")
        if X_test_reindexed.shape[1] <= 1:
            print("⚠️ ERROR: After reindexing, test data has too few features for SHAP analysis!")
            continue
            
        create_shap_plots(model, X_test_reindexed, model_name, shap_dir)


if __name__ == "__main__":
    main()