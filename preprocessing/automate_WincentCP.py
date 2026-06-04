import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads raw dataset from file_path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocesses raw dataset.
    - Scales numerical columns.
    - One-hot encodes multiclass categorical columns.
    Returns:
        preprocessed_df: pd.DataFrame ready for training
    """
    df_processed = df.copy()
    
    # Define column types
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    
    # Scale numerical features
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # One-hot encode categorical features
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Ensure all boolean outputs from get_dummies are converted to 0/1 integers
    # since newer pandas versions use boolean
    for col in df_processed.columns:
        if df_processed[col].dtype == 'bool':
            df_processed[col] = df_processed[col].astype(int)
            
    return df_processed

def save_data(df, save_path):
    """Saves processed dataframe to save_path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Preprocessed data saved to {save_path}")

def run_pipeline(raw_path, processed_path):
    """Runs full preprocessing pipeline."""
    df_raw = load_data(raw_path)
    df_processed = preprocess_data(df_raw)
    save_data(df_processed, processed_path)
    return df_processed

if __name__ == '__main__':
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.abspath(os.path.join(script_dir, '..', 'namadataset_raw', 'heart.csv'))
    processed_path = os.path.abspath(os.path.join(script_dir, 'namadataset_preprocessing', 'heart_preprocessed.csv'))
    
    # Run pipeline
    run_pipeline(raw_path, processed_path)
