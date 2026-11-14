"""
Data Loading and Preprocessing for Phase 2 Autoencoder
Clean, beginner-friendly data loader.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_and_preprocess_data(data_dir='data/processed', models_dir='results/models'):
    """
    Loads train, validation, and test datasets.
    
    1. Loads the 29 features chosen in Phase 1
    2. Applies log1p to 'Amount' to tame extreme spending spikes
    3. Reuses Phase 1's StandardScaler (Zero Data Leakage!)
    4. Returns clean NumPy arrays ready for PyTorch
    """
    # Step 1: Load feature names and Phase 1 scaler
    features_path = os.path.join(models_dir, 'gmm_features.joblib')
    scaler_path   = os.path.join(models_dir, 'gmm_scaler.joblib')
    
    features = joblib.load(features_path)
    scaler   = joblib.load(scaler_path)
    
    # Step 2: Load the CSV files
    train_df = pd.read_csv(os.path.join(data_dir, 'train_legit.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_mixed.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_mixed.csv'))
    
    # Step 3: Prepare the feature matrices
    def clean_features(df):
        X = df[features].copy()
        X['Amount'] = np.log1p(X['Amount'])  # log(1 + Amount)
        return scaler.transform(X)           # Apply Z-score scaling
        
    X_train = clean_features(train_df)
    X_val   = clean_features(val_df)
    X_test  = clean_features(test_df)
    
    # Extract fraud labels (0 = normal, 1 = fraud)
    y_val  = val_df['Class'].values
    y_test = test_df['Class'].values
    
    return {
        'X_train': X_train,
        'X_val':   X_val,
        'X_test':  X_test,
        'y_val':   y_val,
        'y_test':  y_test,
        'features': features,
        'scaler':  scaler
    }


def get_dataloaders(X_train, X_val, batch_size=256):
    """
    Wraps NumPy arrays into PyTorch DataLoaders.
    """
    # Convert numpy arrays to PyTorch float tensors
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor   = torch.tensor(X_val, dtype=torch.float32)
    
    # In an Autoencoder, target is the input itself!
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset   = TensorDataset(val_tensor, val_tensor)
    
    # Training loader: shuffles data every epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation loader: no need to shuffle
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    return train_loader, val_loader
