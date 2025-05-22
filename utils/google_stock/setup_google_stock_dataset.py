import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def prepare_stock_dataset(
    df,
    feature_cols=['Close/Last', 'Open', 'High', 'Low', 'Volume'],
    target_col='Close/Last',
    date_col='Date',
    sequence_length=30,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15
):
    """
    Prepares a stock price dataset with normalization and sequence formatting for time series forecasting.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw stock price data.
        feature_cols (list): List of feature column names.
        target_col (str): The column to predict (must be in feature_cols).
        date_col (str): The column containing dates.
        sequence_length (int): Number of time steps in each input sequence.
        batch_size (int): Batch size for the DataLoaders.
        train_ratio (float): Ratio of samples for the training set.
        val_ratio (float): Ratio of samples for the validation set.

    Returns:
        dict: {
            'train_loader': DataLoader,
            'val_loader': DataLoader,
            'test_loader': DataLoader,
            'min_max': {column_name: (min, max)},
            'dates': corresponding dates for targets,
            'X': full input tensor,
            'Y': full output tensor,
        }
    """
    df = df.copy()

    # Convert price columns to float (remove $ symbols if present)
    for col in ['Close/Last', 'Open', 'High', 'Low']:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Store original min-max for each feature (for potential denormalization)
    min_max = {}
    for col in feature_cols:
        min_val, max_val = df[col].min(), df[col].max()
        min_max[col] = (min_val, max_val)
        df[col] = (df[col] - min_val) / (max_val - min_val)

    # Extract normalized data
    data = df[feature_cols].values.astype(np.float32)

    # Create sequences and targets
    X, Y, dates = [], [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(data[i+sequence_length][feature_cols.index(target_col)])
        dates.append(df[date_col].iloc[i + sequence_length])

    X = torch.tensor(X)  # Shape: (N, sequence_length, num_features)
    Y = torch.tensor(Y).unsqueeze(-1)  # Shape: (N, 1)

    # Split dataset
    dataset = TensorDataset(X, Y)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_size': train_size,
        'val_size': val_size,
        'min_max': min_max,
        'dates': dates,
        'X': X,
        'Y': Y,
    }
