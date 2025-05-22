import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def prepare_country_temperature_dataset(df, country_name='France', sequence_length=12, batch_size=32, train_ratio=0.8):
    """
    Prepares a normalized monthly temperature dataset for a specific country.
    Returns train/test DataLoaders and related metadata.

    Args:
        df (pd.DataFrame): The original dataset containing 'dt', 'AverageTemperature', and 'Country'.
        country_name (str): Country to filter for (default: 'France').
        sequence_length (int): Number of months used in each input sequence (default: 12).
        batch_size (int): Batch size for the DataLoaders (default: 32).
        train_ratio (float): Ratio of data to use for training (default: 0.8).

    Returns:
        dict: {
            'train_loader': DataLoader for training,
            'test_loader': DataLoader for testing,
            'temps': normalized temperature array,
            'min_temp': original minimum temperature,
            'max_temp': original maximum temperature,
            'dates': dates corresponding to Y targets,
            'X': full input tensor,
            'Y': full output tensor,
        }
    """
    df_country = df[["dt", "AverageTemperature", "Country"]].copy()
    df_country['dt'] = pd.to_datetime(df_country['dt'])

    # Filter for the selected country
    country_data = df_country[df_country['Country'] == country_name]

    # Compute monthly average temperatures
    monthly = country_data.groupby(country_data['dt'].dt.to_period('M'))['AverageTemperature'].mean().reset_index()
    monthly['dt'] = monthly['dt'].dt.to_timestamp()

    # Handle missing values by interpolation
    temps = monthly['AverageTemperature'].values
    if np.any(np.isnan(temps)):
        temps = pd.Series(temps).interpolate().values
    temps = temps.astype(np.float32)

    # Store original min and max for denormalization later
    min_temp, max_temp = temps.min(), temps.max()

    # Normalize to [0, 1]
    temps = (temps - min_temp) / (max_temp - min_temp)

    # Create input-output sequences
    X, Y, dates = [], [], []
    for i in range(len(temps) - sequence_length):
        X.append(temps[i:i+sequence_length])
        Y.append(temps[i+sequence_length])
        dates.append(monthly['dt'][i + sequence_length])

    X = torch.tensor(X).unsqueeze(-1)  # Shape: (N, seq_len, 1)
    Y = torch.tensor(Y).unsqueeze(-1).float()  # Shape: (N, 1)

    # Split into train/test datasets
    dataset = TensorDataset(X, Y)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_size': train_size,
        'temps': temps,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'dates': dates,
        'X': X,
        'Y': Y,
    }
