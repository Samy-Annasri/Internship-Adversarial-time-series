import torch
import numpy as np
def evaluate_model_google(model, test_loader, dates, train_size, val_size):
    """
    Evaluates a trained model on the test set and returns real vs predicted values with dates.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        dates (list): Full list of dates (same order as the full dataset).
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.

    Returns:
        dict: {
            'real': np.array of real values,
            'predicted': np.array of predicted values,
            'dates': list of dates corresponding to predictions
        }
    """
    model.eval()
    real_values = []
    predicted_values = []
    test_indices = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            real_values.append(labels.cpu().numpy())
            predicted_values.append(outputs.cpu().numpy())

            batch_size = inputs.shape[0]
            start_idx = train_size + val_size + i * test_loader.batch_size
            end_idx = min(start_idx + batch_size, len(dates))
            test_indices.extend(range(start_idx, end_idx))

    real_values = np.concatenate(real_values).squeeze()
    predicted_values = np.concatenate(predicted_values).squeeze()
    test_dates = [dates[i] for i in test_indices]

    return {
        'real_values': real_values,
        'predicted_values': predicted_values,
        'test_dates': test_dates
    }
