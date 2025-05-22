import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_loader, train_size, dates, batch_size=32):
    model.eval()
    predictions = []
    true_values = []
    test_indices = []

    with torch.no_grad():
        for i, (X_test, Y_test) in enumerate(test_loader):
            output = model(X_test)
            predictions.append(output.cpu().numpy())
            true_values.append(Y_test.cpu().numpy())
            
            current_batch_size = X_test.shape[0]
            start_idx = train_size + i * batch_size
            end_idx = min(start_idx + current_batch_size, len(dates))
            test_indices.extend(range(start_idx, end_idx))
            
    predictions = np.concatenate(predictions, axis=0).squeeze()
    true_values = np.concatenate(true_values, axis=0).squeeze()
    test_dates = [dates[i] for i in test_indices]

    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)

    return {
        "predictions": predictions,
        "true_values": true_values,
        "test_dates": test_dates,
        "metrics": {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
    }

