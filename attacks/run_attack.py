import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_manipulation import scalar_similarity,denormalize
from attacks.fgsm import fgsm_attack
from attacks.bim import bim_attack
from attacks.tca import tca_attack

# func for denormalize the data
def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def plot_rolling_attacks_temp_fr(test_dates, true_vals, pred_vals, adv_vals, label, eps):
    true_smooth = pd.Series(true_vals).rolling(window=12).mean()
    pred_smooth = pd.Series(pred_vals).rolling(window=12).mean()
    adv_smooth = pd.Series(adv_vals).rolling(window=12).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, true_smooth, label='real rolling average', color='blue')
    plt.plot(test_dates, pred_smooth, label='prediction rolling average', color='red')
    plt.plot(test_dates, adv_smooth, label=f'{label} (eps={eps})', color='orange')

    plt.xlabel('Date')
    plt.ylabel('Température (°C)')
    plt.title(f'Temperature Trend – {label} Attack (epsilon={eps})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_attack(model, loss_fn, test_loader, attack_fn, **attack_kwargs):
    model.eval()
    adv_predictions, true_values = [], []

    for X_test, Y_test in test_loader:
        X_test = X_test.clone().detach()
        Y_test = Y_test.clone().detach()
        X_adv = attack_fn(model, loss_fn, X_test, Y_test, **attack_kwargs)
        with torch.no_grad():
            output = model(X_adv)
        adv_predictions.append(output.numpy())
        true_values.append(Y_test.numpy())

    adv_predictions = np.concatenate(adv_predictions).squeeze()
    true_values = np.concatenate(true_values).squeeze()
    return adv_predictions, true_values


def run_and_log_attack(model, model_name, loss_fn, test_loader, attack_fn, attack_name, epsilon_values,
                       temps_min, temps_max, predictions, test_dates, res_tab_temp,
                       alpha=None, num_iter=None):
    predictions_denorm = denormalize(predictions, temps_min, temps_max)

    for eps in epsilon_values:
        kwargs = {'epsilon': eps}
        if alpha is not None:
            kwargs['alpha'] = alpha
        if num_iter is not None:
            kwargs['num_iter'] = num_iter

        adv_preds, true_vals = run_attack(model, loss_fn, test_loader, attack_fn, **kwargs)
        adv_preds_denorm = denormalize(adv_preds, temps_min, temps_max)
        true_vals_denorm = denormalize(true_vals, temps_min, temps_max)

        mae = mean_absolute_error(true_vals_denorm, adv_preds_denorm)
        sim = scalar_similarity(true_vals_denorm, adv_preds_denorm)
        rsme = np.sqrt(mean_squared_error(true_vals_denorm, adv_preds_denorm))

        eps_str = f"{eps:.2f}"
        res_tab_temp.loc[(model_name, 'MAE'), (attack_name, eps_str)] = mae
        res_tab_temp.loc[(model_name, 'SIM'), (attack_name, eps_str)] = sim
        res_tab_temp.loc[(model_name, 'RSME'), (attack_name, eps_str)] = rsme

        print(f"{model_name} | {attack_name} – Epsilon {eps} – Adversarial MAE: {mae:.4f} | SIM: {sim:.4f}")
        plot_rolling_attacks_temp_fr(test_dates, true_vals_denorm, predictions_denorm, adv_preds_denorm,
                                     label=f"{attack_name} ({model_name})", eps=eps)

def run_and_log_attack_google(
    model,
    model_name,
    loss_fn,
    test_loader,
    test_dates,
    predictions_denorm,
    true_values_denorm,
    price_min,
    price_max,
    res_tab,
    epsilon_values=[0.01, 0.1, 0.2],
    colors=None
):
    """
    Runs adversarial attacks (FGSM, BIM, TCA) on a model and plots predictions.

    Args:
        model (torch.nn.Module): The trained model (LSTM, RNN, or CNN).
        model_name (str): The name to use in the result table (e.g., 'LSTM').
        loss_fn: The loss function used (e.g., torch.nn.MSELoss()).
        test_loader: DataLoader for test set.
        test_dates (list): Dates corresponding to predictions.
        predictions_denorm (np.array): Original predictions (denormalized).
        true_values_denorm (np.array): Ground truth values (denormalized).
        price_min (float): Minimum value of the target feature.
        price_max (float): Maximum value of the target feature.
        res_tab (pd.DataFrame): DataFrame to store MAE, SIM, RMSE results.
        epsilon_values (list): List of epsilons to use for the attacks.
        colors (dict): Mapping from attack name to color for plotting.

    Returns:
        None
    """
    if colors is None:
        colors = {
            "FGSM": "red",
            "BIM": "blue",
            "TCA": "orange"
        }

    for eps in epsilon_values:
        alpha = eps / 10
        iters = 10 if eps < 0.2 else 20

        methods = {
            "FGSM": lambda eps: run_attack(model, loss_fn, test_loader, fgsm_attack, epsilon=eps),
            "BIM":  lambda eps: run_attack(model, loss_fn, test_loader, bim_attack, epsilon=eps, alpha=alpha, num_iter=iters),
            "TCA":  lambda eps: run_attack(model, loss_fn, test_loader, tca_attack, epsilon=eps),
        }

        plt.figure(figsize=(12, 5))
        plt.title(f"Adversarial Forecasting – {model_name} – Epsilon = {eps}")

        plt.plot(test_dates, predictions_denorm, label="Original Prediction", linestyle="--", color="black")
        plt.plot(test_dates, true_values_denorm, label="Ground Truth", color="green")

        for method_name, attack_fn in methods.items():
            adv_preds, true_vals = attack_fn(eps)
            adv_preds_denorm = denormalize(adv_preds, price_min, price_max)

            mae = mean_absolute_error(true_values_denorm, adv_preds_denorm)
            sim = scalar_similarity(true_values_denorm, adv_preds_denorm)
            rsme = np.sqrt(mean_squared_error(true_values_denorm, adv_preds_denorm))

            print(f"{method_name} – Epsilon {eps:.2f} – MAE: {mae:.4f} | SIM: {sim:.4f} | RSME: {rsme:.4f}")

            res_tab.loc[(model_name, 'MAE'), (method_name, f"{eps:.2f}")] = mae
            res_tab.loc[(model_name, 'SIM'), (method_name, f"{eps:.2f}")] = sim
            res_tab.loc[(model_name, 'RSME'), (method_name, f"{eps:.2f}")] = rsme

            plt.plot(test_dates, adv_preds_denorm, label=method_name, linewidth=2, color=colors.get(method_name, 'gray'))

        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

