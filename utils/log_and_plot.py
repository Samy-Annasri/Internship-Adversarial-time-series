import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def log_and_plot_predictions(true_values, predictions, test_dates,
                             true_values_rolling, predictions_rolling,
                             model_name, attack_name, epsilon,
                             res_tab, similarity_fn,google=False):
    """
    Calcule les métriques, les enregistre dans res_tab, et trace les courbes.
    """

    # Dénormalisation déjà faite en amont
    mae = mean_absolute_error(true_values, predictions)
    sim = similarity_fn(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))

    eps_str = f"{epsilon:.2f}"

    # Log des résultats dans la table
    res_tab.loc[(model_name, 'MAE'), (attack_name, eps_str)] = mae
    res_tab.loc[(model_name, 'SIM'), (attack_name, eps_str)] = sim
    res_tab.loc[(model_name, 'RSME'), (attack_name, eps_str)] = rmse

    # Affichage console
    print(f"{model_name} | {attack_name} – Epsilon {eps_str} – MAE: {mae:.4f} | SIM: {sim:.4f}")

    # Plot
    if google == False:
      plt.figure(figsize=(12, 6))
      plt.plot(test_dates, true_values_rolling, label='real rolling average', color='blue')
      plt.plot(test_dates, predictions_rolling, label='predict rolling average', color='red')
      plt.xlabel('Date')
      plt.ylabel('Température (°C)')
      plt.title(f'Température en France – {model_name} ({attack_name}, eps={eps_str})')
      plt.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()
    else:
      plt.figure(figsize=(10,6))
      plt.plot(test_dates, true_values, label='Real Values', color='blue')
      plt.plot(test_dates, predictions, label='Predictions', color='red', linestyle='--')
      plt.xlabel('Date')
      plt.ylabel('Stock Price')
      plt.title('Real vs Predicted Stock Prices')
      plt.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()
