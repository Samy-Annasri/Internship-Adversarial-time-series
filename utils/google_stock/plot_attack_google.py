import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_attacks_google(test_dates, true_vals, pred_vals, adv_vals, label, eps):
    true_smooth = pd.Series(true_vals).rolling(window=12).mean()
    pred_smooth = pd.Series(pred_vals).rolling(window=12).mean()
    adv_smooth = pd.Series(adv_vals).rolling(window=12).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, true_smooth, label='real rolling average', color='blue')
    plt.plot(test_dates, pred_smooth, label='prediction rolling average', color='red')
    plt.plot(test_dates, adv_smooth, label=f'{label} (eps={eps})', color='orange')

    plt.xlabel('Date')
    plt.ylabel('stock price')
    plt.title(f'stock price – {label} Attack (epsilon={eps})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()