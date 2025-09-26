"""
Visualization utilites for financial analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class FinancialPlotter:

    def __init__(self, style=None, figsize=(10, 6)):
        sns.set_style("whitegrid")
        self.figsize = figsize

    def plot_backtest_results(self, results):
        """Visualize your backtesting results"""

        required_keys = ['forecasts', 'actuals', 'n_splits']
        for key in required_keys:
            if key not in results:
                raise KeyError(f"Missing required key '{key}' in results")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Scatter plot - Forecasts vs Actuals
        ax1.scatter(results['actuals'], results['forecasts'], alpha=0.7, s=50)
        ax1.plot([min(results['actuals']), max(results['actuals'])],
                 [min(results['actuals']), max(results['actuals'])],
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Prices ($)')
        ax1.set_ylabel('Forecasted Prices ($)')
        ax1.set_title('Black-Scholes Model: Forecasts vs Actual Prices')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add R² score
        correlation = np.corrcoef(
            results['actuals'], results['forecasts'])[0, 1]
        r_squared = correlation ** 2
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="white",
                           alpha=0.8))

        # Plot 2: Time series of predictions
        fold_numbers = range(1, results['n_splits'] + 1)
        ax2.plot(fold_numbers, results['actuals'],
                 'o-', label='Actual', linewidth=2, markersize=6)
        ax2.plot(fold_numbers, results['forecasts'],
                 's-', label='Forecast', linewidth=2, markersize=6)

        # Add error bars
        errors = results['forecasts'] - results['actuals']
        ax2.errorbar(fold_numbers, results['actuals'], yerr=np.abs(errors),
                     fmt='none', alpha=0.5, color='gray')

        ax2.set_xlabel('Fold Number')
        ax2.set_ylabel('Price ($)')
        ax2.set_title('Time Series: Black-Scholes Forecasts vs Actuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return r_squared
