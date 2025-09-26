"""
Backtesting framework for financial models.

This module provides backtesting capabilities for financial models,
including time series cross-validation, multiple evaluation metrics,
risk measures, and support for different forecasting horizons.
It's designed to evaluate the performance of various financial models
using historical data.

Key Features:
    - Time series cross-validation to prevent data leakage
    - Comprehensive error metrics (MAE, RMSE, MAPE)
    - Performance metrics (directional accuracy, Sharpe ratio)
    - Risk metrics (VaR, max/min errors)
    - Parameter statistics and stability analysis
    - Model comparison functionality
    - Flexible forecast horizons

"""

import warnings
from typing import List, Dict
from parameter_estimation import ParameterEstimator
from config import TRADING_DAYS_PER_YEAR
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd


class ModelValidator:
    """
    Backtesting framework for financial models.

    This class provides comprehensive backtesting capabilities including
    multiple evaluation metrics, risk measures, and support for different
    forecasting horizons. It uses time series cross-validation to ensure
    proper evaluation without data leakage.

    Attributes
    ----------
    n_splits : int
        Number of time series cross-validation splits
    forecast_days : int
        Number of days ahead to forecast
    tscv : TimeSeriesSplit
        Scikit-learn time series cross-validation object
    estimator : ParameterEstimator
        Parameter estimation object for financial models

    Methods
    -------
    backtest_model(prices, model='black_scholes')
        Perform comprehensive backtesting of a financial model
    compare_models(prices, models)
        Compare multiple models using backtesting
    print_results(results)
        Print formatted backtesting results

    Example
    -------
    >>> validator = ModelValidator(forecast_days=30, n_splits=5)
    >>> results = validator.backtest_model(prices, "black_scholes")
    >>> validator.print_results(results)
    """

    def __init__(self, forecast_days, n_splits=5):
        """
        Initialize the ModelValidator.

        Parameters
        ----------
        forecast_days : int
            Number of days ahead to forecast. Must be positive.
        n_splits : int, default 5
            Number of time series cross-validation splits. Must be >= 2.

        Raises
        ------
        ValueError
            If forecast_days or n_splits are invalid

        Example
        -------
        >>> validator = ModelValidator(forecast_days=30, n_splits=5)
        """
        self.n_splits = n_splits
        self.forecast_days = forecast_days
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.estimator = ParameterEstimator()

    def backtest_model(self, prices, model: str = 'black_scholes'):
        """
        Perform comprehensive backtesting of a financial model.

        This method implements time series cross-validation to evaluate model
        performance. It estimates model parameters for each training fold and
        generates forecasts for the corresponding test periods.

        Parameters
        ----------
        prices : pd.Series
            Historical price data with datetime index. Must contain at least
            n_splits + 1 data points for proper cross-validation.
        model : str, default 'black_scholes'
            Model type to use for forecasting. Currently supports:
                - 'black_scholes': Geometric Brownian Motion model

        Returns
        -------
        Dict
        Dictionary containing comprehensive backtesting results with keys:
        - 'forecasts': np.ndarray, forecasted prices for each fold
        - 'actuals': np.ndarray, actual prices for each fold
        - 'errors': np.ndarray, forecast errors (forecasts - actuals)
        - 'percentage_errors': np.ndarray, percentage errors
        - 'mae': float, Mean Absolute Error
        - 'rmse': float, Root Mean Square Error
        - 'mape': float, Mean Absolute Percentage Error
        - 'directional_accuracy': float, percentage of correct
                                         direction predictions
        - 'sharpe_ratio': float, risk-adjusted performance measure
        - 'var_95': float, Value at Risk at 95% confidence level
        - 'max_error': float, maximum absolute error
        - 'min_error': float, minimum error
        - 'avg_mu': float, average estimated drift parameter
        - 'avg_sigma': float, average estimated volatility parameter
        - 'mu_std': float, standard deviation of drift estimates
        - 'sigma_std': float, standard deviation of volatility estimates
        - 'n_splits': int, number of cross-validation folds
        - 'forecast_days': int, forecast horizon in days

    Raises
    ------
    ValueError
        If model type is not supported or data is insufficient
    IndexError
        If prices data is too short for cross-validation

    Example
    -------
    >>> validator = ModelValidator(forecast_days=30)
    >>> results = validator.backtest_model(prices, "black_scholes")
    >>> print(f"MAE: ${results['mae']:.2f}")
    """

        forecasts, actuals, parameters = self._run_cross_validation(
            prices, model)

        results = self._calculate_metrics(forecasts, actuals, parameters)

        return results

    def _prepare_returns_data(self, prices):
        """
        Extract and prepare returns data from price series.

        Parameters
        ----------
        prices : pd.Series
            Historical price data

        Returns
        -------
        pd.Series
            Daily log returns, with NaN values removed
        """
        price_relative = prices / prices.shift(periods=1)
        daily_returns = np.log(price_relative).dropna()
        return daily_returns

    def _generate_forecast(self, last_price: float, mu: float,
                           sigma: float, model: str) -> float:
        """
        Generate price forecast based on model type.

        Parameters
        ----------
        last_price : float
            Last observed price in training data
        mu : float
            Estimated drift parameter (annualized)
        sigma : float
            Estimated volatility parameter (annualized)
        model : str
            Model type for forecasting

        Returns
        -------
        float
            Forecasted price

        Raises
        ------
        ValueError
            If model type is not supported
        """
        if model.lower() == "black_scholes":

            dt = self.forecast_days / TRADING_DAYS_PER_YEAR
            forecast = last_price * np.exp((mu - 0.5 * sigma ** 2) * dt)
        else:
            raise ValueError(f"Model '{model}' not implemented yet")

        return forecast

    def _run_cross_validation(self, prices, model):
        """
        Run time series cross-validation for model evaluation.

        Parameters
        ----------
        prices : pd.Series
            Historical price data
        daily_returns : pd.Series
            Daily log returns
        model : str
            Model type for forecasting

        Returns
        -------
        tuple
            Tuple containing (forecasts, actuals, parameters) arrays/lists
        """
        forecasts = np.zeros(self.n_splits)
        actuals = np.zeros(self.n_splits)
        parameters = []

        for fold_idx, (train_idx, test_idx) in enumerate(
                self.tscv.split(prices)):
            fold_result = self._process_fold(prices, train_idx,
                                             test_idx, model)
            forecasts[fold_idx] = fold_result['forecast']
            actuals[fold_idx] = fold_result['actual']
            parameters.append(fold_result['parameters'])

        return forecasts, actuals, parameters

    def _process_fold(self, prices, train_idx, test_idx, model):
        """
        Process a single cross-validation fold.

        Parameters
        ----------
        prices : pd.Series
            Historical price data
        train_idx : array-like
            Training data indices
        test_idx : array-like
            Test data indices
        model : str
            Model type for forecasting

        Returns
        -------
        dict
            Dictionary containing forecast, actual price, and parameters
        """
        train_data = prices.iloc[train_idx]
        test_data = prices.iloc[test_idx]

        mu, sigma = (
            self.estimator.estimate_black_scholes_parameters(train_data))

        # Get last training price and actual test price
        last_price = train_data.iloc[-1].item()
        actual_price = test_data.iloc[0].item()
        forecast = self._generate_forecast(last_price, mu, sigma, model)

        return {
            'forecast': forecast,
            'actual': actual_price,
            'parameters': {'mu': mu, 'sigma': sigma}
        }

    def _calculate_metrics(self, forecasts: np.ndarray, actuals: np.ndarray,
                           parameters: List[Dict]) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Parameters
        ----------
        forecasts : np.ndarray
            Array of forecasted prices
        actuals : np.ndarray
            Array of actual prices
        parameters : List[Dict]
            List of parameter dictionaries from each fold

        Returns
        -------
        Dict
            Dictionary containing all calculated metrics
        """
        error_metrics = self._calculate_error_metrics(forecasts, actuals)
        performance_metrics = self._calculate_performance_metrics(forecasts,
                                                                  actuals)
        risk_metrics = self._calculate_risk_metrics(forecasts, actuals)
        param_stats = self._calculate_parameter_statistics(parameters)

        return {
            'forecasts': forecasts,
            'actuals': actuals,
            'errors': error_metrics['errors'],
            'percentage_errors': error_metrics['percentage_errors'],
            **error_metrics,
            **performance_metrics,
            **risk_metrics,
            **param_stats,
            # Configuration
            'n_splits': self.n_splits,
            'forecast_days': self.forecast_days
        }

    def _calculate_error_metrics(self, forecasts, actuals):
        """
        Calculate basic error metrics.

        Parameters
        ----------
        forecasts : np.ndarray
            Forecasted prices
        actuals : np.ndarray
            Actual prices

        Returns
        -------
        Dict
            Dictionary containing error metrics
        """
        errors = forecasts - actuals
        percentage_errors = (errors / actuals) * 100
        mae = np.mean(np.abs(errors))  # Mean Absolute Error
        rmse = np.sqrt(np.mean(errors ** 2))  # Root Mean Square Error
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs(percentage_errors))
        return {
            'errors': errors,
            'percentage_errors': percentage_errors,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
        }

    def _calculate_performance_metrics(self, forecasts, actuals):
        """
        Calculate performance-related metrics.

        Parameters
        ----------
        forecasts : np.ndarray
            Forecasted prices
        actuals : np.ndarray
            Actual prices

        Returns
        -------
        Dict
            Dictionary containing performance metrics
        """
        directional_accuracy = 100 * np.mean(
            np.sign(forecasts - np.roll(actuals, 1)) ==
            np.sign(actuals - np.roll(actuals, 1))
        )

        forecast_returns = (
            (forecasts - np.roll(actuals, 1)) / np.roll(actuals, 1)
        )
        actual_returns = (actuals - np.roll(actuals, 1)) / np.roll(actuals, 1)

        excess_returns = forecast_returns - actual_returns
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)
                        if np.std(excess_returns) > 0 else 0)
        return {
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
        }

    def _calculate_risk_metrics(self, forecasts, actuals):
        """
        Calculate risk-related metrics.

        Parameters
        ----------
        forecasts : np.ndarray
            Forecasted prices
        actuals : np.ndarray
            Actual prices

        Returns
        -------
        Dict
            Dictionary containing risk metrics
        """
        errors = forecasts - actuals
        var_95 = np.percentile(errors, 5)
        return {
            'var_95': var_95,
            'max_error': np.max(np.abs(errors)),
            'min_error': np.mean(errors),
        }

    def _calculate_parameter_statistics(self, parameters):
        """
        Calculate parameter statistics across folds.

        Parameters
        ----------
        parameters : List[Dict]
            List of parameter dictionaries from each fold

        Returns
        -------
        Dict
            Dictionary containing parameter statistics
        """
        mus = [p['mu'] for p in parameters]
        sigmas = [p['sigma'] for p in parameters]
        return {
            'avg_mu': np.mean(mus),
            'avg_sigma': np.mean(sigmas),
            'mu_std': np.std(mus),
            'sigma_std': np.std(sigmas),
        }

    def compare_models(self, prices: pd.Series,
                       models: List[str]) -> Dict[str, Dict]:
        """
        Compare multiple models using backtesting.

        This method runs backtesting for each specified model and returns
        results for comparison.
        Models that fail are set to None in the results.

        Parameters
        ----------
        prices : pd.Series
            Historical price data with datetime index
        models : List[str]
            List of model names to compare. Each model name should be
            supported by the _generate_forecast method.

        Returns
        -------
        Dict[str, Dict]
            Dictionary where keys are model names and values are backtesting
            results dictionaries. Failed models have None values.

        Example
        -------
        >>> validator = ModelValidator(forecast_days=30)
        >>> models = ["black_scholes", "heston"]
        >>> results = validator.compare_models(prices, models)
        >>> for model, result in results.items():
        ...     if result is not None:
        ...         print(f"{model}: MAE = ${result['mae']:.2f}")
"""

        results = {}
        for model in models:
            try:
                results[model] = self.backtest_model(prices, model)
            except ValueError as e:
                warnings.warn(f"Model '{model}' not implemented: {e}")
                results[model] = None
            except (KeyError, IndexError, AttributeError) as e:
                warnings.warn(f"Data error testing model {model}: {e}")
                results[model] = None
        return results

    def print_results(self, results: dict) -> None:
        """
        Print formatted backtesting results.

        This method takes backtesting results and displays them in a
        well-formatted, human-readable format with sections for different
        types of metrics.

        Parameters
        ----------
        results : dict
            Backtesting results dictionary from backtest_model method.
            Must contain all required keys from the backtest_model output.

        Example
        -------
        >>> validator = ModelValidator(forecast_days=30)
        >>> results = validator.backtest_model(prices, "black_scholes")
        >>> validator.print_results(results)
        """
        print("=" * 60)
        print("BACKTESTING RESULTS")
        print("=" * 60)

        print(f"Forecast Horizon: {results['forecast_days']} days")
        print(f"Number of Folds: {results['n_splits']}")
        print()

        print("ERROR METRICS:")
        print(f"    Mean Absolute Error (MAE): ${results['mae']:.2f}")
        print(f"    Root Mean Square Error (RMSE): ${results['rmse']:.2f}")
        print(f"    Mean Absolute Percentage Error (MAPE):"
              f"{results['mape']:.2f} %")
        print()

        print("PERFORMANCE METRICS:")
        print(f"    Directional Accuracy:"
              f"{results['directional_accuracy']:.1f} %")
        print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print()

        print("RISK METRICS:")
        print(f"    Value at Risk (95%): ${results['var_95']:.2f}")
        print(f"    Maximum Error: ${results['max_error']:.2f}")
        print(f"    Minimum Error: ${results['min_error']:.2f}")
        print()

        print("PARAMETER STATISTICS:")
        print(f"    Average Drift (mu): {results['avg_mu']:.4f}")
        print(f"    Average Volatility (sigma): {results['avg_sigma']:.4f}")
        print(f"    Drift Std Dev: {results['mu_std']:.4f}")
        print(f"    Volatility Std Dev: {results['sigma_std']:.4f}")
        print("=" * 60)
