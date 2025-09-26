"""
Option Greeks calculation module.

This module provides functionality for calculating option Greeks (delta, gamma,
theta, vega, rho) for European options using the Black-Scholes model. Option
Greeks measure the sensitivity of option prices to various underlying factors
such as the underlying asset price, time to expiration, volatility, and
interest rates.

Classes:
    OptionGreeks: Main class for calculating individual option Greeks.

The Greeks calculated are:
    - Delta: Sensitivity to underlying asset price changes
    - Gamma: Rate of change of delta
        (second derivative of price w.r.t. asset price)
    - Theta: Sensitivity to time decay
    - Vega: Sensitivity to volatility changes
    - Rho: Sensitivity to interest rate changes

Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 101, 99, 102, 98])
    >>> greeks = OptionGreeks(prices, strike_price=100, time_expiration=0.25)
    >>> delta = greeks.delta_call()
    >>> all_greeks = greeks.all_greeks()
"""

import numpy as np
from scipy.stats import norm
from parameter_estimation import ParameterEstimator
from black_scholes import BlackScholesModel


class OptionGreeks:
    """
    Calculate option Greeks using the Black-Scholes model.

    This class provides methods to calculate individual option Greeks (delta,
    gamma, theta, vega, rho) for European call and put options.
    The calculations are based on the Black-Scholes model with parameters
    estimated from historical price data.

    Parameters
    ----------
    prices : pandas.Series
        Historical price data for the underlying asset.
        Should be a time series with datetime index containing closing prices.
    strike_price : float
        Strike price of the option (K).
    time_expiration : float
        Time to expiration in years (T).
    force_of_interest : float, optional
        Risk-free interest rate, by default 0.02 (2%).

    Attributes
    ----------
    prices : pandas.Series
        Historical price data.
    strike_price : float
        Option strike price.
    time_expiration : float
        Time to expiration in years.
    force_of_interest : float
        Risk-free interest rate.
    """

    def __init__(self, prices, strike_price, time_expiration,
                 force_of_interest=0.02):
        self.prices = prices
        self.strike_price = strike_price
        self.time_expiration = time_expiration
        self.force_of_interest = force_of_interest

    def _calculate_common_variables(self):
        """
        Calculate common variables used across all Greek calculations.

        This method estimates the Black-Scholes model parameters (drift and
        volatility) from historical price data, creates a Black-Scholes model
        instance, and calculates the auxiliary variables d_plus and d_minus.

        Returns
        -------
        dict
            Dictionary containing:
                - 'mu': Estimated annualized drift parameter
                - 'sigma': Estimated annualized volatility parameter
                - 'last_price': Most recent price from the time series
                - 'd_plus': Black-Scholes auxiliary variable d_plus
                - 'd_minus': Black-Scholes auxiliary variable d_minus
        """
        estimator = ParameterEstimator()
        mu, sigma = estimator.estimate_black_scholes_parameters(
            self.prices)
        last_price = self.prices.iloc[-1].item()
        bs_model = BlackScholesModel(last_price,
                                     self.force_of_interest,
                                     sigma,
                                     self.time_expiration)
        d_plus, d_minus = bs_model.auxiliary_variables(self.strike_price)
        return {
            'mu': mu,
            'sigma': sigma,
            'last_price': last_price,
            'd_plus': d_plus,
            'd_minus': d_minus
        }

    def delta_call(self):
        """
        Calculate the delta of a European call option.

        Delta measures the sensitivity of the option price to changes in the
        underlying asset price. For a call option, delta ranges from 0 to 1.

        Formula: Δ_call = Φ(d_plus)
        Where Φ is the standard normal cumulative distribution function.

        Returns
        -------
        float
            Call option delta.
        """
        values = self._calculate_common_variables()
        d_plus = values['d_plus']
        return norm.cdf(d_plus)

    def delta_put(self):
        """
        Calculate the delta of a European put option.

        Delta measures the sensitivity of the option price to changes in the
        underlying asset price. For a put option, delta ranges from -1 to 0.

        Formula: Δ_put = Δ_call - 1

        Returns
        -------
        float
            Put option delta.
        """
        return self.delta_call() - 1

    def gamma(self):
        """
        Calculate the gamma of an option.

        Gamma measures the rate of change of delta with respect to changes in
        the underlying asset price. Gamma is the same for both call and put
        options with the same strike price and time to expiration.

        Formula: Γ = φ(d_plus) / (S * σ * √T)
        Where φ is the standard normal probability density function.

        Returns
        -------
        float
            Option gamma.
        """
        values = self._calculate_common_variables()
        sigma = values['sigma']
        last_price = values['last_price']
        d_plus = values['d_plus']
        time_expiration = self.time_expiration
        gamma = (norm.pdf(d_plus) /
                 (last_price * sigma * np.sqrt(time_expiration))
                 )
        return gamma

    def theta_call(self):
        """
        Calculate the theta of a European call option.

        Theta measures the sensitivity of the option price to the passage of
        time (time decay). For call options, theta is typically negative,
        indicating that the option loses value as time passes.

        Formula: Θ_call = -S * φ(d_plus) * σ / (2√T)
                            - r * K * e^(-rT) * Φ(d_minus)

        Returns
        -------
        float
            Call option theta.
        """
        values = self._calculate_common_variables()
        sigma = values['sigma']
        last_price = values['last_price']
        d_plus = values['d_plus']
        d_minus = values['d_minus']
        time_expiration = self.time_expiration
        force_of_interest = self.force_of_interest
        strike_price = self.strike_price

        theta_call = (-last_price * norm.pdf(d_plus) * sigma
                      / (2 * np.sqrt(time_expiration))
                      - force_of_interest * strike_price *
                      np.exp(-force_of_interest * time_expiration)
                      * norm.cdf(d_minus))
        return theta_call

    def theta_put(self):
        """
        Calculate the theta of a European put option.

        Theta measures the sensitivity of the option price to the passage of
        time (time decay). For put options, theta can be positive or negative
        depending on moneyness and time to expiration.

        Formula: Θ_put = -S * φ(d_plus) * σ / (2√T)
                            + r * K * e^(-rT) * Φ(-d_minus)

        Returns
        -------
        float
            Put option theta.
        """
        values = self._calculate_common_variables()
        sigma = values['sigma']
        last_price = values['last_price']
        d_plus = values['d_plus']
        d_minus = values['d_minus']
        time_expiration = self.time_expiration
        force_of_interest = self.force_of_interest
        strike_price = self.strike_price

        theta_put = (- last_price * norm.pdf(d_plus) * sigma
                     / (2 * np.sqrt(time_expiration))
                     + force_of_interest * strike_price *
                     np.exp(-force_of_interest * time_expiration)
                     * norm.cdf(-d_minus))
        return theta_put

    def vega(self):
        """
        Calculate the vega of an option.

        Vega measures the sensitivity of the option price to changes in the
        volatility of the underlying asset. Vega is the same for both call and
        put options with the same strike price and time to expiration, and is
        always positive.

        Formula: ν = S * φ(d_plus) * √T

        Returns
        -------
        float
            Option vega.
        """
        values = self._calculate_common_variables()
        last_price = values['last_price']
        d_plus = values['d_plus']
        time_expiration = self.time_expiration
        vega = (last_price * norm.pdf(d_plus)
                * np.sqrt(time_expiration))
        return vega

    def rho_call(self):
        """
        Calculate the rho of a European call option.

        Rho measures the sensitivity of the option price to changes in the
        risk-free interest rate. For call options, rho is typically positive.

        Formula: ρ_call = K * T * e^(-rT) * Φ(d_minus)

        Returns
        -------
        float
            Call option rho.
        """
        values = self._calculate_common_variables()
        d_minus = values['d_minus']
        time_expiration = self.time_expiration
        force_of_interest = self.force_of_interest
        strike_price = self.strike_price
        rho_call = (strike_price * time_expiration *
                    np.exp(- force_of_interest * time_expiration)
                    * norm.cdf(d_minus))
        return rho_call

    def rho_put(self):
        """
        Calculate the rho of a European put option.

        Rho measures the sensitivity of the option price to changes in the
        risk-free interest rate. For put options, rho is typically negative.

        Formula: ρ_put = -K * T * e^(-rT) * Φ(-d_minus)

        Returns
        -------
        float
            Put option rho.
        """
        values = self._calculate_common_variables()
        d_minus = values['d_minus']
        time_expiration = self.time_expiration
        force_of_interest = self.force_of_interest
        strike_price = self.strike_price
        rho_put = (-strike_price * time_expiration *
                   np.exp(- force_of_interest * time_expiration)
                   * norm.cdf(-d_minus)
                   )
        return rho_put

    def all_greeks(self):
        """
        Calculate all option Greeks at once.

        This method calculates all five Greeks
        (delta, gamma, theta, vega, rho)
        for both call and put options
        and returns them in a structured dictionary.

        Returns
        -------
        dict
            Dictionary containing all Greeks with structure:
            {
                'call': {
                    'delta': float,
                    'theta': float,
                    'rho': float
                },
                'put': {
                    'delta': float,
                    'theta': float,
                    'rho': float
                },
                'gamma': float,
                'vega': float
            }
        """

        return {
            'call':
                {'delta': self.delta_call(),
                 'theta': self.theta_call(),
                 'rho': self.rho_call()},
            'put':
                {'delta': self.delta_put(),
                 'theta': self.theta_put(),
                 'rho': self.rho_put()},
            'gamma': self.gamma(),
            'vega': self.vega()
        }
