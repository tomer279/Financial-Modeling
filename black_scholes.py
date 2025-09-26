"""
Black-Scholes option pricing models and core SDE implementations.

This module provides implementations of the Black-Scholes model for pricing
European call and put options. The Black-Scholes model assumes that stock
prices follow a geometric Brownian motion with constant volatility and
risk-free interest rate.

Classes:
    BlackScholesModel:
        Main class for Black-Scholes option pricing calculations.
"""
from typing import Tuple
import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    """Black-Scholes option pricing model"""

    def __init__(self, price: float, force_of_interest: float,
                 volatility: float, time_expiration: float):
        """
        Initialize Black-Scholes model parameters.

        Parameters
        ----------
        price : float
            Current stock price (S_0)
        force_of_interest : float
            Risk-free interest rate (r)
        volatility : float
            Stock price volatility (sigma)
        time_expiration : float
            Time to expiration in years (T)
        """
        self.price = price
        self.force_of_interest = force_of_interest
        self.volatility = volatility
        self.time_expiration = time_expiration

    def call_option(self, strike_price: float) -> float:
        """
        Calculate Black-Scholes price for European call option.

        Uses the formula:
            C = S_0 * Phi(d_plus) - K * e^(-rT) * Phi(d_minus)
        Where Phi is the standard normal CDF.

        Parameters
        ----------
        strike_price : float
            Strike price of the option (K)

        Returns
        -------
        call_price : float
            Call option price

        """
        d_plus, d_minus = self.auxiliary_variables(strike_price)
        call_price = (norm.cdf(d_plus) * self.price
                      - norm.cdf(d_minus) * strike_price * np.exp(
                          -self.force_of_interest * self.time_expiration))
        return call_price

    def put_option(self, strike_price: float) -> float:
        """
        Calculate Black-Scholes price for European put option.

        Uses the formula:
            Phi(-d_minus) * K e^(-rT) - N(-d_plus) * S_0
        Where Phi is the standard normal CDF.

        Parameters
        ----------
        strike_price : float
            Strike price of the option (K)

        Returns
        -------
        put_price : float
            Put option price
        """
        d_plus, d_minus = self.auxiliary_variables(strike_price)
        put_price = (strike_price * np.exp(-self.force_of_interest
                                           * self.time_expiration)
                     * norm.cdf(-d_minus) - self.price * norm.cdf(-d_plus)
                     )
        return put_price

    def auxiliary_variables(self, strike_price: float) -> Tuple[float, float]:
        """
        Calculate Black-Scholes auxiliary variables d_plus and d_minus

        These are the standardized variables used in the Black-Scholes formula:
            d_plus = [ln(S/K) + (r + sig^2 / 2)T] / (sig * sqrt(T))
            d_minus = d_plus - sig * sqrt(T)

        Parameters
        ----------
        strike_price : float
            Strike price of the option (K)

        Returns
        -------
        d_plus : float

        d_minus : float
        """
        price = self.price
        force_of_interest = self.force_of_interest
        volatility = self.volatility
        time_expiration = self.time_expiration

        d_plus = (np.log(price / strike_price)
                  + (force_of_interest + 0.5 * volatility ** 2)
                  * time_expiration) / (volatility * np.sqrt(time_expiration))
        d_minus = d_plus - volatility * np.sqrt(time_expiration)
        return d_plus, d_minus
