"""
Monte Carlo option pricing module.

This module provides Monte Carlo simulation methods for pricing
European options using geometric Brownian motion.
It implements the same underlying stochastic process as the
Black-Scholes model, allowing for validation of analytical formulas through
numerical simulation.
"""

import numpy as np
from parameter_estimation import ParameterEstimator

rng = np.random.default_rng()


class MonteCarloPricer:
    """Monte Carlo option pricing"""

    def __init__(self, num_sim=100_000, force_of_interest=0.02):
        self.num_sim = num_sim
        self.force_of_interest = force_of_interest
        self.estimator = ParameterEstimator()

    def _simulate_final_prices(self, price, time_expiration, volatility):
        """
        Private method to simulate final stock prices.

        Parameters
        ----------
        price : float
            Current stock price
        volatility : float
            Volatility parameter
        time_expiration : float
            Time to expiration in years

        Returns
        -------
        numpy.ndarray
            Array of simulated final stock prices

        """
        random_shocks = rng.normal(size=self.num_sim)
        final_prices = (
            price * np.exp((self.force_of_interest - 0.5 * volatility ** 2)
                           * time_expiration
                           + volatility * np.sqrt(time_expiration)
                           * random_shocks)
        )
        return final_prices

    def price_call(self, price, strike_price, time_expiration, volatility):
        """
        Price a call option using Monte Carlo simulation.

        Parameters
        ----------
        price : float
            Current stock price
        strike_price : float
            Strike price of the option
        time_expiration : float
            Time to expiration in years
        volatility : float
            Volatility parameter

        Returns
        -------
        float
            Monte Carlo estimate of call option price
        """

        final_prices = self._simulate_final_prices(price, time_expiration,
                                                   volatility)

        # Calculate option payoffs
        call_payoffs = np.maximum(final_prices - strike_price, 0)

        # Discount to present value
        call_price_mc = np.exp(-self.force_of_interest * time_expiration) \
            * np.mean(call_payoffs)

        return call_price_mc

    def price_put(self, price, strike_price, time_expiration, volatility):
        """
        Price a put option using Monte Carlo simulation.

        Parameters
        ----------
        price : float
            Current stock price
        strike_price : float
            Strike price of the option
        time_expiration : float
            Time to expiration in years
        volatility : float
            Volatility parameter

        Returns
        -------
        float
            Monte Carlo estimate of put option price
        """
        final_prices = self._simulate_final_prices(price, time_expiration,
                                                   volatility)
        put_payoffs = np.maximum(strike_price - final_prices, 0)

        put_price_mc = np.exp(-self.force_of_interest * time_expiration) \
            * np.mean(put_payoffs)
        return put_price_mc

    def price_options_both(self, price, strike_price, time_expiration,
                           volatility):
        """
        Price both call and put options using Monte Carlo simulation.

        Parameters
        ----------
        price : float
            Current stock price
        strike_price : float
            Strike price of the option
        time_expiration : float
            Time to expiration in years
        volatility : float
            Volatility parameter

        Returns
        -------
        tuple
            (call_price, put_price)

        """
        call_price_mc = self.price_call(price, strike_price, time_expiration,
                                        volatility)
        put_price_mc = self.price_put(price, strike_price, time_expiration,
                                      volatility)
        return call_price_mc, put_price_mc

    def simulate_price_paths(self, price, volatility, time_expiration,
                             num_steps=252):
        """
        Simulate multiple price paths using geometric Brownian motion.

        Parameters
        ----------
        price : float
            Current stock price
        volatility : float
            Volatility parameter
        time_expiration : float
            Time to expiration in years
        num_steps : int
            Number of time steps in the simulation

        Returns
        -------
        numpy.ndarray
            Array of shape (num_sim, num_steps + 1) c
            ontaining simulated price paths
        """

        dt = time_expiration / num_steps

        # Initialize price paths
        paths = np.zeros((self.num_sim, num_steps + 1))
        paths[:, 0] = price

        random_shocks = rng.normal(size=(self.num_sim, num_steps))

        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.force_of_interest - 0.5 * volatility ** 2) * dt
                + volatility * np.sqrt(dt) * random_shocks[:, t-1]
            )
        return paths
