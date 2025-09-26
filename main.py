
import yfinance as yf
from config import (
    FORCE_OF_INTEREST,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_N_SPLITS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_SYMBOL,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    TIME_EXPIRATION
)
from parameter_estimation import ParameterEstimator
from black_scholes import BlackScholesModel
from greeks import OptionGreeks
from monte_carlo import MonteCarloPricer
from backtesting import ModelValidator
from visualization import FinancialPlotter


def load_market_data(symbol=DEFAULT_SYMBOL, start_date=DEFAULT_START_DATE,
                     end_date=DEFAULT_END_DATE):
    """Load market data using yfinance"""
    data = yf.download(symbol, start_date, end_date, auto_adjust=True)
    return data["Close"]


def main():

    # 1. Load data
    print("Loading market data...")
    prices = load_market_data()
    print(f"Loaded {len(prices)} days of data for {DEFAULT_SYMBOL}")

    # 2. Estimate parameters
    print("\nEstimating model parameters...")
    estimator = ParameterEstimator()
    mu, sigma = estimator.estimate_black_scholes_parameters(prices)
    print(f"Estimated drift (μ): {mu:.4f}")
    print(f"Estimated volatility (σ): {sigma:.4f}")

    # 3. Initialize Black-Scholes model
    current_price = prices.iloc[-1].item()
    bs_model = BlackScholesModel(price=current_price,
                                 force_of_interest=FORCE_OF_INTEREST,
                                 volatility=sigma,
                                 time_expiration=TIME_EXPIRATION)

    # 4. Price options
    print("\n=== Option Pricing ===")
    strikes = [current_price * 0.9, current_price * 0.95, current_price,
               current_price * 1.05, current_price * 1.1]

    for K in strikes:
        call_price = bs_model.call_option(K)
        put_price = bs_model.put_option(K)
        moneyness = K / current_price
        print(f"K=${K:.2f} ({moneyness:.1%}): Call=${call_price:.2f},"
              f"Put=${put_price:.2f}")

    # 5. Calculate Greeks
    print(f"\n=== Greeks for K=${strikes[-1]:.2f} ===")
    greek_calc = OptionGreeks(prices, strikes[-1], TIME_EXPIRATION)
    all_greeks = greek_calc.all_greeks()

    print(f"Call Delta: {all_greeks['call']['delta']:.4f}")
    print(f"Put Delta: {all_greeks['put']['delta']:.4f}")
    print(f"Gamma: {all_greeks['gamma']:.4f}")
    print(f"Vega: {all_greeks['vega']:.4f}")
    print(f"Call Theta: {all_greeks['call']['theta']:.4f}")
    print(f"Put Theta: {all_greeks['put']['theta']:.4f}")

    # 6. Monte Carlo validation
    print("\n=== Monte Carlo Validation ===")
    mc_pricer = MonteCarloPricer(num_sim=DEFAULT_NUM_SIMULATIONS)

    for K in strikes:
        analytical_call = bs_model.call_option(K)
        analytical_put = bs_model.put_option(K)

        mc_call, mc_put = mc_pricer.price_options_both(
            current_price, K, TIME_EXPIRATION, sigma)
        call_diff = abs(analytical_call - mc_call)
        put_diff = abs(analytical_put - mc_put)

        print(f"K=${K:.2f}: Call diff=${call_diff:.4f},"
              f"Put diff=${put_diff:.4f}")

    # 7. Backtesting
    print("\n=== Model Validation ===")
    validator = ModelValidator(forecast_days=DEFAULT_FORECAST_DAYS,
                               n_splits=DEFAULT_N_SPLITS)
    results = validator.backtest_model(prices, model='black_scholes')

    # Print comprehensive results
    validator.print_results(results)

    # 8. Visualization
    print("\nGenerating plots...")
    plotter = FinancialPlotter()
    r_squared = plotter.plot_backtest_results(results)
    print(f" R^2 Score:{r_squared:.3f}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
