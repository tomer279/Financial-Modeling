# Financial Modeling Toolkit

**Note**: This project was partially developed with assistance from Cursor.ai, an AI-powered code editor.

A comprehensive Python toolkit for financial modeling, option pricing, and backtesting using the Black-Scholes model and Monte Carlo simulations.

## Features

- **Black-Scholes Option Pricing**: European call and put option pricing
- **Option Greeks**: Delta, Gamma, Theta, Vega, and Rho calculations
- **Monte Carlo Simulations**: Numerical validation of analytical models
- **Parameter Estimation**: Drift and volatility estimation from historical data
- **Backtesting Framework**: Time series cross-validation for model evaluation
- **Visualization**: Comprehensive plotting and analysis tools

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from main import main

# Run the complete analysis pipeline
main()
```

## Usage

### Basic Option Pricing

```python
from black_scholes import BlackScholesModel

# Initialize model
bs_model = BlackScholesModel(
    price=150.0,           # Current stock price
    force_of_interest=0.02, # Risk-free rate
    volatility=0.25,       # Volatility
    time_expiration=0.25   # Time to expiration (years)
)

# Price options
call_price = bs_model.call_option(150.0)  # At-the-money call
put_price = bs_model.put_option(150.0)    # At-the-money put
```

### Option Greeks

```python
from greeks import OptionGreeks

greek_calc = OptionGreeks(prices, strike=150.0, time_expiration=0.25)
all_greeks = greek_calc.all_greeks()

print(f"Delta: {all_greeks['call']['delta']:.4f}")
print(f"Gamma: {all_greeks['gamma']:.4f}")
print(f"Vega: {all_greeks['vega']:.4f}")
```

### Monte Carlo Validation

```python
from monte_carlo import MonteCarloPricer

mc_pricer = MonteCarloPricer(num_sim=100000)
mc_call, mc_put = mc_pricer.price_options_both(
    current_price=150.0,
    strike=150.0,
    time_expiration=0.25,
    volatility=0.25
)
```

### Backtesting

```python
from backtesting import ModelValidator

validator = ModelValidator(forecast_days=30, n_splits=5)
results = validator.backtest_model(prices, model='black_scholes')
validator.print_results(results)
```

## Project Structure
├── main.py # Main execution script

├── config.py # Configuration settings

├── black_scholes.py # Black-Scholes model implementation

├── greeks.py # Option Greeks calculations

├── monte_carlo.py # Monte Carlo simulations

├── parameter_estimation.py # Parameter estimation methods

├── backtesting.py # Backtesting framework

├── visualization.py # Plotting and visualization

└── requirements.txt # Dependencies

## Configuration

Key parameters can be adjusted in `config.py`:

- `DEFAULT_SYMBOL`: Stock symbol for analysis (default: "AAPL")
- `DEFAULT_START_DATE`: Start date for data (default: "2020-01-01")
- `DEFAULT_END_DATE`: End date for data (default: "2024-01-01")
- `DEFAULT_FORECAST_DAYS`: Forecast horizon (default: 30)
- `DEFAULT_N_SPLITS`: Cross-validation splits (default: 5)

## Dependencies

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- yfinance

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer


This software is for educational and research purposes only. It should not be used for actual trading decisions without proper risk management and professional financial advice.

