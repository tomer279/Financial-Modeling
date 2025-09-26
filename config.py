"""
Configuration settings for the financial modeling project.
"""

# Market parameters
FORCE_OF_INTEREST = 0.02
TRADING_DAYS_PER_YEAR = 252
TIME_EXPIRATION = 0.25

# Model parameters
DEFAULT_NUM_SIMULATIONS = 100_000
DEFAULT_N_SPLITS = 5
DEFAULT_FORECAST_DAYS = 30

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-01-01"
DEFAULT_SYMBOL = "AAPL"
