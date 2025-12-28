"""
Data collection for volatility prediction.

Tickers:
- SPY: S&P 500 ETF (for realized vol calculation)
- ^VIX: 30-day implied volatility
- ^VIX3M: 3-month implied volatility
- ^VVIX: VIX of VIX (vol of vol)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


# Data Download

TICKERS = {
    'SPY': 'S&P 500 ETF',
    '^VIX': '30-day implied vol',
    '^VIX3M': '3-month implied vol',
    '^VVIX': 'Vol of VIX',
}


def download_data(start='2010-01-01', end=None, cache=True):
    """ Download OHLC data for SPY and VIX indices.
    Daily data, 15 years, 252x15 = 3,780 rows
    - for lower time-frames need polygon.io data

    @return DataFrame with MultiIndex columns: (ticker, OHLC).
    """
    cache_path = Path(__file__).parent / 'cache.csv'

    if cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True, header=[0, 1])

    print(f"Downloading {list(TICKERS.keys())}...")
    data = yf.download(
        list(TICKERS.keys()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if cache:
        data.to_csv(cache_path)
        print(f"Cached to {cache_path}")

    return data


# Realized Volatility Estimators

def realized_vol_close(prices, window=21):
    """Close-to-close volatility (standard). Annualized using 252 trading days."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252) * 100


def realized_vol_parkinson(high, low, window=21):
    """Parkinson (1980) high-low volatility estimator. More efficient than close-to-close (uses intraday range)."""
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))
    daily_var = factor * log_hl**2
    return np.sqrt(daily_var.rolling(window).mean() * 252) * 100


def realized_vol_garman_klass(open_, high, low, close, window=21):
    """Garman-Klass (1980) OHLC volatility estimator. Most efficient estimator using all OHLC data."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    daily_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(daily_var.rolling(window).mean() * 252) * 100


def build_dataset(data, rv_window=21):
    """Build clean dataset for volatility prediction.

    Returns DataFrame with:
    - RV_21: 21-day realized vol (close-to-close)
    - RV_21_parkinson: Parkinson estimator
    - RV_21_gk: Garman-Klass estimator
    - VIX: 30-day implied vol
    - VIX3M: 3-month implied vol
    - VVIX: Vol of VIX
    - VIX_term: Term structure slope (VIX3M - VIX)
    """
    df = pd.DataFrame(index=data.index)

    # Realized vol from SPY
    spy = data['SPY'] if 'SPY' in data.columns.get_level_values(0) else data.xs('SPY', axis=1, level=1)

    # Handle both MultiIndex formats from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        spy_close = data[('Close', 'SPY')]
        spy_high = data[('High', 'SPY')]
        spy_low = data[('Low', 'SPY')]
        spy_open = data[('Open', 'SPY')]
        vix = data[('Close', '^VIX')]
        vix3m = data[('Close', '^VIX3M')]
        vvix = data[('Close', '^VVIX')]
    else:
        raise ValueError("Expected MultiIndex columns from yfinance")

    # Realized volatility estimators
    df['RV_21'] = realized_vol_close(spy_close, rv_window)
    df['RV_21_parkinson'] = realized_vol_parkinson(spy_high, spy_low, rv_window)
    df['RV_21_gk'] = realized_vol_garman_klass(spy_open, spy_high, spy_low, spy_close, rv_window)

    # Implied volatility
    df['VIX'] = vix
    df['VIX3M'] = vix3m
    df['VVIX'] = vvix

    # Derived features
    df['VIX_term'] = vix3m - vix  # Term structure slope
    df['VRP'] = vix - df['RV_21']  # Volatility risk premium

    return df.dropna()


if __name__ == '__main__':
    data = download_data()
    df = build_dataset(data)
    print(df.tail(10))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")