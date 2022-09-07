import numpy as np
import pandas as pd
from scipy import special

def round_to_nearest_x(arr:np.array, x:int) -> np.array:
    """ Rounds an array or series """
    div = arr / x
    arr = (div).astype(int) if isinstance(arr, pd.Series) else np.int64(div)
    return arr * x

def round_to_nearest_exponent(ser:pd.Series) -> pd.Series:
    """ Rounds a series to closes 10 exponent """
    x = (10**ser.apply(np.log10).round(0).sub(1)).clip(lower = 1)
    return (ser / x).astype(int) * x

def daily_average(ser:pd.Series) -> float:
    """ Calculates the daily average of a pandas series with datetime index"""
    return ser.groupby(ser.index.date).sum().mean()

def rolling_daily_average(ser:pd.Series, per: int = 365) -> pd.Series:
    """ Cacltulates a rolling daily average """
    grp = ser.groupby(ser.index.date)
    return grp.sum().rolling(per, min_periods = min(per, len(grp))).mean().bfill()

def estimated_buy_volume(price_open:pd.Series, price_close:pd.Series, volume:pd.Series, std: pd.Series = None) -> pd.Series:
    """ Estimates buy volume of a bar. """
    price_diff = price_close - price_open
    if std is None:
        std = (
            price_diff
                .expanding()
                .std()
                .bfill()
        )
    buy_volume = (
        price_diff
            .div(std)
            .apply(special.ndtr)
            .mul(volume)
    )    
    return buy_volume 