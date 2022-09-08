from . import utils
from typing import List, Union
from numba import njit
import pandas as pd
import numpy as np


def calculate_dollar_or_volume_bars(
        timebars: pd.DataFrame, 
        rolling_per: Union[None, int] = None, 
        nr_bars_per_day: int = 50,
        dollarbar: bool = True, 
        start_idx: int = 0
    ) -> pd.DataFrame:
    """ 
    Calculate dollar or volume bars from OHLC time bars. 
    Each bar is a sampled by dollar amount or volume instead of elapsed time.
    If dollar amount/volume has changes alot over time a rolling threshold can be used. 
    
    Parameters
    ----------
    timebars: pd.DataFrame
        A data frame containing timebars. Expects columns open, high, low, close and volume  
    rolling_per: int
        The lookback period for calcualting the threshold for each bar.
    nr_bars_per_day: int
        The size of the each bar. Uses daily average volume / 50 if none. 
    dollarbar: bool
        Samples based on dollar amount if True, else volume.
    start_idx: int
        Where to start calculating the bars. Useful if you dont want to recalculate the entire timeseries. For example if you want to add on a timeseries stored in a database.
    Returns
    -------
    bars: pd.DataFrame
        A data frame with dollar bars. 
    """
    timebars.columns = timebars.columns.str.lower()
    timebars.sort_index(inplace = True)
    if rolling_per is None: rolling_per = len(set(timebars.index.date))
    sample_series = 'dollar_value' if dollarbar else 'volume'
    
    # Adds features that can be vectorized
    timebars['dollar_value'] = timebars[['open', 'high', 'low', 'close']].mean(axis = 1) * timebars['volume']
    timebars['buy_volume'] = utils.estimated_buy_volume(timebars['open'], timebars['close'], timebars['volume']).fillna(0)
    
    #if 'bar_threshold' not in timebars.columns:
    threshold = (
        timebars[sample_series]
            .pipe(utils.rolling_daily_average, rolling_per)
            .div(nr_bars_per_day)
            .pipe(utils.round_to_nearest_exponent)
            .rename('bar_threshold')
    )
    timebars = timebars.join(threshold, on = timebars.index.date)

    # Calculates the bars with changing threshold. 
    args = timebars.loc[timebars.index[start_idx]:, ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dollar_value', 'bar_threshold', 'buy_volume']].T.values
    bars = pd.DataFrame(
        data = _calculate_bars(*args, dollarbar=dollarbar),
        columns = ['datetime', 'open', 'high', 'low', 'close', 'vwap', 'volume', '´volume_est_buy_pct', 'dollar_value', 'ticks']
    )
    bars['datetime'] = bars['datetime'].pipe(pd.to_datetime, unit = 'ms')
    return bars.set_index('datetime') if len(bars) > 0 else bars


@njit
def _calculate_bars(
        timestamps: np.array, 
        opens: np.array, 
        highs: np.array, 
        lows: np.array, 
        closes: np.array, 
        volumes: np.array, 
        dollarvalues: np.array, 
        threshholds: np.array, 
        buy_volume: np.array, 
        dollarbar: bool = True
    ) -> List[list]:
    """"
    Helper function for calculate_dollar_or_volume_bars. Wraps the looping part with numba to speed up the calcualtions. 
    """
    bar_start = counter = 0
    threshold, bars = threshholds[0], []
    sample_series = dollarvalues if dollarbar else volumes

    for idx in range(len(opens)):
        counter += sample_series[idx]
        if counter >= threshold:
            bar_volume = np.sum(volumes[bar_start:idx+1])
            bar_buy_volume = np.sum(buy_volume[bar_start:idx+1])

            bars.append([
                timestamps[idx], 
                opens[bar_start], 
                np.max(highs[bar_start:idx+1]), 
                np.min(lows[bar_start:idx+1]), 
                closes[idx], 
                np.sum(dollarvalues[bar_start:idx+1]) / bar_volume, 
                bar_volume, 
                bar_buy_volume / bar_volume,
                np.sum(dollarvalues[bar_start:idx+1]), 
                idx-bar_start
            ])
            counter, bar_start, threshold = 0, idx +1, threshholds[idx]
    return bars