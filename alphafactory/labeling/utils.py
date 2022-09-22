from typing import Optional
import statsmodels.api as sm
import pandas as pd
import numpy as np

    
class ColNames:
    START_DT = 'date'
    END_DT = 'end_date'
    RETURN = 'return'
    SAMPLE_WEIGHT = 'sample_weight'
    LABEL = 'label'   
    PROFIT_TAKING = 'profit_taking'
    STOP_LOSS = 'stop_loss'
    VERTICAL_BARRIER = 'vertical_barrier'
    TARGET = 'target'
    TRIGGER = 'trigger'
    TVAL = 'tval'
    TVAL_ABS = 'abs_val'
    ASSETS = 'asset'
    
    
def get_1d_return(prices:pd.Series) -> pd.Series:
    """Computes the daily return at intraday estimation points."""
    idx = prices.index.searchsorted(prices.index - pd.Timedelta(days=1), side = 'right')
    idx = idx[idx > 0]

    returns = (
        prices.iloc[len(prices) - len(idx):]
      / prices.loc[prices.index[idx - 1]].values 
        - 1
    )
    return returns


def get_daily_volatility(prices: pd.Series, lookback: Optional[int]=100) -> pd.Series:
    """
    Computes the daily volatility at intraday estimation points,
    applying a span of lookback days to an exponentially weighted moving standard deviation.
    This function is used to compute dynamic thresholds for profit taking and stop loss limits.
    """
    return get_1d_return(prices).ewm(span=lookback).std()


def get_mean_returns(prices: pd.Series, lookback: Optional[int] = 100) -> pd.Series:
    return get_1d_return(prices).ewm(span = lookback).mean()


def cumsum_filter(prices: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    The CUSUM filter is a method designed to detect a shift in the
    mean value of a measured quantity away from a target value. The filter is set up to
    identify a sequence of upside or downside divergences from any reset level zero.
    We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.
    One practical aspect that makes CUSUM filters appealing is that multiple events are not
    triggered by prices hovering around a threshold level, which is a flaw suffered by popular
    market signals such as Bollinger Bands. 
    """
    
    events = []
    sum_pos = sum_neg = 0

    log_returns = (
        prices
            .apply(np.log) 
            .diff()
            .dropna()
    )

    for dt, ret in log_returns[1:].iteritems():
        sum_pos = max(0, sum_pos + ret)
        sum_neg = min(0, sum_neg + ret)

        if sum_neg < -threshold:
            sum_neg = 0
            events.append(dt)

        elif sum_pos > threshold:
            sum_pos = 0
            events.append(dt)

    event_dts = pd.DatetimeIndex(events)
    return event_dts


def volatility_based_cumsum_filter(prices: pd.Series, volatility_lookback: Optional[int] = 100) -> pd.DatetimeIndex:
    daily_volatility = get_daily_volatility(prices, lookback = volatility_lookback)
    eval_datetimes = cumsum_filter(
        prices = prices,
        threshold = daily_volatility.mean()
    )
    return eval_datetimes


def find_forward_dates(date_index: pd.DatetimeIndex, timedelta: pd.Timedelta) -> pd.Series:
    """Finds the timestamp of the next bar at or immediately after a timedelta for each index."""
    date_index = date_index.sort_values()
    idx = date_index.searchsorted(date_index + timedelta)
    idx = idx[idx < len(date_index)]
    return pd.Series(
        index = date_index[:len(idx)],
        data = date_index[idx],
        name = 'forward_dates'
    )


def calculate_sample_weights(returns: pd.Series, trade_dates:pd.Series, time_decay: Optional[float] = 1.) -> pd.Series:
    """
    Highly overlapping outcomes would have disproportionate weights if considered equal to non-overlapping outcomes.
    At the same time, labels associated with large absolute returns should be given more importance than labels with negligible absolute returns.
    This function calculates weights by uniqueness and absolute return.
    """
    co_events = numb_concurrent_events(trade_dates)
    log_returns = returns.add(1).apply(np.log)
    
    weights = pd.Series({
        start_date: (log_returns.loc[start_date:end_date] / co_events.loc[start_date:end_date]).sum() 
        for start_date, end_date in trade_dates.iteritems()
    })
    weights = weights.abs() * len(weights) / weights.abs().sum()
   
    if time_decay < 1:
        weights = apply_time_decay_to_sample_weights(weights, last_weight = time_decay)
 
    return weights


def numb_concurrent_events(trade_dates: pd.Series) -> pd.Series:
    """Calculates how many trades are part of each return contribution."""   
    co_events = pd.Series(0, index = trade_dates.index, name = 'num_co_events')
    for start_date, end_date in trade_dates.iteritems():
        co_events.loc[start_date:end_date] +=1
    return co_events


def apply_time_decay_to_sample_weights(weights: pd.Series, last_weight: Optional[float] = .5, exponent: Optional[int] = 1) -> pd.Series:
    """
    Markets are adaptive systems (Lo [2017]). As markets evolve, older examples are less relevant than the newer ones. 
    Consequently, we would typically like sample weights to decay as new observations arrive.
    """
    time_decay = (
        weights
            .sort_index()
            .cumsum()
            .rename('time_decay')
    )
    if last_weight >= 0: 
        slope = ((1. - last_weight) / time_decay.iloc[-1]) ** exponent
    else: 
        slope = 1. / ((last_weight + 1) * time_decay.iloc[-1]) ** exponent
    
    const = 1. - slope * time_decay.iloc[-1]
    time_decay = const + slope * time_decay
    time_decay[time_decay < 0] = 0
    return weights * time_decay 


def ols_reg(arr: np.array) -> float:
    x = sm.add_constant(np.arange(arr.shape[0]))
    return sm.OLS(arr, x).fit()


def calculate_return(start_price:pd.Series, end_price:pd.Series) -> pd.Series:
    return end_price / start_price - 1  