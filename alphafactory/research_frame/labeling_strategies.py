from __future__ import annotations
from .utils import (
    get_daily_volatility, get_mean_returns, calculate_return, 
    apply_time_decay_to_sample_weights,calculate_sample_weights,
    ols_reg, find_forward_dates, volatility_based_cumsum_filter
)
from typing import Optional, Dict, List
from collections.abc import Iterable 
from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime as dt 
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
    
    
@dataclass 
class LabelGenerator(ABC):
    
    """ 
    A base class for calculating forwards returns, classification labels and  
    sample weights for a price series. Diffrent implementations have diffrent
    ways of determening when a trade is ended/closed. 
    
    Parameters
    ----------
    prices : pd.Sereis
        A series of prices indexed with datetime. 
    """
    
    prices: pd.Series
    
    @abstractmethod
    def _find_end_dates(self) -> pd.DataFrame:
        """Finds when each trade is ended."""
        pass  
    
    def __post_init__(self) -> None:
        self.prices = (
            self.prices
                .squeeze()
                .sort_index()
        )
        
    def _calculate_forward_returns(self) -> pd.DataFrame:
        """Calculate the return based on the end date."""
        forward_returns = self._find_end_dates()
        forward_returns[ColNames.RETURN] = calculate_return(
            start_price = self.prices.loc[forward_returns.index],
            end_price = self.prices.loc[forward_returns[ColNames.END_DT].values].values
        ).replace([np.inf, -np.inf], np.nan)
        return forward_returns
    
    def create_labels(self) -> pd.DataFrame:
        labels = self._calculate_forward_returns()
        labels[ColNames.LABEL] = np.sign(labels[ColNames.RETURN])
        return labels
    
    def calculate_sample_weights(self, labels: pd.DataFrame, time_decay: float = 75.) -> pd.DataFrame:
        """
        Calculates and adds sample weights base on uniqness and returns of each observation. 
        If time decay is less than 1 more recent older observations sample weight 
        will be reduced to put more emphasis on recent observations. 
        
        Parameters
        ----------
        labels : pd.DataFrame
            A dataframe containg end dates and forward returns. 
        time_decay : float, optional
            A factor for putting more weight on recent observations. 
            The number represents the scaling factor on the oldest observation. 

        Returns
        -------
        labels : pd.DataFram
            Returns the original dataframe with and additional column 
            containing sample weights. 

        """
        labels[ColNames.SAMPLE_WEIGHT] = calculate_sample_weights(
            labels[ColNames.RETURN], 
            labels[ColNames.END_DT], 
            time_decay
        )
        return labels
    
    
@dataclass 
class FixedTimeLabels(LabelGenerator):
    """
    Calculates forward returns using a fixed time horizon.
    
    Parameters
    ----------
        max_holding_period: pd.Timedelta
            A timedelta object representing how long we are willing to hold each trade.
    """
    
    max_holding_period: pd.Timedelta
    eval_datetimes: Optional[pd.DatetimeIndex] = None
    
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eval_datetimes is None:    
            self.eval_datetimes = self.prices.index 

    def _find_end_dates(self) -> pd.DataFrame:
        return self._find_vertical_barriers(ColNames.END_DT) 
    
    def _find_vertical_barriers(self, col_name:str) -> pd.DataFrame:
        """
        Finds the timestamp of the next price bar at or immediately after a number of days holding_days for each index in df.
        This function creates a series that has all the timestamps of when the vertical barrier is reached.
        """
        return (
            find_forward_dates(self.prices.index, self.max_holding_period)
                .pipe(lambda df: df.loc[df.index.intersection(self.eval_datetimes)])
                .to_frame(col_name)    
        )
        
    
@dataclass 
class TrippleBarrierLabels(FixedTimeLabels):
    """"
    Calculates forward returns using the tripple barrier method. A trade is closed in one of three ways.
    1) A stop loss level is triggered 2) a profit taking level is hit or 3) a maximum amount of time has expiered.
    
    Parameters
    ----------
        prices: pd.Series
            A pandas series with prices, indexed with datetime.
        max_holding_period: pd.Timedelta
            A timedelta object representing how long we are willing to hold each trade.
        barrier_width: float
            A manual scaling factor for profit taking and stop loss targets expressed in standard deviations.
            A higher number means higher profit taking targets and lower stop losses.
        eval_datetimes: pd.DatetimeIndex
            Allows for runing the calculations on a subset only for efficency.
            For example we might want to use some filtering strategy on our observations to reduce noise.  
        volatility_lookback: int
            A lookback window that is used to calculate rolling volatility for target setting.
        
    """
    
    barrier_width: Optional[float] = 1.
    volatility_lookback: Optional[int] = 100
        
    def _find_end_dates(self) -> pd.DataFrame:
        target_returns = self._calculate_target_returns()
        vertical_barriers = self._find_vertical_barriers(ColNames.VERTICAL_BARRIER) 
        horizontal_barriers = self._find_horizontal_barriers(vertical_barriers, target_returns)
        return self._find_first_triggered_barrier(vertical_barriers, horizontal_barriers)

    def _calculate_target_returns(self) -> pd.Series:
        """
        Calculates the target levels for profit taking and stop loss at each timestamp.
        Uses volatility and a manual scaling factor (barrier_width).
        """
        target_returns = (
            get_daily_volatility(self.prices, lookback = self.volatility_lookback)
                .mul(self.barrier_width)
                .add(get_mean_returns(self.prices, lookback = self.volatility_lookback))
                .pipe(lambda df: df.loc[df.index.isin(self.eval_datetimes)])
                .rename(ColNames.TARGET)  
        )
        return target_returns
       
    def _find_horizontal_barriers(self, vertical_barriers: pd.Series, target_returns: pd.Series) -> pd.DataFrame:       
        """
        Finds first time between start date and end date that the stop loss and profit taking target is hit.

        Parameters
        ----------
        vertical_barriers : pd.Series
            A series contating the date of the vertical barrier/max holding period.
        target_returns : pd.Series
            A series containing the target levels for stopp loss and profit taking for each date. 

        Returns
        -------
        horizontal_barriers: pd.DataFrame
            A dataframe containing the dates of when the stopp loss and proftit taking targets was hit
            between the start date and vertical barriers.
            
        """

        def _find_horizontal_barrier_dates(prices:pd.Series, start_date:dt.datetime, end_date:dt.datetime, target:float) -> Dict[str, float]:
            """" Helper function """
            price_path = prices[start_date:end_date] 
            returns = calculate_return(price_path, price_path[start_date])
            return {
                ColNames.START_DT: start_date,
                ColNames.STOP_LOSS: returns[returns < target*-1].index.min(),
                ColNames.PROFIT_TAKING: returns[returns > target].index.min()
            }
       
        data = (
            pd.concat([vertical_barriers, target_returns], axis = 1)
                .dropna(subset = [ColNames.TARGET])
        )
        horizontal_barriers = pd.DataFrame([
            _find_horizontal_barrier_dates(
                prices = self.prices, 
                start_date = row.Index, 
                end_date = getattr(row, ColNames.VERTICAL_BARRIER),
                target = getattr(row, ColNames.TARGET))
            for row in data.itertuples()
        ]).set_index(ColNames.START_DT)
        return horizontal_barriers
   
    def _find_first_triggered_barrier(self, vertical_barriers: pd.DataFrame, horizontal_barrier: pd.DataFrame) -> pd.Series:
        """Finds which of the stop loss, profit taking or max holding periods that are triggered first."""
        barriers = (
            pd.concat([vertical_barriers, horizontal_barrier], axis = 1)
                .dropna(how='all')
                .agg(['min','idxmin'], axis=1)
                .rename(columns = {'min': ColNames.END_DT, 'idxmin': ColNames.TRIGGER})
        )
        self.barrier_distribution = barriers[ColNames.TRIGGER].value_counts()
        return barriers[[ColNames.END_DT]]

        
@dataclass 
class TrendScaningLabels(LabelGenerator):
    
    """ Calculates labels based on the trend from an OLS in a given forward window """
    
    holding_periods: List[pd.Timdelta]
    eval_datetimes: Optional[pd.DatetimeIndex] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.holding_periods, Iterable):
            self.holding_periods = [self.holding_periods]
            
        if self.eval_datetimes is None:    
            self.eval_datetimes = self.prices.index 
            
    def _find_end_dates(self) -> pd.DataFrame:
        data = self._multi_period_trends()
        data[ColNames.TVAL_ABS] =  data[ColNames.TVAL].abs()
        max_idx = (
            data
                .groupby([ColNames.START_DT])
                .max()
                .set_index([ColNames.TVAL_ABS], append = True)
                .index
        )
        data = ( 
            data
                .set_index([ColNames.TVAL_ABS], append = True)
                .loc[max_idx]
                .reset_index(ColNames.TVAL_ABS, drop = True)
        )
        return data
    
    def _multi_period_trends(self):
        return pd.concat([
            self._calculate_trend(holding_period) 
            for holding_period in self.holding_periods
        ])
    
    def _calculate_trend(self, time_period: pd.Timedelta) -> pd.DataFrame:
        forward_dates = (
            find_forward_dates(self.prices.index, time_period)
                .pipe(lambda df: df.loc[df.index.isin(self.eval_datetimes)])
        )
        trend = pd.DataFrame([
            {ColNames.START_DT: start_dt, 
             ColNames.END_DT: end_dt, 
             ColNames.TVAL: ols_reg(self.prices.loc[start_dt:end_dt]).tvalues[1]}
             for start_dt, end_dt in forward_dates.iteritems()
        ]).set_index(ColNames.START_DT)
        return trend

    def create_labels(self) -> pd.DataFrame:
        labels = self._calculate_forward_returns()
        labels[ColNames.LABEL] = np.sign(labels[ColNames.TVAL])
        return labels
    
    def calculate_sample_weights(self, labels: pd.DataFrame = None, time_decay: float = 1.) -> pd.DataFrame:
        labels = labels.rename(columns = {ColNames.TVAL: ColNames.SAMPLE_WEIGHT})
        if time_decay < 1:
            labels[ColNames.SAMPLE_WEIGHT] = apply_time_decay_to_sample_weights(labels[ColNames.SAMPLE_WEIGHT], time_decay)
        return labels
    
def filtered_tripple_barrier_labels(
        prices: pd.Series, 
        max_holding_period: pd.Timedelta,
        barrier_width: float, 
        volatility_lookback: int
    ) -> TrippleBarrierLabels:    
    eval_datetimes = volatility_based_cumsum_filter(prices.squeeze(), volatility_lookback)
    return TrippleBarrierLabels(**locals())
    
def filtered_trend_scaning_labels(
        prices: pd.Series, 
        holding_periods: List[pd.Timdelta] 
    ) -> TrendScaningLabels:    
    eval_datetimes = volatility_based_cumsum_filter(prices.squeeze())
    return TrendScaningLabels(**locals())
