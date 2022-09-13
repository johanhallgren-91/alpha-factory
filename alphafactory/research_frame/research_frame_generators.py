from __future__ import annotations
from .utils import (
    get_daily_volatility, calculate_return, find_forward_dates, ols_reg,
    volatility_based_cumsum_filter, apply_time_decay
)
from .research_frame import ResearchFrame, ColNames
from typing import Optional, Dict, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime as dt 
import pandas as pd
import numpy as np

    
class ResearchFrameGenerator(ABC):
    
    """ An interface for calculating forward returns with diffrent strategies. """
    
    @abstractmethod
    def _find_end_dates(self) -> ResearchFrameGenerator:
        """Finds when each trade is ended."""
        pass  
    
    def __post_init__(self) -> None:
        self.data = pd.DataFrame()
        self.prices.sort_index(inplace = True)
    
    def create_frame(
                self, 
                features: Union[pd.DataFrame, None] = None, 
                time_decay: float = .75, 
                dropna: bool = True, 
                drop_col_tresh: float = .05
    ) -> ResearchFrame:
            """"
            The steps for setting up the frame. If no features are supplied it just calculates forward returns.
            Note that features should be resampled with the same frequenzy as prices and have NaNs filled in. 
            """
            columns = []
            
            self \
                ._find_end_dates() \
                ._calculate_forward_return() \
                ._add_labels() \
                ._add_asset_col() 
            
            if features is not None:
                if isinstance(features, pd.Series): features = features.to_frame()
                self._add_features(features, dropna = dropna, drop_col_tresh = drop_col_tresh) 
                columns = self.data.columns.intersection(features.columns)  
            self._add_sample_weights(time_decay) 
              
            return ResearchFrame(self.data, columns)
             
    def _calculate_forward_return(self) -> ResearchFrameGenerator:
        """Calculate the return based on the end date."""
        self.data[ColNames.RETURN] = calculate_return(
            start_price = self.prices.loc[self.data.index],
            end_price = self.prices.loc[self.data[ColNames.END_DT].values].values
        ) 
        return self
    
    def _add_labels(self) -> ResearchFrameGenerator:
        self.data[ColNames.LABEL] = np.sign(self.data[ColNames.RETURN])
        return self
    
    def _add_asset_col(self) -> ResearchFrameGenerator:
        self.data[ColNames.ASSETS] = self.asset_name
        return self
            
    def _add_features(self, features: pd.DataFrame, dropna: bool, drop_col_tresh: float) -> ResearchFrameGenerator:
        """
        Adds features to the returns. 
        Note that features should be resampled with the same frequenzy as prices and have NaNs filled in. 
        """
        if self.data.index.dtype != features.index.dtype:
            raise TypeError('Missmatch in index data types')
        
        if isinstance(features, pd.DataFrame):
            features = features[features.isnull().mean().loc[lambda x: x < drop_col_tresh].index]
       
        self.data = self.data.merge(features, how = 'left', left_index = True, right_index = True, validate = 'one_to_one')
        if dropna: 
            self.data.dropna(inplace = True)
        return self
    
    def _add_sample_weights(self, time_decay: Optional[float] = 1.) -> ResearchFrameGenerator:
        """
        Highly overlapping outcomes would have disproportionate weights if considered equal to non-overlapping outcomes.
        At the same time, labels associated with large absolute returns should be given more importance than labels with negligible absolute returns.
        This function calculates weights by uniqueness and absolute return.
        """
        co_events = self._numb_concurrent_events()
        log_returns = self.data[ColNames.RETURN].add(1).apply(np.log)
        
        weights = pd.Series({
            start_date: (log_returns.loc[start_date:end_date] / co_events.loc[start_date:end_date]).sum() 
            for start_date, end_date in self.data[ColNames.END_DT].iteritems()
        })
        weights = weights.abs() * len(weights) / weights.abs().sum()
       
        if time_decay < 1:
            weights *= apply_time_decay(weights, last_weight = time_decay)
     
        self.data[ColNames.SAMPLE_WEIGHT] = weights
        return self

    def _numb_concurrent_events(self) -> pd.Series:
        """Calculates how many trades are part of each return contribution."""   
        co_events = pd.Series(0, index = self.data.index, name = 'num_co_events')
        for start_date, end_date in self.data[ColNames.END_DT].iteritems():
            co_events.loc[start_date:end_date] +=1
        return co_events
            

@dataclass
class FixedTimeFrameGenerator(ResearchFrameGenerator):
    """
    Calculates forward returns using a fixed time horizon for a single asset.
    
    Arguments:
    ----------
        prices: A pandas series with prices, indexed with datetime.
        asset_name: The name of the asset.
        max_holding_time: A timedelta object representing how long we are willing to hold each trade.
    """
    
    prices: pd.Series
    asset_name: str
    max_holding_time: pd.Timedelta
   
    def _find_end_dates(self) -> FixedTimeFrameGenerator:
        self._add_vertical_barriers(ColNames.END_DT) 
        return self
    
    def _add_vertical_barriers(self, col_name:str) -> FixedTimeFrameGenerator:
        """
        Finds the timestamp of the next price bar at or immediately after a number of days holding_days for each index in df.
        This function creates a series that has all the timestamps of when the vertical barrier is reached.
        """
        self.data = find_forward_dates(self.prices.index, self.max_holding_time).to_frame(col_name)
        return self

   
@dataclass
class TrippleBarrierFrameGenerator(FixedTimeFrameGenerator):
    """"
    Calculates forward returns using the tripple barrier method. A trade is closed in one of three ways.
    1) A stop loss level is triggered 2) a profit taking level is hit or 3) a maximum amount of time has expiered.
    
    Arguments:
    ----------
        prices: A pandas series with prices, indexed with datetime.
        asset_name: The name of the asset.
        max_holding_time: A timedelta object representing how long we are willing to hold each trade.
        barrier_width: A manual scaling factor for profit taking and stop loss targets expressed in standard deviations.
                       A higher number means higher profit taking targets and lower stop losses.
        eval_datetimes: Allows for runing the calculations on a subset only for efficency.
                        For example we might want to use some filtering strategy on our observations to reduce noise.  
        volatility_lookback: A lookback window that is used to calculate rolling volatility for target setting.
        
    """
    
    #prices: pd.Series
    #asset_name: str
    #max_holding_time: Optional[pd.Timedelta] = pd.Timedelta(days = 1, hours = 0, minutes = 0)
    barrier_width: Optional[float] = 1.
    eval_datetimes: Optional[pd.DatetimeIndex] = None
    volatility_lookback: Optional[int] = 100
        
    def _find_end_dates(self) -> TrippleBarrierFrameGenerator:
        self \
            ._add_vertical_barriers(ColNames.VERTICAL_BARRIER) \
            ._calculate_target_returns() \
            ._add_horizontal_barriers() \
            ._find_first_triggered_barrier()
        return self

    def _calculate_target_returns(self) -> TrippleBarrierFrameGenerator:
        """
        Calculates the target levels for profit taking and stop loss at each timestamp.
        Uses volatility and a manual scaling factor (barrier_width).
        """
        target_returns = (
            get_daily_volatility(self.prices, lookback = self.volatility_lookback)
                .mul(self.barrier_width)
                .rename(ColNames.TARGET)
        )
        self.data = self.data.join(target_returns).dropna(subset = [ColNames.TARGET])
        return self
       
    def _add_horizontal_barriers(self) -> TrippleBarrierFrameGenerator:       
        """Finds first time between start date and end date that the stop loss and profit taking target is hit."""
        def _find_horizontal_barrier_dates(prices:pd.Series, start_date:dt.datetime, end_date:dt.datetime, target:float) -> Dict[str, float]:
            """"Used in _add_horizontal_barriers. """
            price_path = prices[start_date:end_date] 
            returns = calculate_return(price_path, price_path[start_date])
            return {
                ColNames.START_DT: start_date,
                ColNames.STOP_LOSS: returns[returns < target*-1].index.min(),
                ColNames.PROFIT_TAKING: returns[returns > target].index.min()
            }
       
        if self.eval_datetimes is not None:
            self.data = self.data.loc[self.data.index.intersection(self.eval_datetimes)]
           
        horizontal_barriers = pd.DataFrame([
            _find_horizontal_barrier_dates(self.prices, row.Index, getattr(row, ColNames.VERTICAL_BARRIER), getattr(row, ColNames.TARGET))
            for row in self.data.itertuples()
        ])
        self.data = self.data.join(horizontal_barriers.set_index(ColNames.START_DT)).drop(ColNames.TARGET, axis = 1)
        return self
   
    def _find_first_triggered_barrier(self) -> TrippleBarrierFrameGenerator:
        """Finds which of the stop loss, profit taking or max holding periods that are triggered first."""
        barriers = [ColNames.STOP_LOSS, ColNames.PROFIT_TAKING, ColNames.VERTICAL_BARRIER]
        self.data[[ColNames.END_DT, ColNames.TRIGGER]] = (
            self.data[barriers]
                .dropna(how='all')
                .agg(['min','idxmin'], axis=1)
        )
        self.barrier_distribution = self.data[ColNames.TRIGGER].value_counts()
        self.data = self.data.dropna(subset = ColNames.END_DT).drop(barriers + [ColNames.TRIGGER], axis = 1)
        return self
   

@dataclass
class TrendScaningFrameGenerator(ResearchFrameGenerator):
    
    """ Calculates labels based on the trend from an OLS in a given forward window """
    
    prices: pd.Series
    asset_name: str
    time_periods: List[pd.Timdelta]
    eval_datetimes: Optional[pd.DatetimeIndex] = None
    
    def _find_end_dates(self) -> TrendScaningFrameGenerator:
        data = pd.concat([self._calculate_trend(time_period) for time_period in self.time_periods])
        data['abs_val'] = data[ColNames.TVAL].abs()
        
        max_vals = data.groupby([ColNames.START_DT]).max()
        idx = max_vals.set_index(['abs_val'], append = True).index
        self.data  = (
            data
                .set_index(['abs_val'], append = True)
                .loc[idx]
                .reset_index('abs_val', drop = True)
        )
        return self
        
    def _calculate_trend(self, time_period: pd.Timedelta) -> pd.DataFrame:
        fwrd_dts = find_forward_dates(self.prices.index, time_period)
        if self.eval_datetimes is not None:
            fwrd_dts = fwrd_dts.loc[fwrd_dts.index.intersection(self.eval_datetimes)]
            
        trend = pd.DataFrame([
            {ColNames.START_DT: start_dt, 
             ColNames.END_DT: end_dt, 
             ColNames.TVAL: ols_reg(self.prices.loc[start_dt:end_dt]).tvalues[1]}
             for start_dt, end_dt in fwrd_dts.iteritems()
        ])
        return trend.set_index(ColNames.START_DT)

    def _add_labels(self) -> TrendScaningFrameGenerator:
        self.data[ColNames.LABEL] = np.sign(self.data[ColNames.TVAL])
        return self
        
    def _add_sample_weights(self, time_decay: float = 1.) -> TrendScaningFrameGenerator:
        self.data = self.data.rename(columns = {ColNames.TVAL:ColNames.SAMPLE_WEIGHT})
        if time_decay < 1: 
            self.data[ColNames.SAMPLE_WEIGHT] *=  apply_time_decay(self.data[ColNames.SAMPLE_WEIGHT], last_weight = time_decay)
        return self 
        
    
def filtered_tripple_barrier_frame(
        prices: pd.Series, 
        asset_name: str, 
        max_holding_time: pd.Timedelta, 
        barrier_width: int, 
        features: pd.DataFrame, 
        time_decay: float, 
        volatility_lookback: Optional[int] = 100
) -> ResearchFrame:
    
    eval_datetimes = volatility_based_cumsum_filter(prices, volatility_lookback)
    generator = TrippleBarrierFrameGenerator(prices, asset_name, max_holding_time, barrier_width, eval_datetimes,volatility_lookback)
    return generator.create_frame(features, time_decay)

def filtered_trend_scaning_frame(
        prices: pd.Series, 
        asset_name: str, 
        time_periods: List[pd.Timdelta], 
        features: pd.DataFrame, 
        time_decay: float, 
        volatility_lookback: Optional[int] = 100
) -> ResearchFrame:
    
    eval_datetimes = volatility_based_cumsum_filter(prices, volatility_lookback)
    generator = TrendScaningFrameGenerator(prices, asset_name, time_periods, eval_datetimes)
    return generator.create_frame(features, time_decay)