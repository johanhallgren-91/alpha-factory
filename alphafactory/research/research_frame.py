from __future__ import annotations
from .utils import (
    get_daily_volatility, calculate_return, find_forward_dates, ols_reg,
    volatility_based_cumsum_filter, apply_time_decay, return_ser)
from typing import Optional, Dict, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
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
    ASSETS = 'asset'
   
     
class ResearchFrame(ABC):
    """ An interface for calculating forward returns with diffrent strategies. """
       
    @abstractmethod
    def _find_end_dates(self) -> ResearchFrame:
        """Finds when each trade is ended."""
        pass
    
    def __post_init__(self) -> None:
        self.data = pd.DataFrame()
        self.prices.sort_index(inplace = True)
        
        self \
            ._find_end_dates() \
            ._calculate_forward_return() \
            ._add_labels()
            
        self.data[ColNames.ASSETS] = self.asset_name
        
    def setup_frame(
            self, features: pd.DataFrame, time_decay: float = .75, 
            dropna: bool = True, drop_col_tresh: float = .95
        ) -> ResearchFrame:
        """"
        The steps for setting up the frame. If no features are supplied it just calculates forward returns.
        Note that features should be resampled with the same frequenzy as prices and have NaNs filled in. 
        """
        self \
            ._add_features(features, dropna = dropna, drop_col_tresh = drop_col_tresh) \
            ._add_sample_weights(time_decay) 
        return self
        
    def _calculate_forward_return(self) -> ResearchFrame:
        """Calculate the return based on the end date."""
        self.data[ColNames.RETURN] = calculate_return(
            start_price = self.prices.loc[self.data.index],
            end_price= self.prices.loc[self.end_dates.values].values
        ) 
        return self
    
    def _add_sample_weights(self, time_decay: Optional[float] = 1.) -> ResearchFrame:
        """
        Highly overlapping outcomes would have disproportionate weights if considered equal to non-overlapping outcomes.
        At the same time, labels associated with large absolute returns should be given more importance than labels with negligible absolute returns.
        This function calculates weights by uniqueness and absolute return.
        """
        co_events = self._numb_concurrent_events()
        log_returns = self.forward_returns.add(1).apply(np.log)
        
        return_weights = pd.Series(index=self.data.index, dtype='float')
        for start_date, end_date in self.end_dates.iteritems():
            return_weights.loc[start_date] = (log_returns.loc[start_date:end_date] / co_events.loc[start_date:end_date]).sum()
        return_weights = return_weights.abs() * len(return_weights) / return_weights.abs().sum()
       
        if time_decay < 1:
            return_weights *= apply_time_decay(return_weights, last_weight = time_decay)
     
        self.data[ColNames.SAMPLE_WEIGHT] = return_weights
        return self

    def _numb_concurrent_events(self) -> pd.Series:
        """Calculates how many trades are part of each return contribution."""   
        co_events = pd.Series(0, index = self.data.index, name = 'num_co_events')
        for start_date, end_date in self.end_dates.iteritems():
            co_events.loc[start_date:end_date] +=1
        return co_events
        
    def _add_features(self, features: pd.DataFrame, dropna: bool = True, drop_col_tresh = .95) -> ResearchFrame:
        """
        Adds features to the returns. 
        Note that features should be resampled with the same frequenzy as prices and have NaNs filled in. 
        """
        if self.data.index.dtype != features.index.dtype:
            raise TypeError('Missmatch in index data types')
        
        if isinstance(features, pd.DataFrame):
            features = features[features.isnull().mean().loc[lambda x: x < drop_col_tresh].index]
        self.feature_columns = features.columns
        self.data = self.data.join(features)
        if dropna: 
            self.data.dropna(inplace = True)
        return self
    
    def _add_labels(self) -> ResearchFrame:
        self.data[ColNames.LABEL] = np.sign(self.forward_returns)
        return self
      
    @property
    def forward_returns(self) ->  Union[pd.Series, None]:
        return return_ser(self.data, ColNames.RETURN)
    
    @property
    def end_dates(self) ->  Union[pd.Series, None]:
        return return_ser(self.data, ColNames.END_DT)
    
    @property
    def sample_weights(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.SAMPLE_WEIGHT)
    
    @property
    def labels(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.LABEL)
    
    @property
    def features(self) -> Union[pd.Series, None]:
        if hasattr(self, 'feature_columns'):
            return self.data[self.feature_columns]
        return None
    
    @property
    def assets(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.ASSETS)
    
    def __len__(self):
        return self.assets.nunique()
    
    def __repr__(self) -> str:
        return self.asset_name
 
    
@dataclass
class FixedTimeFrame(ResearchFrame):
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
   
    def _find_end_dates(self) -> FixedTimeFrame:
        self._add_vertical_barriers(ColNames.END_DT) 
        return self
    
    def _add_vertical_barriers(self, col_name:str) -> FixedTimeFrame:
        """
        Finds the timestamp of the next price bar at or immediately after a number of days holding_days for each index in df.
        This function creates a series that has all the timestamps of when the vertical barrier is reached.
        """
        self.data = find_forward_dates(self.prices.index, self.max_holding_time).to_frame(col_name)
        return self

   
@dataclass
class TrippleBarrierFrame(FixedTimeFrame):
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
        
    def _find_end_dates(self) -> TrippleBarrierFrame:
        self \
            ._add_vertical_barriers(ColNames.VERTICAL_BARRIER) \
            ._calculate_target_returns() \
            ._add_horizontal_barriers() \
            ._find_first_triggered_barrier()

        return self

    def _calculate_target_returns(self) -> TrippleBarrierFrame:
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
       
    def _add_horizontal_barriers(self) -> TrippleBarrierFrame:       
        """Finds first time between start date and end date that the stop loss and profit taking target is hit."""
        def _find_horizontal_barrier_dates(prices:pd.Series, start_date:datetime, end_date:datetime, target:float) -> Dict[str, float]:
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
   
    def _find_first_triggered_barrier(self) -> TrippleBarrierFrame:
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
class TrendScaningFrame(ResearchFrame):
    """ Calculates labels based on the trend from an OLS in a given forward window """
    
    prices: pd.Series
    asset_name: str
    time_periods: List[pd.Timdelta]
    eval_datetimes: Optional[pd.DatetimeIndex] = None
    
    def _find_end_dates(self) -> TrendScaningFrame:
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

    def _add_labels(self) -> TrendScaningFrame:
        self.data[ColNames.LABEL] = np.sign(self.data[ColNames.TVAL])
        return self
        
    def _add_sample_weights(self, time_decay: float = 1.) -> TrendScaningFrame:
        self.data = self.data.rename(columns = {ColNames.TVAL:ColNames.SAMPLE_WEIGHT})
        if time_decay < 1: 
            self.data[ColNames.SAMPLE_WEIGHT] *=  apply_time_decay(self.data[ColNames.SAMPLE_WEIGHT], last_weight = time_decay)
        return self 
    
@dataclass
class MultiAssetFrame(ResearchFrame):
    """ Creates a combined frame for multiple assets """
    research_frames: List[ResearchFrame]
    
    def __post_init__(self):
        self.data = pd.concat([frame.data for frame in self.research_frames])
        self.feature_columns = sorted(set(
            [x for y in [frame.feature_columns for frame in self.research_frames if hasattr(frame, 'feature_columns')] for x in y]
        ))
        
    def _find_end_dates(self) -> MultiAssetFrame:
        return self
 
    def __len__(self):
        return len(self.research_frames)
    
    
def filtered_tripple_barrier_frame(
        prices: pd.Series, asset_name: str, max_holding_time: pd.Timedelta, 
        barrier_width: int, volatility_lookback: Optional[int] = 100
    ) -> TrippleBarrierFrame:
    
    eval_datetimes = volatility_based_cumsum_filter(prices, volatility_lookback)
    return TrippleBarrierFrame(prices, asset_name, max_holding_time, barrier_width, eval_datetimes,volatility_lookback)


def filtered_trend_scaning_frame(
        prices: pd.Series, asset_name: str, time_periods:List[pd.Timdelta], 
        volatility_lookback: Optional[int] = 100
    ) -> TrendScaningFrame:
    
    eval_datetimes = volatility_based_cumsum_filter(prices, volatility_lookback)
    return TrendScaningFrame(prices, asset_name, time_periods, eval_datetimes)