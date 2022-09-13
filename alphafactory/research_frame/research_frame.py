from __future__ import annotations
from typing import List, Union, Tuple, Dict
from dataclasses import dataclass
from .utils import return_ser
import datetime as dt 
import pandas as pd
import itertools
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
  
    
@dataclass
class ResearchFrame:
    
    """ Wraps a dataframe and adds domain specific attributes and methods. """
    
    data: pd.DataFrame
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None: self.feature_columns = [] 
        
    def cross_sectional_grouping(self, freq: str = None) -> Tuple[dt.datetime, pd.Series, pd.Series]:
        """G roups the data by time period and feature across all assets for cross sectional study. """
        period_group = self.data.index if freq is None else pd.Grouper(freq = freq, origin = 'start')
        group = itertools.product(self.feature_columns, self.data.groupby(period_group))
        for feature, (period, df) in group:
            yield period, df[feature], df[ColNames.RETURN]

    def longitudinal_grouping(self) -> Tuple[str, pd.Series, pd.Series]:
        """ Groups the data by asset and feature across time for longitudinal study. """
        group = itertools.product(self.feature_columns, self.data.groupby(self.assets))
        for feature, (asset, df) in group:
            yield asset, df[feature], df[ColNames.RETURN]

    def clf_kwargs(self, classification: bool = True, feature_subset: List[str] = None) -> Dict[str, pd.DataFrame]:
        if feature_subset is None: feature_subset = self.features.columns
        return {
            'X': self.features[feature_subset], 
            'y': self.labels if classification else self.forward_returns, 
            'sample_weight': self.sample_weight
        }
    
    def purged_cv_split(self, n_splits: int, embargo_time: pd.Timedelta) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
       """Splits forward returns in train and test sets."""
       def _purge_train_set(train:pd.Series, test:pd.Series, embargo_time:pd.Timedelta) -> pd.Series:
          """Removes any overlapps between train and test set from the train set."""
          dt_min, dt_max = (
              test.index.min() - embargo_time,
              test.max() + embargo_time
          )
          starts_within = train.loc[dt_min:dt_max].index
          end_within = train[train.between(dt_min, dt_max)].index
          envelops = train[(train.index <= dt_min) & (dt_max <= train)].index
        
          drop_idx = starts_within.union(end_within).union(envelops)
          return train.drop(drop_idx)
      
       end_dates = self.end_dates
       end_dates.sort_index(inplace = True)
       indices = np.arange(len(end_dates))  
       
       uniq_idx = np.unique(end_dates.index)
       test_starts=[
          (i[0], i[-1]) for i in np.array_split(np.arange(len(uniq_idx)), n_splits)
       ]
     
       for uniq_start_dt, uniq_end_dt in test_starts:
           start_idx, end_idx = (
               end_dates.index.searchsorted(uniq_idx[uniq_start_dt], side = 'left'),
               end_dates.index.searchsorted(uniq_idx[uniq_end_dt], side = 'right')
           )
           test_iloc = indices[start_idx:end_idx]
           train_iloc = np.setdiff1d(indices, test_iloc)
           test, train = end_dates.iloc[test_iloc], end_dates.iloc[train_iloc]
           train = _purge_train_set(train, test, embargo_time)
           yield train.index, test.index

    @property
    def forward_returns(self) ->  Union[pd.Series, None]:
        return return_ser(self.data, ColNames.RETURN)
    
    @property
    def end_dates(self) ->  Union[pd.Series, None]:
        return return_ser(self.data, ColNames.END_DT)
    
    @property
    def sample_weight(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.SAMPLE_WEIGHT)
    
    @property
    def labels(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.LABEL)
    
    @property
    def label_distribution(self):
        return self.labels.value_counts(normalize = True).mul(100).round(1).astype(str).add('%')

    @property
    def features(self) -> Union[pd.Series, None]:
        if len(self.feature_columns) > 0:
            return self.data[self.feature_columns]
    
    @property
    def assets(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.ASSETS)
    
    @property
    def nr_assets(self) -> int:
        return self.assets.nunique()
    
    @property
    def shape(self):
        return self.data.shape    
    
    def __repr___(self):
        return self.data.head().to_markdown()
    
    def __add__(self, research_frame) -> ResearchFrame:
        return ResearchFrame(
            data = pd.concat([self.data, research_frame.data]).drop_duplicates(),
            feature_columns = sorted(set(self.feature_columns + research_frame.feature_columns))
        )   
  
    
def multi_asset_frame(research_frames: List[ResearchFrame]) -> ResearchFrame:
    multi_asset_frame = research_frames[0]
    for frame in research_frames[1:]:
        multi_asset_frame += frame
    return multi_asset_frame