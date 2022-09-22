from .base_labels import BaseLabels
from .utils import (
    apply_time_decay_to_sample_weights, 
    find_forward_dates,
    ols_reg,
    ColNames
)
from collections.abc import Iterable 
from dataclasses import dataclass
import pandas as pd
import numpy as np

       
@dataclass 
class TrendScaningLabels(BaseLabels):
    
    """ Calculates labels based on the trend from an OLS in a given forward window """
    
    holding_periods: list[pd.Timedelta] = pd.Timedelta(days = 1)
    
    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.holding_periods, Iterable):
            self.holding_periods = [self.holding_periods]
            
    def _find_end_dates(self) -> pd.DataFrame:
        data = self._multi_period_trends()
        data[ColNames.TVAL_ABS] =  data[ColNames.TVAL].abs()
        max_idx = (
            data
                .groupby([ColNames.START_DT])[ColNames.TVAL_ABS]
                .max()
                .to_frame()
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
    
    def _multi_period_trends(self) -> pd.DataFrame:
        return pd.concat([
            self._calculate_trend(holding_period) 
            for holding_period in self.holding_periods
        ])
    
    def _calculate_trend(self, time_period: pd.Timedelta) -> pd.DataFrame:
        forward_dates = (
            find_forward_dates(self.prices.index, time_period)
                .pipe(lambda df: df.loc[df.index.isin(self.eval_datetimes)])
                .to_frame(ColNames.END_DT)
        )
        forward_dates[ColNames.TVAL] = forward_dates.apply(lambda row: 
            ols_reg(self.prices.loc[row.name:row[ColNames.END_DT]].values).tvalues[1], 
            axis = 1, 
            result_type = 'expand'
        )
        forward_dates.index.names = [ColNames.START_DT]
        return forward_dates
    
    def create_labels(self) -> pd.DataFrame:
        labels = self._calculate_forward_returns()
        labels[ColNames.LABEL] = np.sign(labels[ColNames.TVAL])
        return labels
    
    def calculate_sample_weights(self, labels: pd.DataFrame = None, time_decay: float = 1.) -> pd.DataFrame:
        labels = labels.rename(columns = {ColNames.TVAL: ColNames.SAMPLE_WEIGHT})
        if time_decay < 1:
            labels[ColNames.SAMPLE_WEIGHT] = apply_time_decay_to_sample_weights(labels[ColNames.SAMPLE_WEIGHT], time_decay)
        return labels