from .utils import group_by_per_column
from .base_frame import BaseFrame
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass
from scipy import stats
import pandas as pd


@dataclass
class StatsFrame(BaseFrame):
                
    def spearman_corr(self, cross_sectional: bool = True, multiprocess: bool = False) -> pd.DataFrame:
        """" Calculates the spearman correlation between each asset, feature and forwards returns."""
        if cross_sectional and self.nr_assets < 5:
            raise ValueError('To few assets in dataframe.')
            
        _spearman = lambda feature, group: pd.Series(
            stats.spearmanr(feature.values, self.forward_returns.loc[group.index].values),
            index = ['corr', 'pval']
        )
        return group_by_per_column(
            df = self.features, 
            grouper = self.dates if cross_sectional else self.assets, 
            func = _spearman, 
            multiprocess = multiprocess
        )
    
    def quantiles(self, quantiles: int = 5, cross_sectional: bool = True, multiprocess: bool = False) -> pd.DataFrame:
        if cross_sectional and self.nr_assets < quantiles:
            raise ValueError('To few assets in dataframe.')
        
        _qcut = lambda feature, _: pd.qcut(feature, quantiles, labels = False, duplicates = 'drop') + 1
        return group_by_per_column(
            df = self.features, 
            grouper = self.dates if cross_sectional else self.assets,
            func = _qcut, 
            multiprocess = multiprocess
        )

    def check_stationarity(self, cutoff = .05, multiprocess: bool = False) -> pd.DataFrame: 
        _is_stationary = lambda feature, _:  adfuller(feature.values)[1] < cutoff
        return group_by_per_column(
            df = self.features, 
            grouper = self.assets, 
            func = _is_stationary,
            multiprocess = multiprocess
        )
