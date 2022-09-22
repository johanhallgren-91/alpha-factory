from .base_frame import BaseFrame
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass
from scipy import stats
import datetime as dt 
import pandas as pd
import itertools


@dataclass
class StatsFrame(BaseFrame):
            
    def cross_sectional_grouping(self, freq: str = None) -> tuple[dt.datetime, pd.Series, pd.Series]:
        """G roups the data by time period and feature across all assets for cross sectional study. """
        period_group = self.data.index if freq is None else pd.Grouper(freq = freq, origin = 'start')
        group = itertools.product(self.feature_columns, self.data.groupby(period_group))
        ret_col = self.forward_returns.name
        for feature, (period, df) in group:
            yield period, df[feature], df[ret_col]

    def longitudinal_grouping(self) -> tuple[str, pd.Series, pd.Series]:
        """ Groups the data by asset and feature across time for longitudinal study. """
        group = itertools.product(self.feature_columns, self.data.groupby(self.assets))
        ret_col = self.forward_returns.name
        for feature, (asset, df) in group:
            yield asset, df[feature], df[ret_col]

    def longitudinal_spearman_corr(self) -> pd.DataFrame:
        """" Calculates the spearman correlation between each asset, feature and forwards returns."""
        return self._spearman_corr(self.longitudinal_grouping())
    
    def cross_sectional_spearman_corr(self, freq: str = None) -> pd.DataFrame:
        """" Calculcates the cross-sectional spearman correlation for each asset across all assets during the given period."""
        return self._spearman_corr(self.cross_sectional_grouping(freq))
    
    def _spearman_corr(self, group) -> pd.DataFrame:
        def _corr_dict(group, feature, returns) -> dict[str, any]:
            corr, pval =  stats.spearmanr(feature, returns)
            return {'group': group, 'feature': feature.name, 'corr': corr, 'p-value': pval}
        return pd.DataFrame([_corr_dict(*args) for args in group])

    def longitudinal_quantile_returns(self) -> pd.DataFrame:
        return self._mean_quantile_returns(self.longitudinal_grouping())
    
    def cross_sectional_quantile_returns(self, freq: str = None) -> pd.DataFrame:
        return self._mean_quantile_returns(self.cross_sectional_grouping(freq))
    
    def _mean_quantile_returns(self, group, quantiles: int = 5) -> pd.DataFrame:
        quantile_func = lambda group, feature, ret: (
            pd.qcut(feature, quantiles, labels = False, duplicates = 'drop')
                .add(1)
                .to_frame('quantile')
                .assign(
                    group = group, 
                    feature = feature.name)
                .join(ret)
        )
        quantiles = pd.concat(
            quantile_func(asset, feature, ret) 
            for asset, feature, ret in group
        )
        return quantiles.groupby(['group', 'feature', 'quantile']).mean() 
  
    def check_stationarity(self, cutoff = .05) -> pd.DataFrame: 
        return pd.DataFrame([{
            'feature': feature.name, 'group': group, 'is_stationary': adfuller(feature)[1] < cutoff}
             for group, feature, _ in self.longitudinal_grouping()
        ])