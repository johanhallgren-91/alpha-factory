from .utils import (
    volatility_based_cumsum_filter, 
    calculate_sample_weights, 
    calculate_return, 
    ColNames
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

    
@dataclass 
class BaseLabels(ABC):
    
    """ 
    A base class for calculating forwards returns, classification labels and  
    sample weights for a price series. Diffrent implementations have diffrent
    ways of determening when a trade is ended/closed. 
    
    Parameters
    ----------
    prices : pd.Sereis
        A series of prices indexed with datetime. 
    eval_datetimes: Optional[pd.DatetimeIndex]
        Alows to specify a subest of observations to evaluate. 
    """
    
    prices: pd.Series
    eval_datetimes: Optional[pd.DatetimeIndex] = None
    
    @abstractmethod
    def _find_end_dates(self) -> pd.DataFrame:
        """Finds when each trade is ended."""
        pass  
    
    def __post_init__(self) -> None:
        self.prices = (
            self.prices
                .squeeze()
                .astype(float)
                .sort_index()
        )
        if self.eval_datetimes is None:    
            self.eval_datetimes = self.prices.index 
            
    def _calculate_forward_returns(self, apply_filter: bool = False) -> pd.DataFrame:
        """Calculate the return based on the end date."""
        if apply_filter: 
            self._apply_filter()
            
        forward_returns = self._find_end_dates()
        forward_returns[ColNames.RETURN] = calculate_return(
            start_price = self.prices.loc[forward_returns.index],
            end_price = self.prices.loc[pd.DatetimeIndex(forward_returns[ColNames.END_DT])].values
        ).replace([np.inf, -np.inf], np.nan).astype(float)
        forward_returns.index.names = [ColNames.START_DT]
        return forward_returns
    
    def create_labels(self, apply_filter: bool = False) -> pd.DataFrame:
        labels = self._calculate_forward_returns(apply_filter)
        labels[ColNames.LABEL] = np.sign(labels[ColNames.RETURN]).astype('int8')
        return labels
    
    def calculate_sample_weights(self, labels: pd.DataFrame, time_decay: float = 1.) -> pd.DataFrame:
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
        ).astype(float)
        return labels
    
    def _apply_filter(self):
        self.eval_datetimes = volatility_based_cumsum_filter(self.prices)