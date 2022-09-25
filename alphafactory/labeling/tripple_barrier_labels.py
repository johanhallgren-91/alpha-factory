from .fixed_time_labels import FixedTimeLabels
from .utils import ( 
    get_daily_volatility, 
    get_mean_returns, 
    calculate_return, 
    mp_apply,
    ColNames
)
from dataclasses import dataclass
import pandas as pd


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
    
    barrier_width: float = 1.
    volatility_lookback: int = 100
    multiprocess: bool = False
    
    def _find_end_dates(self) -> pd.DataFrame:
        target_returns = self._calculate_target_returns()
        vertical_barriers = self._find_vertical_barriers(ColNames.VERTICAL_BARRIER) 
        horizontal_barriers = self._find_horizontal_barriers(vertical_barriers, target_returns)
        return self._find_first_triggered_barrier(vertical_barriers, horizontal_barriers)

    def _calculate_target_returns(self, include_mean: bool = False) -> pd.Series:
        """
        Calculates the target levels for profit taking and stop loss at each timestamp.
        Uses volatility and a manual scaling factor (barrier_width).
        """
        target_returns = (
            get_daily_volatility(self.prices, lookback = self.volatility_lookback)
                .mul(self.barrier_width)
                .add(include_mean * get_mean_returns(self.prices, lookback = self.volatility_lookback).abs())
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

        def _find_horizontal_barrier_dates(row: pd.Series, prices: pd.Series) -> dict[str, float]:
            """" Helper function """
            price_path = prices.loc[row.name: row[ColNames.VERTICAL_BARRIER]]
            returns = calculate_return(price_path, price_path.at[row.name])
            return {
                ColNames.STOP_LOSS: returns[returns < row[ColNames.TARGET]*-1].index.min(),
                ColNames.PROFIT_TAKING: returns[returns > row[ColNames.TARGET]].index.min()
            }
            
        data = pd.concat([vertical_barriers, target_returns], axis = 1).dropna()
        return mp_apply(data, self.multiprocess, _find_horizontal_barrier_dates, prices = self.prices, axis = 1, result_type = 'expand')

                          
    def _find_first_triggered_barrier(self, vertical_barriers: pd.DataFrame, horizontal_barrier: pd.DataFrame) -> pd.DataFrame:
        """Finds which of the stop loss, profit taking or max holding periods that are triggered first."""
        barriers = (
            pd.concat([vertical_barriers, horizontal_barrier], axis = 1)
                .dropna(how='all')
                .parallel_apply(lambda x: x.min(), axis = 1)
                .to_frame(ColNames.END_DT)
        )
        return barriers