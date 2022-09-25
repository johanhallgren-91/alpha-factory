from .utils import find_forward_dates, ColNames
from .base_labels import BaseLabels
from dataclasses import dataclass
import pandas as pd

    
@dataclass 
class FixedTimeLabels(BaseLabels):
    """
    Calculates forward returns using a fixed time horizon.
    
    Parameters
    ----------
        max_holding_period: pd.Timedelta
            A timedelta object representing how long we are willing to hold each trade.
    """
    
    max_holding_period: pd.Timedelta = pd.Timedelta(days = 1)
    
    def _find_end_dates(self) -> pd.DataFrame:
        return self._find_vertical_barriers(ColNames.END_DT) 
    
    def _find_vertical_barriers(self, col_name: str) -> pd.DataFrame:
        """
        Finds the timestamp of the next price bar at or immediately after a number of days holding_days for each index in df.
        This function creates a series that has all the timestamps of when the vertical barrier is reached.
        """
        return (
            find_forward_dates(self.prices.index, self.max_holding_period)
                .pipe(lambda df: df.loc[df.index.intersection(self.eval_datetimes)])
                .to_frame(col_name)    
        )