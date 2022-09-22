from alphafactory.labeling.utils import get_1d_return, find_forward_dates
from alphafactory.tests.fixtures import daily_doubling_prices
import pandas as pd


def test_daily_doubling_return(daily_doubling_prices):
    assert all(get_1d_return(daily_doubling_prices) == 1)


def test_daily_fwd_dates(daily_doubling_prices):
    date_idx = daily_doubling_prices.index
    exp = pd.Series(
        index = date_idx[:-1], 
        data = date_idx.shift()[:-1]
    )
    fwd = find_forward_dates(date_idx, pd.Timedelta(days = 1))
    assert all(exp==fwd)
    
