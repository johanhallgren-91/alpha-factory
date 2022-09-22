import pandas as pd
import pytest

@pytest.fixture
def daily_doubling_prices():
    dates = pd.date_range(start = '2020-01-01', end = '2020-01-10', freq = 'd')
    prices = pd.Series(index = dates, data = [2**x for x in range(1, len(dates)+1)])
    return prices