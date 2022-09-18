from alphafactory.research_frame.labeling_strategies import FixedTimeLabels
from alphafactory.research_frame.research_frame import ColNames
import pandas as pd
import pytest

@pytest.fixture
def prices():
    dates = pd.date_range(start = '2020-01-01', end = '2020-01-10', freq = 'd')
    prices = pd.Series(index = dates, data = [2**x for x in range(1, len(dates)+1)])
    return prices

def test_fixed_time_frame(prices) -> None:
    expected_df = pd.DataFrame(
        index = prices.index[:-1],
        data = {
            ColNames.END_DT: prices.index[1:],
            ColNames.RETURN: [1.] * (len(prices)-1),
            ColNames.LABEL: [1.] * (len(prices)-1),
        }    
    )
    fixed_time_labels = FixedTimeLabels(prices = prices, max_holding_period = pd.Timedelta(days = 1)).create_labels()
    assert (fixed_time_labels==expected_df).all().all()

