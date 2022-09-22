from alphafactory.labeling import FixedTimeLabels
from alphafactory.labeling.utils import ColNames
from alphafactory.tests.fixtures import daily_doubling_prices
import pandas as pd


def test_fixed_time_frame(daily_doubling_prices) -> None:
    expected_df = pd.DataFrame(
        index = daily_doubling_prices.index[:-1],
        data = {
            ColNames.END_DT: daily_doubling_prices.index[1:],
            ColNames.RETURN: [1.] * (len(daily_doubling_prices)-1),
            ColNames.LABEL: [1.] * (len(daily_doubling_prices)-1),
        }    
    )
    fixed_time_labels = FixedTimeLabels(prices = daily_doubling_prices, max_holding_period = pd.Timedelta(days = 1)).create_labels()
    assert (fixed_time_labels==expected_df).all().all()

