from alphafactory.research.research_frame import FixedTimeFrameGenerator, ColNames
import pandas as pd
import pytest

@pytest.fixture
def prices():
    dates = pd.date_range(start = '2020-01-01', end = '2020-01-10', freq = 'd')
    prices = pd.Series(index = dates, data = [2**x for x in range(1, len(dates)+1)])
    return prices

def test_fixed_time_frame_default(prices) -> None:
    expected_df = pd.DataFrame(
        index = prices.index[:-1],
        data = {
            ColNames.END_DT: prices.index[1:],
            ColNames.RETURN: [1.] * (len(prices)-1),
            ColNames.LABEL: [1.] * (len(prices)-1),
            ColNames.ASSETS: ['test'] * (len(prices)-1)
        }    
    )
    ftr = FixedTimeFrameGenerator(prices, 'test', pd.Timedelta(days = 1)).create_frame()
    assert (ftr.data.drop('sample_weight',axis=1)==expected_df).all().all()

def test_numb_concurrent_events(prices) -> None:
    expected_ser = pd.Series(
        index = prices.index[:-1],
        data = [1] + [2] * (len(prices)-2)
    )
    ftr = FixedTimeFrameGenerator(prices, 'test', pd.Timedelta(days = 1))
    ftr.create_frame()
    numb_concurrent_events = ftr._numb_concurrent_events()
    assert (numb_concurrent_events==expected_ser).all()
    


