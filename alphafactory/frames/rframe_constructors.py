from ..labeling import TrippleBarrierLabels, TrendScaningLabels 
from ..labeling.base_labels import BaseLabels
from ..labeling.utils import ColNames
from . import ResearchFrame
import pandas as pd


def create_research_frame(
        asset_name: str,
        labels: BaseLabels,
        filter_labels: bool, 
        features: pd.DataFrame, 
        time_decay: float,
        max_pct_na: float,
    ) -> ResearchFrame:
    """
    Takes a label generator and features a dataframe with features to create a research frame. 
    """
    frame = (
        labels.create_labels(filter_labels)
            .merge(features, how = 'left', left_index = True, right_index = True, validate = 'one_to_one')
            .pipe(lambda df: df[df.isnull().mean().loc[lambda x: x < max_pct_na].index])
            .dropna()
            .pipe(labels.calculate_sample_weights, time_decay)
            .assign(**{ColNames.ASSETS: asset_name.upper()})
    )
    return ResearchFrame(frame, frame.columns.intersection(features.columns).to_list())


def create_tripple_barrier_frame(
        asset_name: str,
        prices: pd.Series, 
        features: pd.DataFrame, 
        time_decay: float,
        max_pct_na: float, 
        max_holding_period: pd.Timedelta,
        barrier_width: float, 
        volatility_lookback: int,
        filter_labels: bool
    ) -> ResearchFrame:   
    """ Convinence function to create a filtered tripple barrier research frame"""
    labels = TrippleBarrierLabels(
        prices = prices, 
        max_holding_period = max_holding_period, 
        barrier_width = barrier_width, 
        volatility_lookback = volatility_lookback
    )
    frame = create_research_frame(
        asset_name = asset_name, 
        labels = labels, 
        filter_labels = filter_labels,
        features = features, 
        time_decay = time_decay, 
        max_pct_na = max_pct_na
    )
    return frame


def create_trend_scaning_frame(
        asset_name: str,
        prices: pd.Series, 
        features: pd.DataFrame, 
        time_decay: float,
        max_pct_na: float,
        holding_periods: list[pd.Timdelta],
        volatility_lookback: int, 
        filter_labels: bool
    ) -> ResearchFrame:  
    """ Convinence function to create a filtered tripple barrier research frame"""
    labels = TrendScaningLabels(
        prices = prices, 
        holding_periods = holding_periods
    )
    frame = create_research_frame(
        asset_name = asset_name, 
        labels = labels, 
        filter_labels = filter_labels,
        features = features, 
        time_decay = time_decay, 
        max_pct_na = max_pct_na
    )
    return frame


