from __future__ import annotations
from ..labeling.utils import ColNames
from .utils import return_ser
from dataclasses import dataclass
from typing import Union
import pandas as pd


@dataclass
class BaseFrame:
    
    """ Wraps a dataframe and adds domain specific attributes and methods. """
    
    data: pd.DataFrame
    feature_columns: list[str] = None
    
    def __post_init__(self) -> None:
        if self.feature_columns is None: self.feature_columns = [] 
        for col in self.feature_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='ignore')
                
    @property
    def forward_returns(self) ->  Union[pd.Series, None]:
        return return_ser(self.data, ColNames.RETURN)
    
    @property
    def end_dates(self) ->  Union[pd.Series, None]:
        return return_ser(self.data, ColNames.END_DT)
    
    @property
    def sample_weight(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.SAMPLE_WEIGHT)
    
    @property
    def labels(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.LABEL)
    
    @property
    def label_distribution(self) -> pd.Series:
        return self.labels.value_counts(normalize = True).mul(100).round(1).astype(str).add('%')

    @property
    def features(self) -> Union[pd.Series, None]:
        if len(self.feature_columns) > 0:
            return self.data[self.feature_columns]
    
    @property
    def assets(self) -> Union[pd.Series, None]:
        return return_ser(self.data, ColNames.ASSETS)
    
    @property
    def nr_assets(self) -> int:
        return self.assets.nunique()
    
    @property
    def shape(self) -> tuple[int, int]: 
        return self.data.shape    
    
    def __repr___(self) -> str:
        return self.data.head().to_markdown()
    
    def __add__(self, research_frame) -> BaseFrame:
        return self.__class__(
            data = pd.concat([self.data, research_frame.data]).drop_duplicates(),
            feature_columns = sorted(set(self.feature_columns + research_frame.feature_columns))
        )   
   
    
def multi_asset_frame(frames: list[BaseFrame]) -> BaseFrame:
    multi_asset_frame = frames[0]
    for frame in frames[1:]:
        multi_asset_frame += frame
    return multi_asset_frame