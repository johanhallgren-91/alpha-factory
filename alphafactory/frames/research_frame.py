from __future__ import annotations
from .stats_frame import StatsFrame
from .utils import (
    get_feature_importance, 
    feature_decomposition, 
    fit_and_score,
    score_clf
)
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from functools import partial
from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np


@dataclass
class FeatureImportance:
    feature_importance: pd.DataFrame
    out_of_sample_score: Optional[float] = None
    out_of_bag_score: Optional[float] = None
    
    
@dataclass 
class ResearchFrame(StatsFrame):
    
    clf: Optional[BaseEstimator] = None 
    n_spits: Optional[int] = None
    embargo_time: Optional[pd.Timdelta] = None
    scoring: Optional[str] = None
    
    def __post_init__(self) -> None:
        self.data = self.data.sort_index(level = self.dates.name) 
        self.has_config = False
        
    def set_config(self, clf: BaseEstimator, n_splits: int, embargo_time: pd.Timedelta, scoring: str) -> None:
        self.clf = clf
        self.n_splits = n_splits
        self.embargo_time = embargo_time
        self.scoring = scoring
        self.fit_and_score = partial(
            fit_and_score, 
            clf = self.clf, 
            scoring = self.scoring
        )
        self.has_config = True
        
    def check_config(self):
        if not self.has_config:
            raise AttributeError('You need to set config before running this method.')  
            
    def decompose_features(self, threshold: float = .95) -> pd.DataFrame:
        self.features, self.orig_features  = feature_decomposition(self.features, threshold), self.features
        return self.features
    
    def get_original_features(self) -> pd.DataFrame:
        if hasattr(self, 'orig_features'):
            self.features = self.orig_features
        return self.features
    
    def clf_kwargs(self, feature_subset: Optional[list[str]] = None, classification: bool = True) -> dict[str, pd.DataFrame]:
        if feature_subset is None: feature_subset = self.features.columns
        return {
            'X': self.features[feature_subset], 
            'y': self.labels if classification else self.forward_returns, 
            'sample_weight': self.sample_weight
        }
    
    def purged_cv_split(self) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
       """Splits forward returns in train and test sets."""
       def _purge_train_set(train:pd.Series, test:pd.Series, embargo_time:pd.Timedelta) -> pd.Series:
          """Removes any overlapps between train and test set from the train set."""
          dt_min, dt_max = (
              test.index.min() - embargo_time,
              test.max() + embargo_time
          )
          starts_within = train.loc[dt_min:dt_max].index
          end_within = train[train.between(dt_min, dt_max)].index
          envelops = train[(train.index <= dt_min) & (dt_max <= train)].index
        
          drop_idx = starts_within.union(end_within).union(envelops)
          return train.drop(drop_idx)
      
       if not all(self.data.index == self.data.sort_index(level = self.dates.name).index):
           raise IndexError('Index need to be sorted by datetime')
       
       end_dates = self.end_dates.reset_index(self.assets.name, drop = True)
       indices = np.arange(len(end_dates))  
       
       uniq_idx = np.unique(end_dates.index)
       test_starts=[
          (i[0], i[-1]) for i in np.array_split(np.arange(len(uniq_idx)), self.n_splits)
       ]
     
       for uniq_start_dt, uniq_end_dt in test_starts:
           start_idx, end_idx = (
               end_dates.index.searchsorted(uniq_idx[uniq_start_dt], side = 'left'),
               end_dates.index.searchsorted(uniq_idx[uniq_end_dt], side = 'right')
           )
           test_iloc = indices[start_idx:end_idx]
           train_iloc = np.setdiff1d(indices, test_iloc)
           test, train = end_dates.iloc[test_iloc], end_dates.iloc[train_iloc]
           train = _purge_train_set(train, test, self.embargo_time)
           yield train.index, test.index
           
    def purged_cv_score(self, feature_subset: Optional[list[str]] = None) -> pd.Series:
        """ Purges the train set and crossvalidates a classifier. """    
        self.check_config()
        Xysw = self.clf_kwargs(feature_subset=feature_subset)
        scores = [
            self.fit_and_score(train_idx = train, test_idx = test, **Xysw) 
            for train, test in self.purged_cv_split()
        ]
        return pd.Series({'mean': np.mean(scores), 'std': np.std(scores) * len(scores)**-.5}, name = 'cv_score')
        
    def feature_importance_mdi(self) -> FeatureImportance:
        self.check_config()
        fit = self.clf.fit(**self.clf_kwargs())
        return FeatureImportance(
            get_feature_importance(fit, self.features.columns),
            fit.oob_score_,  
            self.purged_cv_score()
        )
        
    def feature_importance_sfi(self) -> FeatureImportance:
        """
        Single feature importance (SFI) is a cross-section predictive-importance (out-of-sample) method.
        It computes the OOS performance score of each feature in isolation.
        """
        self.check_config()
        return FeatureImportance(
            pd.DataFrame.from_dict(
                {col: self.purged_cv_score([col]) for col in tqdm(self.features.columns)}, 
                orient = 'index')
        )
    
    def feature_importance_mda(self) -> FeatureImportance:
        """
        Fits a classifier, derives its performance OOS according to some performance
        score (accuracy, negative log-loss, etc.); third, it permutates each column of the
        features matrix (X), one column at a time, deriving the performance OOS after each column's
        permutation. The importance of a feature is a function of the loss in performance caused by
        its column's permutation.

        """
        self.check_config()
        Xysw = self.clf_kwargs()
        score, score_col = pd.Series(), pd.DataFrame(columns = self.features.columns)
        for i, (train, test) in tqdm(enumerate(self.purged_cv_split()), total=self.n_splits):
            s, fit = self.fit_and_score(train_idx = train, test_idx = test, **Xysw, return_clf = True)
            score.loc[i] = s
           
            for col in self.features.columns:
                X_test_ = Xysw['X'].loc[test].copy(deep = True)
                np.random.shuffle(X_test_[col].values)
                score_col.loc[i, col] = score_clf(
                    fit = fit,
                    X_test = X_test_,
                    y_test = Xysw['y'].loc[test],
                    sample_weight = Xysw['sample_weight'].loc[test].values,
                    scoring = self.scoring
                )
            
        denominator = -score_col if self.scoring == 'neg_log_loss' else (1.-score_col)
        feature_importance = (
            score_col
                .mul(-1)
                .add(score, axis = 0)
                .div(denominator, axis = 0)
        )
        feature_importance = pd.concat({
            'mean': feature_importance.mean(),
            'std': feature_importance.std() * len(feature_importance)**-.5
            }, axis = 1
        )
        return FeatureImportance(feature_importance, score.mean())