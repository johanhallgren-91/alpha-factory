from .utils import get_feature_importance, score_clf, is_stationary, feature_decomposition, fit_and_score
from ..research_frame.research_frame import ResearchFrame
from sklearn.base import BaseEstimator
from typing import List, Dict
from dataclasses import dataclass
from functools import partial
from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm


@dataclass
class FeatureImportance:
    feature_importance: pd.DataFrame
    out_of_sample_score: float = None
    out_of_bag_score: float = None


@dataclass
class FeatureResearch:
    
    research_frame: ResearchFrame
    clf: BaseEstimator
    classification: bool
    n_splits: int
    embargo_time: pd.Timedelta
    scoring: str
    
    def __post_init__(self):
        
        self.X = self.research_frame.features
        self.y = self.research_frame.labels if self.classification else self.research_frame.forward_returns
        self.sample_weight = self.research_frame.sample_weight
        
        self.purged_cv_split = partial(
            self.research_frame.purged_cv_split,
            n_splits = self.n_splits,
            embargo_time = self.embargo_time
        )
        self.fit_and_score = partial(
            fit_and_score,
            clf = self.clf, 
            scoring = self.scoring
        )
        self.cv_split = partial(
            self.research_frame.purged_cv_split,
            n_splits = self.n_splits,
            embargo_time = self.embargo_time
        )
    
    def clf_kwargs(self, feature_subset: List[str] = None) -> Dict[str, pd.DataFrame]:
        if feature_subset is None: feature_subset = self.X.columns
        return {
            'X': self.X[feature_subset], 
            'y': self.y,
            'sample_weight': self.sample_weight
        }
    
    def decompose_features(self, threshold: float = .95) -> pd.DataFrame:
        self.X, self.orig_X  = feature_decomposition(self.X, threshold), self.X
        return self.X

    def get_original_features(self) -> pd.DataFrame:
        if hasattr(self, 'orig_X'):
            self.X = self.orig_X
        return self.X
    
    def purged_cv_score(self, feature_subset: List[str] = None) -> pd.Series:
        """ Purges the train set and crossvalidates a classifier. """    
        fit_and_score_p = partial(self.fit_and_score, **self.clf_kwargs(feature_subset))
        scores = [fit_and_score_p(train_idx = train, test_idx = test) for train, test in self.cv_split()]
        return pd.Series({'mean': np.mean(scores), 'std': np.std(scores)*len(scores)**-.5}, name = 'cv_score')
        
    def feature_importance_mdi(self) -> FeatureImportance:
        fit = self.clf.fit(**self.clf_kwargs())
        return FeatureImportance(
            get_feature_importance(fit, self.X.columns),
            fit.oob_score_,  
            self.purged_cv_score()
        )
        
    def feature_importance_sfi(self) -> FeatureImportance:
        """
        Single feature importance (SFI) is a cross-section predictive-importance (out-of-sample) method.
        It computes the OOS performance score of each feature in isolation.
        """
        return FeatureImportance(
            pd.DataFrame.from_dict(
                {col: self.purged_cv_score([col]) for col in tqdm(self.X.columns)}, 
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
        fit_and_score_p = partial(self.fit_and_score, **self.clf_kwargs(), return_clf = True)
        score, score_col = pd.Series(), pd.DataFrame(columns = self.X.columns)
        for i, (train, test) in tqdm(enumerate(self.cv_split()), total=self.n_splits):
            s, fit = fit_and_score_p(train_idx = train, test_idx = test)
            score.loc[i] = s
           
            for col in self.X.columns:
                X_test_ = self.X.loc[test].copy(deep = True)
                np.random.shuffle(X_test_[col].values)
                score_col.loc[i, col] = score_clf(
                    fit = fit,
                    X_test = X_test_,
                    y_test = self.y.loc[test],
                    sample_weight = self.sample_weight.loc[test].values,
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

    def longitudinal_spearman_corr(self) -> pd.DataFrame:
        """" Calculates the spearman correlation between each asset, feature and forwards returns."""
        return self._spearman_corr(self.research_frame.longitudinal_grouping())
    
    def cross_sectional_spearman_corr(self, freq: str = None) -> pd.DataFrame:
        """" Calculcates the cross-sectional spearman correlation for each asset across all assets during the given period."""
        return self._spearman_corr(self.research_frame.cross_sectional_grouping(freq))
    
    def _spearman_corr(self, group) -> pd.DataFrame:
        def _corr_dict(group, feature, returns) -> Dict[str, any]:
            corr, pval =  stats.spearmanr(feature, returns)
            return {'group': group, 'feature': feature.name, 'corr': corr, 'p-value': pval}
        return pd.DataFrame([_corr_dict(*args) for args in group])

    def longitudinal_quantile_returns(self) -> pd.DataFrame:
        return self._mean_quantile_returns(self.research_frame.longitudinal_grouping())
    
    def cross_sectional_quantile_returns(self, freq: str = None) -> pd.DataFrame:
        return self._mean_quantile_returns(self.research_frame.cross_sectional_grouping(freq))
    
    def _mean_quantile_returns(self, group, quantiles: int = 5) -> pd.DataFrame:
        quantile_func = lambda group, feature, ret: (
            pd.qcut(feature, quantiles, labels = False)
                .add(1)
                .to_frame('quantile')
                .assign(
                    group = group, 
                    feature = feature.name)
                .join(ret)
        )
        quantiles = pd.concat(
            quantile_func(asset, feature, ret) 
            for asset, feature, ret in group
        )
        return quantiles.groupby(['group', 'feature', 'quantile']).mean()  
        
    def check_stationarity(self) -> pd.DataFrame: 
        return pd.DataFrame([{
            'feature': feature.name, 'group': group, 'is_stationary': is_stationary(feature)}
             for group, feature, _ in self.research_frame.longitudinal_grouping()
        ])
        