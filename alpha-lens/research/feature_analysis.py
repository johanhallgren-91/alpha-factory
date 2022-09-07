from research.utils import get_feature_importance, plot_feature_importances, score_clf, fit_and_score
from research.research_frame import ResearchFrame
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
from dataclasses import dataclass
from functools import partial
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class PurgedKFold:
    
    research_frame: ResearchFrame
    n_splits: int
    embargo_time: pd.Timedelta
    
    def split(self) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """Splits forward returns in train and test sets."""
       
        end_dates = self.research_frame.end_dates
        end_dates.sort_index(inplace = True)
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
            train = self._purge_train_set(train, test, self.embargo_time)
            yield train.index, test.index
    
        
    @staticmethod
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


@dataclass
class FeatureResearch(PurgedKFold):
    
    clf: BaseEstimator
    scoring: str
    
    def __post_init__(self):
        self.X = self.research_frame.features
        self.y = self.research_frame.labels
        self.sample_weight = self.research_frame.sample_weights
        
    def purged_cv_score(self, feature_columns: List[str]) -> Dict[str, float]:
        """ Purges the train set and crossvalidates a classifier. """            
        partial_fit = partial(
            fit_and_score, 
            clf = self.clf, 
            X = self.X[feature_columns],
            y = self.y, 
            sample_weight = self.sample_weight, 
            scoring = self.scoring
        )
        scores = [partial_fit(train_idx = train, test_idx = test) for train, test in self.split()]
        return {'mean': np.mean(scores), 'std': np.std(scores)*len(scores)**-.5}
    
    def decompose_features(self, threshold: float = .95) -> pd.DataFrame:
        pca_pipe = make_pipeline(
            StandardScaler(), 
            PCA(n_components = threshold)
        )
        orth_X = pca_pipe.fit_transform(self.X)
        self.X = pd.DataFrame(
            data = orth_X, 
            index = self.X.index, 
            columns = [f'PCA_{i}' for i in range(1, orth_X.shape[1] + 1)]
        )
        return self.X
    
    def feature_importance_mdi(self):
        fit = self.clf.fit(**self.research_frame.classifier_args)
        self.mdi = {
            'feature_importance': get_feature_importance(fit, self.research_frame.feature_columns),
            'out_of_bag_score': fit.oob_score_, 
            'out_of_sample_score': self.purged_cv_score(self.research_frame.feature_columns)
        }
        return self.mdi
       
    def feature_importance_sfi(self) -> pd.DataFrame:
        """
        Single feature importance (SFI) is a cross-section predictive-importance (out-of-sample) method.
        It computes the OOS performance score of each feature in isolation.
        """
        self.sfi =  pd.DataFrame({
            col: self.purged_cv_score([col]) for col in self.research_frame.feature_columns}
        ).T
        return self.sfi
    

    def feature_importance_mda(self) -> tuple:
        """
        Fits a classifier, derives its performance OOS according to some performance
        score (accuracy, negative log-loss, etc.); third, it permutates each column of the
        features matrix (X), one column at a time, deriving the performance OOS after each column's
        permutation. The importance of a feature is a function of the loss in performance caused by
        its column's permutation.

        """
        score, score_col = pd.Series(), pd.DataFrame(columns = self.X.columns)
        for i, (train_idx, test_idx) in enumerate(self.split()):
            fit = self.clf.fit(
                    X = self.X.loc[train_idx],
                    y = self.y.loc[train_idx],
                    sample_weight = self.sample_weight.loc[train_idx].values
                )
            score.loc[i] = score_clf(
                fit = fit,
                X_test = self.X.loc[test_idx],
                y_test = self.y.loc[test_idx],
                sample_weights = self.sample_weight.loc[test_idx].values,
                scoring = self.scoring
            )
           
            for col in self.X.columns:
                X_test_ = self.X.loc[test_idx].copy(deep = True)
                np.random.shuffle(X_test_[col].values)
                score_col.loc[i, col] = score_clf(
                    fit = fit,
                    X_test = X_test_,
                    y_test = self.y.loc[test_idx],
                    sample_weights = self.sample_weight.loc[test_idx].values,
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
        self.mda= {
            'feature_importance':feature_importance,
            'out_of_sample_score': score.mean()
        }
        return self.mda
        
    def spearman_corr(self) -> pd.DataFrame:
        def corr_dict(df, ret, feature, asset):
            corr, pval =  stats.spearmanr(df[ret], df[feature])
            return {'asset': asset, 'feature': feature, 'corr': corr, 'p-value': pval}
        
        self.spearman_corr = pd.DataFrame([
            corr_dict(df, self.research_frame.forward_returns.name, col, asset) 
                for asset, df in self.research_frame.data.groupby(self.research_frame.assets) 
                for col in self.research_frame.features.columns
        ])
        return self.spearman_corr
