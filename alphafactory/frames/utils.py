from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from typing import Union, Tuple
import pandas as pd
import numpy as np
from ..labeling.utils import mp_apply

def score_clf(
        fit: BaseEstimator, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        sample_weight: np.array, 
        scoring: str
) -> float:
    """ Returns the scores of an classifier """
    if scoring == 'neg_log_loss':
        return -log_loss(y_test, fit.predict_proba(X_test), sample_weight = sample_weight, labels = fit.classes_)
    elif scoring == 'accuracy':
        return accuracy_score(y_test, fit.predict(X_test), sample_weight = sample_weight)
    else:
        raise ValueError('Scoring method not implemented')


def fit_and_score(
        clf: BaseEstimator, 
        train_idx: pd.DatetimeIndex, 
        test_idx: pd.DatetimeIndex, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: pd.Series, 
        scoring: str,
        return_clf: bool = False
) -> Union[float, Tuple[float, BaseEstimator]]:
    """ Fits a classifier on train data and scores it on test data """
    fit = clf.fit(
        X = X.loc[train_idx],
        y = y.loc[train_idx],
        sample_weight = sample_weight.loc[train_idx].values
    )
    score = score_clf(
        fit = fit, 
        X_test = X.loc[test_idx], 
        y_test = y.loc[test_idx],
        sample_weight = sample_weight.loc[test_idx].values,
        scoring = scoring
    )
    return score if not return_clf else (score, fit) 


def feature_decomposition(features: pd.DataFrame, threshold: float = .95) -> pd.DataFrame:
    """ Runs PCA on a dataframe """
    pca_pipe = make_pipeline(
        StandardScaler(), 
        PCA(n_components = threshold)
    )
    orth_features = pca_pipe.fit_transform(features)
    orth_features= pd.DataFrame(
        data = orth_features, 
        index = features.index, 
        columns = [f'PCA_{i}' for i in range(1, orth_features.shape[1] + 1)]
    )
    return orth_features


def get_feature_importance(fit: BaseEstimator, feature_names: list) -> pd.Series:
    """
    Gets the feature importance from a fited tree classeifier.    
    """
    feature_importance = pd.DataFrame(
        data = [tree.feature_importances_ for tree in fit.estimators_],
        columns = feature_names
    )
    feature_importance.replace(0, np.nan, inplace = True)
    feature_importance = pd.concat({
        'mean': feature_importance.mean(),
        'std': feature_importance.std() * len(feature_importance)**-.5
        }, axis = 1
    )
    feature_importance /= feature_importance['mean'].sum()
    return feature_importance


def return_ser(df: pd.DataFrame, col_name: str):
    if col_name in df.index.names: 
        return df.index.get_level_values(col_name)
    return df[col_name] if col_name in df.columns else None


def group_by_per_column(
        df: pd.DataFrame, grouper: Union[pd.Series, str], 
        func: callable, multiprocess: bool = False) -> pd.DataFrame:
    """
    Applies a function of to each group and function in the dataframe. 
    Allows for multiprocessing calculations. 
    """
    grouper = df.groupby(grouper)
    mp_grouper = len(grouper) > len(df.columns)
    return mp_apply(grouper, mp_grouper * multiprocess,
        lambda group: mp_apply(group, (not mp_grouper) * multiprocess,
            lambda column: func(column, group)
        )
    )


def down_sample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    index_name = df.index.names[0]
    idx = (
        df
            .reset_index()
            .groupby(df.index.to_period(freq))
            [index_name]
            .idxmax()
    )
    return df.iloc[idx]