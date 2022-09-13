from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from statsmodels.tsa.stattools import adfuller
from typing import Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def score_clf(
        fit: BaseEstimator, 
        X_test:pd.DataFrame, 
        y_test:pd.Series,
        sample_weight:np.array, 
        scoring: str
) -> float:
    """ Returns the scores of an classifier """
    if scoring == 'neg_log_loss':
        return -log_loss(y_test, fit.predict_proba(X_test), sample_weight = sample_weight, labels = fit.classes_)
    return accuracy_score(y_test, fit.predict(X_test), sample_weight = sample_weight)


def fit_and_score(
        clf: BaseEstimator, 
        train_idx: pd.DatetimeIndex, 
        test_idx: pd.DatetimeIndex, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight:pd.Series, 
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


def get_feature_importance(fit: BaseEstimator, feature_names:list) -> pd.Series:
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


def plot_feature_importances(feature_importance:pd.DataFrame, oob:float = 0, oos:float = 0) -> None:
    plt.figure(figsize = (10, len(feature_importance)/5.))
    feature_importance.sort_values('mean', inplace = True, ascending = True)
    ax = feature_importance['mean'].plot(
        kind = 'barh',
        alpha = .25,
        xerr = feature_importance['std']#,
        #error_kw = {'ecolor':'r'}
    )
 
    plt.xlim([0, feature_importance.sum(axis = 1).max()])
    plt.axvline(1./len(feature_importance), linewidth = 1, color = 'r', linestyle = 'dotted')

    ax.get_yaxis().set_visible(False)
    for patch, feature in zip(ax.patches, feature_importance.index):
        ax.text(
            patch.get_width() / 2, patch.get_y() + patch.get_height() / 2,     
            feature, ha = 'center', va = 'center', color = 'black'
        )
    plt.title(f'oob = {str(round(oob, 4))} | oos = {str(round(oos, 4))}')
    plt.show()


def set_up_bagging_classifier(
        n_estimators:int = 1000, max_samples:float = 1., min_w_leaf:float = 0.
    ) -> BaggingClassifier:
    """Creates a bagging classifier."""
    clf = DecisionTreeClassifier(
        criterion = 'entropy',
        max_features = 1.,
        class_weight = 'balanced',
        min_weight_fraction_leaf = min_w_leaf
    )
    clf = BaggingClassifier(
        base_estimator = clf,
        n_estimators = n_estimators,
        max_features = 1.,
        max_samples = max_samples,
        oob_score = True,
        n_jobs = -1
    )
    return clf


def is_stationary(ser: pd.Series, cutoff: float = .05) -> bool:
    return adfuller(ser)[1] < cutoff
