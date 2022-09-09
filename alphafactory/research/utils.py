from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from statsmodels.tsa.stattools import adfuller
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from typing import Optional, Union
import statsmodels.api as sm
import pandas as pd
import numpy as np


def get_1d_return(prices:pd.Series) -> pd.Series:
    """Computes the daily return at intraday estimation points."""
    idx = prices.index.searchsorted(prices.index - pd.Timedelta(days=1))
    idx = idx[idx > 0]

    returns = (
        prices.iloc[len(prices) - len(idx):]
      / prices.loc[prices.index[idx - 1]].values 
        - 1
    )
    return returns


def get_daily_volatility(prices: pd.Series, lookback: Optional[int]=100) -> pd.Series:
    """
    Computes the daily volatility at intraday estimation points,
    applying a span of lookback days to an exponentially weighted moving standard deviation.
    This function is used to compute dynamic thresholds for profit taking and stop loss limits.
    """
    return get_1d_return(prices).ewm(span=lookback).std()


def get_autocorr(prices:pd.Series, lookback:Optional[int]=100) -> pd.Series:
    """Daily Autocorr Estimates"""
    return get_1d_return(prices).rolling(lookback).apply(lambda x: x.autocorr(), raw=False)


def cumsum_filter(prices: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    The CUSUM filter is a method designed to detect a shift in the
    mean value of a measured quantity away from a target value. The filter is set up to
    identify a sequence of upside or downside divergences from any reset level zero.
    We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.
    One practical aspect that makes CUSUM filters appealing is that multiple events are not
    triggered by prices hovering around a threshold level, which is a flaw suffered by popular
    market signals such as Bollinger Bands. 
    """
    
    events = []
    sum_pos = sum_neg = 0

    log_returns = (
        prices
            .apply(np.log) 
            .diff()
            .dropna()
    )

    for dt, ret in log_returns[1:].iteritems():
        sum_pos = max(0, sum_pos + ret)
        sum_neg = min(0, sum_neg + ret)

        if sum_neg < -threshold:
            sum_neg = 0
            events.append(dt)

        elif sum_pos > threshold:
            sum_pos = 0
            events.append(dt)

    event_dts = pd.DatetimeIndex(events)
    return event_dts


def calculate_return(start_price:pd.Series, end_price:pd.Series) -> pd.Series:
    return end_price / start_price - 1  


def find_forward_dates(date_index: pd.DatetimeIndex, timedelta: pd.Timedelta) -> pd.Series:
    """Finds the timestamp of the next bar at or immediately after a timedelta for each index."""
    date_index = date_index.sort_values()
    idx = date_index.searchsorted(date_index + timedelta)
    idx = idx[idx < len(date_index)]
    return pd.Series(
        index = date_index[:len(idx)],
        data = date_index[idx],
        name = 'forward_dates'
    )

    
def volatility_based_cumsum_filter(prices: pd.Series, volatility_lookback: Optional[int] = 100) -> pd.DatetimeIndex:
    daily_volatility = get_daily_volatility(prices, lookback = volatility_lookback)
    eval_datetimes = cumsum_filter(
        prices = prices,
        threshold = daily_volatility.mean()
    )
    return eval_datetimes


def return_ser(df: pd.DataFrame, col_name:str) -> Union[pd.Series, None]:
    return df[col_name] if col_name in df.columns else None


def apply_time_decay(weights: pd.Series, last_weight: Optional[float] = .5, exponent: Optional[int] = 1) -> pd.Series:
    """
    Markets are adaptive systems (Lo [2017]). As markets evolve, older examples are less relevant than the newer ones. 
    Consequently, we would typically like sample weights to decay as new observations arrive.
    """
    time_decay = (
        weights
            .sort_index()
            .cumsum()
            .rename('time_decay')
    )
    if last_weight >= 0: 
        slope = ((1. - last_weight) / time_decay.iloc[-1]) ** exponent
    else: 
        slope = 1. / ((last_weight + 1) * time_decay.iloc[-1]) ** exponent
    
    const = 1. - slope * time_decay.iloc[-1]
    time_decay = const + slope * time_decay
    time_decay[time_decay < 0] = 0
    return time_decay  


def score_clf(
        fit: BaseEstimator, X_test:pd.DataFrame, y_test:pd.Series,
        sample_weights:np.array, scoring: str
) -> float:
    """ Returns the scores of an classifier """
    if scoring == 'neg_log_loss':
        return -log_loss(y_test, fit.predict_proba(X_test), sample_weight = sample_weights, labels = fit.classes_)
    return accuracy_score(y_test, fit.predict(X_test), sample_weight = sample_weights)


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


def fit_and_score(
        clf: BaseEstimator, train_idx: pd.DatetimeIndex, test_idx: pd.DatetimeIndex, 
        X: pd.DataFrame, y: pd.Series, sample_weight:pd.Series, scoring: str
    ) -> float:
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
        sample_weights = sample_weight.loc[test_idx].values,
        scoring = scoring
    )
    return score


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


def ols_reg(arr: np.array) -> float:
    x = sm.add_constant(np.arange(arr.shape[0]))
    return sm.OLS(arr, x).fit()


def is_stationary(ser: pd.Series, cutoff: float = .05) -> bool:
    return adfuller(ser)[1] < cutoff