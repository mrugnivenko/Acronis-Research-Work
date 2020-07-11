import numpy as np
import math
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE


def error(y_hat, y):
    """
    An relative error
    :param y_hat: predicted values
    :param y: estimated values
    :return: error
    """
    out = np.abs(y_hat - y) / y.max()
    return out


def kth_smallest(arr, k, l=0, r=None):
    """
    returns the k'th smallest elem in array
    :param arr: array which we discover
    :param k: the k which means the k'th smallest element will be returned
    :param l: left bound
    :param r: right bound
    :return: the k'th smallest element
    """
    if r is None:
        r = len(arr) - 1

    # If k is smaller than number of
    # elements in array
    if 0 < k <= r - l + 1:

        # Partition the array around last
        # element and get position of pivot
        # element in sorted array
        pos = _partition(arr, l, r)

        # If position is same as k
        if pos - l == k - 1:
            return arr[pos]
        if pos - l > k - 1:  # If position is more,
            # recur for left subarray
            return kth_smallest(arr, k, l, pos - 1)

            # Else recur for right subarray
        return kth_smallest(arr, k - pos + l - 1, pos + 1, r)

        # If k is more than number of
    # elements in array
    return sys.maxsize


def _partition(arr, l, r):
    """
    Standard partition process of QuickSort().
    It considers the last element as pivot and
    moves all smaller element to left of it
    and greater elements to right
    :param arr: array
    :param l: left bound
    :param r: right bound
    :return:
    """
    x = arr[r]
    i = l
    for j in range(l, r):
        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[r] = arr[r], arr[i]
    return i


def relative_mae(y, y_hat, tail=0.05, abs_val=True, vectorize=False):
    """
    scince MinMaxSacaler is not stable for systems with oultliers we perform normlizing with
    division by mean(). Several instances are periodical, and we must discard several sero-close values
    :param y: vector with true values
    :param y_hat: vector with predicted values
    :param tail: precent of least values to discard
    :param abs_val: whether we like to get abs values, or not(maybe for hist
    :param vectorize: states if we get one float for all values of vector for each moment
    :return: vector or float with error
    """
    _y = y[y > 0]
    _p = math.ceil(len(_y) * tail)
    mmin = kth_smallest(_y, _p)
    _mu = _y[_y > mmin].mean()
    if not vectorize:
        if abs_val:
            return np.mean(np.abs(y - y_hat)) / _mu
        else:
            return np.mean(y - y_hat) / _mu
    else:
        if abs_val:
            return np.abs(y - y_hat) / _mu
        else:
            return (y - y_hat) / _mu


def get_validation_score(model, data: pd.DataFrame, train_size, test_size, error_func=relative_mae, n=None,
                         seasonal={'lags': [], 'avg': 15}):
    """
    :param model: class with .fit(data) and y_hat=model.predict(data)
    :param data: DataFrame with colums['target', 'time_stamp',...]
    :param train_size: length of train piece of data
    :param test_size: length of test piece of data
    :param error_func: func(y, y_hat) which returns error
    :param n: number of validation slices
    :return: score on validation for time series
    """
    # defining ength of data both for test and train
    seas = 0
    if seasonal['lags']:
        seas = max(seasonal['lags'])

    sh = min(test_size, train_size)
    if sh < seas:
        print('test or train is < max(seas["lags"])')
    sh -= seas
    if n is None:
        n = math.ceil((data.shape[0] - (test_size + train_size - seas)) / sh)
    if n <= 0:
        print('too little data for validation')
        return np.nan

    if data.shape[0] < sh * (n-1) + test_size + train_size - seas:
        print(f'not enough datan need {sh * (n-1) + test_size + train_size - seas}, got {data.shape[0]}')
        return np.nan
    _out = []
    for i in range(n):
        tr_st = sh * i
        tr_end = sh * i + train_size
        te_st = sh * i + train_size - seas
        te_end = sh * i + train_size + test_size - seas
        if data.shape[0] >= te_end:
            model.fit(data[tr_st: tr_end])
            # let's account seasonality
            o = model.predict(data[te_st: te_end])
            y_hat = model.scaler_y.inverse_transform(o['predictions_sc'].values.reshape((-1, 1)))
            y = model.scaler_y.inverse_transform(o['target_der_sc'].values.reshape((-1, 1)))
            _out.append(error_func(y, y_hat))
    return sum(_out) / len(_out)


def get_val_score_and_num(model, data: pd.DataFrame, train_size, test_size, error_func=MAE, n=None,
                         seasonal={'lags': [], 'avg': 15}):
    """
    :param model: class with .fit(data) and y_hat=model.predict(data)
    :param data: DataFrame with colums['target', 'time_stamp',...]
    :param train_size: length of train piece of data
    :param test_size: length of test piece of data
    :param error_func: func(y, y_hat) which returns error
    :param n: number of validation slices
    :return: score on validation for time series
    """
    # defining ength of data both for test and train
    seas = 0
    if seasonal['lags']:
        seas = max(seasonal['lags'])

    sh = min(test_size, train_size)
    if sh < seas:
        print('test or train is < max(seas["lags"])')
    sh -= seas
    if n is None:
        n = math.floor((data.shape[0] - (test_size + train_size - seas)) / sh)
    if n <= 0:
        print('too little data for validation')
        return np.nan

    if data.shape[0] < sh * (n-1) + test_size + train_size - seas:
        print(f'not enough datan need {sh * (n-1) + test_size + train_size - seas}, got {data.shape[0]}')
        return np.nan
    _out = []
    _n_out = []
    for i in range(n):
        tr_st = sh * i
        tr_end = sh * i + train_size
        te_st = sh * i + train_size - seas
        te_end = sh * i + train_size + test_size - seas
        if data.shape[0] >= te_end:
            model.fit(data[tr_st: tr_end])
            # let's account seasonality
            o = model.predict(data[te_st: te_end])
            y_hat = model.scaler_y.inverse_transform(o['predictions_sc'].values.reshape((-1, 1)))
            y = model.scaler_y.inverse_transform(o['target_der_sc'].values.reshape((-1, 1)))
            _out.append(error_func(y, y_hat))
            _n_out.append(sum(model.est.coef_ > 0))
    return sum(_out) / len(_out), sum(_n_out) / len(_n_out)
