import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, '../')
import lib.pe_preprocessing as prepro
import lib.la_utils as la_utils
import lib.pe_utils as utils


def get_multiplied_data(df: pd.DataFrame, dots=5,
                        seasonal={'lags': [0], 'avg': 15},
                        cols_not_to_mult=['time_stamp', 'target', 'target_der']):
    """
    After getting scaled data we are to multipy them to take into consideration previous moments
    :param df: pd.DataFrame() object we are dealing with. there must be additionam columns as ['time_stamp', 'target',
     'target_der'], and
    :param dots: the number of dots to deal with
    :param target: target to be just cut. shape (n, 1)
    :return: matrix of shape (n * dots, m-dots). And a vector of shape (n-dots, 1)
    """
    if df.shape == (0, 0):
        return df

    _out = df.copy()
    # let's add lag's
    _out.drop(columns=cols_not_to_mult, inplace=True)
    _init_cols = _out.columns.values
    for _d in range(1, dots):
        for _c in _init_cols:
            _out[f'{_c}__{_d}'] = _out[_c].shift(_d).values
            _out[f'target__{_d}'] = df['target_der'].shift(_d).values

    # also need to shift other columns
    _out = pd.concat([_out, df[cols_not_to_mult]], axis=1)
    _avg = int(seasonal['avg'])
    # then lets's add seasoning compsonent
    for lag in seasonal['lags']:
        for _c in _init_cols:
            _out[f'{_c}__seas_{lag}__avg={_avg}'] = _out[_c].shift(lag).rolling(_avg).mean()
        _out[f'target__seas_{lag}__avg={_avg}'] = _out['target_der'].shift(lag).rolling(_avg).mean()
    return _out


def get_df_with_given_features(df: pd.DataFrame, features: list, must_have_columns=[], inplace=True):
    """
    simple func to drop all columns but given
    :param df: dataframe which must
    :param features: list of features we need in df
    :param inplace: like in pd.DataFrame.drop
    :param must_have_columns: there are sum columns but features
    :return: the same as pd.DataFrame.drop(inplace=inplace)
    """
    col_to_leave = must_have_columns + features
    col_to_drop = []
    for c in df.columns.values:
        if c not in col_to_leave:
            col_to_drop.append(c)
    out = df.drop(columns=col_to_drop, inplace=inplace)
    return out


def get_df_divided_by_col(df, column, not_to_div_col):
    d_values = df[column].values
    for c in df.columns.values:
        if c not in not_to_div_col:
            df[c] = np.divide(df[c].values, d_values)
    df.drop(columns=[column], inplace=True)
    return d_values


def preprocess_la(df, train_size, test_size, scaler_x=MinMaxScaler(), scaler_y=MinMaxScaler(), dots=5,
                  n_sigma_outliers=np.inf, seasonal={'lags': [60 * 24], 'avg': 20}, features=None,
                  col_to_dive=None):
    """
    now there is a difficult issue to preprocess data and get 4 of them
    :param df: data with columns 'time_stamp' and 'target'
    :param train_size:
    :param test_size:
    :param scaler:
    :param dots:
    :return: test_data, train_data, train_target, test_target
    """
    # let's first get rid of NaN
    # check input for sizes
    dict_to_rn = {}
    for c in df.columns:
        dict_to_rn[c] = la_utils.reformat_feature_name(c)
    df.rename(columns=dict_to_rn, inplace=True)
    if (test_size + train_size) > df.shape[0]:
        test_size = int(df.shape[0] * test_size / (test_size + train_size))
        train_size = int(df.shape[0] * train_size / (test_size + train_size))


    # then let's split for test and train
    train_data, test_data = prepro.train_test_split(data=df, test_size=test_size, train_size=train_size, dots=dots)
    # then let's calculate derive for particular columns
    train_data, test_data = prepro.get_test_train_derivate(train_data, test_data)
    # here we are to drop reloads
    _reloads_ts = train_data[train_data['target_der'] < 0]['time_stamp'].values
    train_data = prepro.get_df_with_droped_reloads(train_data, _reloads_ts, reload_window=60 * 10)
    _reloads_ts = test_data[test_data['target_der'] < 0]['time_stamp'].values
    test_data = prepro.get_df_with_droped_reloads(test_data, _reloads_ts, reload_window=60 * 10)
    # TODO if uncoment next string you'll get lower accuracy, it seems to be due to lack of previous 'target_der'
    # in mult_data, so I guess it's an important part
    # rm outliers for better scaling
    train_data = prepro.get_df_with_droped_outliers_s(train_data, n_sigma_outliers)
    # let's scale
    scaling = prepro.get_scaled_df(train_data, test_data, scaler_x=scaler_x, scaler_y=scaler_y)

    train_df_diff_sc, test_df_diff_sc, _, diff_scaler_y = scaling

    # now, if it's needed, let's drop extra features
    must_have = ['time_stamp', 'target', 'target_der']
    if features is not None:
        dict_to_rn = {}
        for c in train_df_diff_sc.columns:
            dict_to_rn[c] = la_utils.reformat_feature_name(c)
        train_df_diff_sc.rename(columns=dict_to_rn, inplace=True)
        test_df_diff_sc.rename(columns=dict_to_rn, inplace=True)

        if type(features) != list:
            features = list(features.reshape(-1))
        get_df_with_given_features(df=train_df_diff_sc, features=features, must_have_columns=must_have)
        get_df_with_given_features(df=test_df_diff_sc, features=features, must_have_columns=must_have)

    not_dive_col = ['time_stamp', 'target']
    if col_to_dive is not None:
        d_values_tr = get_df_divided_by_col(df=train_df_diff_sc, column=col_to_dive, not_to_div_col=not_dive_col)
        d_values_te = get_df_divided_by_col(df=test_df_diff_sc, column=col_to_dive, not_to_div_col=not_dive_col)

    # and after scaling and dropping let's multiply columns to use lags
    train_df_diff_sc_m = get_multiplied_data(train_df_diff_sc, dots=dots, seasonal=seasonal)
    test_df_diff_sc_m = get_multiplied_data(test_df_diff_sc, dots=dots, seasonal=seasonal)

    if col_to_dive is not None:
        train_df_diff_sc_m['div'] = d_values_tr
        test_df_diff_sc_m['div'] = d_values_te

    # train_df_diff_sc_m.dropna(inplace=True)
    # test_df_diff_sc_m.dropna(inplace=True)

    # deal with gaps in data to avoid incorrect usage of multiple dots
    # test_df_diff_sc_m = prepro.manage_gaps_after_mult(test_df_diff_sc_m, dots)
    # train_df_diff_sc_m = prepro.manage_gaps_after_mult(train_df_diff_sc_m, dots)

    #print(train_df_diff_sc_m)
    train_df_diff_sc_m.dropna(inplace=True)
    test_df_diff_sc_m.dropna(inplace=True)

    return test_df_diff_sc_m, train_df_diff_sc_m, scaler_x, scaler_y


def fill_misses(df, column='time_stamp', step=60):
    """
    sometimes in time series there is no data for several timestamps. this may cause unstable behavior usein AR and
    SAR models
    Time stamps must be a series with a constant step
    This function fills gaps with nans so the will be no wrong lags
    :param df: pd.DataFrame(), there must be a column to fill, 'time_stamp' default. dtype=float, int
    :param column: the column where gaps can lead to wrong results. dtype=int, float
    :param step: step of the time series
    """
    _d = df.copy()
    _d.set_index(column, inplace=True)
    idx = np.linspace(min(_d.index), max(_d.index), (max(_d.index) - min(_d.index)) // step + 1, dtype=int)
    _d = _d.reindex(idx, fill_value=np.nan)
    _d.reset_index(drop=True, inplace=True)
    _d[column] = idx
    return _d
