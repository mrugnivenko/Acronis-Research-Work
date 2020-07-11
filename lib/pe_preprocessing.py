import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model, metrics
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(1, '../')
import lib.pe_model as model


class Normalizer:

    def __init__(self, axis=0, dots=5, typ='target'):
        self.axis = axis
        self.dots = dots
        self.typ = typ
        self.X_max = None
        self.X_min = None
        self.X_mean = None
        self.dic = None
        self.rest = None

    def fit(self, X):
        if self.typ == 'target':
            self.X_max = X.max(axis=self.axis)
            self.X_min = X.min(axis=self.axis)
            self.X_mean = X.mean(axis=self.axis)
        else:
            self.dic = {'X_max': [], 'X_min': [], 'X_mean': [], 'rest': []}
            for e in range(len(X[0])):
                column = [row[e] for row in X]
                self.dic['rest'].append(column[:self.dots - 1])
                for i in range(self.dots):
                    self.dic['X_max'].append(max(column[i:]))
                    self.dic['X_min'].append(min(column[i:]))
                    self.dic['X_mean'].append(np.mean(column[i:]))

    def transform(self, X):
        if self.typ == 'target':
            self.rest = X[:self.dots - 1]
            return ((X - self.X_mean) / (self.X_max - self.X_min))[self.dots - 1:]
        else:
            glob = 0
            tmp = []
            for e in range(len(X[0])):
                column = [row[e] for row in X]
                tmp2 = column
                tmp.append((tmp2[self.dots - 1:] - self.dic['X_mean'][glob]) / (
                        self.dic['X_max'][glob] - self.dic['X_min'][glob]))
                for i in range(self.dots - 1):
                    tmp.append((tmp2[self.dots - i - 1 - 1:-i - 1] - self.dic['X_mean'][glob + i + 1]) / (
                            self.dic['X_max'][glob + i + 1] - self.dic['X_min'][glob + i + 1]))
                glob = glob + self.dots
            return np.array(tmp).T

    def inverse_transform(self, X):
        if self.typ == 'target':
            return np.append(self.rest, X * (self.X_max - self.X_min) + self.X_mean)
        else:
            return np.append(self.rest, (X[0]) * (self.X_max[0] - self.X_min[0]) + self.X_mean[0])


def get_multiplied_data(df: pd.DataFrame, dots=5, cols_not_to_mult=['time_stamp', 'target', 'target_der']):
    """
    Afrer getting scaled data we are to multipy them to take into consideration previous moments
    :param dots: the number of dots to deal with
    :param data: matrix to be multiplied. shape is (n, m)
    :param target: target to be just cut. shape (n, 1)
    :return: matrix of shape (n * dots, m-dots). And a vector of shape (n-dots, 1)
    """
    if df.shape == (0, 0):
        return df
    _out = df.copy()

    _out.drop(columns=cols_not_to_mult, inplace=True)
    _init_cols = _out.columns.values
    for _d in range(1, dots):
        for _c in _init_cols:
            _out[_c+'__'+str(_d)] = _out[_c].shift(_d).values
            _out['target__'+str(_d)] = df['target_der'].shift(_d).values

    # also need to shift other columns
    out = pd.concat([_out, df[cols_not_to_mult]], axis=1)

    return out


def train_test_split(test_size: int, train_size: int, data: pd.DataFrame, dots=5) -> list:
    """
    this function split data for train and test determinate.
    The last part of fixed size for test and the first part of fixed size for train
    This function exists for extrapolation tasks as
    :param test_size: size for test
    :param train_size: size for train
    :param data: data frame or numpy ndarray with data
    :param target: target data - also data, but further for other purpose
    :param dots: amount of dots for each metric to consider. if more than one (for example 7) so 6
                previous moments will be considered
    :return: train_data, test_data, train_target, test_target
    """
    whole_size = test_size + train_size
    max_minutes = max(data.index)
    # discard data
    min_minutes = max_minutes - whole_size
    min_to_spl = min_minutes + train_size

    data_to_split = data[-test_size - train_size:]
    # target_to_split = target[target.index >= min_minutes]
    # split train, no to do with dots
    if test_size != 0:
        train_data = data[-test_size - train_size: -test_size]
    else:
        train_data = data[-test_size - train_size:]
    # split test, adding a bit of dots
    if test_size != 0:
        test_data = data[-test_size:]
    else:
        test_data = data[:0]
    return [train_data, test_data]


def get_test_train_derivate(train_df: pd.DataFrame, test_df: pd.DataFrame, metric_not_to_diff=['abgw_conns']) -> list:
    """
    It is not easy to get proper derivation so this function does this. It gets train/test data and target.
    Data is to be copied and transformed to data on derivate
    Target is to be just .diff()
    :param train_df: pandas df which is to be trnsformed to derivation.
    But not columns that starts with "metric_not_to_diff"
    :param test_df: pandas df which is to be trnsformed to derivation.
    But not columns that starts with "metric_not_to_diff"
    :param metric_not_to_diff: since we got pandas df there may be data we should not transform.
    the list with first parts of the columns name should be passed
    :return: train_data_diff, test_data_diff, train_target_diff, test_target_diff
    """
    if not train_df.empty:
        for m in metric_not_to_diff:
            for c in train_df.drop(columns=['time_stamp', 'target']).columns.values:
                if not (c.startswith(m)):
                    train_df[c] = train_df[c].diff()
    train_df['target_der'] = train_df['target'].diff()
    train_df.dropna(inplace=True)

    if not test_df.empty:
        for m in metric_not_to_diff:
            for c in test_df.drop(columns=['time_stamp', 'target']).columns.values:
                if not (c.startswith(m)):
                    test_df[c] = test_df[c].diff()
    test_df['target_der'] = test_df['target'].diff()
    test_df.dropna(inplace=True)

    return train_df, test_df


def get_scaled_df(train_df: np.ndarray, test_df: np.ndarray, scaler_x, scaler_y) -> list:
    """
    To scale test and train data this function is written
    :param train_df: data to be scaled with first, X-scaler
    :param test_df:
    :param scaler_x: scaler for data
    :param scaler_y: scaler for target value
    :return: list with input, but scaled
    """
    if train_df.shape[0] > 0:
        # no feature data away
        info_cols = ['time_stamp', 'target', 'target_der']
        info_df_train = train_df[info_cols]
        # extract important data
        x_train = train_df.drop(columns=info_cols)
        cols = x_train.columns.values
        x_train = x_train.values
        # fit and add non features
        scaler_x.fit(x_train)
        x_train = scaler_x.transform(x_train)
        train_df_sc = pd.DataFrame(x_train, columns=cols)

        train_df_sc = pd.concat([train_df_sc.reset_index(drop=True), info_df_train.reset_index(drop=True)], axis=1)
        # with target it's simple
        scaler_y.fit(train_df['target_der'].values.reshape((-1, 1)))
        train_df_sc['target_der'] = scaler_y.transform(train_df_sc['target_der'].values.reshape((-1, 1)))
        train_df = train_df_sc

    if test_df.shape[0] > 0:
        # no feature data away
        info_cols = ['time_stamp', 'target', 'target_der']
        info_df_test = test_df[info_cols]
        # extract important data
        x_test = test_df.drop(columns=info_cols)
        cols = x_test.columns.values
        x_test = x_test.values
        # fit and add non features
        x_test = scaler_x.transform(x_test)
        test_df_sc = pd.DataFrame(x_test, columns=cols)
        test_df_sc = pd.concat([test_df_sc.reset_index(drop=True), info_df_test.reset_index(drop=True)], axis=1)
        # with target it's simple
        test_df_sc['target_der'] = scaler_y.transform(test_df_sc['target_der'].values.reshape((-1, 1)))
        test_df = test_df_sc

    return train_df, test_df, scaler_x, scaler_y


def get_split_points(df, dots=5):
    df_apr = df[df['_dif'] - dots*60 < 5*60][df['_dif'] - dots*60 > 0]
    list_of_splits = df_apr.index.tolist()
    split_points = []
    if list_of_splits:
        start = list_of_splits[0]
        split_points.append(start)
        for j in list_of_splits[1:]:
            if df_apr[df_apr.index == j]['_dif'].values[0] - df_apr[df_apr.index == start]['_dif'].values[0] != 0 or j - start != 1:
                split_points.append(j)
            start = j
    return split_points


def put_new_data_in_df(df, split_points):
    for end_index in split_points:
        start_index = end_index - 1

        start_point = int(df[df.index == start_index]['time_stamp'].values[0])
        end_point = int(df[df.index == end_index]['time_stamp'].values[0])
        split = [i for i in range(start_point + 60, end_point) if i % 60 == 0]

        for point in split:
            dict_for_point = {}
            for column in df.drop(columns=['time_stamp']):

                w = (df[df.index == end_index][column].values[0] - df[df.index == start_index][column].values[0]) / (
                        df[df.index == end_index]['time_stamp'].values[0] -
                        df[df.index == start_index]['time_stamp'].values[0])
                b = df[df.index == end_index][column].values[0] - w * df[df.index == end_index]['time_stamp'].values[0]

                dict_for_point[column] = [point*w + b]
            df_point = pd.DataFrame(dict_for_point)
            df = pd.concat([df, df_point])

    return df


def manage_gaps_after_mult(df, dots=5):
    """
    If there is a gap in data, model will use as previous moments data from so far
    :param df: data frame with column 'time_stamp'
    :param dots: we want to feed dots moments to our model
    :return: df with no incorrect data, there is a column 'time_stamp'
    """
    if df.shape[0] == 0:
        return df
    df['_dif'] = df['time_stamp'].diff(dots)
    # выкидываем те точки, у которых нет данных за предыдущие > 5 минут
    df.drop(df[df['_dif'] - dots * 60 > 0].index, inplace=True)

    df.reset_index(drop=True, inplace=True)
    # TO DO if uncomment it everything goes to hell
    # split_points = get_split_points(df, dots)
    # df = put_new_data_in_df(df, split_points)

    df.sort_values(by=['time_stamp'], inplace=True)
    if '_dif' in df.columns.values:
        df.drop(columns=['_dif'], inplace=True)
    return df


def get_df_with_droped_outliers_s(df, n_s, col='target_der'):
    s = np.sqrt(np.var(df[col].values))
    m = np.mean(df[col].values)

    df = df[df[col] < m + n_s * s]
    return df


def get_df_with_droped_reloads(data, reloads, reload_window=600, inplace=True):
    out = data
    for rel in reloads:
        out = out[(out['time_stamp'] > rel + reload_window) | (out['time_stamp'] < rel)]
    return out


def preprocess1(df, train_size, test_size, scaler_x=MinMaxScaler(), scaler_y=MinMaxScaler(), dots=5,
                n_sigma_outliers=np.inf):
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
    df.dropna(inplace=True)
    # check input for sizes
    if (test_size + train_size) > df.shape[0]:
        test_size = int(df.shape[0] * test_size / (test_size + train_size))
        train_size = int(df.shape[0] * train_size / (test_size + train_size))

    # then let's split for test and train
    train_data, test_data = train_test_split(data=df, test_size=test_size, train_size=train_size, dots=dots)

    # then let's calculate derive for particular columns
    train_data, test_data = get_test_train_derivate(train_data, test_data)
    # here we are to drop reloads
    _reloads_ts = train_data[train_data['target_der'] < 0]['time_stamp'].values
    train_data = get_df_with_droped_reloads(train_data, _reloads_ts, reload_window=60 * 10)

    _reloads_ts = test_data[test_data['target_der'] < 0]['time_stamp'].values
    test_data = get_df_with_droped_reloads(test_data, _reloads_ts, reload_window=60 * 10)
    # TO DO if uncoment next string you'll get lower accuracy, it seems to be due to lack of previous 'target_der'
    # in mult_data, so I guess it's an important part
    # rm outliers for better scaling
    train_data = get_df_with_droped_outliers_s(train_data, n_sigma_outliers)
    # let's scale
    scaling = get_scaled_df(train_data, test_data, scaler_x=scaler_x, scaler_y=scaler_y)

    train_df_diff_sc, test_df_diff_sc, _, diff_scaler_y = scaling
    # and after scaling let's multiply columns to use previous moments

    train_df_diff_sc_m = get_multiplied_data(train_df_diff_sc, dots=dots)
    test_df_diff_sc_m = get_multiplied_data(test_df_diff_sc, dots=dots)

    train_df_diff_sc_m.dropna(inplace=True)
    test_df_diff_sc_m.dropna(inplace=True)
    # here we add a column with ones to train bias
    ones = np.ones((train_df_diff_sc_m.shape[0], 1))
    train_df_diff_sc_m = pd.concat([train_df_diff_sc_m, pd.DataFrame(ones, columns=['ones'])],
                                   axis=1)
    ones = np.ones((test_df_diff_sc_m.shape[0], 1))
    test_df_diff_sc_m = pd.concat([test_df_diff_sc_m,  pd.DataFrame(ones, columns=['ones'])],
                                  axis=1)

    # deal with gaps in data to avoid incorrect usage of multiple dots
    test_df_diff_sc_m = manage_gaps_after_mult(test_df_diff_sc_m, dots)
    train_df_diff_sc_m = manage_gaps_after_mult(train_df_diff_sc_m, dots)

    train_df_diff_sc_m.dropna(inplace=True)
    test_df_diff_sc_m.dropna(inplace=True)

    return test_df_diff_sc_m, train_df_diff_sc_m, scaler_x, scaler_y
