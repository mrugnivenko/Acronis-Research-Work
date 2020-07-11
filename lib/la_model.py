import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import sys
import itertools
from sklearn.metrics import mean_absolute_error as MAE

sys.path.insert(1, '../')
import lib.la_prepro as prepro
import lib.la_utils as la_utils
import lib.pe_utils as utils


def get_best_model():
    return ModelLA(estimator=linear_model.HuberRegressor(), scaler_x=MinMaxScaler(),
                   scaler_y=MinMaxScaler(), dots=3, q=2, imp_features=6, k_test=4, k_train=2
                   )


class ModelLA:
    """.
    It uses some new features, as:
    - seasonal approach
    - usesonly some of features
    """

    def __init__(self, estimator=linear_model.HuberRegressor(), scaler_x=MinMaxScaler(),
                 scaler_y=MinMaxScaler(), dots=3, q=0, n_sigma_outliers=np.inf,
                 seasonal={'lags': [], 'avg': 15}, imp_features=None,
                 col_to_dive=None, k_test=5, k_train=5, step=60
                 ):
        """
        It uses some features, as:
        - seasonal approach
        - sesonaly some of features
        - fill gaps in data
        :param estimator: must have methods .fit( , ); .predict(). HubertReegression as default
        because stable for outliers
        :param scaler_x: must have .fit(), .transform() methods. This sclaler transforms features
        :param scaler_y: must have .fit(), .transform() methods. This sclaler transforms target
        :param dots: AKA p for AR models.
        :param n_sigma_outliers: if there is outliers we can discard data with target greater than .mean() + n * .std()
        This is exactly what gonna happen
        :param seasonal: dict {'lags':[], 'avg': int}. 'lags' set an array with big lags which be used for estimation.
        'avg' is a length of window. I suppose it's worth to use not real lags, but average of some window.
        :param imp_features: list of important features. If int is passed only most "important will be used"
        if list of strings so the passed feature will be used.
        :param col_to_dive: one day I've decided do divide all the features and target by some feature. So to conductit
        name of the feature is needed.
        """
        self.step = step
        self.est = estimator
        self.scaler = MinMaxScaler()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dots = dots
        self.n_s = n_sigma_outliers
        self.features = None
        self.train_size = None
        self.test_size = None
        self.seasonal = seasonal
        self.q = q
        if q > 0:
            self.train_f_hist = pd.DataFrame()
            self.test_err_hist = pd.DataFrame()
        self.k_test = k_test
        if k_test is None and k_train is not None:
            self.k_test = k_train
        self.k_train = k_train
        if k_train is None and k_test is not None:
            self.k_train = k_test
        if type(imp_features) == int:
            self.imp_features = la_utils.get_imp_values(imp_features)
        else:
            self.imp_features = imp_features

        self.col_to_dive = col_to_dive
        self.dr_cols = None
        if self.col_to_dive is None:
            self.dr_cols = ['time_stamp', 'target', 'target_der']
        else:
            self.dr_cols = ['time_stamp', 'target', 'target_der', 'div']

    def fit(self, df, train_size=None):
        """
        fit's model's estimator
        :param train_size: takes tha last %train_size% rows to fit
        :param df: data frame with column 'target' and others for data
        :return: nothing
        """
        _df_c = df.copy()
        _df_c = prepro.fill_misses(_df_c, step=self.step)
        _df_c.reset_index(drop=True, inplace=True)
        if train_size is None:
            train_size = _df_c.shape[0]
        _, _train_df, self.scaler_x, self.scaler_ym = prepro.preprocess_la(df=_df_c, train_size=train_size,
                                                                           test_size=0,
                                                                           scaler_x=self.scaler_x,
                                                                           scaler_y=self.scaler_y,
                                                                           dots=self.dots,
                                                                           n_sigma_outliers=self.n_s,
                                                                           seasonal=self.seasonal,
                                                                           features=self.imp_features,
                                                                           col_to_dive=self.col_to_dive
                                                                           )
        self.est.fit(_train_df.drop(columns=self.dr_cols).values,
                     _train_df['target_der'].values)
        self.features = list(_train_df.drop(columns=self.dr_cols).columns.values)
        if self.q != 0:
            self.train_f_hist = pd.DataFrame(columns=self.features + [f'err_{q + 1}' for q in range(self.q)])
            self.features = list(self.train_f_hist.columns.values)
            for k in range(self.k_train):
                err = _train_df['target_der'].values.reshape(-1) - \
                      self.est.predict(_train_df.drop(columns=self.dr_cols).values).reshape(-1)
                for q in range(self.q):
                    _train_df[f'err_{q + 1}'] = np.zeros((_train_df.shape[0], 1))
                    err_ = np.roll(err, q + 1)
                    err_[0:q + 1] = [0] * (q + 1)
                    _train_df[f'err_{q + 1}'] = err_
                self.est.fit(_train_df.drop(columns=self.dr_cols).values,
                             _train_df['target_der'].values)
                self.train_f_hist.loc[k + 1] = self.est.coef_
        self.train_size = _train_df.shape[0]

    def predict(self, df, test_size=None):
        """
        fit's model's estimator
        :param test_size: takes tha last %train_size% rows to predict
        :param df: data frame with column 'target' and others for data
        :return: df with columns ['time_stamp', 'target_der', 'predictions']
        """
        _df_c = df.copy()
        _df_c = prepro.fill_misses(_df_c, step=self.step)
        _df_c.reset_index(drop=True, inplace=True)
        if test_size is None:
            test_size = _df_c.shape[0]
        else:
            test_size += self.dots - 1
        _test_df, _, self.scaler_x, self.scaler_y = prepro.preprocess_la(df=_df_c, train_size=0,
                                                                         test_size=test_size,
                                                                         scaler_x=self.scaler_x,
                                                                         scaler_y=self.scaler_y,
                                                                         dots=self.dots,
                                                                         n_sigma_outliers=self.n_s,
                                                                         seasonal=self.seasonal,
                                                                         features=self.imp_features,
                                                                         col_to_dive=self.col_to_dive)
        self.test_size = _test_df.shape[0]
        if self.q != 0:
            # adding columns for errors
            for q in range(self.q):
                _test_df[f'err_{q + 1}'] = np.zeros((_test_df.shape[0], 1))

        _out = pd.DataFrame(columns=['time_stamp', 'target_der_sc', 'predictions_sc', 'target'])
        _out['time_stamp'] = _test_df['time_stamp']
        _out['target_der_sc'] = _test_df['target_der']
        _out['predictions_sc'] = self.est.predict(_test_df.drop(columns=self.dr_cols).values)

        if self.col_to_dive is not None:
            _out['target_der_sc'] = np.multiply(_out['target_der_sc'].values, _test_df['div'].values)
            _out['predictions_sc'] = np.multiply(_out['predictions_sc'].values, _test_df['div'].values)

        _out['target'] = _test_df['target']

        if self.q != 0:
            self.test_err_hist = pd.DataFrame(columns=['MAE'])
            for k in range(self.k_test):
                err = _test_df['target_der'].values.reshape(-1) - \
                      _out['predictions_sc'].values.reshape(-1)
                for q in range(self.q):
                    _test_df[f'err_{q + 1}'] = np.zeros((_test_df.shape[0], 1))
                    err_ = np.roll(err, q + 1)
                    err_[0:q + 1] = [0] * (q + 1)
                    _test_df[f'err_{q + 1}'] = err_
                _out['predictions_sc'] = self.est.predict(_test_df.drop(columns=self.dr_cols).values)
                self.test_err_hist.loc[k + 1] = MAE(_out['predictions_sc'].values.reshape(-1),
                                                    _out['target_der_sc'].values.reshape(-1))
        return _out
