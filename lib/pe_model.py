import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.decomposition import PCA

sys.path.insert(1, '../')
import lib.pe_preprocessing as prepro
import lib.pe_utils as utils


class PCAModel:
    def __init__(self, pca=PCA(0.98), estimator=linear_model.LinearRegression(n_jobs=-1), scaler_x=MinMaxScaler(),
                 scaler_y=MinMaxScaler(), dots=5, n_sigma_outliers=np.inf):
        self.est = estimator
        self.scaler = MinMaxScaler()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dots = dots
        self.n_s = n_sigma_outliers
        self.pca = pca
        self.features = None

    def fit(self, df, train_size=None):
        """
        fit's model's estimator
        :param train_size: takes tha last %train_size% rows to fit
        :param df: data frame with column 'target' and others for data
        :return: nothing
        """
        _df_c = df.copy()
        _df_c.reset_index(drop=True, inplace=True)
        if train_size is None:
            train_size = _df_c.shape[0]
        _, _train_df, self.scaler_x, self.scaler_y = prepro.preprocess1(df=_df_c, train_size=train_size,
                                                                        test_size=0, scaler_x=self.scaler_x,
                                                                        scaler_y=self.scaler_y, dots=self.dots,
                                                                        n_sigma_outliers=self.n_s)

        _y = _train_df['target_der'].values
        _x = _train_df.drop(columns=['time_stamp', 'target', 'target_der']).values
        self.pca.fit(_x)
        _explained_variance_ratio_ = self.pca.explained_variance_ratio_
        _train_df = self.pca.transform(_x)
        self.est.fit(_train_df, _y)
        self.features = _train_df.drop(columns=['time_stamp', 'target', 'target_der']).columns.values
        return _explained_variance_ratio_


    def predict(self, df, test_size=None):
        """
        fit's model's estimator
        :param test_size: takes tha last %train_size% rows to predict
        :param df: data frame with column 'target' and others for data
        :return: df with columns ['time_stamp', 'target_der', 'predictions']
        """
        _df_c = df.copy()
        _df_c.reset_index(drop=True, inplace=True)
        if test_size is None:
            test_size = _df_c.shape[0]
        else:
            test_size += self.dots - 1
        _test_df, _, self.scaler_x, self.scaler_y = prepro.preprocess1(df=_df_c, train_size=0,
                                                                       test_size=test_size, scaler_x=self.scaler_x,
                                                                       scaler_y=self.scaler_y, dots=self.dots,
                                                                       n_sigma_outliers=self.n_s)


        _out = pd.DataFrame(columns=['time_stamp', 'target_der_sc', 'predictions_sc', 'target'])
        _out['time_stamp'] = _test_df['time_stamp']
        _out['target_der_sc'] = _test_df['target_der']
        _out['target'] = _test_df['target']
        _test_df = self.pca.transform(_test_df.drop(columns=['time_stamp', 'target', 'target_der']).values)
        _out['predictions_sc'] = self.est.predict(_test_df)
        return _out


class Model1_Scaled_Derive:
    def __init__(self, estimator=linear_model.Ridge(alpha=0.01), scaler_x=MinMaxScaler(),
                 scaler_y=MinMaxScaler(), dots=5, n_sigma_outliers=1.5):
        self.est = estimator
        self.scaler = MinMaxScaler()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dots = dots
        self.n_s = n_sigma_outliers
        self.features = None
        self.train_size = None
        self.test_size = None

    def fit(self, df, train_size=None):
        """
        fit's model's estimator
        :param train_size: takes tha last %train_size% rows to fit
        :param df: data frame with column 'target' and others for data
        :return: nothing
        """
        _df_c = df.copy()
        _df_c.reset_index(drop=True, inplace=True)
        if train_size is None:
            train_size = _df_c.shape[0]
        _, _train_df, self.scaler_x, self.scaler_y = prepro.preprocess1(df=_df_c, train_size=train_size,
                                                                        test_size=0, scaler_x=self.scaler_x,
                                                                        scaler_y=self.scaler_y, dots=self.dots,
                                                                        n_sigma_outliers=self.n_s)
        self.train_size = _train_df.shape[0]
        self.est.fit(_train_df.drop(columns=['time_stamp', 'target', 'target_der']).values,
                     _train_df['target_der'].values)
        self.features = _train_df.drop(columns=['time_stamp', 'target', 'target_der']).columns.values

    def predict(self, df, test_size=None):
        """
        fit's model's estimator
        :param test_size: takes tha last %train_size% rows to predict
        :param df: data frame with column 'target' and others for data
        :return: df with columns ['time_stamp', 'target_der', 'predictions']
        """
        _df_c = df.copy()
        _df_c.reset_index(drop=True, inplace=True)
        if test_size is None:
            test_size = _df_c.shape[0]
        else:
            test_size += self.dots - 1
        _test_df, _, self.scaler_x, self.scaler_y = prepro.preprocess1(df=_df_c, train_size=0,
                                                                       test_size=test_size, scaler_x=self.scaler_x,
                                                                       scaler_y=self.scaler_y, dots=self.dots,
                                                                       n_sigma_outliers=self.n_s)
        self.test_size = _test_df.shape[0]
        _out = pd.DataFrame(columns=['time_stamp', 'target_der_sc', 'predictions_sc', 'target'])
        _out['time_stamp'] = _test_df['time_stamp']
        _out['target_der_sc'] = _test_df['target_der']
        _out['predictions_sc'] = self.est.predict(_test_df.drop(columns=['time_stamp', 'target', 'target_der']).values)
        _out['target'] = _test_df['target']
        return _out


def plotting_and_mse(df: pd.core.frame.DataFrame,
                     test_size: int,
                     train_size: int,
                     print_plot=False,
                     estimator=linear_model.LinearRegression(n_jobs=-1),
                     scaler=MinMaxScaler(),
                     dots=1):
    """
    The last model for predictions. After splitting for test and train we get data with derivatives
    Then derivate is scaled and predicted. Then as we get drivate it is time to integrade and turn for integration
    With integration the target predicts
    :param df: data for test and train and a column with 'target'
    :param test_size: size of test. The last test_size points will be used for test
    :param train_size: size of train data. The losest for test piece of data will be taken. Dont forget to pass extra
    :param print_plot: should we print MSE and make plots?
    :param estimator: estimator to predict derivate, default - LinearRegressor
    :param scaler: a string with a message which scaler is to be used. If 'Standart', then StandartSaler from sklearn
                    if Normalizer, then our owns
    :param dots: how many values of one metric we shold use. If more than one so some extra values from
                    previous observations will be added
    :return: dictionary with fields:    'pred_sc_der_test' - predictions for scaled derivation on test
                                        'exp_sc_der_test' - expected values of scaled derivation
                                        'pred_sc_der_train' - predictions for scaled derivation on train
                                        'exp_sc_der_test' - expected values of scaled derivation on train
                                        'pred_train' -  predictions on train data
                                        'exp_train' - target, train data
                                        'pred_test' - predictions on test data
                                        'exp_test' - target, test data
    """

    prepro_res = prepro.preprocess1(df=df, train_size=train_size,
                                    test_size=test_size, scaler=scaler, dots=dots
                                    )
    train_data, test_data, train_target, test_target, scaler_x, scaler_y = prepro_res

    linear_regressor = estimator
    linear_regressor.fit(train_data, train_target)

    predictions_sc_test = linear_regressor.predict(test_data)
    predictions_sc_train = linear_regressor.predict(train_data)

    out = {'pred_sc_der_test': predictions_sc_test, 'exp_sc_der_test': test_target,
           'pred_sc_der_train': predictions_sc_train, 'exp_sc_der_train': train_target
           }

    if print_plot:
        x_test = list(range(test_data.shape[0]))
        utils.pred_exp_plotter(x=x_test,
                               expected=out['exp_sc_der_test'],
                               predicted=out['pred_sc_der_test'],
                               y_axis='scaled', kind='TEST', title='Scaled derivate'
                               )
        x_train = list(range(train_data.shape[0]))
        utils.pred_exp_plotter(x=x_train,
                               expected=out['exp_sc_der_train'],
                               predicted=out['pred_sc_der_train'],
                               y_axis='scaled', kind='TRAIN', title='Scaled derivate'
                               )
    y_test = scaler_y.inverse_transform(predictions_sc_test)
    y_train = scaler_y.inverse_transform(predictions_sc_train)
    integr_test = utils.get_inegral(init=test_target.tolist()[0][0], values=y_test)

    integr_train = utils.get_inegral(init=train_target.tolist()[0][0], values=y_train)

    out['pred_train'] = integr_train
    out['exp_train'] = train_target

    out['pred_test'] = integr_test
    out['exp_test'] = test_target
    if print_plot:
        utils.pred_exp_plotter(x=x_test, predicted=integr_test[:-1],
                               expected=out['exp_test'],
                               title='Target', y_axis='target', kind='TEST'
                               )

        utils.pred_exp_plotter(x=x_train, predicted=integr_train[:-1],
                               expected=out['exp_train'],
                               title='Target', y_axis='target', kind='TRAIN'
                               )
    return out
