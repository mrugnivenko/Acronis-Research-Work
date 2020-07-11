from datetime import datetime
import pytz
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae

sys.path.append('../')
import lib.pe_config as Config
import lib.pe_down_loading_data_frame as loader
import lib.pe_df_writer as writer
import lib.pe_utils as utils


def _datetime_to_unix(dt):
    return time.mktime(dt.timetuple())


def datetime_str_to_ts(date_time, form="%d/%m/%Y %H:%M") -> float:
    """
    This function converts date_time string  to time stamp
    :param date_time: string with date_time which is to be converted
    :param form: string with format
    :return: time stamp
    """
    dt = datetime.strptime(date_time, form)

    ts = _datetime_to_unix(dt)
    return ts


def get_date_time_from_stamp(ts: float) -> datetime:
    """
    converts time_stamp to date time, useful fo plotting
    :param ts: time stamp
    :return: string with date time
    """
    # tz = pytz.timezone(Config.get_time_zone())
    out = datetime.fromtimestamp(ts)
    return out


def get_datetime_str_from_stamp(ts: float, form='%d/%m/%Y %H:%M') -> str:
    """
    converts time stamp to string with date_time
    :param ts: time stamp
    :param form: string with format
    :return: string with date time
    """
    tz = pytz.timezone(Config.get_time_zone())
    out_str = datetime.fromtimestamp(ts, tz).strftime(form)
    out = datetime.strptime(out_str, form)
    newformat = out.strftime(form)
    return newformat


def get_inegral(init: float, values: np.ndarray):
    """
    returns a list with integrtet values
    :param init: a constant for integration
    :param values: values of function to integrate
    :return: a list with integrtet values, constant is {init}
    """
    out = [init]
    for i in range(values.shape[0]):
        to_app = float(out[-1]) + float(values[i])
        out.append(to_app)
    return out


def pred_exp_plotter(x, expected, predicted, title, y_axis, x_axis='min', kind='TEST'):
    """
    Quite often we need to plot predicted and test lines together so there is a function to do it
    :param x: values for x axis
    :param expected: 1-dimensional data, data to test
    :param predicted: 1-dimensional data, predicted
    :param title: a piece of title, the other part is kind
    :param y_axis: lable for y-axis
    :param x_axis: lable for x-axis
    :param kind: TEST as default, title is to be like {title + kind}
    :return: nothing, just plots
    """
    _ = plt.plot(x, predicted, label='Predicted')
    _ = plt.plot(x, expected, c='red', label='Expected')
    _ = plt.legend(shadow=True, fontsize='medium', loc='bottom left')
    plt.title(title + ', ' + kind + '')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def get_df_with_renamed_col_for_var(df_var, to_remove=('node',), separator='_'):
    """
    always we got df with hudge column names, so we need to make them shorter, removing some extra information
    :param df_var: df with variance data
    :param to_remove: itterative with list of features fo remove in evry column name
    :param separator: swparator for features
    :return: the same df but with shorter column names
    """
    out = df_var.copy()
    for c in df_var.columns.values:
        if c.startswith('abgw'):
            pieces = c.split(separator)
            new_pieces = pieces
            for p in pieces:
                for r in to_remove:
                    if p.startswith(r):
                        new_pieces.remove(p)
            c_new = separator.join(new_pieces)

            out = out.rename(columns={c: c_new})
    return out


def get_instance_for_dc(dc: str, inst_num: int) -> str:
    """
    it's easy to get instance with only number and dc
    :param dc: name of dc
    :param inst_num: number of instance
    :return: name of instance
    """
    instance = Config.get_inst_format(dc).replace("$", format(inst_num, '02d'))
    return instance


def get_file_name(kind, dc, inst_num, st_time, en_time, dots=None):
    """
    helps to make file names uniform-like for experiments with dc and inst
    :param dots: number of dots we observing now
    :param kind: soe short description
    :param dc: dc name
    :param inst_num: number of instance
    :param st_time: reforested string with start time of experiment
    :param en_time: reforested string with end time of experiment
    :return: sting for file name
    """
    file_name = f'{kind}__{dc}__{inst_num}__{st_time}__{en_time}'
    if dots is not None:
        file_name += f'__{dots}'
    return file_name


def get_list_with_reloads(df: pd.DataFrame, target='target', time_feature='time_stamp'):
    """
    :param df: df where we wnat to find Reloads in certain column. It must have column 'time_stamp'
    :param target: the column to find reloads
    :param time_feature: column name with time feature. this feature, when reload occurs, is the answer
    :return: list of reloads time_stamp
    """
    out = df[df[target].shift(-1) - df[target] < 0][time_feature].tolist()
    return out


def get_upd_anomaly_list(dc: str, inst_num: int, start_time: str,
                         finish_time: str, step_size: str, time_feature='time_stamp'
                         ):
    """
    :param dc:
    :param inst_num:
    :param start_time:
    :param time_feature:
    :param step_size:
    :param finish_time:
    :return: list of update times
    """
    df = loader.get_df_with_upd_metrics(dc=dc, inst_num=inst_num, start_time=start_time,
                                        finish_time=finish_time, step_size=step_size
                                        )
    df_no_nan = df.fillna(0)
    upd_cols = []
    for column in df.columns.values:
        if column != 'time_stamp':
            upd_cols.append(column)
    out = []
    for c in upd_cols:
        zero_one_points = df_no_nan[df_no_nan[c].shift(-1) - df_no_nan[c] < 0][time_feature].tolist()
        out += zero_one_points

    out = list(sorted(set(out)))
    return out


def get_data_with_date(start_time, finish_time, dc, querys, inst_num, step='1m', dots=5, data_file_name=None):
    """
    it is a common situation to load data before experiments, now it is in one function
    :param querys:
    :param dots:
    :param data_file_name:
    :param start_time: time of the beginning for collecting data
    :param finish_time: end time for collecting data
    :param dc: data center to collect data
    :param inst_num: number of instance
    :param step: frequency of data collection
    :return: 'df' with data and 'y' with target
    """

    st_time = start_time.replace(" ", "").replace("/", "").replace(":", "")
    en_time = finish_time.replace(" ", "").replace("/", "").replace(":", "")
    instance = utils.get_instance_for_dc(dc, inst_num)

    target_query = 'sum(abgw_req_latency_ms_sum{dc="' + dc + '", instance="' + instance + '", req="OpenFile"})' + \
                   ' + sum(abgw_req_latency_ms_sum{dc="' + dc + '", instance="' + instance + '", req="Append"})'

    if data_file_name is None:
        data_file_name = utils.get_file_name(kind=f'data_model', dc=dc,
                                             inst_num=inst_num, st_time=st_time, en_time=en_time, dots=dots
                                             )
    out = loader.get_normalized_time_series(query=target_query, finish=finish_time,
                                            start=start_time, step=step)
    if not out.empty:
        out.rename(columns={out.columns.values[0]: 'target'}, inplace=True)
        # now let's download all the other metrics
    for q in querys:
        adj = loader.get_normalized_time_series(query=q, finish=finish_time,
                                                start=start_time, step=step
                                                )
        if adj.empty:
            print(f'empty query: {q}')
        else:
            out = pd.merge(out, adj, how='outer', on='time_stamp')
    writer.write_df(df=out, file_name=data_file_name)
    return out


def get_querys(dc, instance):
    out = [
        ['sum(abgw_req_latency_ms_count{dc="' + dc + '", instance="' + instance + '", req="OpenFile"})' + \
         ' + sum(abgw_req_latency_ms_count{dc="' + dc + '", instance="' + instance + '", req="Append"})',
         'abgw_req_latency_ms_count__Append+OpenFile'],
        ['abgw_iop_latency_ms_count{dc="' + dc + '", instance="' + instance + '", err="OK", proxied="0", iop="open('
                                                                              'pcs)"}',
            'abgw_iop_latency_ms_count__open(pcs)'],
        [
            'abgw_iop_latency_ms_sum{dc="' + dc + '", instance="' + instance + '", err="OK", proxied="0", iop="open('
                                                                               'pcs)"}',
            'abgw_iop_latency_ms_sum__open(pcs)'],
        ['sum(abgw_conns{dc="' + dc + '", instance="' + instance + '"})',
         'abgw_conns'],
        ['abgw_account_lookup_errs_total{dc="' + dc + '", instance="' + instance + '", err="OK"}',
         'abgw_account_lookup_errs_total'],
        ['abgw_account_pull_errs_total{dc="' + dc + '", instance="' + instance + '", err="OK"}',
         'abgw_account_pull_errs_total'],
        ['abgw_accounts{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_accounts'],
        ['abgw_append_throttle_delay_ms_total{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_append_throttle_delay_ms_total'],
        ['abgw_fds{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_fds'],
        ['abgw_file_lookup_errs_total{dc="' + dc + '", instance="' + instance + '", err="OK"}',
         'abgw_file_lookup_errs_total'],
        ['abgw_files{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_files'],
        ['abgw_read_bufs{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_read_bufs'],
        ['abgw_read_bufs_bytes{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_read_bufs_bytes'],
        ['abgw_read_bytes_total{dc="' + dc + '", instance="' + instance + '",proxied="0"}',
         'abgw_read_bytes_total'],
        ['abgw_read_reqs_total{dc="' + dc + '", instance="' + instance + '"}',
         'abgw_read_reqs_total']
    ]
    return out


def load_var_data(dc, inst, start_time, end_time, pri=True, step='1m'):
    """
    this function aims at loading data for variance - long and time -consuming process
    :param dc: data center name
    :param inst: int with number of instance
    :param start_time: time of the beginning '01/01/2020 00:01'
    :param end_time: time of the ending '01/01/2020 00:01'
    :param pri: boolean variable, if True, then prints some states
    :param step: sting with step size, eg '1m'
    :return: path to data
    """
    st_time = start_time.replace(" ", "").replace("/", "").replace(":", "")
    en_time = end_time.replace(" ", "").replace("/", "").replace(":", "")
    if pri:
        print(f'\nvar:\nDC={dc}\nINSTANCE={inst}')
    var_file_name = utils.get_file_name(kind=f'for_variance/var_data', dc=dc, inst_num=inst,
                                        st_time=st_time, en_time=en_time)
    if not os.path.isfile(f'../datasets/{var_file_name}.csv'):
        instance = utils.get_instance_for_dc(dc, inst)
        # if not os.path.isfile(var_file_name)
        # 'abgw_req_latency_ms_bucket{dc="' + dc + '"' + ', instance="'+ instance + '", req="Append"}'
        bq_a = 'abgw_req_latency_ms_bucket{le="$", dc="'+dc+'", instance="' + instance + '", req="Append"}'
        bq_o = 'abgw_req_latency_ms_bucket{le="$", dc="'+dc+'", instance="' + instance + '", req="OpenFile"}'
        bq_desc = 'abgw_req_latency_ms_bucket{le="$", dc="'+dc+'", instance="' + instance + '", req="OpenFile+Append"} '
        bucket_querys_a = [bq_a.replace('$', str(le)) for le in Config.get_var_numbers_1() + ['+Inf']]
        bucket_querys_o = [bq_o.replace('$', str(le)) for le in Config.get_var_numbers_1() + ['+Inf']]
        bucket_querys_desc = [bq_desc.replace('$', str(le)) for le in Config.get_var_numbers_1() + ['+Inf']]
        bucket_querys = []
        for i in range(len(bucket_querys_desc)):
            bucket_querys.append([f'sum({bucket_querys_o[i]}) + sum({bucket_querys_a[i]})', bucket_querys_desc[i]])
        _ = utils.get_data_with_date(start_time=start_time, finish_time=end_time, dc=dc,
                                      querys=bucket_querys, inst_num=inst, step='1m', dots=5,
                                      data_file_name=var_file_name,
                                      )
    else:
        if pri:
            print('\tvar data already loaded')
    return f'../datasets/{var_file_name}.csv'


def load_model_data(dc, inst, start_time, end_time, pri=True):
    """
    this function aims at loading data for variance - long and time -consuming process
    :param dc: data center name
    :param inst: int with number of instance
    :param start_time: time of the beginning '01/01/2020 00:01'
    :param end_time: time of the ending '01/01/2020 00:01'
    :param pri: boolean variable, if True, then prints some states
    :return: path to data
    """
    st_time = start_time.replace(" ", "").replace("/", "").replace(":", "")
    en_time = end_time.replace(" ", "").replace("/", "").replace(":", "")
    if pri:
        print(f'\nmodel:\nDC={dc}\nINSTANCE={inst}')
    data_file_name = utils.get_file_name(kind=f'for_model/data_model', dc=dc, inst_num=inst,
                                         st_time=st_time, en_time=en_time
                                         )
    if not os.path.isfile(f'../datasets/{data_file_name}.csv'):
        instance = utils.get_instance_for_dc(dc, inst)
        querys = utils.get_querys(dc, instance)
        data_file_name = utils.get_file_name(kind=f'for_model/data_model', dc=dc, inst_num=inst,
                                             st_time=st_time, en_time=en_time
                                             )

        df = utils.get_data_with_date(start_time=start_time, finish_time=end_time, dc=dc,
                                      querys=querys, inst_num=inst, step='1m', dots=5,
                                      data_file_name=data_file_name,
                                      )
    else:
        if pri:
            print('\tdata for model is already loaded')
    return f'../datasets/{data_file_name}.csv'
