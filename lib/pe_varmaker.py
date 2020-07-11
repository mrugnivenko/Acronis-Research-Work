from datetime import datetime
import pandas as pd
import sys
import numpy as np
sys.path.append('../lib/')

import lib.pe_utils as utils
import lib.pe_config as Config
import lib.pe_down_loading_data_frame as loader

'''
Нужно переписать эту функцию так, чтобы потом можно было без стоаха смотреть на неё.

Воспользуйся pycharm, постарайся чтобы было как можно меньше подчёркиваний

Обязательно чтобы на выходе был стобец с дисперсией и time_stamp. Последний будет представлять собой ключ для 
последующей обработки 
'''


def var_maker(start_time, finis_time, query_bucket, metric_start, metric_end, step='1m', data_path=None):
    if data_path is None:
        df = loader.get_normalized_time_series(query=query_bucket,
                                               finish=finis_time,
                                               start=start_time,
                                               step=step
                                               )
    else:
        df = pd.read_csv(data_path)
    df = utils.get_df_with_renamed_col_for_var(df)
    df1 = pd.DataFrame(df[metric_start + '+Inf' + metric_end])
    df1['summ'] = df[metric_start + '+Inf' + metric_end].diff()
    list_of_reloads_indexes = df1[df1['summ'] < 0].index.tolist()
    list_of_previous_indexes = [i - 1 for i in list_of_reloads_indexes]
    df1['summ'].iloc[list_of_reloads_indexes] += df[metric_start + '+Inf' + metric_end].iloc[list_of_previous_indexes]

    prevmetric = df[metric_start + str(1) + metric_end]
    # я запихнул массив, который начинается с двойки в переменную ниже
    numbers_2 = Config.get_var_numbers_2()
    for num in numbers_2:
        metric = metric_start + str(num) + metric_end
        df[metric] = df[metric] - prevmetric
        prevmetric = df[metric] + prevmetric
    ts = df['time_stamp']

    # я запихнул массив, который начинается с единицы в переменную ниже
    # я думю нет смысла хранить эти оба массива в Config, лучше понть как правильно работать, используя только один
    numbers_1 = Config.get_var_numbers_1()
    for num in numbers_1:
        metric = metric_start + str(num) + metric_end
        # я пытался убрать это окончаие 'WTF', но ничего не получилось
        # вообще нужно всего две колонки, тактчто их оставлять не обязательно, но процесс должен быть рациональным
        df[metric + 'WTF'] = df[metric].diff()
        df[metric + 'WTF'].iloc[list_of_reloads_indexes] += df[metric].iloc[list_of_previous_indexes]
        del df[metric]
    del df[metric_start + '+Inf' + metric_end]

    df = df.drop(df.index[0])
    df['summ'] = df1['summ']
    df['mean'] = df[metric_start + str(1) + metric_end + 'WTF']
    lastnum = 1
    for num in numbers_2:
        metric = metric_start + str(num) + metric_end + 'WTF'
        df['mean'] = df['mean'] + df[metric] * ((num + lastnum) / 2 + 1)
        lastnum = num
    df['mean'] = df['mean'] / df['summ']
    ts = ts.drop(ts.index[0])
    ts = ts.apply(datetime.fromtimestamp)
    df['date'] = ts

    df['variance'] = (1 - df['mean']) ** 2 * df[metric_start + str(1) + metric_end + 'WTF']
    for num in numbers_2:
        metric = metric_start + str(num) + metric_end + 'WTF'
        df['variance'] = df['variance'] + (num - df['mean']) ** 2 * df[metric]
    df['variance'] = df['variance'] / df['summ']

    return df


def get_percentile(path_to_df: str, percentiles=[0.99, 0.95, 0.9]):
    """
    here is a function that returns a pf.DataFrame() with percentiles
    :param path_to_df: path for df with data from bucket
        here must be the following columns:
        abgw_...bucket{le="666", dc="us3", .....}
        'time_stamp'
    :param percentiles: percentiles we want to get
    :returns: pd.DataFrame() with following columsn:
        'time_stamp'
        f'{100 * p}%' for p in percentiles
    """

    # firstl let's load data
    buck_df = pd.read_csv(path_to_df)
    if 'target' in buck_df.columns.values:
        buck_df.drop(columns=['target'], inplace=True)
    # then let's rename columns abgw_req_latency_ms_bucket{le="666", dc="us3",
    # instance="us3-acs1-stor20.vstoragedomain", req="OpenFile+Append"} -> 666
    dict_for_rn = {}
    for col in buck_df.drop(columns=['time_stamp']):
        filters = col.split('{')[1].split('}')[0].split(', ')
        for f in filters:
            if f.startswith('le='):
                if f[4: -1] == '+Inf':
                    dict_for_rn[col] = np.inf
                else:
                    dict_for_rn[col] = int(f[4: -1])
    buck_df.rename(columns=dict_for_rn, inplace=True)
    # here is df with columns ['time_stamp', 1, 2, 3, 4, 5, 10, ...., np.inf]
    # now let's go and get out df
    out = pd.DataFrame(columns=[f'{100 * p}%' for p in percentiles] + ['time_stamp'])
    out['time_stamp'] = buck_df['time_stamp']
    # <YOUR CODE>

    for p in percentiles:
        out[f'{100 * p}%'] = np.ones(buck_df.shape[0]) * p
    return out


def percentile_maker(start_time, finis_time, query_bucket, metric_start, metric_end, anomaly, step='1m',
                     data_path=None):
    if data_path is None:
        df = loader.get_normalized_time_series(query=query_bucket,
                                               finish=finis_time,
                                               start=start_time,
                                               step=step
                                               )
    else:
        df = pd.read_csv(data_path)

    df = utils.get_df_with_renamed_col_for_var(df)
    df1 = pd.DataFrame.copy(df)
    df = df[df.columns[2:]]
    df = df.diff()
    df['summ'] = df[metric_start + '120000' + metric_end]
    list_of_reloads_indexes = df[df['summ'] < 0].index.tolist()
    list_of_previous_indexes = [i - 1 for i in list_of_reloads_indexes]
    df.iloc[list_of_reloads_indexes] += df1.iloc[list_of_previous_indexes]
    df['target'] = df1['target']
    df['time_stamp'] = df1['time_stamp']
    df = df.drop(df.index[0])
    df.dropna(inplace=True)
    df = df[df['time_stamp'].isin([anomaly[0] - shift * 60 for shift in range(60 * 24)] +
                                  [anomaly[1] + shift * 60 for shift in range(1, 60 * 24)])]

    numbers_1 = conf.get_var_numbers_1()
    df['99th_percentile'] = df[df.columns[:33]].apply(lambda row: sum(1 - (row // (row['summ'] * 0.99))), axis=1)
    df.dropna(inplace=True)
    df['99th_percentile'] = df['99th_percentile'].apply(lambda x: numbers_1[int(x) - 1] if int(x) > 0 else numbers_1[0])
    return df