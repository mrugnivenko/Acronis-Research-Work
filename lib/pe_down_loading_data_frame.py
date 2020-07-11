import pandas as pd
from datetime import datetime
import numpy as np
import requests
import time
import sys
sys.path.append('../')
import lib.pe_config as Config
import lib.pe_df_reader as reader
import lib.pe_utils as utils
import lib.pe_df_writer as writer
_path = r'config.ini'


def _datetime_to_unix(dt):
    return time.mktime(dt.timetuple())


def _stepsize_to_unix(s):
    """
    stepsize to unix
    """
    seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return int(s[:-1]) * seconds_per_unit[s[-1]]


def _dmy_to_unix(star_time, finish_time, step):
    dt_start = datetime.strptime(star_time, "%d/%m/%Y %H:%M")
    dt_end = datetime.strptime(finish_time, "%d/%m/%Y %H:%M")

    ts_start = _datetime_to_unix(dt_start)
    ts_end = _datetime_to_unix(dt_end)
    out = {'start': ts_start, 'end': ts_end, 'step': _stepsize_to_unix(step)}
    return out


def _get_number_of_query(end, start, step):
    out = (end - start) / step
    out += 1
    return out


def _get_fragmentation(end, start, step):
    out = []
    number_of_querys = _get_number_of_query(end, start, step)
    req_step = min(Config.get_critial_querys(), number_of_querys)
    req_step *= step
    i = 0
    while start + i * req_step < end:
        out.append(start + i * req_step)
        i += 1
    out.append(end)

    return out


def get_time_series(query: str, start: str, finish: str, step: str):
    ts = _dmy_to_unix(start, finish, step)
    fragmentation = _get_fragmentation(ts['end'], ts['start'], ts['step'])
    delta = ts['step'] / 2

    frames = []
    for call in range(len(fragmentation) - 1):
        r = requests.get(f'{Config.get_url()}/api/v1/query_range',
                         params={
                             'query': query[0] if type(query) == list else query,
                             'start': fragmentation[call],
                             'end': fragmentation[call + 1] - delta,
                             'step': ts['step']
                         }
                         )
        res = r.json()
        for result in res['data']['result']:
            df = pd.DataFrame.from_dict(result['metric'], orient='index').T  # got metrics
            if len(result['values']) - 1 != 0:
                df = df.append([df] * (len(result['values']) - 1), ignore_index=True)
            # multiplied metrics for time column

            val_d = result['values']
            # got values = time + value in the other df
            if type(query) == list:
                metric_name = query[0]
            else:
                try:
                    metric_name = df['__name__'][0]
                except KeyError:
                    metric_name = query[:35] + '...'
            val_df = pd.DataFrame(val_d, columns=['time', metric_name])
            # got df with 2 columns of metrics and time

            # gonna join columns in 2 df
            df = val_df.T.append(df.T).T
            # appended to df list
            frames.append(df)

        # 1 request done
        if frames:
            out = pd.concat(frames, ignore_index=True)
            if '__name__' in out.columns.values:
                out = out.drop(columns=['__name__'])
            col_to_drop = Config.get_labels_to_exclude()
            for col in col_to_drop:
                if col in tuple(out.columns.values):
                    out.drop(columns=[col], inplace=True)

            out[metric_name] = out[metric_name].astype(float)
        else:
            out = pd.DataFrame()
    return out


def _query(query: str) -> np.ndarray:
    request = requests.get(f'{Config.get_url()}/api/v1/query',
                           params={'query': query}
                           )
    return request.json()['data']['result']


def get_normalized_time_series(query: str, finish: str, start: str, step: str, target_starts_with='abgw') -> pd.DataFrame:
    """
    this is te function we are to get df with
    :param query: The query - an opportunity to make in manual
    :param start: string with start time like '22/07/2019 13:10'
    :param finish: string with finish time like '22/07/2019 13:10'
    :param step: step size like '1m', no less
    :return: df with columns multiplied by categories like if there are 6 dc and 2 instance
    we got 6 columns with metrics an so one for other features
    """
    out = get_time_series(query, start, finish, step)
    if out.empty:
        return out
    # the_next_level_python all I can say is the order of columns is vital important
    df_columns = out.columns

    target = ''
    for c in out.columns.values:
        if c.startswith(target_starts_with):
            target = c
    if target == '':
        target = out.columns.values[1]

    col_to_group = df_columns.drop([target, 'time'])
    out.set_index(['time'] + col_to_group.to_list(), inplace=True)
    out = out.stack().unstack(level=0).T

    new_columns = []
    for col_old in out.columns:
        new_name = target
        for col_name, col_val in zip(col_to_group, col_old):
            new_name += f'_{col_name}={col_val}'

        new_columns.append(new_name)
    out.columns = new_columns

    out.sort_index(inplace=True)
    out['time_stamp'] = out.index
    out.reset_index(drop=True, inplace=True)

    if type(query) == list:
        rn_d = {}
        col_to_rn = out.drop(columns=['time_stamp']).columns.values
        if len(col_to_rn) > 1:
            for i, c in enumerate(col_to_rn):
                rn_d[c] = f'{c}VERSION{i}'
        else:
            rn_d[col_to_rn[0]] = query[1]
        out.rename(columns=rn_d, inplace=True)
    if out.shape[1] != 2:
        print(f'QUERY {query} returned multiple time series')
    return out


def get_df_with_upd_metrics(dc: str, inst_num: int, start_time: str, finish_time: str, step_size: str):

    st_time = start_time.replace(" ", "").replace("/", "").replace(":", "")
    en_time = finish_time.replace(" ", "").replace("/", "").replace(":", "")
    upd_file_name = utils.get_file_name(kind='upd_data', dc=dc, inst_num=inst_num,
                                        st_time=st_time, en_time=en_time
                                        )
    upd_features = Config.get_upd_features()
    upd_list = []
    df_with_upd = reader.read_df(file_name=upd_file_name)
    if type(df_with_upd) == str:
        df_with_upd = pd.DataFrame()
        for upd_feature in upd_features:
            inst_str = utils.get_instance_for_dc(dc, inst_num)
            upd_query = '%s{dc="%s", instance="%s"}' % (upd_feature, dc, inst_str)
            df_with_feature = get_normalized_time_series(query=upd_query, finish=finish_time,
                                                         start=start_time, step=step_size
                                                         )
            if df_with_upd.empty:
                df_with_upd = df_with_feature
            else:
                if not df_with_feature.empty:
                    df_with_upd = pd.merge(df_with_upd, df_with_feature, how='outer', on='time_stamp')
        df_with_upd.fillna(0, inplace=True)
        writer.write_df(df=df_with_upd, file_name=upd_file_name)
    return df_with_upd
