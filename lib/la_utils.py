import pandas as pd
import itertools
from datetime import datetime


def is_real_feature(x):
    lp = x.split('__')[-1]
    try:
        float(lp)
        return False
    except ValueError:
        return True


def get_imp_values(num):
    var_df = pd.read_csv('../datasets/Aleksander/variance_of_feature_coeffs.csv')
    mask = [is_real_feature(x) for x in var_df['feature'].to_list()]
    real_features = list(itertools.compress(var_df['feature'].to_list(), mask))
    if 'target' in real_features:
        real_features.remove('target')
    if num > 0:
        out = real_features[:num]
    else:
        out = real_features
    return out


def reformat_feature_name(name):
    nn = name.split('{')
    if len(nn) == 1:
        return name
    else:
        if '__' in name:
            end = name.split('__')[-1]
            out = nn[0] + '__' + end
        else:
            end = ''
            out = nn[0]
    return out


def get_alpha(df, score_col, alpha_col, desireble_score=10):
    _d = df[df[score_col] >= desireble_score].sort_values(by=[score_col])
    out = _d[alpha_col].values[0]
    return out


def get_stagnation_point(df, num_col, loss_col, stagnation_percent=5, crit_flatten=4):
    _df = df[[num_col, loss_col]].sort_values(by=[num_col])
    _df = _df.groupby(num_col, as_index=False).mean()
    nums = _df[num_col].values.reshape(-1)
    loss = _df[loss_col].values.reshape(-1)
    dived = loss.min()
    loss = loss / dived
    out_n = -1
    last_loss = loss[0]
    last_n = nums[0]
    st_in_row = 0
    # print(num_col)
    # print(_df.shape)
    for i in range(len(nums) - 1):
        num_sh = nums[i+1] - last_n
        loss_sh = last_loss - loss[i+1]
        # print(f'loss_sh = {loss_sh:.4f}, loss[i] = {loss[i]:.4f}, num_sh={num_sh}')
        # print(f'perc_sh = {loss_sh / (loss[i] * num_sh) * 100}\n')
        if loss_sh / (last_loss * num_sh) >= stagnation_percent / 100.:
            out_n = nums[i+1]
            st_in_row = 0
            last_loss = loss[i+1]
            last_n = nums[i+1]
        else:
            st_in_row += 1
            if nums[i+1] - last_n >= crit_flatten:
                return out_n
    return out_n


def get_resampled_df(df, mins_res=5):
    data = df.copy()
    data.index = data['time_stamp'].apply(lambda x: datetime.fromtimestamp(x))
    date_df = pd.DataFrame()
    date_df['time_stamp'] = data['time_stamp']
    date_df.index = data.index
    data.drop(columns=['time_stamp'], inplace=True)
    data = data.resample(f'{mins_res}T').mean()
    data = pd.merge(date_df, data, right_index=True, left_index=True)
    data.reset_index(drop=True, inplace=True)
    return data