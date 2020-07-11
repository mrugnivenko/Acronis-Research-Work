import pandas as pd
import numpy as np
import sys
sys.path.append('../lib/')
_path = r'config.ini'


def _find_reboots(df):
    list_of_reboots = list(df[df['target_der'] < 0].index)
    return list_of_reboots


def _delete_rebots(df, list_of_reboots):
    df_without_reboots = df.drop(list_of_reboots)
    return df_without_reboots


def _find_var_and_mean(df_without_reboots,column):
    sigma = df_without_reboots[column].var()**0.5
    mean = df_without_reboots[column].mean()
    return [sigma, mean]


def get_var_and_mean(df, column):
    list_of_reboots = _find_reboots(df)
    df_without_reboots = _delete_rebots(df, list_of_reboots)
    patrametrs = _find_var_and_mean(df_without_reboots, column)
    return patrametrs


def get_anomaly_dots(df, sigma: float, mu: float, n: float, col: str):
    anomaly_df = df[df[col] > mu + n * sigma]
    return anomaly_df['time_stamp'].tolist()


def join_to_mult_window(an_list: list, win_len=30, min_dots=5):
    """
    this function gets a list of dots and returns all the possible intervals of fixed length
    in every interval there are not less then min_dots objects
    :param an_list: list of anomaly dots
    :param win_len: length of window, where we search for anomaly dots
    :param min_dots: min amount of dots in window, to assume as anomaly
    :return: list of tuples [(begin, end), ...] with local anomalies. they may overlap
     """
    out = []
    s_an_list = sorted(an_list)
    _len = len(s_an_list)
    for init_i in range(_len):
        for fin_i in range(min(_len, init_i + min_dots-1), _len):
            if 0 < s_an_list[fin_i] - s_an_list[init_i] <= win_len:
                out.append((s_an_list[init_i], s_an_list[fin_i]))
    return out


def merge_intervals(intervals):
    """
    from stack overflow. it merge intervals. usefull for overlapping intervals returned by join_to_mult_window
    :param intervals: list of tuples [(s, f), ...] with intervals start and end
    :return: list of tuples with after merge
    """
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged


def join_to_window(an_list: list, win_len=30, min_dots=5):
    """
    our anomaly approach is the follows: we got outiers and we wand group them. For every anomaly point there must exist
    a window with not less then min_dots outliers
    :param an_list: list of outliers
    :param win_len: of window where must be some group of outliers
    :param min_dots: amount of outliers in window
    """
    _intervals = join_to_mult_window(an_list, win_len=win_len, min_dots=min_dots)
    out = merge_intervals(_intervals)
    return out


def get_df_with_droped_reloads(data, reloads, reload_window=600):
    out = data
    for rel in reloads:
        out = out[(out['time_stamp'] > rel + reload_window) | (out['time_stamp'] < rel)]
    return out


def get_df_with_fixed_pred(df: pd.DataFrame, reloads: list, reload_window=60 * 12):
    out = df.copy()
    for rel in reloads:
        rel_w = rel + reload_window
        print(f'rel_w = {max(df["time_stamp"])}')
        mask = (out['time_stamp'] > rel_w).astype(int).values
        y = np.mean(out[out['time_stamp'] >= rel & (out['time_stamp'] <= rel_w)]['target'].values)
        y_hat = out[out['time_stamp'] == rel_w]['predicted_target'].values[0]
        out['predicted_target'] += mask * (y - y_hat)
    return out


def get_df_with_no_reloads(df: pd.DataFrame, reloads: list, metric: str):
    if reloads:
        last_piece = [max(reloads), max(df['time_stamp'])]
        pairs = []
        for i in range(len(reloads)-1):
            pairs.append([reloads[i], reloads[i + 1]])
        pairs.append(last_piece)
        out = df[df['time_stamp'] <= min(reloads)]
        for section in pairs:
            bias = np.max(df[df['time_stamp'] >= section[0] - 60 * 60 * 12 & (df['time_stamp'] <= section[0])][metric].values)
            new_= df[df['time_stamp'] > section[0]]
            new_ = new_[new_['time_stamp'] <= section[1]]
            new_[metric] = new_[metric].apply(lambda x: x + bias)
            out = pd.concat([out, new_], ignore_index=True)
        return out
    else:
        return df


def intersection_of_two_anomalies(anomaly_1, anomaly_2):
    if anomaly_1[0] <= anomaly_2[0]:
        if anomaly_2[0] < anomaly_1[1] <= anomaly_2[1]:
            return [anomaly_2[0], anomaly_1[1]]
        if anomaly_1[1] > anomaly_2[0] and anomaly_1[1] > anomaly_2[1]:
            return [anomaly_2[0], anomaly_2[1]]
        if anomaly_1[1] < anomaly_2[0]:
            return []
        if anomaly_1[1] == anomaly_2[0]:
            return [anomaly_1[1], anomaly_1[1]]
    if anomaly_2[0] <= anomaly_1[0]:
        if anomaly_1[0] < anomaly_2[1] <= anomaly_1[1]:
            return [anomaly_1[0], anomaly_2[1]]
        if anomaly_2[1] > anomaly_1[0] and anomaly_2[1] > anomaly_1[1]:
            return [anomaly_1[0], anomaly_1[1]]
        if anomaly_2[1] < anomaly_1[0]:
            return []
        if anomaly_2[1] == anomaly_2[0]:
            return [anomaly_1[1], anomaly_1[1]]


def one_vs_all(one_anomaly, list_of_anomaly):
    list_of_intersections = []
    final_list = []
    for anomaly in list_of_anomaly:
        list_of_intersections.append(intersection_of_two_anomalies(one_anomaly, anomaly))
    for i in range(len(list_of_intersections)):
        if list_of_intersections[i]:
            final_list.append(list_of_intersections[i])
    return final_list


def discard_small_windows(a: list, min_dots=3, step=60):
    for w in a:
        if abs(w[1] - w[0]) < (min_dots - 1) * step:
            a.remove(w)
    return a


def intersection_of_two_lists_of_anomalies(lst1, lst2, min_dots=3, step=60):
    list_of_intercections = []
    final_list = []
    for one_anomaly in lst1:
        list_of_intercections.append(one_vs_all(one_anomaly, lst2))
    for list_of_anomaly_list in list_of_intercections:
        for anomaly_list in list_of_anomaly_list:
            final_list.append(anomaly_list)
    final_list = discard_small_windows(final_list, min_dots=min_dots, step=step)
    return final_list


def intersection_of_three_lists_of_anomalies(lst1, lst2, lst3, min_dots=3, step=60):
    lst12 = intersection_of_two_lists_of_anomalies(lst1, lst2)
    print(lst12)
    out = intersection_of_two_lists_of_anomalies(lst12, lst3)
    out = discard_small_windows(out, min_dots=min_dots, step=step)
    return out


def find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, mode, min_dots=3):
    s, mu = get_var_and_mean(plot_data, mode)
    an = get_anomaly_dots(plot_data, s, mu, number_of_sigms, mode)
    list_of_anomalys = join_to_window(an, win_len=60 * 30, min_dots=min_dots)
    return list_of_anomalys


def find_interval_of_anomaly_with_mode(plot_data, number_of_sigms, mode, min_dots=3):
    if mode in ['main_error', 'variance', 'mean']:
        list_of_anomalys = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, mode,
                                                                     min_dots=min_dots)
        return list_of_anomalys

    if mode == 'and of mean and main_error':
        list_of_anomalys_main_error = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                                'main_error', min_dots=min_dots)
        list_of_anomalys_mean = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                          'mean', min_dots=min_dots)
        inters = intersection_of_two_lists_of_anomalies(list_of_anomalys_main_error, list_of_anomalys_mean)
        return inters

    if mode == 'all and':
        list_of_anomalys_main_error = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                                'main_error', min_dots=min_dots)
        list_of_anomalys_mean = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                          'variance', min_dots=min_dots)
        inters = intersection_of_two_lists_of_anomalies(list_of_anomalys_main_error, list_of_anomalys_mean)
        return inters

    if mode == 'or':
        list_of_all_anomaly = []
        list_of_anomalys_main_error = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                                'main_error', min_dots=min_dots)
        list_of_anomalys_variance = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                              'variance', min_dots=min_dots)
        list_of_anomalys_mean = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                          'mean', min_dots=min_dots)
        for anomaly in list_of_anomalys_main_error:
            list_of_all_anomaly.append(anomaly)
        for anomaly in list_of_anomalys_variance:
            list_of_all_anomaly.append(anomaly)
        for anomaly in list_of_anomalys_mean:
            list_of_all_anomaly.append(anomaly)
        a = list_of_all_anomaly
        return discard_small_windows(a, min_dots=min_dots)
