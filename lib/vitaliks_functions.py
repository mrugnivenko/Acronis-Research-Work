import sys
sys.path.insert(1, '../')
import lib.pe_df_analyzer as analyzer

def intersection_of_two_anomalies(anomaly_1, anomaly_2):
    print("Stop using intersection_of_two_anomalies function from lib.vitaliks_functions, \nuse lib.pe_df_analyzer")
    if anomaly_1[0] <= anomaly_2[0]:
        if anomaly_1[1] > anomaly_2[0] and anomaly_1[1] <= anomaly_2[1]:
            return [anomaly_2[0], anomaly_1[1]]
        if anomaly_1[1] > anomaly_2[0] and anomaly_1[1] > anomaly_2[1]:
            return [anomaly_2[0], anomaly_2[1]]
        if anomaly_1[1] < anomaly_2[0]:
            return []
        if anomaly_1[1] == anomaly_2[0]:
            return [anomaly_1[1], anomaly_1[1]]
    if anomaly_2[0] <= anomaly_1[0]:
        if anomaly_2[1] > anomaly_1[0] and anomaly_2[1] <= anomaly_1[1]:
            return [anomaly_1[0], anomaly_2[1]]
        if anomaly_2[1] > anomaly_1[0] and anomaly_2[1] > anomaly_1[1]:
            return [anomaly_1[0], anomaly_1[1]]
        if anomaly_2[1] < anomaly_1[0]:
            return []
        if anomaly_2[1] == anomaly_2[0]:
            return [anomaly_1[1], anomaly_1[1]]


def one_vs_all(one_anomaly, list_of_anomaly):
    print("Stop using one_vs_all function from lib.vitaliks_functions, \nuse lib.pe_df_analyzer")
    list_of_intersections = []
    final_list = []
    for anomaly in list_of_anomaly:
        list_of_intersections.append(intersection_of_two_anomalies(one_anomaly, anomaly))
    for i in range(len(list_of_intersections)):
        if list_of_intersections[i]:
            final_list.append(list_of_intersections[i])
    return final_list


def intersection_of_two_lists_of_anomalies(lst1, lst2):
    print("Stop using intersection_of_two_lists_of_anomalies function from lib.vitaliks_functions, \nuse lib.pe_df_analyzer")
    list_of_intercections = []
    final_list = []
    for one_anomaly in lst1:
        list_of_intercections.append(one_vs_all(one_anomaly, lst2))
    for list_of_anomaly_list in list_of_intercections:
        for anomaly_list in list_of_anomaly_list:
            final_list.append(anomaly_list)
    return final_list


def intersection_of_three_lists_of_anomalies(lst1, lst2, lst3):
    print("Stop using intersection_of_three_lists_of_anomalies function from lib.vitaliks_functions, \nuse lib.pe_df_analyzer")
    lst12 = intersection_of_two_lists_of_anomalies(lst1, lst2)
    print(lst12)
    return intersection_of_two_lists_of_anomalies(lst12, lst3)

def find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, mode):
    print("Stop using find_interval_of_anomaly_with_simple_mode function from lib.vitaliks_functions, \nuse lib.pe_df_analyzer")
    s, mu = analyzer.get_var_and_mean(plot_data, mode)
    an = analyzer.get_anomaly_dots(plot_data, s, mu, number_of_sigms, mode)
    list_of_anomalys = analyzer.join_to_window(an, win_len=60 * 30, min_dots=5)
    return list_of_anomalys


def find_interval_of_anomaly_with_mode(plot_data, number_of_sigms, mode):
    print("Stop using find_interval_of_anomaly_with_mode function from lib.vitaliks_functions, \nuse lib.pe_df_analyzer")
    if mode in ['main_error', 'variance', 'mean']:
        list_of_anomalys = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, mode)
        return list_of_anomalys

    if mode == 'and of mean and main_error':
        list_of_anomalys_main_error = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                                'main_error')
        list_of_anomalys_mean = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, 'mean')
        return intersection_of_two_lists_of_anomalies(list_of_anomalys_main_error, list_of_anomalys_mean)

    if mode == 'all and':
        list_of_anomalys_main_error = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                                'main_error')
        list_of_anomalys_mean = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, 'variance')
        return intersection_of_two_lists_of_anomalies(list_of_anomalys_main_error, list_of_anomalys_mean)

    if mode == 'or':
        list_of_all_anomaly = []
        list_of_anomalys_main_error = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms,
                                                                                'main_error')
        list_of_anomalys_variance = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, 'variance')
        list_of_anomalys_mean = find_interval_of_anomaly_with_simple_mode(plot_data, number_of_sigms, 'mean')
        for anomaly in list_of_anomalys_main_error:
            list_of_all_anomaly.append(anomaly)
        for anomaly in list_of_anomalys_variance:
            list_of_all_anomaly.append(anomaly)
        for anomaly in list_of_anomalys_mean:
            list_of_all_anomaly.append(anomaly)
        return list_of_all_anomaly
