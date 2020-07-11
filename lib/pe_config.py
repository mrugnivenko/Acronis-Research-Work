import configparser
import os

_path = r'config.ini'


def create_config():
    """
    Create a config file
    """
    config = configparser.ConfigParser()
    config.add_section('API')
    config.set('API', 'URL', 'http://dashboard.stg.corp.acronis.com:9090')
    config.set('API', 'CRITICAl_QUERYS', '10000')
    config.set('API', 'LABELS_TO_EXCLUDE', '__name__ job node')
    upd_features = ['application_build_info',
                    'pcs_process_build_info',
                    'prometheus_build_info',
                    'node_exporter_build_info'
                    ]

    upd_features_str = '__'.join(upd_features)
    config.set('API', 'UPD_FEATURES', upd_features_str)

    config.add_section('QUERY')
    config.add_section('DC:INST_FORMAT')

    dc_ins_format = {'au2-acs1': 'au2-acs1-stor$.vstoragedomain',
                     'eu2-acs1': 'eu2-acs1-stor$.vstoragedomain',
                     'eu3-acs1': 'eu3-acs1-stor$.vstoragedomain',
                     'eu5-acs1': 'eu5-acs1-stor$.vstoragedomain',
                     'eu9-acs1': 'eu9-acs1-stor$.vstoragedomain',
                     'jp2-acs1': 'jp2-acs1-stor$.vstoragedomain',
                     'ne1-aas01': 'ne1-aas01-si$.vstoragedomain',
                     'nissan': 'jp2-nissan-stor$.vstoragedomain',
                     'ru2-acs1': 'ru2-acs1-stor$.vstoragedomain',
                     'sg2': 'stor-$.vstoragedomain',
                     'uc1-aas01': 'uc1-aas01-si$.vstoragedomain',
                     'us2-acs1': 'us2-acs1-stor$.vstoragedomain',
                     'us3': 'us3-acs1-stor$.vstoragedomain',
                     'us3-acs2': 'us3-acs2-stor$.vstoragedomain',
                     'us6-acs1': 'us6-acs1-stor$.vstoragedomain',
                     'us6-acs2': 'us6-acs2-stor$.vstoragedomain'
                     }

    dcs = []
    for dc in dc_ins_format:
        dcs.append(dc)
        config.set('DC:INST_FORMAT', dc, dc_ins_format[dc])
    dcs_str = '__'.join(dcs)

    config.add_section('DC:INST_RANGE')
    dc_range_of_inst = {'au2-acs1': list(range(1, 11)) + list(range(13,22)),
                        'eu2-acs1': list(range(1, 19)),
                        'eu3-acs1': list(range(1, 32)),
                        'eu5-acs1': list(range(1, 10)),
                        'eu9-acs1': list(range(1, 19)),
                        'jp2-acs1': list(range(1, 16)),
                        'ne1-aas01': list(range(1, 4)),
                        'nissan': list(range(1, 8)),
                        'ru2-acs1': list(range(1, 29)) + list(range(30, 31)),
                        'sg2': list(range(1, 12)) + list(range(13, 23)),
                        'uc1-aas01': list(range(1, 4)),
                        'us2-acs1': list(range(1, 13)) + list(range(19, 23)) + list(range(27, 30)),
                        'us3': list(range(1, 6)) + list(range(7, 22)),
                        'us3-acs2': list(range(1, 21)),
                        'us6-acs1': list(range(1, 30)),
                        'us6-acs2': list(range(1, 31))
                        }
    for dc in dc_range_of_inst:
        config.set('DC:INST_RANGE', dc, '__'.join([str(i) for i in dc_range_of_inst[dc]]))


    config.set('QUERY', 'DCS', dcs_str)
    config.set('QUERY', 'TIME_ZONE', 'Europe/Moscow')

    config.add_section('VARMAKER')

    numbers_2 = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000,
                 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
    numbers_1 = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000,
                 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
    config.set('VARMAKER', 'NUMBERS_1', '__'.join([str(i) for i in numbers_1]))
    config.set('VARMAKER', 'NUMBERS_2', '__'.join([str(i) for i in numbers_2]))

    config.add_section('METRIC:OTHER_FEATURES')
    metrics = []
    metric_features = [['abgw_write_reqs_total', ['']],
                       ['abgw_read_reqs_total', ['']],
                       ['abgw_iop_latency_ms_count', [', err="OK", job="abgw", iop="isync", proxied="0"']],
                       ]
    for metr_feat in metric_features:
        metrics.append(metr_feat[0])
        config.set('METRIC:OTHER_FEATURES', metr_feat[0], '__'.join(metr_feat[1]))
    metrics_str = '__'.join(metrics)
    config.set('QUERY', 'METRICS', metrics_str)
    with open(_path, "w") as config_file:
        config.write(config_file)


def get_inst_range_for_dc(dc: str) -> list:
    """
    :return: list of insance numbers for dc
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('DC:INST_RANGE', dc)
    out = out_str.split("__")
    out_int = [int(el) for el in out]
    return out_int


def get_inst_format(dc: str) -> str:
    """
    To be able create inctance for dc properly we are to khow forma fo every dc
    :param dc:
    :return: string with '$' one should replace with 01 or other number
    """
    if not os.path.exists(_path):
        create_config()
    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('DC:INST_FORMAT', dc)
    return out_str

def get_var_numbers_1() -> list:
    """
    in varmaker requests to bucket occurs so numbers are needed
    :return: the first list of numbers
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('VARMAKER', 'NUMBERS_1')
    out_str = out_str.split("__")
    out = [int(i) for i in out_str]
    return out


def get_var_numbers_2() -> list:
    """
    in varmaker requests to bucket occurs so numbers are needed
    :return: the second list of numbers
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('VARMAKER', 'NUMBERS_2')
    out_str = out_str.split("__")
    out = [int(i) for i in out_str]
    return out


def get_feature_for_metric(metric: str) -> list:
    """
    for some labels some features are required, so it gets a list of features for this metric
    :param metric: metric to get additional features(not dc and instance)
    :return: get additional features(not dc and instance)
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('METRIC:OTHER_FEATURES', metric)
    out = out_str.split("__")
    return out


def get_metrics() -> list:
    """
    what are the metrics to observe
    :return: list of metrics to observe
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('QUERY', 'METRICS')
    out = out_str.split("__")
    return out


def get_inst_for_dc(dc) -> list:
    """
    There is no universal rule how combine dc and number to get instance, so here we can get instances
    :param dc:  string with proper dc name
    :return: list of instances as  strings for this dc
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out_str = config.get('DC:INST', dc)
    out = out_str.split("__")
    return out


def get_time_zone() -> str:
    """
    Converting time stamp to datetime requires time zone
    :return: string wit time zone
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out = config.get('QUERY', "TIME_ZONE")
    return out


def get_dcs() -> list:
    """
    what are the dcs we are to dscover?
    :return: dcs_list
    """
    if not os.path.exists(_path):
        create_config()
    config = configparser.ConfigParser()
    config.read(_path)

    out_str = config.get('QUERY', "DCS")
    out = out_str.split("__")
    return out


def get_upd_features() -> list:
    """
    there is some metrics are responsible for updates
    :return: list with some metrics, which are important fo updates
    """

    if not os.path.exists(_path):
        create_config()
    config = configparser.ConfigParser()
    config.read(_path)

    out_str = config.get('API', "UPD_FEATURES")
    out = out_str.split("__")
    return out


def get_url() -> str:
    """
    url for dashboard with data
    :return: URL
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out = config.get('API', "URL")
    return out


def get_critial_querys() -> int:
    """
    It is impossible to download many data for one time
    :return: critical querys as int
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out = int(config.get('API', "CRITICAl_QUERYS"))
    return out


def get_labels_to_exclude():
    """
    :return: critical querys
    """
    if not os.path.exists(_path):
        create_config()

    config = configparser.ConfigParser()
    config.read(_path)
    out = config.get('API', "LABELS_TO_EXCLUDE")
    out = out.split('__')
    return out
