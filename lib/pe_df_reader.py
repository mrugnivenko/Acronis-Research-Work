import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def read_df(file_name: str, folder_path='../datasets/') -> pd.DataFrame:
    """
    Opens df with name "file_name".csv' in folder with ds
    :param file_name: name of file to be written
    :param folder_path: path from the place you run file to the dir with data sets
    :return: df
    """
    full_file_name = file_name + '.csv'
    path_to_file = folder_path + full_file_name
    try:
        out = pd.read_csv(path_to_file)
    except FileNotFoundError:
        out = 'FileNotFoundError'
    return out
