import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def write_df(df: pd.DataFrame, file_name: str, folder_path='../datasets/'):
    """
    save pd df as csv
    :param folder_path: path from the place you run file to the dir with data sets
    :param df: pandas df
    :param file_name: string with name of file, for example name of dc
    :return: nothing
    """
    file_name = file_name + '.csv'
    path_to_file = folder_path + file_name
    df.to_csv(path_or_buf=path_to_file,
              index=False
              )
