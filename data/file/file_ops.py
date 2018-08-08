
import pandas as pd

class File_ops(object):

    def __init__(self, file_path, file_name):
        self.file_path = file_path + '/' + file_name

    def read_csv_file(self):
        ts_df = pd.read_csv(self.file_path)
        return ts_df
