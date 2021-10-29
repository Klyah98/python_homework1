import os
import shutil
import pandas as pd


def json_from_csv_bytes(input_string: bytes):
    if os.path.exists('../data'):
        shutil.rmtree('../data')
    os.mkdir('../data')
    with open('../data/input_data.csv', 'wb') as file:
        file.write(input_string)
    data = pd.read_csv('../data/input_data.csv')
    return data.to_json()
