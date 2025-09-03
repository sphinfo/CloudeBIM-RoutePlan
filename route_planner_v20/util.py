# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import os
import csv
import geojson # type: ignore
import time
import logging
import numpy as np # type: ignore
from functools import wraps
from route_planner_v20.constants import START_BLOCK, LEFT_BLOCK, RIGHT_BLOCK



def read_csv(full_path):
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File not found, {full_path}')

    rows = []
    with open(full_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [dict(row) for row in reader]
    return rows

#TO SPH geojosn reader
def read_geojson(full_path):
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File not found, {full_path}')

    rows = []
    with open(full_path, 'r') as jsonfile:
        reader = geojson.loads(jsonfile.read())
        features = reader['features']
        for feature in features:
            rows.append(feature.properties)
    return rows

def read_input_files(input_file_path: str):
    cell_data = {}

    for block_type in [START_BLOCK, LEFT_BLOCK, RIGHT_BLOCK]:
        cell_data[block_type] = read_geojson(f'{input_file_path}/grid_cells_{block_type}BL.json')

    outline_data = {
        LEFT_BLOCK: read_csv(f'{input_file_path}/region1.csv'),
        RIGHT_BLOCK: read_csv(f'{input_file_path}/region2.csv')
    }

    return cell_data, outline_data


def file_name(input_path):
    try:
        return input_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    except:
        try:
            return input_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
        except:
            return 'unknown'


def log_decorator(log_type):
    def wrapper(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            logging.info(f'{log_type} 완료.. 소요시간: {time.time() - start}초')
            return result
        return decorator
    return wrapper


def dist_node(node: dict, next_node):
    x1, y1 = node['x'], node['y']
    x2, y2 = next_node['x'], next_node['y']
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def dist_each_node(df0: list):
    return [dist_node(df0[i], df0[i + 1]) for i in range(len(df0) - 1)]
