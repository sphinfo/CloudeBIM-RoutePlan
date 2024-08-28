# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import os
import csv
import pyodbc # type: ignore
import geojson
import time
import numpy as np
from functools import wraps


MDB_DRIVER = '{Microsoft Access Driver (*.mdb)}'


def get_data(cur, table_name):
    rows = cur.execute(f'SELECT * FROM {table_name};').fetchall()
    columns = [column[0] for column in cur.description]
    return [{column: row[i] for i, column in enumerate(columns)} for row in rows]



def read_mdb(full_path, table_name):
    con, cur, result = None, None, []
    try:
        if not os.path.exists(full_path):
            raise Exception(f'File not found, {full_path}')

        con = pyodbc.connect(f'DRIVER=Microsoft Access Driver (*.mdb, *.accdb);DBQ={full_path};')
        cur = con.cursor()

        return get_data(cur, table_name) if isinstance(table_name, str) else {tb: get_data(cur, tb) for tb in table_name}
    except Exception as e:
        print(str(e))
        raise e
    finally:
        if cur is not None:
            cur.close()
        if con is not None:
            con.close()


def read_csv(full_path):
    if not os.path.exists(full_path):
        raise Exception(f'File not found, {full_path}')

    rows = []
    with open(full_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [dict(row) for row in reader]
    return rows
#TO SPH geojosn reader
def read_geojson(full_path):
    if not os.path.exists(full_path):
        raise Exception(f'File not found, {full_path}')

    rows = []
    with open(full_path, 'r') as jsonfile:
        reader = geojson.loads(jsonfile.read())
        features = reader['features']
        for feature in features:
            etc = ''
            coordinates = feature.geometry.coordinates
            for parts in coordinates:
                for coordinate in parts:
                    etc += str(coordinate[0])+","+str(coordinate[1])+",0"
                etc+";"
            feature.properties["etc"] = etc
            rows.append(feature.properties)
    return rows
#TO SPH min_dist연산
def calculate_min_dist_center_node(linelist):
    distances = []
    for i in range(len(linelist)-1):
        x1, y1 = float(linelist[i]['x0']), float(linelist[i]['y0'])
        x2, y2 = float(linelist[i+1]['x0']), float(linelist[i+1]['y0'])
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(distance)
    return max(distances)

def file_name(input_path):
    try:
        return input_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    except:
        try:
            return input_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
        except:
            return 'unknown'


def to_rgba(rgb: str, alpha=1):
    return (int(rgb[0], 16) * int(rgb[1], 16) / 255, int(rgb[2], 16) * int(rgb[3], 16)  / 255, int(rgb[4], 16) * int(rgb[5], 16)  / 255, alpha)


def log_decorator(log_type):
    def wrapper(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            import logging
            start = time.time()
            result = func(*args, **kwargs)
            logging.info(f'{log_type} 완료.. 소요시간: {time.time() - start}초')
            return result
        return decorator
    return wrapper