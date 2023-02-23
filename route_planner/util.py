# COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import os
import csv, pyodbc
import geojson
import time
from functools import wraps

MDB_DRIVER = '{Microsoft Access Driver (*.mdb)}'

def read_mdb(full_path, table_name):
    con, cur, result = None, None, []
    try:
        if not os.path.exists(full_path):
            raise Exception(f'File not found, {full_path}')

        # connect to mdb
        con = pyodbc.connect(f'DRIVER=Microsoft Access Driver (*.mdb, *.accdb);DBQ={full_path};')
        cur = con.cursor()

        # run a query and get the results (json)
        SQL = f'SELECT * FROM {table_name};' # your query goes here
        rows = cur.execute(SQL).fetchall()
        columns = [column[0] for column in cur.description]
        result = [{column: row[i] for i, column in enumerate(columns)} for row in rows]
        return result
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

def file_name(input_path):
    try:
        return input_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
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
