
# COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import logging
from os import makedirs
from datetime import datetime
from route_planner.util import file_name
from route_planner.arguments import args
from route_planner.version import __version__
from route_planner.constants import LOGGING_LEVEL, LOGGING_PATH, LOGGING_FLAG

VERSION = __version__


print(f'Route Planner Module.. version: {VERSION}')
print('COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.')

if LOGGING_FLAG:
    LOGGING_MAP = {
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARN,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
    }

    MSG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    makedirs(LOGGING_PATH, exist_ok=True)
    makedirs(f'{LOGGING_PATH}/{file_name(args.get("input_path"))}', exist_ok=True)
    file_handler = logging.FileHandler(filename=f'{LOGGING_PATH}/{file_name(args.get("input_path"))}/{args.get("execute_type")}_{args.get("equip_type")}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt=MSG_FORMAT, datefmt=DATETIME_FORMAT))
    logger = logging.getLogger()
    logger.setLevel(LOGGING_MAP.get(LOGGING_LEVEL))
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARN)
    logger.addHandler(file_handler)
    matplotlib_logger.addHandler(file_handler)
