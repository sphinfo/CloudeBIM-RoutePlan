
# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import logging
from os import makedirs
from datetime import datetime
from route_planner_v20.util import file_name
from route_planner_v20.arguments import args
from route_planner_v20.version import __version__
from route_planner_v20.constants import LOGGING_LEVEL, LOGGING_PATH, LOGGING_FLAG

VERSION = __version__


print(f'Route Planner Module.. version: {VERSION}')
print('COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.')

if LOGGING_FLAG:
    LOGGING_MAP = {
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARN,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
    }
    logging_path = args['logging_path'] if args.get('logging_path') else LOGGING_PATH
    MSG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    makedirs(logging_path, exist_ok=True)

    file_handler = logging.FileHandler(filename=f'{logging_path}/{datetime.now().strftime("%Y%m%d%H%M%S")}.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt=MSG_FORMAT, datefmt=DATETIME_FORMAT))
    logger = logging.getLogger()
    logger.setLevel(LOGGING_MAP.get(LOGGING_LEVEL))
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARN)
    logger.addHandler(file_handler)
    matplotlib_logger.addHandler(file_handler)
