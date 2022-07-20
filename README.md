# Module Structure
Path                                                | Decription
--------------------------------------------------- | ---------------------------------------------------
input                                               | Input File 디렉토리
-- *.MDB                                            | MDB or CSV 파일
log                                                 | log 디렉토리
-- {input_file_name}                                | input file 명 디렉토리
    -- {execute_type}_{equip_type}_{timestamp}.log  | 로그 파일, ex) block_dozer_20211014153108.log
output                                              | Output file 디렉토리
-- csv                                              | csv 디렉토리
-- png                                              | png 디렉토리
route_planner                                       | python3.9 기반 소스 코드
process.py                                          | module 실행 파일(main, wrapper)
requirements.txt                                    | python package 목록 



# Constants
Name            | Description           | Default Value
TABLE_NAME      | MDB Read Table Name   | eVolume
COLOR_LIST      | RGB Color List        | ['DDD9C3','C5BE97','948B54','8DB4E3','538ED5','B8CCE4','376091','E6B9B8','D99795','D7DABC','C2D69A','CCC0DA','93CDDD','FCD5B4']
LOGGING_LEVEL   | Logging level         | DEBUG
LOGGING_PATH    | Logging path          | ./log
LOGGING_FLAG    | Logging 여부          | True
OUTPUT_PATH     | Output path           | ./output
DPI             | PNG DPI               |  300
FONT            | PNG default font      | {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 8}
RED_FONT        | PNG virtual cell font | {'family': 'serif', 'color':  'red', 'weight': 'normal', 'size': 8}
