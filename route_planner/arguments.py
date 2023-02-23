# COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import sys
import argparse

parser = argparse.ArgumentParser(description='Route Planner Module', allow_abbrev=False)

parser.add_argument('--execute_type', type=str,   required=True, choices=['all', 'block', 'alloc', 'route', 'movement_plan'], help='실행(all,block,alloc,route,movement_plan)')
parser.add_argument('--input_path', type=str, required=True, help='Input File 경로')
parser.add_argument('--input_file_type', type=str, default='csv', required=True, choices=['csv', 'mdb', 'json'], help='Input File 형식(csv, mdb, geojson)')
parser.add_argument('--output_path', type=str, required=True, help='Output File 경로')
parser.add_argument('--output_file_type', type=str, default='csv', required=False, choices=['csv', 'all'], help='Output File 형식(csv, all)')

parser.add_argument('--sub_type', type=str,   required=False, choices=['fill', 'cut'], help='dozer fill or cut')
parser.add_argument('--equip_type', type=str,   required=False, choices=['dozer', 'roller','grader','paver'], help='장비(dozer, roller, grader, paver)')
parser.add_argument('--start_block_name', type=str, required=False, help='시작 블록 이름')
parser.add_argument('--work_direction', type=int, required=False, help='시작 방향')

parser.add_argument('--e', type=int, required=False, help='1회 작업 역량')
parser.add_argument('--s', type=int, required=False, help='각 셀의 종축길이')
parser.add_argument('--l', type=int, required=False, help='종축 1회 최대 정지거리')
parser.add_argument('--compaction_count', type=int, required=False, help='누적 다짐 회수')
parser.add_argument('--allowable_error_height', type=float, required=False, help='허용오차높이')

parser.add_argument('--borrow_pit_x', type=float, required=False, help='토취장 X좌표')
parser.add_argument('--borrow_pit_y', type=float, required=False, help='토취장 Y좌표')
parser.add_argument('--borrow_pit_z', type=float, required=False, help='토취장 Z좌표')
parser.add_argument('--dumping_area_x', type=float, required=False, help='사토장 X좌표')
parser.add_argument('--dumping_area_y', type=float, required=False, help='사토장 Y좌표')
parser.add_argument('--dumping_area_z', type=float, required=False, help='사토장 Z좌표')
parser.add_argument('--borrow_pit_cut_vol', type=float, required=False, help='토취장 절토량')
parser.add_argument('--dumping_area_fill_vol', type=float, required=False, help='사토장 성토량')

args = {k: v for k, v in parser.parse_args().__dict__.items() if v is not None}

def mandatory_args(__args: dict, *keys):
    for key in keys:
        if key not in __args:
            raise Exception(f'--{key} required')

if args['execute_type'] != 'movement_plan':
    mandatory_args(args, 'equip_type', 'start_block_name', 'work_direction')
    if 'block' not in args['execute_type']:
        mandatory_args(args, 'e', 's', 'l')
        if 'dozer' == args['equip_type']:
            mandatory_args(args, 'allowable_error_height', 'sub_type')
        if 'roller' == args['equip_type'] and any([args['execute_type'] == x for x in ['all', 'route']]):
            mandatory_args(args, 'compaction_count')
else:
    args['equip_type'] = 'movement'
    mandatory_args(args, 'borrow_pit_x', 'borrow_pit_y', 'borrow_pit_z', 'dumping_area_x', 'dumping_area_y', 'dumping_area_z', 'borrow_pit_cut_vol', 'dumping_area_fill_vol')

