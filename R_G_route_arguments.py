import sys
import argparse

parser = argparse.ArgumentParser(description='R G Route Planner Module', allow_abbrev=False)
parser.add_argument('--input_file', type=str, required=True, help='입력 파일 경로')
parser.add_argument('--output_file', type=str, required=True, help='출력 파일 경로')
parser.add_argument('--equipment_width', type=float,   required=True, help='장비 폭')
parser.add_argument('--attachment_width', type=float,   required=True, help='어테치먼트 폭')
parser.add_argument('--safety_line', type=float,   required=True, help='안전거리')
parser.add_argument('--x_min', type=float,   required=True, help='중복도 최소범위')
parser.add_argument('--turning_radius', type=float,   required=True, help='회전반경')
parser.add_argument('--starting_position', type=str,   required=True, choices=['1', '2'], help='작업 진행 방향')
parser.add_argument('--starting_direction', type=str,   required=True, choices=['A', 'B'], help='작업 시작 방향')
parser.add_argument('--cycle_num', type=int,   default=1, required=False, help='싸이클 횟수')
parser.add_argument('--line_change_way', type=int, default=1, required=False, choices=[1,2,3], help='1:후진 후 변경, 2:후진 중 변경, 3:3점회전법')
parser.add_argument('--input_target_gpkg_file', type=str, required=False, help='입력 타겟모델 파일 경로')
parser.add_argument('--input_ground_gpkg_file', type=str, required=False, help='입력 지형모델 파일 경로')

args = {k: v for k, v in parser.parse_args().__dict__.items() if v is not None}