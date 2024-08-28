# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import argparse

parser = argparse.ArgumentParser(description='Route Planner Module', allow_abbrev=False)
#parser.add_argument('--input_path', type=str, required=True, help='Input File 경로')
parser.add_argument('--input_cell_file', type=str, required=True, help='입력 그리드 파일 경로(*.json)')
parser.add_argument('--input_line_file', type=str, required=True, help='입력 외곽선 파일 경로(*.csv)')
parser.add_argument('--output_file', type=str, required=True, help="출력 파일 경로")
parser.add_argument('--Blade_Capacity', type=int, required=True, help='블레이드 버킷 용량')
parser.add_argument('--Blade_Width', type=float, required=True, help='블레이드 폭')
parser.add_argument('--Equipment_Width', type=float, required=True, help='장비 폭')
parser.add_argument('--needed_dist', type=float, required=False, default=2.5, help='라인변경에 필요한 거리(상수 변경예정)')
parser.add_argument('--Reapted_rate', type=float, required=False, help='중복도')
parser.add_argument('--Starting_Direction', type=str, required=False, help='작업 시작 방향')
parser.add_argument('--Starting_Point', type=str, required=False, help='작업 진행 방향')
parser.add_argument('--Min_Fwdist', type=float, required=False, default=5.0, help='최소 전진 거리(상수 변경예정)')
parser.add_argument('--Min_Cendist', type=float, required=False, help='중심선 노드간 최소거리(내부 셀에서 연산)')
parser.add_argument('--Obstacle_Cell', type=str, required=False, default='-', help='장애물 셀 지정')
parser.add_argument('--Start_Line', type=int, required=True, help='시작라인')
parser.add_argument('--execute_type', type=str,  default='route', required=False, choices=['alloc', 'route'], help='실행(alloc,route)')


args = {k: v for k, v in parser.parse_args().__dict__.items() if v is not None}

def mandatory_args(__args: dict, *keys):
    for key in keys:
        if key not in __args:
            raise Exception(f'--{key} required')

#mandatory_args(args, 'input_path')