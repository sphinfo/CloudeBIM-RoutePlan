# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import argparse

parser = argparse.ArgumentParser(description='Route Planner Module', allow_abbrev=False)


parser.add_argument('--blade_capacity', type=float, required=True, help='블레이드 버킷 용량(e)')
parser.add_argument('--blade_width', type=float, required=True, help='블레이드 폭')
parser.add_argument('--equipment_width', type=float, required=True, help='장비 폭')
parser.add_argument('--equipment_length', type=float, required=True, help='장비 길이')
parser.add_argument('--repeated_rate', type=float, required=True, help='중복도')
parser.add_argument('--min_fwdist', type=float, required=True, help='최소 전진 거리(s)')

parser.add_argument('--equipment', type=str, required=False,  default='dozer', choices=['dozer', 'grader'], help='장비 종류,')
parser.add_argument('--min_cendist', type=float, required=False, help='중심선 노드간 최소거리(내부 셀에서 연산)')
parser.add_argument('--obstacle_cell', type=str, required=False, help='장애물 셀 지정')
parser.add_argument('--required_line_change_distance', type=float, required=False, default='2.5', help='라인변경에 필요한 거리')

# equipment가 grader일 경우 필수
parser.add_argument('--turning_radius', type=float, required=False, help='회전 반경')
parser.add_argument('--grader_front_length', type=float, required=False, help='그레이더 전방길이')
parser.add_argument('--grader_rear_length', type=float, required=False, help='그레이더 후방길이')


parser.add_argument('--safety_line_df1', type=float, required=False, default=0, help='좌측 안전 거리')
parser.add_argument('--safety_line_df2', type=float, required=False, default=0, help='우측 안전 거리')

parser.add_argument('--input_path', type=str, required=True, help='입력 파일 경로(*.json)')
parser.add_argument('--output_file', type=str, required=True, help="출력 파일")
parser.add_argument('--logging_path', type=str, required=False, help="로깅 경로")
parser.add_argument('--visual_mode', required=False, default=False, action=argparse.BooleanOptionalAction, help='Debugging Visual ON')

args = {k: v for k, v in parser.parse_args().__dict__.items()}
