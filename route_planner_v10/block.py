# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import re
import math
import logging
from typing import Any
import numpy as np # type: ignore
from route_planner_v10.util import dist_each_node, calculate_min_dist_center_node, calculate_s_num
from route_planner_v10.constants import MIN_FWDISTANCE, NEEDED_DISTANCE
from route_planner_v10.exception import InputDataError


class Block(object):
    # cells: [{block: dict}], order by 행 asc, 열 asc
    BLOCK_NAME_REGEX = re.compile('BL_[0-9]+_[0-9]+')

    @staticmethod
    def sort_cells(cells: list):
        return sorted(cells, key=lambda x: (list(Block.get_bl_i_j(x))[1], list(Block.get_bl_i_j(x))[0]))

    @staticmethod
    def get_bl_i_j(block: dict):
        return map(lambda x: int(block.get('block_name').split('_')[x]), [1, 2])

    @staticmethod
    def get_block_names(blocks: list) -> list:
        return [block.get("block_name") for block in blocks]

    @staticmethod
    def get_block_by_name(block_items: dict, block_name: str):
        i, j = map(lambda x: int(block_name.split('_')[x]), [1, 2])
        return block_items[j][i]

    @staticmethod
    def check_moveable(block: dict, obstacle_cells: list):
        return block['yn'] == 'Y' and block['block_name'] not in obstacle_cells

    # cells: [{block: dict}], order by 행 asc, 열 asc
    def sort_cells(cells: list):
        return sorted(cells, key=lambda x: (list(Block.get_bl_i_j(x))[1], list(Block.get_bl_i_j(x))[0]))

    @staticmethod
    def get_bl(cell_data: list, end_line: int, front_cells: int):
        blocks = {}
        # RENAME_MAP = {'BLName': 'block_name', 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'Y,N': 'yn'}
        RENAME_MAP = {'BLName': 'block_name', 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'YN': 'yn',
                      'X1coord': 'x1', 'X2coord': 'x2', 'X3coord': 'x3', 'X4coord': 'x4',
                      'Y1coord': 'y1', 'Y2coord': 'y2', 'Y3coord': 'y3', 'Y4coord': 'y4',}
        # overflow_end_line = []
        for cell in cell_data:
            cell['totalVol'] = cell['fillVol']-cell['cutVol']
            block_name, yn = map(cell.get, ['BLName', 'YN'])
            if block_name is None:
                continue
            if not Block.BLOCK_NAME_REGEX.match(block_name):
                raise InputDataError(f'Block name only allows BL_number_number, block_name: {block_name}')

            i, j = map(lambda x: int(block_name.split('_')[x]), [1, 2])

            # end_line({end_line})보다 행 번호가 큰 경우 정보 추가 하지 않음(셀 정보 삭제)
            # if end_line is not None:
            #     if j > end_line :
            #         overflow_end_line.append(block_name)
            #         continue

            blocks.setdefault(j, {})[i] = {RENAME_MAP[k]: float('0' if (v == '-' or v is None) else Block.valid_float(k, v)) if k in ['cutVol', 'fillVol', 'totalVol'] else v for k, v in cell.items() if k in RENAME_MAP}

            for _k in ['XTcoord', 'YTcoord', 'ZTcoord', 'XBcoord', 'YBcoord', 'ZBcoord']:
                if yn == 'Y':
                    _v = cell.get(_k)
                    if not _v:
                        raise InputDataError(f'Required value, k: {_k}: {_v}')
                    Block.valid_float(_k, _v)
        
        n, _ = Block.get_n_m(blocks)
        _end_line = n if end_line is None else end_line
        _end_line -= front_cells
        logging.getLogger('block').info(f'v1.3.0 end_line: {_end_line}, 최대행: {n}, 입력된 end_line({end_line}), front_cells: {front_cells}')
        for _n in list(blocks.keys()):
            if _n > _end_line:
                logging.getLogger('block').info(f'v1.3.0 end_line({_end_line})보다 행 번호가 큰 행({_n}) 삭제 -> {blocks[_n]}')
                del blocks[_n]
        return blocks
    
    @staticmethod
    def get_n_m(block_items: dict):
        # return N 최대 행 번호, M 최대 열 번호
        return max(block_items.keys()), max(block_items[1].keys())

    @staticmethod
    def get_repeat_count(cell_data: list, e):
        vs = sum([c.get('total_vol') for c in cell_data])
        return math.ceil(abs(vs / e))

    @staticmethod
    def check_accessible(block: dict, block_items: dict, obstacle_cells: list):
        i, j = Block.get_bl_i_j(block)
        accessible = True
        for x in range(j - 1, j - 3, -1):
        # for x in range(j - 1, j - s_num - 1, -1):
            if x < 1:
                accessible = False
                break
            block_x = block_items[x][i]
            accessible_x = Block.check_moveable(block_x, obstacle_cells)
            if x > 0:
                if not accessible_x:
                    accessible = False
                    break
            else:
                accessible = False
                break
        return accessible

    @staticmethod
    def valid_float(key: str, value: Any):
        try:
            return float(value)
        except:
            raise InputDataError(f'Allowed only real number, key: {key}, value: {value}')
    
    @staticmethod
    def valid_integer(key: str, value: Any):
        try:
            return int(value)
        except:
            raise InputDataError(f'Allowed only integer, key: {key}, value: {value}')

    # 1번, 2번 유형 셀 
    @staticmethod
    def get_cell_1_2(block_items: dict, allocate_cell_names: list, obstacle_cells: list):
        case_1_cells, case_2_cells = {}, {}

        for j, _v in block_items.items():
            for _, block in _v.items():
                block_name = block.get('block_name')
                if Block.check_accessible(block, block_items, obstacle_cells):
                    # 진입 가능하고 할당되지 않았을 경우 case_1_cells에 추가
                    if block_name not in set(allocate_cell_names):
                        case_1_cells[block_name] = block
                else:
                    # 2번 유형 진입 불가이고, 절성토량 존재하고 이동 불가인 셀
                    if abs(block.get('total_vol')) > 0 and not Block.check_moveable(block, obstacle_cells):
                        case_2_cells[block_name] = block

        return case_1_cells, case_2_cells


    @staticmethod
    def valid_parameter(args: dict, cell_data: list, model_line_data: list):
        (
            e, s, h, l, obstacle_cells, blade_width, equipment_width,
            start_line, required_line_change_distance, equipment_length,
            equipment, end_line, safety_line_df1, safety_line_df2,
            turning_radius, repeated_rate, grader_front_length, grader_rear_length
        ) = map(
            args.get, ['Blade_Capacity', 'Min_Fwdist', 'needed_dist', 'Min_Cendist', 'Obstacle_Cell',
                           'Blade_Width', 'Equipment_Width', 'Start_Line', 'required_line_change_distance',
                           'equipment_length', 'equipment', 'end_line', 'safety_line_df1', 'safety_line_df2',
                           'turning_radius', 'Repeated_rate', 'grader_front_length', 'grader_rear_length']) 

        # v1.3.0
        cell_size = (1 - repeated_rate) * blade_width

        if equipment == 'grader':
            # for _k, _p in [('turning_radius', turning_radius), ('Repeated_rate', repeated_rate), ('grader_front_length', grader_front_length), ('grader_rear_length', grader_rear_length)]:
            for _k, _p in [('turning_radius', turning_radius), ('grader_front_length', grader_front_length), ('grader_rear_length', grader_rear_length)]:
                if _p is None:
                    logging.getLogger('block').error(f'Required value when grader, k: {_k}: {_p}')            
                    raise InputDataError(f'Required value, k: {_k}: {_p}')
            # v1.1.0 그레이더 일 경우 라인변경에 필요한 거리 다시 계산
            logging.getLogger('block').debug(f'required_line_change_distance: {required_line_change_distance} -> {turning_radius * 1.3 * (1 - repeated_rate) / 7.5 * 10}')
            required_line_change_distance = turning_radius * 1.3 * (1 - repeated_rate) / 7.5 * 10

            # v1.3.0
            front_cells = math.ceil(grader_front_length / cell_size)
            back_cells  = math.ceil(grader_rear_length  / cell_size)
        else:
            front_cells = back_cells = 0

        l = l if l else calculate_min_dist_center_node(model_line_data)

        # TODO: 추가적으로 현재 MDB파일의 Inpu_Parameter에서 needed distance와 Min Fwdistance값을 입력값이 아닌 고정값으로 변환하고 싶습니다.
        # 그 값은 우선 needed distance=2.5m, Min Fwdistance=5m로 설정하고 테스트를 통해서 결정할 예정입니다.
        h = h if h else NEEDED_DISTANCE
        s = s if s else MIN_FWDISTANCE

        obstacle_cells = [] if not obstacle_cells else [cell for cell in obstacle_cells.replace(' ', '').split(',') if Block.BLOCK_NAME_REGEX.match(cell)]

        h_num, s_num = math.ceil(h / l), math.ceil(s / l)
        gap = max(blade_width, equipment_width) / 2

        outline_data = {'df_l': [], 'df_r': [], 'org_df_l': [], 'org_df_r': []}
        #for mld in sorted([model_line for model_line in model_line_data if model_line.get('No')], key=lambda x: x.get('No')):
        No = 1
        # print(model_line_data)
        center_nodes = []
        for mld in model_line_data:
            df0, df1, df2 = map(lambda n: {
                'x': Block.valid_float(f'x{n}', mld.get(f'x{n}')),
                'y': Block.valid_float(f'y{n}', mld.get(f'y{n}')),
                'z': Block.valid_float(f'z{n}', mld.get(f'z{n}')),
                # 'z':-10.0,
                'No': No
            }, ['0', '1', '2'])
            outline_data['org_df_l'].append(df1)
            outline_data['org_df_r'].append(df2)
            offeset_df_l = Block.offset_point(outline=df1, center=df0, gap=gap, safety_line_df=safety_line_df1)
            offeset_df_r = Block.offset_point(outline=df2, center=df0, gap=gap, safety_line_df=safety_line_df2)
            offeset_df_l.update({'No': No})
            offeset_df_r.update({'No': No})
            outline_data['df_l'].append(offeset_df_l)
            outline_data['df_r'].append(offeset_df_r)
            center_nodes.append(df0)
            No+=1

        return {
            'e': e,
            's': s,
            'h': h,
            'l': l,
            'distances': dist_each_node(center_nodes),
            'obstacle_cells': obstacle_cells,
            'blade_width': blade_width, 
            'equipment': equipment,
            'equipment_width': equipment_width,
            'equipment_length': equipment_length,
            'required_line_change_distance': required_line_change_distance,
            'start_line': int(start_line),
            'end_line': end_line,
            'safety_line_df1': safety_line_df1,
            'safety_line_df2': safety_line_df2,
            'turning_radius': turning_radius,
            'repeated_rate': repeated_rate,
            'h_num': h_num,
            's_num': s_num,
            'gap': gap,
            'cell_size': cell_size,
            'front_cells': front_cells,
            'back_cells': back_cells,
        }, Block.get_bl(cell_data, end_line, front_cells), outline_data


    # point: (x, y, z)
    # direction_vector: (x_direction_vector, y_direction_vector, z_direction_vector)
    def move_point_parallel(point: tuple, direction_vector: tuple, gap: float, safety_line_df: float) -> tuple:
        return {
            'x': str(point[0] + gap * direction_vector[0]),
            'y': str(point[1] + gap * direction_vector[1]),
            'z': str(point[2] + gap * direction_vector[2]),
            'safe_x': str(point[0] + (gap + safety_line_df) * direction_vector[0]),
            'safe_y': str(point[1] + (gap + safety_line_df) * direction_vector[1]),
            'safe_z': str(point[2] + (gap + safety_line_df) * direction_vector[2])
        }

    def calculate_direction_vector(outline_point: tuple, center_point: tuple) -> tuple:
        x1, y1, z1 = outline_point
        x2, y2, z2 = center_point
        # 벡터
        direction_vector = (x2 - x1, y2 - y1, z2 - z1)
        # 두 점사이 거리
        magnitude = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2)
        # 벡터에 거리를 나눠 길이가 1인 정규화벡터
        return (direction_vector[0] / magnitude, direction_vector[1] / magnitude, direction_vector[2] / magnitude)

    # outline: {'x': float, 'y': float, 'z': float}
    # center: {'x': float, 'y': float, 'z': float}
    def offset_point(outline: dict, center: dict, gap: float, safety_line_df: float): # gap(dist) = 평행이동 할 거리
        outline_point = tuple(map(lambda x: outline.get(x), ['x', 'y', 'z']))
        center_point = tuple(map(lambda x: center.get(x), ['x', 'y', 'z']))
        direction_vector = Block.calculate_direction_vector(outline_point, center_point)
        return Block.move_point_parallel(outline_point, direction_vector, gap, safety_line_df)
