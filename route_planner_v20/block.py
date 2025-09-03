# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import re
import copy
import math
import logging
from typing import Any
import numpy as np # type: ignore
from route_planner_v20.util import dist_each_node
from route_planner_v20.exception import InputDataError
from route_planner_v20.constants import START_BLOCK, LEFT_BLOCK, RIGHT_BLOCK


class Block(object):
    # cells: [{block: dict}], order by 행 asc, 열 asc
    BLOCK_NAME_REGEX = re.compile('[LRS]BL_[0-9]+_[0-9]+')
    DEFAULT_START_LINE = 1
    BOTTOM_LEFT_COORD = ('x1', 'y1', 'z1')
    BOTTOM_RIGHT_COORD = ('x2', 'y2', 'z2')

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
    def merge_bl(block_items: dict, org_blocks: set):
        merged_block_items = {}

        for block_type, blocks in block_items.items():
            for j, v in blocks.items():
                if j > 0:
                    for i, bl in v.items():
                        if bl.get('block_name') not in org_blocks:
                            merged_block_items.setdefault(j, {})[f'{block_type}{i}'] = bl

        return merged_block_items

    @staticmethod
    def get_bl(cell_data: list):
        blocks, org_blocks = {}, []

        # RENAME_MAP = {'BLName': 'block_name', 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'Y,N': 'yn'}
        RENAME_MAP = {'BLName': 'block_name', 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'YN': 'yn',
                      'X1coord': 'x1', 'X2coord': 'x2', 'X3coord': 'x3', 'X4coord': 'x4', 'X5coord': 'x5',
                      'Y1coord': 'y1', 'Y2coord': 'y2', 'Y3coord': 'y3', 'Y4coord': 'y4', 'Y5coord': 'y5',
                      'Z1coord': 'z1', 'Z2coord': 'z2', 'Z3coord': 'z3', 'Z4coord': 'z4', 'Z5coord': 'z5',}
        
        # overflow_end_line = []
        for cell in cell_data:
            # cell['totalVol'] = cell['fillVol']-cell['cutVol']
            cell['totalVol'] = (float(cell['fillVol']) if cell['fillVol'] else 0.0) - (float(cell['cutVol']) if cell['cutVol'] else 0.0)
            block_name, yn = map(cell.get, ['BLName', 'YN'])

            if block_name is None:
                continue
            if not Block.BLOCK_NAME_REGEX.match(block_name):
                raise InputDataError(f'Block name only allows BL_number_number, block_name: {block_name}')

            i, j = map(lambda x: int(block_name.split('_')[x]), [1, 2])

            if 'OriginalBlock' in cell:
                org_block_name = cell.get('OriginalBlock')
                if Block.BLOCK_NAME_REGEX.match(org_block_name):
                    org_blocks.append(org_block_name)

            # blocks.setdefault(j, {})[i] = {RENAME_MAP[k]: float('0' if (v == '-' or v is None or v == '') else Block.valid_float(k, v)) if k in ['cutVol', 'fillVol', 'totalVol'] else v for k, v in cell.items() if k in RENAME_MAP}
            blocks.setdefault(j, {})[i] = {RENAME_MAP[k]: float('0' if (v == '-' or v is None or v == '') else Block.valid_float(k, v)) if k in ['cutVol', 'fillVol', 'totalVol', 'XTcoord', 'YTcoord', 'ZTcoord', 'XBcoord', 'YBcoord', 'ZBcoord', 'X1coord', 'X2coord', 'X3coord', 'X4coord', 'Y1coord', 'Y2coord', 'Y3coord', 'Y4coord'] else v for k, v in cell.items() if k in RENAME_MAP}
            
            for _k in ['XTcoord', 'YTcoord', 'ZTcoord', 'XBcoord', 'YBcoord', 'ZBcoord']:
                if yn == 'Y':
                    _v = cell.get(_k)
                    if not _v:
                        raise InputDataError(f'Required value, k: {_k}: {_v}')
                    Block.valid_float(_k, _v)
            # v2.0.0 (x1 + x2) / 2
            blocks[j][i]['x1x2_2'] = (blocks[j][i]['x1'] + blocks[j][i]['x2']) / 2

        return blocks, org_blocks
    
    # (x1 + x2) / 2 가 동일할 경우 반대편 블럭 설정
    # LBL
    # x4 x3
    # x1 x2
    
    # x2 x1
    # x3 x4
    # RBL

    # LBL_2_0을 넣는다고하면 
    # x3 -> x1
    # x4 -> x2
    # x1 -> x3
    # x2 -> x4
    @staticmethod
    def set_bottom_by_opposite(lbl: dict, rbl: dict):
        for r in range(0, -3, -1):
            if r + 1 in lbl and abs(r) + 1 in rbl:
                for left_i, left_block in lbl[r + 1].items():
                    for right_block in rbl[abs(r) + 1].values():
                        if left_block['x1x2_2'] is not None and right_block['x1x2_2'] is not None and left_block['x1x2_2'] == right_block['x1x2_2']:
                            lbl.setdefault(r, {})[left_i] = copy.deepcopy(right_block)
                            temp_x, temp_y, temp_z = lbl[r][left_i]['x1'], lbl[r][left_i]['y1'], lbl[r][left_i]['z1']
                            lbl[r][left_i]['x1'], lbl[r][left_i]['y1'], lbl[r][left_i]['z1'] = lbl[r][left_i]['x3'], lbl[r][left_i]['y3'], lbl[r][left_i]['z3']
                            lbl[r][left_i]['x3'], lbl[r][left_i]['y3'], lbl[r][left_i]['z3'] = temp_x, temp_y, temp_z
                            temp_x, temp_y, temp_z = lbl[r][left_i]['x2'], lbl[r][left_i]['y2'], lbl[r][left_i]['z2']

                            lbl[r][left_i]['x2'], lbl[r][left_i]['y2'], lbl[r][left_i]['z2'] = lbl[r][left_i].get('x4'), lbl[r][left_i].get('y4'), lbl[r][left_i].get('z4')
                            lbl[r][left_i]['x4'], lbl[r][left_i]['y4'], lbl[r][left_i]['z4'] = temp_x, temp_y, temp_z
                            temp_x, temp_y, temp_z = lbl[r][left_i]['x_t'], lbl[r][left_i]['y_t'], lbl[r][left_i]['z_t']
                            lbl[r][left_i]['x_t'], lbl[r][left_i]['y_t'], lbl[r][left_i]['z_t'] = lbl[r][left_i]['x_b'], lbl[r][left_i]['y_b'], lbl[r][left_i]['z_b']
                            lbl[r][left_i]['x_b'], lbl[r][left_i]['y_b'], lbl[r][left_i]['z_b'] = temp_x, temp_y, temp_z
                            lbl[r][left_i]['x1x2_2'] = (lbl[r][left_i]['x1'] + lbl[r][left_i]['x2']) / 2 if lbl[r][left_i]['x2'] is not None else None
                            break
            if r + 1 in rbl and abs(r) + 1 in lbl:
                for right_i, right_block in rbl[r + 1].items():
                    for left_block in lbl[abs(r) + 1].values():
                        if left_block['x1x2_2'] is not None and right_block['x1x2_2'] is not None and left_block['x1x2_2'] == right_block['x1x2_2']:
                            rbl.setdefault(r, {})[right_i] = copy.deepcopy(left_block)
                            temp_x, temp_y, temp_z = rbl[r][right_i]['x1'], rbl[r][right_i]['y1'], rbl[r][right_i]['z1']
                            rbl[r][right_i]['x1'], rbl[r][right_i]['y1'], rbl[r][right_i]['z1'] = rbl[r][right_i]['x3'], rbl[r][right_i]['y3'], rbl[r][right_i]['z3']
                            rbl[r][right_i]['x3'], rbl[r][right_i]['y3'], rbl[r][right_i]['z3'] = temp_x, temp_y, temp_z
                            temp_x, temp_y, temp_z = rbl[r][right_i]['x2'], rbl[r][right_i]['y2'], rbl[r][right_i]['z2']
                            rbl[r][right_i]['x2'], rbl[r][right_i]['y2'], rbl[r][right_i]['z2'] = rbl[r][right_i].get('x4'), rbl[r][right_i].get('y4'), rbl[r][right_i].get('z4')
                            rbl[r][right_i]['x4'], rbl[r][right_i]['y4'], rbl[r][right_i]['z4'] = temp_x, temp_y, temp_z
                            temp_x, temp_y, temp_z = rbl[r][right_i]['x_t'], rbl[r][right_i]['y_t'], rbl[r][right_i]['z_t']
                            rbl[r][right_i]['x_t'], rbl[r][right_i]['y_t'], rbl[r][right_i]['z_t'] = rbl[r][right_i]['x_b'], rbl[r][right_i]['y_b'], rbl[r][right_i]['z_b']
                            rbl[r][right_i]['x_b'], rbl[r][right_i]['y_b'], rbl[r][right_i]['z_b'] = temp_x, temp_y, temp_z
                            rbl[r][right_i]['x1x2_2'] = (rbl[r][right_i]['x1'] + rbl[r][right_i]['x2']) / 2 if rbl[r][right_i]['x2'] is not None else None
                            break

            logging.getLogger('block').debug([f'{r}행 생성 LBL_{k}_{r} -> {v.get("block_name")}' for k, v in lbl.get(r, {}).items()])
            logging.getLogger('block').debug([f'{r}행 생성 RBL_{k}_{r} -> {v.get("block_name")}' for k, v in rbl.get(r, {}).items()])

    @staticmethod
    def get_n_m(block_items: dict):
        # return N 최대 행 번호, M 최대 열 번호
        return max(block_items.keys()), max([max(v.keys()) for v in block_items.values()])

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
        converted_1_cells = {}
        for j, _v in block_items.items():
            for _, block in _v.items():
                block_name = block.get('block_name')
                if Block.check_accessible(block, block_items, obstacle_cells):
                    # v2.0.0 진입 가능, 절성토량 존재, 이동 가능(Y)
                    if abs(block.get('total_vol')) > 0 and Block.check_moveable(block, obstacle_cells):
                        case_1_cells[block_name] = block
                        converted_1_cells.setdefault(j, []).append(block)
                    # 진입 가능하고 할당되지 않았을 경우 case_1_cells에 추가
                    # if block_name not in set(allocate_cell_names):
                    #     case_1_cells[block_name] = block
                    #     converted_1_cells.setdefault(j, []).append(block)
                # else:
                    # 2번 유형 진입 불가이고, 절성토량 존재하고 이동 불가인 셀
                    # if abs(block.get('total_vol')) > 0 and not Block.check_moveable(block, obstacle_cells):
                    #     case_2_cells[block_name] = block
                    # v2.0.0 2번 유형: 할당되지 않은 셀, 이동 불가(Y)
                if block_name not in set(allocate_cell_names) and not Block.check_moveable(block, obstacle_cells):
                    case_2_cells[block_name] = block

        return case_1_cells, case_2_cells, converted_1_cells


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


    def get_outline(model_line_data: list, gap: float, safety_line_df1: float, safety_line_df2: float):
        np.seterr(divide='ignore', invalid='ignore')
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
            'data': outline_data,
            'distance': dist_each_node(center_nodes)
        }

    def valid_parameter(args: dict, cell_data: dict, outline_data: dict):
        (
            blade_width, equipment_width, equipment_length, 
            repeated_rate, min_fwdist, equipment, obstacle_cell, 
            required_line_change_distance, turning_radius, grader_front_length,
            grader_rear_length, safety_line_df1, safety_line_df2
        ) =  map(
            args.get, ['blade_width', 'equipment_width', 'equipment_length', 
                       'repeated_rate', 'min_fwdist', 'equipment', 'obstacle_cell', 
                       'required_line_change_distance', 'turning_radius', 'grader_front_length', 
                       'grader_rear_length', 'safety_line_df1', 'safety_line_df2'])

        cell_size = (1 - repeated_rate) * blade_width

        if equipment == 'grader':
            for _k, _p in [('turning_radius', turning_radius), ('grader_front_length', grader_front_length), ('grader_rear_length', grader_rear_length)]:
                if _p is None:
                    logging.getLogger('block').error(f'Required value when grader, k: {_k}: {_p}')            
                    raise InputDataError(f'Required value, k: {_k}: {_p}')

            logging.getLogger('block').debug(f'required_line_change_distance: {required_line_change_distance} -> {turning_radius * 1.3 * (1 - repeated_rate) / 7.5 * 10}')
            required_line_change_distance = turning_radius * 1.3 * (1 - repeated_rate) / 7.5 * 10

            front_cells = math.ceil(grader_front_length / cell_size)
            back_cells  = math.ceil(grader_rear_length  / cell_size)
        else:
            front_cells = back_cells = 0

        
        obstacle_cell = [] if not obstacle_cell else [cell for cell in obstacle_cell.replace(' ', '').split(',') if Block.BLOCK_NAME_REGEX.match(cell)]

        h_num = math.ceil(required_line_change_distance / cell_size)
        s_num = math.ceil(min_fwdist / cell_size)
        space = math.ceil(equipment_length / cell_size)

        start_line = 1 + h_num + space
        start_line += front_cells

        gap = max(blade_width, equipment_width) / 2

        params = {k: v for k, v in args.items()}
        params.update({
            'obstacle_cell': obstacle_cell,
            'required_line_change_distance': required_line_change_distance,
            'start_line': int(start_line),
            'h_num': h_num,
            's_num': s_num,
            'space': space,
            'cell_size': cell_size,
            'front_cells': front_cells,
            'back_cells': back_cells,
            'gap': gap
        })

        start_blocks, org_blocks = Block.get_bl(cell_data[START_BLOCK])
        left_blocks, _ = Block.get_bl(cell_data[LEFT_BLOCK])
        right_blocks, _ = Block.get_bl(cell_data[RIGHT_BLOCK])

        block_items = {
            START_BLOCK: start_blocks,
            LEFT_BLOCK: left_blocks,
            RIGHT_BLOCK: right_blocks
        }
        outline_items = {
            LEFT_BLOCK: Block.get_outline(outline_data[LEFT_BLOCK], gap, safety_line_df1, safety_line_df2),
            RIGHT_BLOCK: Block.get_outline(outline_data[RIGHT_BLOCK], gap, safety_line_df1, safety_line_df2)
        }
        return params, block_items, outline_items, org_blocks