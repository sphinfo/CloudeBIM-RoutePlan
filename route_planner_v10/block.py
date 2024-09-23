# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import re
import math
import logging
from typing import Any
import numpy as np # type: ignore

from route_planner_v10.constants import MIN_FWDISTANCE, NEEDED_DISTANCE


class Block(object):
    # cells: [{block: dict}], order by 행 asc, 열 asc
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
    def check_moveable(block: dict, obstacle_cells: list):
        return block['yn'] == 'Y' and block['block_name'] not in obstacle_cells

    # cells: [{block: dict}], order by 행 asc, 열 asc
    def sort_cells(cells: list):
        return sorted(cells, key=lambda x: (list(Block.get_bl_i_j(x))[1], list(Block.get_bl_i_j(x))[0]))

    @staticmethod
    def get_bl(cell_data: list):
        blocks = {}
        p = re.compile('BL_[0-9]+_[0-9]+')
        # RENAME_MAP = {'BLName': 'block_name', 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'Y,N': 'yn'}
        #TO SPH Y,N -> YN, totalVol = fillVol=cutVol로 변경
        RENAME_MAP = {'BLName': 'block_name', 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'YN': 'yn',
                      'X1coord': 'x1', 'X2coord': 'x2', 'X3coord': 'x3', 'X4coord': 'x4',
                      'Y1coord': 'y1', 'Y2coord': 'y2', 'Y3coord': 'y3', 'Y4coord': 'y4',}
        for cell in cell_data:
            cell['totalVol'] = cell['fillVol']-cell['cutVol']
            block_name, yn = map(cell.get, ['BLName', 'YN'])
            if block_name is None:
                continue
            if not p.match(block_name):
                raise Exception(f'Block name only allows BL_number_number, block_name: {block_name}')

            i, j = map(lambda x: int(block_name.split('_')[x]), [1, 2])
            blocks.setdefault(j, {})[i] = {RENAME_MAP[k]: float('0' if (v == '-' or v is None) else Block.valid_float(k, v)) if k in ['cutVol', 'fillVol', 'totalVol'] else v for k, v in cell.items() if k in RENAME_MAP}

            for _k in ['XTcoord', 'YTcoord', 'ZTcoord', 'XBcoord', 'YBcoord', 'ZBcoord']:
                if yn == 'Y':
                    _v = cell.get(_k)
                    if not _v:
                        raise Exception(f'Required value, k: {_k}: {_v}')
                    Block.valid_float(_k, _v)

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
    def check_accessible(block: dict, block_items: dict, s_num: int, obstacle_cells: list):
        i, j = Block.get_bl_i_j(block)
        accessible = Block.check_moveable(block, obstacle_cells)
        for x in range(j - 1, j - s_num - 1, -1):
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
            raise Exception(f'Allowed only real number, key: {key}, value: {value}')
    
    @staticmethod
    def valid_integer(key: str, value: Any):
        try:
            return int(value)
        except:
            raise Exception(f'Allowed only integer, key: {key}, value: {value}')


    #SPH 입력 변수 변경으로.. 수정
    @staticmethod
    def valid_parameter(param: list, cell_data: list, model_line_data: list):
        if len(param) != 1:
            logging.getLogger('block').error(f'Only one line of input is allowed')
            raise Exception('Only one line of input is allowed')

        e, s, h, l, obstacle_cells, blade_width, equipment_width, start_line = map(
            param[0].get, ['Blade_Capacity', 'Min_Fwdist', 'needed_dist', 'Min_Cendist', 'Obstacle_Cell', 'Blade_Width', 'Equipment_Width', 'Start_Line']) 

        for _k, _p in [('Blade_Capacity', e), ('Min_Cendist', l), ('blade_width', blade_width), ('equipment_width', equipment_width), ('start_line', start_line)]:
            if not _p:
                logging.getLogger('block').error(f'Required value, k: {_k}: {_p}')
                raise Exception(f'Required value, k: {_k}: {_p}')

        # TODO: 추가적으로 현재 MDB파일의 Inpu_Parameter에서 needed distance와 Min Fwdistance값을 입력값이 아닌 고정값으로 변환하고 싶습니다.
        # 그 값은 우선 needed distance=2.5m, Min Fwdistance=5m로 설정하고 테스트를 통해서 결정할 예정입니다.
        h = h if h else NEEDED_DISTANCE
        s = s if s else MIN_FWDISTANCE

        obstacle_cells = [] if not obstacle_cells else obstacle_cells.replace(' ', '').split(',')

        # valid float
        for _k, _v in [('Min_Fwdist', h), ('needed_dist', h), ('Min_Cendist', l), ('blade_width', blade_width), ('equipment_width', equipment_width), ('Blade_Capacity', e)]:
            Block.valid_float(_k, _v)

        # valid int
        for _k, _v in [('start_line', start_line)]:
            Block.valid_integer(_k, _v)

        h_num, s_num = math.ceil(h / l), math.ceil(s / l)
        gap = max(blade_width, equipment_width) / 2

        outline_data = {'df_l': [], 'df_r': [], 'org_df_l': [], 'org_df_r': []}
        #for mld in sorted([model_line for model_line in model_line_data if model_line.get('No')], key=lambda x: x.get('No')):
        No = 1
        for mld in model_line_data:
            df0, df1, df2 = map(lambda n: {
                'x': Block.valid_float(f'x{n}', mld.get(f'x{n}')),
                'y': Block.valid_float(f'y{n}', mld.get(f'y{n}')),
                #'z': Block.valid_float(f'z{n}', mld.get(f'z{n}'))
                'z':-10.0,
                'No': No
            }, ['0', '1', '2'])
            outline_data['org_df_l'].append(df1)
            outline_data['org_df_r'].append(df2)
            offeset_df_l = Block.offset_point(outline=df1, center=df0, gap=gap)
            offeset_df_r = Block.offset_point(outline=df2, center=df0, gap=gap)
            offeset_df_l.update({'No': No})
            offeset_df_r.update({'No': No})
            outline_data['df_l'].append(offeset_df_l)
            outline_data['df_r'].append(offeset_df_r)
            No+=1

        return {
            'e': e,
            's': s,
            'h': h,
            'l': l,
            'obstacle_cells': obstacle_cells,
            'blade_width': blade_width, 
            'equipment_width': equipment_width,
            'start_line': int(start_line),
            'h_num': h_num,
            's_num': s_num,
            'gap': gap
        }, Block.get_bl(cell_data), outline_data


    # point: (x, y, z)
    # direction_vector: (x_direction_vector, y_direction_vector, z_direction_vector)
    def move_point_parallel(point: tuple, direction_vector: tuple, gap: float) -> tuple:
        return {'x': str(point[0] + gap * direction_vector[0]), 'y': str(point[1] + gap * direction_vector[1]), 'z': str(point[2] + gap * direction_vector[2])}

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
    def offset_point(outline: dict, center: dict, gap: float): # gap(dist) = 평행이동 할 거리
        outline_point = tuple(map(lambda x: outline.get(x), ['x', 'y', 'z']))
        center_point = tuple(map(lambda x: center.get(x), ['x', 'y', 'z']))
        direction_vector = Block.calculate_direction_vector(outline_point, center_point)

        return Block.move_point_parallel(outline_point, direction_vector, gap)
