# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import math
import csv
import logging
import json
from os import makedirs
from shapely import BufferCapStyle # type: ignore
from shapely.geometry import LineString, Polygon # type: ignore

from route_planner_v20.block import Block
from route_planner_v20.util import log_decorator
from route_planner_v20.constants import SHOW_ALLOC_CELL_FLAG
from route_planner_v20.exception import InputDataError, RouteCreationError

class RoutePlan():
    def __init__(self, param: dict):
        self.route_plan = []

        for k, v in param.items():
            setattr(self, k, v)

    def add_route(self, coord: dict, allocate_cell_name: str, cell_name: str):
        coord.update({'allocate_cell_name': allocate_cell_name, 'cell_name': cell_name})
        
        # for c in ['x', 'y', 'z']:
        #     coord[c] = None if float(coord[c]) == 0 else coord[c]

        self.route_plan.append(coord)
        logging.getLogger(f'plan').debug(json.dumps(coord, ensure_ascii=False))

    # 단일 경로 추가
    def add_single_route_plan(self, coord: dict, forward: bool, allocate_cell_name: str, cell_name: str):
        # coord: {'x':0, 'y':0, 'z': 0}
        direction = 1 if forward else -1
        coord.update({'direction': direction})

        # 마지막 경로와 좌표가 같을 경우 경로에 추가하지 않음
        latest_route = self.route_plan[-1] if self.route_plan else {}
        if latest_route.get('x') != coord.get('x') or latest_route.get('y') != coord.get('y')  or latest_route.get('z') != coord.get('z'):
            self.add_route(coord=coord, allocate_cell_name=allocate_cell_name, cell_name=cell_name)
        else:
            logging.getLogger(f'plan').debug(f'마지막 경로와 좌표가 동일하여 경로 생성 skip - {cell_name}')

    # 경로 추가
    def add_route_plan(self, block: dict, forward: bool, allocate_cell_name: str, cell_name: str):
        # 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'Y,N': 'yn',
        direction = 1 if forward else -1
        x_t, y_t, z_t, x_b, y_b, z_b = map(block.get, ['x_t', 'y_t', 'z_t', 'x_b', 'y_b', 'z_b'])
        latest_route = self.route_plan[-1] if self.route_plan else {}

        if forward:
            # BOTTOM -> TOP
            # 마지막 경로와 좌표가 같을 경우 경로에 추가하지 않음
            if latest_route.get('x') != x_b or latest_route.get('y') != y_b  or latest_route.get('z') != z_b:
                self.add_route(coord={'x': x_b, 'y': y_b, 'z': z_b, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}-B')
            else:
                logging.getLogger(f'plan').debug(f'마지막 경로와 좌표가 동일하여 경로 생성 skip - {cell_name}-B')
            self.add_route(coord={'x': x_t, 'y': y_t, 'z': z_t, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}-T')
        else:
            # TOP -> BOTTOM
            if latest_route.get('x') != x_t or latest_route.get('y') != y_t  or latest_route.get('z') != z_t:
                self.add_route(coord={'x': x_t, 'y': y_t, 'z': z_t, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}-T')
            else:
                logging.getLogger(f'plan').debug(f'마지막 경로와 좌표가 동일하여 경로 생성 skip - {cell_name}-T')
            self.add_route(coord={'x': x_b, 'y': y_b, 'z': z_b, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}-B')


    @log_decorator('계획 경로 알고리즘 CSV 저장')
    def save_output_csv(self, output_file: str):
        makedirs(output_file.rsplit('/', 1)[0], exist_ok=True)
        with open(output_file, 'w', newline='\n', encoding='utf-8') as csvfile:
            headers = ['x', 'y', 'direction', 'z1', 'allocate_cell_name', 'cell_name'] if SHOW_ALLOC_CELL_FLAG else ['x', 'y', 'direction', 'z1']
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for i, v in enumerate(self.route_plan):
                # v.update({'Timeline': i + 1})
                if 'z' in v:
                    v['z1'] = v.pop('z')
                if not SHOW_ALLOC_CELL_FLAG:
                    v.pop('allocate_cell_name')
                    v.pop('cell_name')
            writer.writerows(self.route_plan)

    # converted_block: BlName key dict
    @log_decorator('스타트라인 계획 경로 알고리즘')
    def calc_start_route_plan(self, block_items: dict, allocate_cell: dict, allocate_cell_names: list, block_type: str):
        # M: 전체 셀데이터 열번호 중 최고값, N: 할당셀 행번호 중 최고값
        self.block_items = block_items
        self.allocate_cell = allocate_cell
        self.allocate_cell_names = allocate_cell_names
        self.N, self.M = max(self.allocate_cell.keys()), max(self.block_items[1].keys())

        logging.getLogger(f'plan-{block_type}').debug(f'불도저 버켓용량(e): {self.blade_capacity}')
        logging.getLogger(f'plan-{block_type}').debug(f'장애물셀: {self.obstacle_cell}, Start Line: {self.start_line}(1 + h_num({self.h_num}) + space({self.space}) + front_cells({self.front_cells}))')
        logging.getLogger(f'plan-{block_type}').debug(f'라인변경에 필요한 거리(required_line_change_distance): {self.required_line_change_distance}, 장비길이(equipment_length): {self.equipment_length}')
        logging.getLogger(f'plan-{block_type}').debug(f'최대 열 번호 M = {self.M}, 최대 행 번호 N = {self.N}')

        first_j = next(iter(self.allocate_cell))
        first_i = next(iter(self.allocate_cell[first_j]))

        i, j, j_min = 0, 1, None
        i_cur, j_cur = Block.get_bl_i_j(self.allocate_cell.get(first_j, {}).get(first_i, {}).get('cells', [])[0])

        while(j <= self.N):
            alloc_is_first = True
            logging.getLogger(f'plan-{block_type}').debug(f'i = {i}, j = {j}')
            i += (-1 * int(math.pow(-1, j)))
            logging.getLogger(f'plan-{block_type}').debug(f'다음 열로 이동, i: {i}')

            ran = range(i, 0, -1) if (-1 * int(math.pow(-1, j))) == -1 else range(i, self.M + 1, 1)
            exist_alloc = False
            for i in ran:
                # 현재 할당셀(AL_i_j)에 할당된 셀이 존재하는가? 
                logging.getLogger(f'plan-{block_type}').debug(f'현재 할당셀({block_type}AL_{i}_{j})에 할당된 셀이 존재하는가? {self.allocate_cell.get(j, {}).get(i) is not None}')
                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) > 0:
                    exist_alloc = True
                    break
                logging.getLogger(f'plan-{block_type}').debug(f'다음 열로 이동, i({i}) -> i({i + (-1 * int(math.pow(-1, j)))})')

            if not exist_alloc:
                j += 1
                logging.getLogger(f'plan-{block_type}').debug(f'다음 행으로 이동, j({j})+=1')
                continue

            initial_i = i
            for i in range(i, 0, -1) if (-1 * int(math.pow(-1, j))) == -1 else range(i, self.M + 1, 1):
                # AL_i_j에 할당된 셀이 있는가? 없을 경우 다음 열 이동
                if i != initial_i:
                    logging.getLogger(f'plan-{block_type}').debug(f'{block_type}AL_i_j({block_type}AL_{i}_{j})에 할당된 셀이 있는가? {not len(self.allocate_cell.get(j, {}).get(i, {}).get("cells", [])) == 0}')

                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) == 0:
                    logging.getLogger(f'plan-{block_type}').debug(f'다음 열 이동')
                    continue

                i_next, j_next = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                j_max = list(Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells')[-1]))[1]

                # i_next ==i_cur And j_next==j_cur
                logging.getLogger(f'plan-{block_type}').debug(f'i_next({i_next}) ==i_cur({i_cur}) And j_next({j_next})==j_cur({j_cur}): {not (i_next != i_cur or j_next != j_cur)}')
                if i_next != i_cur or j_next != j_cur:
                    j_next -= 1
                    logging.getLogger(f'plan-{block_type}').debug(f'j_cur({j_cur}) >= j_next({j_next}) - h_num : {j_cur >= j_next - self.h_num}')
                    if j_cur >= j_next - self.h_num:
                        # BL_(i_cur)_(j_cur)의 후방 이동점부터 BL_(i_cur)_(j_next-h_num)의 후방 이동점까지 후진 경로 생성
                        logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_cur})_({j_cur})의 후방 이동점부터 {block_type}BL_({i_cur})_({j_next}-{self.h_num})의 후방 이동점까지 후진 경로 생성')
                        self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}-B')
                        for _j in range(j_cur - 1, j_next - self.h_num - 1, -1):
                            self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')
                        
                        # j_cur = j_next- h_num
                        j_cur = j_next- self.h_num

                    # BL_(i_cur)_(j_cur) 의 후방 이동점에서 BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                    logging.getLogger(f'plan-{block_type}').debug(f'BL_i_cur({i_cur})_j_cur({j_cur}) 의 후방 이동점에서 BL_i_next({i_next})_j_next({j_next}-1) 의 전방 이동점으로 전진경로 생성')
                    self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                    self.add_single_route_plan(coord={'x': self.block_items[j_next - 1][i_next]['x_t'], 'y': self.block_items[j_next - 1][i_next]['y_t'], 'z': self.block_items[j_next - 1][i_next]['z_t']}, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_next - 1][i_next].get("block_name")}-T')

                # j_min= AL_i_j의 행번호가 가장 낮은 셀의 행번호
                _, j_min = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                logging.getLogger(f'plan-{block_type}').debug(f'j_min({j_min})= {block_type}AL_i({i})_j({j})의 행번호가 가장 낮은 셀의 행번호')

                # BL_(i_next)_(j_max)의 전방 이동점까지 전진경로 생성
                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_max})의 전방 이동점 까지 전진경로 생성')
                for block in self.allocate_cell[j][i]['cells']:
                    self.add_route_plan(block=block, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{block.get("block_name")}')

                # 반복 횟수 만큼 반복
                repeat_count = self.allocate_cell[j][i]['repeat_count']
                repeat_route = []
                # 반복횟수[R] 만족했는가?
                logging.getLogger(f'plan-{block_type}').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {0}, {not (repeat_count > 0)}')
                for r in range(1, repeat_count + 1):
                    if not repeat_route:
                        before_route_index = len(self.route_plan)
                        # BL_(i_next)_(j_min)의 후방 이동점까지 후진경로 생성
                        logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_min})의 후방 이동점까지 후진경로 생성')
                        for _j in range(j_max, j_min - 1, -1):
                            self.add_route_plan(block=self.block_items[_j][i_next], forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_next].get("block_name")}')

                        # BL_(i_next)_(j_max)의 전방 이동점까지 전진경로 생성
                        logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_max})의 전방 이동점 까지 전진경로 생성')
                        for block in self.allocate_cell[j][i]['cells']:
                            self.add_route_plan(block=block, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{block.get("block_name")}')

                        repeat_route = self.route_plan[before_route_index:len(self.route_plan)]
                    else:
                        logging.getLogger(f'plan-{block_type}').debug(f'반복으로 인한 동일 경로 추가')
                        self.route_plan.extend(repeat_route)
                    # 반복횟수[R] 만족했는가?
                    logging.getLogger(f'plan-{block_type}').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {r}, {not (repeat_count > r)}')

                i_cur, j_cur = i_next, j_max

                # v1.3.0
                # 열이 2개만 있을 경우 비효율적으로 후진하는 상황 방지.
                # M == 2 인 경우  경로 생성 알고리즘 추가  (일반 셀로 할당 후 일반 경로로 셀 생성)
                logging.getLogger(f'plan-{block_type}').debug(f'M == 2? : {self.M == 2}')
                if self.M == 2:
                    # AL_i_j가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가?
                    logging.getLogger(f'plan-{block_type}').debug(f'할당셀({block_type}AL_{i}_{j})가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가? {alloc_is_first}')
                    if alloc_is_first:
                        alloc_is_first = False
                    else:
                        for _ in range(j, self.N):
                            j += 1
                            # 현재 할당셀(AL_i_j)에 할당된 셀이 존재하는가? 
                            logging.getLogger(f'plan-{block_type}').debug(f'현재 할당셀({block_type}AL_{i}_{j})에 할당된 셀이 존재하는가? {self.allocate_cell.get(j, {}).get(i) is not None}')
                            if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) > 0:
                                # j_max= AL_i_j 에서 행번호가 가장 높은 값
                                j_max = list(Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells')[-1]))[1]
                                logging.getLogger(f'plan-{block_type}').debug(f'j_max({j_max})= {block_type}AL_i({i})_j({j}) 에서 행번호가 가장 높은 값')
                                # BL_(i_cur)_(j_max)의 전방 이동점까지 전진 경로 생성
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_cur})_({j_max})의 전방 이동점 까지 전진경로 생성')
                                for block in self.allocate_cell[j][i]['cells']:
                                    self.add_route_plan(block=block, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{block.get("block_name")}')
                                j_cur = j_max
                                break

        return self.route_plan


    # converted_block: BlName key dict
    @log_decorator('계획 경로 알고리즘')
    def calc_route_plan(self, block_items: dict, allocate_cell: dict, allocate_cell_names: list, allocate_outline_cell: dict, outline_items: dict, block_type: str):
        # M: 전체 셀데이터 열번호 중 최고값, N: 할당셀 행번호 중 최고값
        self.block_items = block_items
        self.allocate_cell = allocate_cell
        self.allocate_cell_names = allocate_cell_names
        self.outline_data = outline_items['data']
        self.allocate_outline_cell = allocate_outline_cell
        self.N, self.M = max(self.allocate_cell.keys()), max(self.block_items[1].keys())

        logging.getLogger(f'plan-{block_type}').debug(f'불도저 버켓용량(e): {self.blade_capacity}')
        logging.getLogger(f'plan-{block_type}').debug(f'장애물셀: {self.obstacle_cell}, Start Line: {self.start_line}(1 + h_num({self.h_num}) + space({self.space}) + front_cells({self.front_cells}))')
        logging.getLogger(f'plan-{block_type}').debug(f'라인변경에 필요한 거리(required_line_change_distance): {self.required_line_change_distance}, 장비길이(equipment_length): {self.equipment_length}')
        logging.getLogger(f'plan-{block_type}').debug(f'최대 열 번호 M = {self.M}, 최대 행 번호 N = {self.N}')

        first_j = next(iter(self.allocate_cell))
        first_i = next(iter(self.allocate_cell[first_j]))

        i, j, j_min = 0, 1, None
        i_cur, j_cur = Block.get_bl_i_j(self.allocate_cell.get(first_j, {}).get(first_i, {}).get('cells', [])[0])

        while(j <= self.N):
            alloc_is_first = True
            logging.getLogger(f'plan-{block_type}').debug(f'i = {i}, j = {j}')
            i += (-1 * int(math.pow(-1, j)))
            logging.getLogger(f'plan-{block_type}').debug(f'다음 열로 이동, i: {i}')

            ran = range(i, 0, -1) if (-1 * int(math.pow(-1, j))) == -1 else range(i, self.M + 1, 1)
            exist_alloc = False
            for i in ran:
                # 현재 할당셀(AL_i_j)에 할당된 셀이 존재하는가? 
                logging.getLogger(f'plan-{block_type}').debug(f'현재 할당셀({block_type}AL_{i}_{j})에 할당된 셀이 존재하는가? {self.allocate_cell.get(j, {}).get(i) is not None}')
                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) > 0:
                    exist_alloc = True
                    break
                logging.getLogger(f'plan-{block_type}').debug(f'다음 열로 이동, i({i}) -> i({i + (-1 * int(math.pow(-1, j)))})')

            if not exist_alloc:
                j += 1
                logging.getLogger(f'plan-{block_type}').debug(f'다음 행으로 이동, j({j})+=1')
                continue

            initial_i = i
            for i in range(i, 0, -1) if (-1 * int(math.pow(-1, j))) == -1 else range(i, self.M + 1, 1):
                # AL_i_j에 할당된 셀이 있는가? 없을 경우 다음 열 이동
                if i != initial_i:
                    logging.getLogger(f'plan-{block_type}').debug(f'{block_type}AL_i_j({block_type}AL_{i}_{j})에 할당된 셀이 있는가? {not len(self.allocate_cell.get(j, {}).get(i, {}).get("cells", [])) == 0}')

                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) == 0:
                    logging.getLogger(f'plan-{block_type}').debug(f'다음 열 이동')
                    continue

                i_next, j_next = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                j_max = list(Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells')[-1]))[1]

                # i_next ==i_cur And j_next==j_cur
                logging.getLogger(f'plan-{block_type}').debug(f'i_next({i_next}) ==i_cur({i_cur}) And j_next({j_next})==j_cur({j_cur}): {not (i_next != i_cur or j_next != j_cur)}')
                if i_next != i_cur or j_next != j_cur:
                    j_next -= 1
                    ####
                    # 직전 경로가 외단라인 경로인가?
                    latest_allocate_cell_name = self.route_plan[-1]['allocate_cell_name']
                    logging.getLogger(f'plan-{block_type}').debug(f'직전 경로가 외단라인 경로인가? {self.route_plan[-1]["allocate_cell_name"].startswith(f"{block_type}OL")}, latest alloc cell: {latest_allocate_cell_name}')
                    if latest_allocate_cell_name.startswith(f'{block_type}OL'):
                        # TODO line_num1이 1인가?
                        line_num1 = int(latest_allocate_cell_name.split('_')[1])
                        logging.getLogger(f'plan-{block_type}').debug(f'line_num1({line_num1})이 1인가? {line_num1 == 1}')

                        last_alloc_i, last_alloc_j = map(lambda x: int(latest_allocate_cell_name.split('_')[x]), [1, 2])
                        target_coord_info, offset_coord_info, safety_line_df = (Block.BOTTOM_LEFT_COORD, Block.BOTTOM_RIGHT_COORD, self.safety_line_df1) if line_num1 == 1 else (Block.BOTTOM_RIGHT_COORD, Block.BOTTOM_LEFT_COORD, self.safety_line_df2)
                        block_i, block_j = Block.get_bl_i_j(self.allocate_outline_cell[last_alloc_i][last_alloc_j][0])
                        target_block = self.block_items[int(block_j) - 1][block_i]
                        #  {'x':'y':  'z': 'safe_x': 'safe_y':  'safe_z': str(point[2] + (gap + safety_line_df) * direction_vector[2])}
                        offset_coord = Block.offset_point(
                            outline={'x': target_block[target_coord_info[0]], 'y': target_block[target_coord_info[1]], 'z': target_block[target_coord_info[2]]},
                            center={'x': target_block[offset_coord_info[0]], 'y': target_block[offset_coord_info[1]], 'z': target_block[offset_coord_info[2]]},
                            gap=self.gap, safety_line_df=safety_line_df)

                        logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_{last_alloc_i}_ {last_alloc_j}({Block.get_block_names(self.allocate_outline_cell[last_alloc_i][last_alloc_j])})의 최하단 블록 {"좌측하단" if line_num1 == 1 else "우측하단"} 좌표를 {"우측하단 좌표" if line_num1 == 1 else "좌측하단 좌표"} 방향으로 gap만큼 offset하고  cell_size만큼 아래로 내린 좌표까지 후진경로 생성')
                        self.add_single_route_plan(coord={'x': offset_coord['x'], 'y': offset_coord['y'], 'z': offset_coord['z']}, forward=False, allocate_cell_name=latest_allocate_cell_name, cell_name=target_block['block_name'])
                        # TODO end

                        # |i_cur-i_next|<=1
                        logging.getLogger(f'plan-{block_type}').debug(f'|i_cur({i_cur})-i_next({i_next})|<=1 : {abs(i_cur - i_next) <= 1}')
                        if abs(i_cur - i_next) <= 1:
                            
                            # TODO 
                            # LOL_{line_num1}_ [ j_cur ]의 최하단 블록 좌측하단 좌표를 우측방향으로 gap 만큼 offset한 좌표를 cell_size만큼 아래로 내린 좌표부터 LOL_{line_num1}_ [ j_next ]의 최하단 블록 좌측하단 좌표를 우측방향으로 gap만큼 offset 하고 (cell_size + h_num)  만큼 아래로 내린 좌표까지 후진경로 생성
                            logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_{last_alloc_i}_ {last_alloc_j}({Block.get_block_names(self.allocate_outline_cell[last_alloc_i][last_alloc_j])})의 최하단 블록 {"좌측하단" if line_num1 == 1 else "우측하단"} 좌표를 {"우측하단 좌표" if line_num1 == 1 else "좌측하단 좌표"} 방향으로 gap만큼 offset하고  cell_size+h_num만큼 아래로 내린 좌표까지 후진경로 생성')
                            target_block = self.block_items[int(block_j) - 1 - self.h_num][block_i]
                            offset_coord = Block.offset_point(
                                outline={'x': target_block[target_coord_info[0]], 'y': target_block[target_coord_info[1]], 'z': target_block[target_coord_info[2]]},
                                center={'x': target_block[offset_coord_info[0]], 'y': target_block[offset_coord_info[1]], 'z': target_block[offset_coord_info[2]]},
                                gap=self.gap, safety_line_df=safety_line_df)
                            self.add_single_route_plan(coord={'x': offset_coord['x'], 'y': offset_coord['y'], 'z': offset_coord['z']}, forward=False, allocate_cell_name=latest_allocate_cell_name, cell_name=target_block['block_name'])
                            # TODO end

                            # j_cur -= h_num
                            j_cur -= self.h_num
                            
                            # BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                            logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_next-1})의 전방 이동점으로 전진경로 생성')
                            __block = self.block_items[j_next - 1][i_next]
                            self.add_single_route_plan(coord={'x': __block['x_t'], 'y': __block['y_t'], 'z': __block['z_t']}, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{__block.get("block_name")}-T')
                        else:
                            # i_cur += - 1*(-1)^j
                            logging.getLogger(f'plan-{block_type}').debug(f'i_cur: {i_cur}, j: {j}')
                            i_cur += -1 * pow(-1, j)
                            logging.getLogger(f'plan-{block_type}').debug(f'i_cur += -1 * pow(-1, j), i_cur: {i_cur}')
                            # BL_(i_cur)_(j_cur-h_num)의 후방 이동점으로 후진 경로 생성
                            logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_cur})_({j_cur-self.h_num})의 후방 이동점으로 후진경로 생성')
                            __block = self.block_items[j_cur-self.h_num][i_cur]
                            self.add_single_route_plan(coord={'x': __block['x_b'], 'y': __block['y_b'], 'z': __block['z_b']}, forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{__block.get("block_name")}-B')

                            #j_cur -= h_num
                            j_cur -= self.h_num
                            
                            logging.getLogger(f'plan-{block_type}').debug(f'j_cur({j_cur}) >= j_next({j_next}) - h_num : {j_cur >= j_next - self.h_num}')
                            if j_cur >= j_next - self.h_num:
                                # BL_(i_cur)_(j_cur)의 후방 이동점부터 BL_(i_cur)_(j_next-h_num)의 후방 이동점까지 후진 경로 생성
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_cur})_({j_cur})의 후방 이동점부터 {block_type}BL_({i_cur})_({j_next}-{self.h_num})의 후방 이동점까지 후진 경로 생성')
                                self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                                for _j in range(j_cur - 1, j_next - self.h_num - 1, -1):
                                    self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')
                                
                                # j_cur = j_next- h_num
                                j_cur = j_next- self.h_num
                            
                            # BL_(i_cur)_(j_cur) 의 후방 이동점에서 BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                            self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                            self.add_single_route_plan(coord={'x': self.block_items[j_next - 1][i_next]['x_t'], 'y': self.block_items[j_next - 1][i_next]['y_t'], 'z': self.block_items[j_next - 1][i_next]['z_t']}, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_next - 1][i_next].get("block_name")}-T')
                    else:
                        logging.getLogger(f'plan-{block_type}').debug(f'j_cur({j_cur}) >= j_next({j_next}) - h_num : {j_cur >= j_next - self.h_num}')
                        if j_cur >= j_next - self.h_num:
                            # BL_(i_cur)_(j_cur)의 후방 이동점부터 BL_(i_cur)_(j_next-h_num)의 후방 이동점까지 후진 경로 생성
                            logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_cur})_({j_cur})의 후방 이동점부터 {block_type}BL_({i_cur})_({j_next}-{self.h_num})의 후방 이동점까지 후진 경로 생성')
                            self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                            for _j in range(j_cur - 1, j_next - self.h_num - 1, -1):
                                self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')
                            
                            # j_cur = j_next- h_num
                            j_cur = j_next- self.h_num

                        # BL_(i_cur)_(j_cur) 의 후방 이동점에서 BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                        self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                        self.add_single_route_plan(coord={'x': self.block_items[j_next - 1][i_next]['x_t'], 'y': self.block_items[j_next - 1][i_next]['y_t'], 'z': self.block_items[j_next - 1][i_next]['z_t']}, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[j_next - 1][i_next].get("block_name")}-T')

                # j_min= AL_i_j의 행번호가 가장 낮은 셀의 행번호
                _, j_min = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                logging.getLogger(f'plan-{block_type}').debug(f'j_min({j_min})= {block_type}AL_i({i})_j({j})의 행번호가 가장 낮은 셀의 행번호')

                # BL_(i_next)_(j_max)의 전방 이동점까지 전진경로 생성
                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_max})의 전방 이동점 까지 전진경로 생성')
                for block in self.allocate_cell[j][i]['cells']:
                    self.add_route_plan(block=block, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{block.get("block_name")}')

                # 반복 횟수 만큼 반복
                repeat_count = self.allocate_cell[j][i]['repeat_count']
                repeat_route = []
                # 반복횟수[R] 만족했는가?
                logging.getLogger(f'plan-{block_type}').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {0}, {not (repeat_count > 0)}')
                for r in range(1, repeat_count + 1):
                    if not repeat_route:
                        before_route_index = len(self.route_plan)
                        # BL_(i_next)_(j_min)의 후방 이동점까지 후진경로 생성
                        logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_min})의 후방 이동점까지 후진경로 생성')
                        for _j in range(j_max, j_min - 1, -1):
                            self.add_route_plan(block=self.block_items[_j][i_next], forward=False, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_next].get("block_name")}')

                        # BL_(i_next)_(j_max)의 전방 이동점까지 전진경로 생성
                        logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_next})_({j_max})의 전방 이동점 까지 전진경로 생성')
                        for block in self.allocate_cell[j][i]['cells']:
                            self.add_route_plan(block=block, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{block.get("block_name")}')

                        repeat_route = self.route_plan[before_route_index:len(self.route_plan)]
                    else:
                        logging.getLogger('plan').debug(f'반복으로 인한 동일 경로 추가')
                        self.route_plan.extend(repeat_route)
                    # 반복횟수[R] 만족했는가?
                    logging.getLogger(f'plan-{block_type}').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {r}, {not (repeat_count > r)}')

                i_cur, j_cur = i_next, j_max

                # 열이 2개만 있을 경우 비효율적으로 후진하는 상황 방지.
                # M == 2 인 경우  경로 생성 알고리즘 추가  (일반 셀로 할당 후 일반 경로로 셀 생성)
                logging.getLogger(f'plan-{block_type}').debug(f'M == 2? : {self.M == 2}')
                if self.M == 2:
                    # AL_i_j가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가?
                    logging.getLogger(f'plan-{block_type}').debug(f'할당셀({block_type}AL_{i}_{j})가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가? {alloc_is_first}')
                    if alloc_is_first:
                        alloc_is_first = False
                    else:
                        for _ in range(j, self.N):
                            j += 1
                            # 현재 할당셀(AL_i_j)에 할당된 셀이 존재하는가? 
                            logging.getLogger(f'plan-{block_type}').debug(f'현재 할당셀({block_type}AL_{i}_{j})에 할당된 셀이 존재하는가? {self.allocate_cell.get(j, {}).get(i) is not None}')
                            if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) > 0:
                                # j_max= AL_i_j 에서 행번호가 가장 높은 값
                                j_max = list(Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells')[-1]))[1]
                                logging.getLogger(f'plan-{block_type}').debug(f'j_max({j_max})= {block_type}AL_i({i})_j({j}) 에서 행번호가 가장 높은 값')
                                # BL_(i_cur)_(j_max)의 전방 이동점까지 전진 경로 생성
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}BL_({i_cur})_({j_max})의 전방 이동점 까지 전진경로 생성')
                                for block in self.allocate_cell[j][i]['cells']:
                                    self.add_route_plan(block=block, forward=True, allocate_cell_name=f'{block_type}AL_{i}_{j}', cell_name=f'{block.get("block_name")}')
                                j_cur = j_max
                                break
                else: 
                    # AL_i_j가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가?
                    logging.getLogger(f'plan-{block_type}').debug(f'할당셀({block_type}AL_{i}_{j})가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가? {alloc_is_first}')
                    if alloc_is_first:
                        alloc_is_first = False
                        if j % 2 == 1:
                            if self.allocate_outline_cell.get(1, {}).get(j):
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_1_{j} 작업')
                                j_cur = self.outline(j, self.allocate_outline_cell[1][j], self.outline_data['df_l'], self.outline_data['df_r'], i_cur, j_cur, 'df1', outline_items['distance'], self.safety_line_df1, block_type)
                                before_i_cur = i_cur
                                i_cur += 1 * pow(-1, j)
                                logging.getLogger(f'plan-{block_type}').debug(f'i_cur+=1*(-1)^j({j}) -> i_cur: {i_cur} before_i_cur: {before_i_cur}')
                            else:
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_1_{j} 작업 - Skip')
                        else:
                            if self.allocate_outline_cell.get(2, {}).get(j):
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_2_{j} 작업')
                                j_cur = self.outline(j, self.allocate_outline_cell[2][j], self.outline_data['df_r'], self.outline_data['df_l'], i_cur, j_cur, 'df2', outline_items['distance'], self.safety_line_df2, block_type)
                                before_i_cur = i_cur
                                i_cur += 1 * pow(-1, j)
                                logging.getLogger(f'plan-{block_type}').debug(f'i_cur+=1*(-1)^j({j}) -> i_cur: {i_cur} before i_cur: {before_i_cur}')
                            else:
                                logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_2_{j} 작업 - Skip')
            # j%2==1
            if j % 2 == 1:
                if self.allocate_outline_cell.get(2, {}).get(j):
                    logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_2_{j} 작업')
                    j_cur = self.outline(j, self.allocate_outline_cell[2][j], self.outline_data['df_r'], self.outline_data['df_l'], i_cur, j_cur, 'df2', outline_items['distance'], self.safety_line_df2, block_type)
                    before_i_cur = i_cur
                    i_cur += -1 * pow(-1, j)
                    logging.getLogger(f'plan-{block_type}').debug(f'다음 열로 이동 i_cur+=-1*(-1)^j({j}) -> i_cur: {i_cur} before_i_cur: {before_i_cur}')
                else:
                    logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_2_{j} 작업 - Skip')
            else:
                if self.allocate_outline_cell.get(1, {}).get(j):
                    logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_1_{j} 작업')
                    j_cur = self.outline(j, self.allocate_outline_cell[1][j], self.outline_data['df_l'], self.outline_data['df_r'], i_cur, j_cur, 'df1', outline_items['distance'], self.safety_line_df1, block_type)
                    before_i_cur = i_cur
                    i_cur += -1 * pow(-1, j)
                    logging.getLogger(f'plan-{block_type}').debug(f'다음 열로 이동 i_cur+=-1*(-1)^j({j}) -> i_cur: {i_cur} before_i_cur: {before_i_cur}')
                else:
                    logging.getLogger(f'plan-{block_type}').debug(f'{block_type}OL_1_{j} 작업 - Skip')

        # 마지막 OL 경로의 후진 경로삭제
        for idx in range(len(self.route_plan) - 1, -1, -1):
            if self.route_plan[idx].get('direction') == -1:
                logging.getLogger(f'plan-{block_type}').debug(f'마지막 {block_type}OL 경로의 후진 경로삭제 - {json.dumps(self.route_plan[-1], ensure_ascii=False)}')
                del self.route_plan[-1]
            else:
                break

        return self.route_plan


    def check_outline(self, df: list, idx: int, block_type: str):
        logging.getLogger(f'plan-{block_type}').debug(f'df[{idx}] 경로 추가')
        if idx >= len(df):
            raise InputDataError(f'Not exist No.{idx + 1} in Model_line_Data table, Error: list index out of range')
        

    # block_items: {j: {i: {block}}}
    # alloc_outline_data: [{block}...]
    # center_data: [{'x': 237516.453, 'y': 425177.98, 'z': 0.0}...]
    # outline_data: [{'x': 237516.453, 'y': 425177.98, 'z': 0.0}...]
    def outline(self, j: int, alloc_outline_data: list, target_outline_data: list, opposite_outline_data: list, i_cur: int, j_cur: int, df_name: str, distances: list, safety_line_df: float, block_type: str):

        # line_num1 = 현재 작업중인 OL 라인 번호
        # line_num2 = 현재 작업중이지 않은 OL 라인 번호
        line_num1 = 1 if df_name == 'df1' else 2
        line_num2 = 2 if line_num1 == 1 else 1
        outline_name = f"{block_type}{'OL_1' if df_name == 'df1' else 'OL_2'}_{j}"

        gap_1 = self.gap + self.safety_line_df1
        gap_2 = self.gap + self.safety_line_df2

        j_min_block = sorted(alloc_outline_data, key=lambda x: list(Block.get_bl_i_j(x))[1])[0]

        # j_min = OL_2_j의 셀 중 가장 행 번호가 낮은 셀의 행 번호, j_max = OL_2_j의 셀 중 가장 행 번호가 높은 셀의 행 번호 
        j_items = [list(Block.get_bl_i_j(block))[1] for block in alloc_outline_data]
        j_min, j_max = min(j_items) - 1, max(j_items)
        logging.getLogger(f'plan-{block_type}OL').debug(f'j_min: {j_min}, j_max: {j_max}, alloc_outline_data: {Block.get_block_names(alloc_outline_data)}')

        # 직전 작업 경로가 OL 작업인가?
        latest_allocate_cell_name = self.route_plan[-1]['allocate_cell_name']
        outline_prefix = f'{block_type}OL'
        logging.getLogger(f'plan-{block_type}OL').debug(f'직전 경로가 외단라인 경로인가? {latest_allocate_cell_name.startswith(outline_prefix)}, latest alloc cell: {latest_allocate_cell_name}')
        if latest_allocate_cell_name.startswith(outline_prefix):
            # TODO start
            # linenum1 1일때 LOL_{line_num2}_ [ j_cur ]의 최상단 블록의 우측 하단 좌표를 좌측방향으로 gap 만큼 offset 한 좌표부터 LOL_{line_num2}_ [ j_cur] 의 최하단 블록의 우측 하단 좌표를  좌측 방향으로 gap 만큼 offset하고 cell_size만큼 아래로 내린 좌표까지 후진경로 생성
            # linenum1 1 아닐때  LOL_{line_num2}_ [ j_cur ]의 최상단 블록의 좌측 하단 좌표를 우측 방향으로 gap 만큼 offset 한 좌표부터 LOL_{line_num2}_ [ j_cur] 의 최하단 블록의 좌측 하단 좌표를  우측 방향으로 gap 만큼 offset하고 cell_size만큼 아래로 내린 좌표까지 후진경로 생성
            last_alloc_i, last_alloc_j = map(lambda x: int(latest_allocate_cell_name.split('_')[x]), [1, 2])
            reverse_last_alloc_i = 2 if last_alloc_i == 1 else 1
            logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}OL_{reverse_last_alloc_i}_{last_alloc_j} ({Block.get_block_names(self.allocate_outline_cell[reverse_last_alloc_i][last_alloc_j])}) 의 최상단 블록의 {"우측 하단" if line_num1 == 1 else "좌측 하단"} 좌표를 {"좌측 하단 좌표" if line_num1 == 1 else "우측 하단 좌표"} 방향으로 gap 만큼 offset 한 좌표부터 {block_type}OL_{reverse_last_alloc_i}_{last_alloc_j} 의 최하단 블록의 {"우측 하단" if line_num1 == 1 else "좌측 하단"} 좌표를  {"좌측 하단 좌표" if line_num1 == 1 else "우측 하단 좌표"} 방향으로 gap 만큼 offset하고 cell_size만큼 아래로 내린 좌표까지 후진경로 생성')

            target_coord_info, offset_coord_info = (Block.BOTTOM_RIGHT_COORD, Block.BOTTOM_LEFT_COORD) if line_num1 == 1 else (Block.BOTTOM_LEFT_COORD, Block.BOTTOM_RIGHT_COORD)

            for _j in range(len(self.allocate_outline_cell[reverse_last_alloc_i][last_alloc_j]), 0, -1):
                target_block = self.allocate_outline_cell[reverse_last_alloc_i][last_alloc_j][_j - 1]
                offset_coord = Block.offset_point(
                    outline={'x': target_block[target_coord_info[0]], 'y': target_block[target_coord_info[1]], 'z': target_block[target_coord_info[2]]},
                    center={'x': target_block[offset_coord_info[0]], 'y': target_block[offset_coord_info[1]], 'z': target_block[offset_coord_info[2]]},
                    gap=self.gap, safety_line_df=safety_line_df)
                self.add_single_route_plan(coord={'x': offset_coord['x'], 'y': offset_coord['y'], 'z': offset_coord['z']}, forward=False, allocate_cell_name=latest_allocate_cell_name, cell_name=target_block['block_name'])
            
            block_i, block_j = Block.get_bl_i_j(self.allocate_outline_cell[reverse_last_alloc_i][last_alloc_j][0])
            target_block = self.block_items[int(block_j) - 1][block_i]
            #  {'x':'y':  'z': 'safe_x': 'safe_y':  'safe_z': str(point[2] + (gap + safety_line_df) * direction_vector[2])}
            offset_coord = Block.offset_point(
                outline={'x': target_block[target_coord_info[0]], 'y': target_block[target_coord_info[1]], 'z': target_block[target_coord_info[2]]},
                center={'x': target_block[offset_coord_info[0]], 'y': target_block[offset_coord_info[1]], 'z': target_block[offset_coord_info[2]]},
                gap=self.gap, safety_line_df=safety_line_df)

            self.add_single_route_plan(coord={'x': offset_coord['x'], 'y': offset_coord['y'], 'z': offset_coord['z']}, forward=False, allocate_cell_name=latest_allocate_cell_name, cell_name=target_block['block_name'])
            # TODO end

            dd = int(math.pow(-1, line_num1))
            # LBL_(i_cur +dd)_(j_min - h_num)의 후방이동점 까지 후진경로 생성
            logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}BL_(i_cur({i_cur}) + dd({dd}))_(j_min({j_min})-h_num({self.h_num}))의 후방 이동점 까지 후진경로 생성')
            self.add_single_route_plan(coord={'x': self.block_items[j_min - self.h_num][i_cur + dd]['x_b'], 'y': self.block_items[j_min - self.h_num][i_cur + dd]['y_b'], 'z': self.block_items[j_min - self.h_num][i_cur + dd]['z_b']}, forward=False, allocate_cell_name=outline_name, cell_name=f'{self.block_items[j_min - self.h_num][i_cur + dd].get("block_name")}-B')
        else:
            # LBL_(i_cur)_(j_cur)의 전방 이동점에서부터 LBL_(i_cur )_(j_min-h_num)의 후방 이동점 까지 후진경로 생성
            logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}BL_(i_cur({i_cur}))_(j_cur({j_cur}) 전방 이동점 부터 ({block_type}BL_(i_cur){i_cur}_(j_min({j_min}) - h_num({self.h_num})){j_min - self.h_num})의 후방 이동점 까지 후진경로 생성')
            for _j in range(j_cur, j_min - self.h_num - 1, -1):
                self.add_route_plan(self.block_items[_j][i_cur], forward=False, allocate_cell_name=outline_name, cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')

        # line_num1이 1인가?

        logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}OL_{line_num1}_{j} ({Block.get_block_names(self.allocate_outline_cell[line_num1][j])}) 의 최하단 블록의 {"우측 하단" if line_num1 == 1 else "좌측 하단"} 좌표를 {"좌측 하단 좌표" if line_num1 == 1 else "우측 하단 좌표"} 방향으로 gap 만큼 offset 하고 cell_size만큼 아래로 내린 좌표로 전진경로 생성')
        target_coord_info, offset_coord_info = (Block.BOTTOM_RIGHT_COORD, Block.BOTTOM_LEFT_COORD) if line_num1 == 1 else (Block.BOTTOM_LEFT_COORD, Block.BOTTOM_RIGHT_COORD)

        block_i, block_j = Block.get_bl_i_j(self.allocate_outline_cell[line_num1][j][0])
        target_block = self.block_items[int(block_j) - 1][block_i]
        #  {'x':'y':  'z': 'safe_x': 'safe_y':  'safe_z': str(point[2] + (gap + safety_line_df) * direction_vector[2])}
        offset_coord = Block.offset_point(
            outline={'x': target_block[target_coord_info[0]], 'y': target_block[target_coord_info[1]], 'z': target_block[target_coord_info[2]]},
            center={'x': target_block[offset_coord_info[0]], 'y': target_block[offset_coord_info[1]], 'z': target_block[offset_coord_info[2]]},
            gap=self.gap, safety_line_df=safety_line_df)

        self.add_single_route_plan(coord={'x': offset_coord['x'], 'y': offset_coord['y'], 'z': offset_coord['z']}, forward=True, allocate_cell_name=outline_name, cell_name=target_block['block_name'])
        
        logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}OL_{line_num1}_{j}({Block.get_block_names(self.allocate_outline_cell[line_num1][j])}) 의 최상단 블록의 {"우측 하단" if line_num1 == 1 else "좌측 하단"} 좌표를 {"좌측 하단 좌표" if line_num1 == 1 else "우측 하단 좌표"} 방향으로 gap 만큼 offset 좌표로 전진경로 생성')
        for _target_block in self.allocate_outline_cell[line_num1][j]:
            _offset_coord = Block.offset_point(
                outline={'x': _target_block[target_coord_info[0]], 'y': _target_block[target_coord_info[1]], 'z': _target_block[target_coord_info[2]]},
                center={'x': _target_block[offset_coord_info[0]], 'y': _target_block[offset_coord_info[1]], 'z': _target_block[offset_coord_info[2]]},
                gap=self.gap, safety_line_df=safety_line_df)
            self.add_single_route_plan(coord={'x': _offset_coord['x'], 'y': _offset_coord['y'], 'z': _offset_coord['z']}, forward=True, allocate_cell_name=outline_name, cell_name=_target_block['block_name'])

        # 반복횟수[R] 만족했는가?
        repeat_count = Block.get_repeat_count(alloc_outline_data, self.blade_capacity)
        logging.getLogger(f'plan-{block_type}OL').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {1}, {not (repeat_count > 1)}')
        repeat_route = []
        for r in range(2, repeat_count + 1):
            if not repeat_route:
                before_route_index = len(self.route_plan)
                # 1일 때 No LOL_{line_num1}_ [ j_cur]의 최하단 블록 우측 하단 좌표를 좌측방향으로 gap 만큼 offset하고 cell_size만큼 내린 좌표까지 후진경로 생성
                # No LOL_{line_num1}_ [ j_cur]의 최하단 블록 좌측 하단 좌표를 우측방향으로 gap 만큼 offset하고 cell_size만큼 내린 좌표까지 후진경로 생성
                logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}OL_{line_num1}_{j}({Block.get_block_names(self.allocate_outline_cell[line_num1][j])}) 의 최하단 블록의 {"우측 하단" if line_num1 == 1 else "좌측 하단"} 좌표를 {"좌측 하단 좌표" if line_num1 == 1 else "우측 하단 좌표"} 방향으로 gap 만큼 offset cell_size만큼 내린 좌표까지 후진경로 생성')
                for _j in range(len(self.allocate_outline_cell[line_num1][j]), 0, -1):
                    _target_block = self.allocate_outline_cell[line_num1][j][_j - 1]
                    _offset_coord = Block.offset_point(
                        outline={'x': _target_block[target_coord_info[0]], 'y': _target_block[target_coord_info[1]], 'z': _target_block[target_coord_info[2]]},
                        center={'x': _target_block[offset_coord_info[0]], 'y': _target_block[offset_coord_info[1]], 'z': _target_block[offset_coord_info[2]]},
                        gap=self.gap, safety_line_df=safety_line_df)
                    self.add_single_route_plan(coord={'x': _offset_coord['x'], 'y': _offset_coord['y'], 'z': _offset_coord['z']}, forward=False, allocate_cell_name=outline_name, cell_name=_target_block['block_name'])

                block_i, block_j = Block.get_bl_i_j(self.allocate_outline_cell[line_num1][j][0])
                _target_block = self.block_items[int(block_j) - 1][block_i]
                #  {'x':'y':  'z': 'safe_x': 'safe_y':  'safe_z': str(point[2] + (gap + safety_line_df) * direction_vector[2])}
                _offset_coord = Block.offset_point(
                    outline={'x': _target_block[target_coord_info[0]], 'y': _target_block[target_coord_info[1]], 'z': _target_block[target_coord_info[2]]},
                    center={'x': _target_block[offset_coord_info[0]], 'y': _target_block[offset_coord_info[1]], 'z': _target_block[offset_coord_info[2]]},
                    gap=self.gap, safety_line_df=safety_line_df)

                self.add_single_route_plan(coord={'x': _offset_coord['x'], 'y': _offset_coord['y'], 'z': _offset_coord['z']}, forward=False, allocate_cell_name=outline_name, cell_name=_target_block['block_name'])

                logging.getLogger(f'plan-{block_type}OL').debug(f'{block_type}OL_{line_num1}_{j}({Block.get_block_names(self.allocate_outline_cell[line_num1][j])}) 의 최상단 블록의 {"우측 하단" if line_num1 == 1 else "좌측 하단"} 좌표를 {"좌측 하단 좌표" if line_num1 == 1 else "우측 하단 좌표"} 방향으로 gap 만큼 offset 좌표로 전진경로 생성')
                for _target_block in self.allocate_outline_cell[line_num1][j]:
                    _offset_coord = Block.offset_point(
                        outline={'x': _target_block[target_coord_info[0]], 'y': _target_block[target_coord_info[1]], 'z': _target_block[target_coord_info[2]]},
                        center={'x': _target_block[offset_coord_info[0]], 'y': _target_block[offset_coord_info[1]], 'z': _target_block[offset_coord_info[2]]},
                        gap=self.gap, safety_line_df=safety_line_df)
                    self.add_single_route_plan(coord={'x': _offset_coord['x'], 'y': _offset_coord['y'], 'z': _offset_coord['z']}, forward=True, allocate_cell_name=outline_name, cell_name=_target_block['block_name'])

                repeat_route = self.route_plan[before_route_index:len(self.route_plan)]
            else:
                logging.getLogger(f'plan-{block_type}OL').debug(f'반복으로 인한 동일 경로 추가')
                self.route_plan.extend(repeat_route)
            # 반복횟수[R] 만족했는가?
            logging.getLogger(f'plan-{block_type}OL').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {r}, {not (repeat_count > r)}')
        return j_max
