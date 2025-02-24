# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import math
import csv
import logging
import json
from os import makedirs
from shapely import BufferCapStyle # type: ignore
from shapely.geometry import LineString, Polygon # type: ignore

from route_planner_v10.block import Block
from route_planner_v10.util import log_decorator, calculate_h_num
from route_planner_v10.constants import SHOW_ALLOC_CELL_FLAG
from route_planner_v10.exception import InputDataError


class DozerRoutePlan:
    def __init__(self, block_items: dict, allocate_cell: dict, allocate_outline_cell: dict, outline_data: dict, allocate_cell_names: list):
        # 셀 정보
        self.block_items = block_items
        # 할당셀
        self.allocate_cell = allocate_cell
        # 할당셀 외단라인
        self.allocate_outline_cell = allocate_outline_cell
        # outline 좌표 정보
        self.outline_data = outline_data
        self.allocate_cell_names = allocate_cell_names
        self.route_plan = []
        self.e = None
        self.gap = None
        # M: 전체 셀데이터 열번호 중 최고값, N: 할당셀 행번호 중 최고값
        self.N, self.M = max(allocate_cell.keys()), max(self.block_items[1].keys())
        self.timeline = 0

    def add_route(self, coord: dict, allocate_cell_name: str, cell_name: str):
        coord.update({'allocate_cell_name': allocate_cell_name, 'cell_name': cell_name})
        
        # for c in ['x', 'y', 'z']:
        #     coord[c] = None if float(coord[c]) == 0 else coord[c]

        self.route_plan.append(coord)
        logging.getLogger('plan').debug(json.dumps(coord, ensure_ascii=False))

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
            logging.getLogger('plan').debug(f'마지막 경로와 좌표가 동일하여 경로 생성 skip - {cell_name}')

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
                logging.getLogger('plan').debug(f'마지막 경로와 좌표가 동일하여 경로 생성 skip - {cell_name}-B')
            self.add_route(coord={'x': x_t, 'y': y_t, 'z': z_t, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}-T')
        else:
            # TOP -> BOTTOM
            if latest_route.get('x') != x_t or latest_route.get('y') != y_t  or latest_route.get('z') != z_t:
                self.add_route(coord={'x': x_t, 'y': y_t, 'z': z_t, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}-T')
            else:
                logging.getLogger('plan').debug(f'마지막 경로와 좌표가 동일하여 경로 생성 skip - {cell_name}-T')
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
                v['z1'] = v.pop('z')
                if not SHOW_ALLOC_CELL_FLAG:
                    v.pop('allocate_cell_name')
                    v.pop('cell_name')
            writer.writerows(self.route_plan)


    # 도저 성토 계획 경로 산출
    # converted_block: BlName key dict
    @log_decorator('도저 계획 경로 알고리즘')
    def calc_route_plan(self, param: dict):
        (
            e, s, h, l, obstacle_cells, h_num, s_num, gap,
            required_line_change_distance,
            distances, safety_line_df1, safety_line_df2
        )  = map(
            param.get, ['e', 's', 'h', 'l', 'obstacle_cells', 'h_num', 's_num', 'gap',
                        'required_line_change_distance',
                        'distances', 'safety_line_df1', 'safety_line_df2'])
        
        self.e = e
        self.gap = gap

        logging.getLogger('plan').debug(f'불도저 버켓용량(e): {e}, 최소전진거리(s): {s}, 라인변경에 필요한 거리(h): {h}, 중심선 노드간 최소거리(l): {l}')
        logging.getLogger('plan').debug(f'장애물셀: {obstacle_cells}, gap: {gap}')
        logging.getLogger('plan').debug(f'라인변경에 필요한 셀 칸수 산정(H_num): {h_num}, 최소 할당하는셀의 개수 산정(S_num): {s_num}')
        logging.getLogger('plan').debug(f'최대 할당셀 열 번호 M = {self.M}, 최대 할당셀 행 번호 N = {self.N}')

        first_j = next(iter(self.allocate_cell))
        first_i = next(iter(self.allocate_cell[first_j]))

        i, j, j_min = 0, 1, None
        i_cur, j_cur = Block.get_bl_i_j(self.allocate_cell.get(first_j, {}).get(first_i, {}).get('cells', [])[0])

        while(j <= self.N):
            alloc_is_first = True
            logging.getLogger('plan').debug(f'i = {i}, j = {j}')
            i += (-1 * int(math.pow(-1, j)))
            logging.getLogger('plan').debug(f'다음 열로 이동, i: {i}')

            ran = range(i, 0, -1) if (-1 * int(math.pow(-1, j))) == -1 else range(i, self.M + 1, 1)
            exist_alloc = False
            for i in ran:
                # 현재 할당셀(AL_i_j)에 할당된 셀이 존재하는가? 
                logging.getLogger('plan').debug(f'현재 할당셀(AL_{i}_{j})에 할당된 셀이 존재하는가? {self.allocate_cell.get(j, {}).get(i) is not None}')
                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) > 0:
                    exist_alloc = True
                    break
                logging.getLogger('plan').debug(f'다음 열로 이동, i({i}) -> i({i + (-1 * int(math.pow(-1, j)))})')

            if not exist_alloc:
                j += 1
                logging.getLogger('plan').debug(f'다음 행으로 이동, j({j})+=1')
                continue

            initial_i = i
            for i in range(i, 0, -1) if (-1 * int(math.pow(-1, j))) == -1 else range(i, self.M + 1, 1):
                # AL_i_j에 할당된 셀이 있는가? 없을 경우 다음 열 이동
                if i != initial_i:
                    logging.getLogger('plan').debug(f'AL_i_j(AL_{i}_{j})에 할당된 셀이 있는가? {not len(self.allocate_cell.get(j, {}).get(i, {}).get("cells", [])) == 0}')

                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) == 0:
                    logging.getLogger('plan').debug(f'다음 열 이동')
                    continue

                i_next, j_next = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                j_max = list(Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells')[-1]))[1]

                # i_next ==i_cur And j_next==j_cur
                logging.getLogger('plan').debug(f'i_next({i_next}) ==i_cur({i_cur}) And j_next({j_next})==j_cur({j_cur}): {not (i_next != i_cur or j_next != j_cur)}')
                if i_next != i_cur or j_next != j_cur:
                    # v1.1.0
                    h_num = calculate_h_num(j_cur, required_line_change_distance, distances)
                    j_next -= 1

                    # v1.1.0 직전 경로가 외단라인 경로인가?
                    latest_allocate_cell_name = self.route_plan[-1]['allocate_cell_name']
                    logging.getLogger('plan').debug(f'직전 경로가 외단라인 경로인가? {self.route_plan[-1]["allocate_cell_name"].startswith("OL")}, latest alloc cell: {latest_allocate_cell_name}')
                    if latest_allocate_cell_name.startswith('OL'):
                        # df [j_cur]를 df0[j_cur] 방향으로 gap 만큼 offset 한 좌표 부터 df [j_cur-1]를 df0[j_cur-1] 방향으로 gap 만큼 offset 한 좌표까지 후진 경로 생성
                        df_name, target_outline_data = ('df1', self.outline_data['df_l']) if latest_allocate_cell_name.startswith('OL_1') else ('df2', self.outline_data['df_r'])
                        logging.getLogger('plan').debug(f'{df_name} [j_cur]를 df0[j_cur] 방향으로 gap 만큼 offset 한 좌표 부터 {df_name} [j_cur-1]를 df0[j_cur-1] 방향으로 gap 만큼 offset 한 좌표까지 후진 경로 생성')
                        for _j in range(j_cur, j_cur - 2, -1):
                            self.check_outline(target_outline_data, _j)
                            self.add_single_route_plan(coord={'x': target_outline_data[_j].get('x'), 'y': target_outline_data[_j].get('y'), 'z': target_outline_data[_j].get('z')}, forward=False, allocate_cell_name=latest_allocate_cell_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

                        # |i_cur-i_next|<=1
                        if abs(i_cur - i_next) <= 1:
                            # df [j_cur-1]를 df0[j_cur-1] 방향으로 gap 만큼 offset 한 좌표부터 df [j_next-h_num-1]를 df0[j_next-h_num-1] 방향으로 gap 만큼 offset 한 좌표까지 후진 경로 생성
                            logging.getLogger('plan').debug(f'{df_name} [j_cur-1]를 df0[j_cur-1] 방향으로 gap 만큼 offset 한 좌표 부터 {df_name} [j_cur-h_num-1]를 df0[j_cur-h_num-1] 방향으로 gap 만큼 offset 한 좌표까지 후진 경로 생성')
                            for _j in range(j_cur, j_cur - h_num- 2, -1):
                                self.check_outline(target_outline_data, _j)
                                self.add_single_route_plan(coord={'x': target_outline_data[_j].get('x'), 'y': target_outline_data[_j].get('y'), 'z': target_outline_data[_j].get('z')}, forward=False, allocate_cell_name=latest_allocate_cell_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

                            # j_cur -= h_num
                            j_cur -= h_num
                            
                            # BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                            logging.getLogger('plan').debug(f'BL_({i_next})_({j_next-1})의 전방 이동점으로 전진경로 생성')
                            __block = self.block_items[j_next - 1][i_next]
                            self.add_single_route_plan(coord={'x': __block['x_t'], 'y': __block['y_t'], 'z': __block['z_t']}, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{__block.get("block_name")}-T')
                        else:
                            # i_cur += - 1*(-1)^j
                            i_cur += -1 * pow(-1, j)

                            # BL_(i_cur)_(j_cur-h_num)의 후방 이동점으로 후진 경로 생성
                            logging.getLogger('plan').debug(f'BL_({i_cur})_({j_cur-h_num})의 후방 이동점으로 후진경로 생성')
                            __block = self.block_items[j_cur-h_num][i_cur]
                            self.add_single_route_plan(coord={'x': __block['x_b'], 'y': __block['y_b'], 'z': __block['z_b']}, forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{__block.get("block_name")}-B')

                            #j_cur -= h_num
                            j_cur -= h_num
                            
                            # h_num =calculate_h_num(j_next, required_line_change_distance)
                            h_num =calculate_h_num(j_next, required_line_change_distance, distances)
                            logging.getLogger('plan').debug(f'j_cur: {j_cur}, h_num: {h_num}')

                            logging.getLogger('plan').debug(f'j_cur >= j_next({j_next}) - h_num : {j_cur >= j_next - h_num}')
                            if j_cur >= j_next - h_num:
                                # BL_(i_cur)_(j_cur)의 후방 이동점부터 BL_(i_cur)_(j_next-h_num)의 후방 이동점까지 후진 경로 생성
                                logging.getLogger('plan').debug(f'BL_({i_cur})_({j_cur})의 후방 이동점부터 BL_({i_cur})_({j_next}-{h_num})의 후방 이동점까지 후진 경로 생성')
                                self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                                for _j in range(j_cur - 1, j_next - h_num - 1, -1):
                                    self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')
                                
                                # j_cur = j_next- h_num
                                j_cur = j_next- h_num
                            
                            # BL_(i_cur)_(j_cur) 의 후방 이동점에서 BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                            self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                            self.add_single_route_plan(coord={'x': self.block_items[j_next - 1][i_next]['x_t'], 'y': self.block_items[j_next - 1][i_next]['y_t'], 'z': self.block_items[j_next - 1][i_next]['z_t']}, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_next - 1][i_next].get("block_name")}-T')

                    else:
                        # h_num =calculate_h_num(j_next, required_line_change_distance)
                        h_num =calculate_h_num(j_next, required_line_change_distance, distances)
                        logging.getLogger('plan').debug(f'j_cur: {j_cur}, h_num: {h_num}')

                        logging.getLogger('plan').debug(f'j_cur >= j_next({j_next}) - h_num : {j_cur >= j_next - h_num}')
                        if j_cur >= j_next - h_num:
                            # BL_(i_cur)_(j_cur)의 후방 이동점부터 BL_(i_cur)_(j_next-h_num)의 후방 이동점까지 후진 경로 생성
                            logging.getLogger('plan').debug(f'BL_({i_cur})_({j_cur})의 후방 이동점부터 BL_({i_cur})_({j_next}-{h_num})의 후방 이동점까지 후진 경로 생성')
                            self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}-B')
                            for _j in range(j_cur - 1, j_next - h_num - 1, -1):
                                self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')
                            
                            # j_cur = j_next- h_num
                            j_cur = j_next- h_num
                        
                        # BL_(i_cur)_(j_cur) 의 후방 이동점에서 BL_(i_next)_(j_next-1) 의 전방 이동점으로 전진경로 생성
                        self.add_single_route_plan(coord={'x': self.block_items[j_cur][i_cur]['x_b'], 'y': self.block_items[j_cur][i_cur]['y_b'], 'z': self.block_items[j_cur][i_cur]['z_b']}, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_cur][i_cur].get("block_name")}-B')
                        self.add_single_route_plan(coord={'x': self.block_items[j_next - 1][i_next]['x_b'], 'y': self.block_items[j_next - 1][i_next]['y_b'], 'z': self.block_items[j_next - 1][i_next]['z_b']}, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_next - 1][i_next].get("block_name")}-B')

                # j_min= AL_i_j의 행번호가 가장 낮은 셀의 행번호
                _, j_min = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                logging.getLogger('plan').debug(f'j_min({j_min})= AL_i({i})_j({j})의 행번호가 가장 낮은 셀의 행번호')

                # BL_(i_next)_(j_max)의 전방 이동점까지 전진경로 생성
                logging.getLogger('plan').debug(f'BL_({i_next})_({j_max})의 전방 이동점 까지 전진경로 생성')
                for block in self.allocate_cell[j][i]['cells']:
                    self.add_route_plan(block=block, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{block.get("block_name")}')

                # 반복 횟수 만큼 반복
                repeat_count = self.allocate_cell[j][i]['repeat_count']
                # 반복횟수[R] 만족했는가?
                logging.getLogger('plan').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {0}, {not (repeat_count > 0)}')
                for r in range(1, repeat_count + 1):
                    # BL_(i_next)_(j_min)의 후방 이동점까지 후진경로 생성
                    logging.getLogger('plan').debug(f'BL_({i_next})_({j_min})의 후방 이동점까지 후진경로 생성')
                    for _j in range(j_max, j_min - 1, -1):
                        self.add_route_plan(block=self.block_items[_j][i_next], forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_next].get("block_name")}')

                    # BL_(i_next)_(j_max)의 전방 이동점까지 전진경로 생성
                    logging.getLogger('plan').debug(f'BL_({i_next})_({j_max})의 전방 이동점 까지 전진경로 생성')
                    for block in self.allocate_cell[j][i]['cells']:
                        self.add_route_plan(block=block, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{block.get("block_name")}')

                    # 반복횟수[R] 만족했는가?
                    logging.getLogger('plan').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {r}, {not (repeat_count > r)}')

                i_cur, j_cur = i_next, j_max
                # AL_i_j가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가?
                logging.getLogger('plan').debug(f'할당셀(AL_{i}_{j})가 해당 행의 할당셀들 중 첫 번째로 경로 생성이 되는 할당셀 인가? {alloc_is_first}')
                if alloc_is_first:
                    alloc_is_first = False
                    if j % 2 == 1:
                        if self.allocate_outline_cell.get(1, {}).get(j):
                            logging.getLogger('plan').debug(f'OL_1_{j} 작업')
                            j_cur = self.outline(j, self.allocate_outline_cell[1][j], self.outline_data['df_l'], self.outline_data['df_r'], i_cur, j_cur, 'df1', distances, required_line_change_distance, safety_line_df1)
                            bf_i_cur = i_cur
                            i_cur += 1 * pow(-1, j)
                            logging.getLogger('plan').debug(f'i_cur+=1*(-1)^j({j}) -> i_cur: {i_cur} bf_i_cur: {bf_i_cur}')
                        else:
                            logging.getLogger('plan').debug(f'OL_1_{j} 작업 - Skip')
                    else:
                        if self.allocate_outline_cell.get(2, {}).get(j):
                            logging.getLogger('plan').debug(f'OL_2_{j} 작업')
                            j_cur = self.outline(j, self.allocate_outline_cell[2][j], self.outline_data['df_r'], self.outline_data['df_l'], i_cur, j_cur, 'df2', distances, required_line_change_distance, safety_line_df2)
                            bf_i_cur = i_cur
                            i_cur += 1 * pow(-1, j)
                            logging.getLogger('plan').debug(f'i_cur+=1*(-1)^j({j}) -> i_cur: {i_cur} before i_cur: {bf_i_cur}')
                        else:
                            logging.getLogger('plan').debug(f'OL_2_{j} 작업 - Skip')
            # j%2==1
            if j % 2 == 1:
                if self.allocate_outline_cell.get(2, {}).get(j):
                    logging.getLogger('plan').debug(f'OL_2_{j} 작업')
                    j_cur = self.outline(j, self.allocate_outline_cell[2][j], self.outline_data['df_r'], self.outline_data['df_l'], i_cur, j_cur, 'df2', distances, required_line_change_distance, safety_line_df2)
                    bf_i_cur = i_cur
                    i_cur += -1 * pow(-1, j)
                    logging.getLogger('plan').debug(f'다음 열로 이동 i_cur+=-1*(-1)^j({j}) -> i_cur: {i_cur} bf_i_cur: {bf_i_cur}')
                else:
                    logging.getLogger('plan').debug(f'OL_2_{j} 작업 - Skip')
            else:
                if self.allocate_outline_cell.get(1, {}).get(j):
                    logging.getLogger('plan').debug(f'OL_1_{j} 작업')
                    j_cur = self.outline(j, self.allocate_outline_cell[1][j], self.outline_data['df_l'], self.outline_data['df_r'], i_cur, j_cur, 'df1', distances, required_line_change_distance, safety_line_df1)
                    bf_i_cur = i_cur
                    i_cur += -1 * pow(-1, j)
                    logging.getLogger('plan').debug(f'다음 열로 이동 i_cur+=-1*(-1)^j({j}) -> i_cur: {i_cur} bf_i_cur: {bf_i_cur}')
                else:
                    logging.getLogger('plan').debug(f'OL_1_{j} 작업 - Skip')

        # 마지막 OL 경로의 후진 경로삭제
        for idx in range(len(self.route_plan) - 1, -1, -1):
            if self.route_plan[idx].get('direction') == -1:
                logging.getLogger('plan').debug(f'마지막 OL 경로의 후진 경로삭제 - {json.dumps(self.route_plan[-1], ensure_ascii=False)}')
                del self.route_plan[-1]
            else:
                break
        
        # shapely로 장애물 검사
        intersected_blocks, intersected_block_names = self.check_obstacle(obstacle_cells)

        return self.route_plan, intersected_blocks, intersected_block_names


    def check_outline(self, df: list, idx: int):
        logging.getLogger('plan').debug(f'df[{idx}] 경로 추가')
        if idx >= len(df):
            raise InputDataError(f'Not exist No.{idx + 1} in Model_line_Data table, Error: list index out of range')
    
    @log_decorator('shapely로 장애물 검사')
    def check_obstacle(self, obstacle_cells: int):
        logging.getLogger('plan').debug(f'장애물셀: {obstacle_cells}')
        obstacle_polygons, intersected_blocks, intersected_block_names = [], [], []
        for block_name in obstacle_cells:
            block = Block.get_block_by_name(self.block_items, block_name)

            coords = []
            for i in range(1, 6): 
                x, y = map(lambda x: block.get(f'{x}{i}'), ['x', 'y']) 
                if x is not None and y is not None:
                    coords.append((x, y))

            if len(coords) >= 3:
                obstacle_polygons.append((block_name, Polygon(coords)))

        for idx in range(0, len(self.route_plan) - 1):
            line = LineString([(self.route_plan[idx]['x'], self.route_plan[idx]['y']), (self.route_plan[idx + 1]['x'], self.route_plan[idx + 1]['y'])])

            for block_name, polygon in obstacle_polygons:
                # if line.touches(polygon):
                #     logging.getLogger('plan').warning(f'한점에서 만남: True, Block: {block_name}, timeline : {idx}, line: {self.route_plan[idx]["cell_name"]} -> {self.route_plan[idx + 1]["cell_name"]}')
                if not line.touches(polygon) and line.buffer(self.gap, cap_style=BufferCapStyle.flat).intersects(polygon):
                    logging.getLogger('plan').warning(f'교차: {block_name}, timeline : {idx}, 한점만 만나는가?: {line.touches(polygon)}, line: {self.route_plan[idx]["cell_name"]} -> {self.route_plan[idx + 1]["cell_name"]}')
                    intersected_blocks.append((block_name, idx))
                    intersected_block_names.append(block_name)
        
        return intersected_blocks, list(set(intersected_block_names))


    # block_items: {j: {i: {block}}}
    # alloc_outline_data: [{block}...]
    # center_data: [{'x': 237516.453, 'y': 425177.98, 'z': 0.0}...]
    # outline_data: [{'x': 237516.453, 'y': 425177.98, 'z': 0.0}...]
    def outline(self, j: int, alloc_outline_data: list, target_outline_data: list, opposite_outline_data: list,i_cur: int, j_cur: int, df_name: str, distances: list, required_line_change_distance: float, safety_line_df: float):

        # line_num1 = 현재 작업중인 OL 라인 번호
        # line_num2 = 현재 작업중이지 않은 OL 라인 번호
        line_num1 = 1 if df_name == 'df1' else 2
        line_num2 = 2 if line_num1 == 1 else 1

        # j_min = OL_2_j의 셀 중 가장 행 번호가 낮은 셀의 행 번호, j_max = OL_2_j의 셀 중 가장 행 번호가 높은 셀의 행 번호 
        j_items = [list(Block.get_bl_i_j(block))[1] for block in alloc_outline_data]
        j_min, j_max = min(j_items), max(j_items)
        # v1.1.0 
        h_num = calculate_h_num(j_min, required_line_change_distance, distances)
        j_min -= 1
        
        repeat_count = Block.get_repeat_count(alloc_outline_data, self.e)
        outline_name = f"{'OL_1' if df_name == 'df1' else 'OL_2'}_{j}"

        logging.getLogger('plan').debug(f'j_min: {j_min},  j_max: {j_max}, h_num: {h_num}')
        
        # 직전 작업 경로가 OL 작업인가?
        latest_allocate_cell_name = self.route_plan[-1]['allocate_cell_name']
        logging.getLogger('plan').debug(f'직전 경로가 외단라인 경로인가? {latest_allocate_cell_name.startswith("OL")}, latest alloc cell: {latest_allocate_cell_name}')
        if latest_allocate_cell_name.startswith('OL'):
            # df_{line_num2}[j_cur] 을 df0[j_cur] 방향으로 gap_{line_num2}만큼 offset 한 좌표부터 df_{line_num2}[j_min-1] 을 df0[j_min-1] 방향으로 gap_{line_num2}만큼 offset 한 좌표까지 후진경로 생성
            logging.getLogger('plan').debug(f'df_{line_num2}[{j_cur}] 을 df0[{j_cur}] 방향으로 gap_{line_num2}만큼 offset 한 좌표부터 df_{line_num2}[{j_min}-1] 을 df0[{j_min}-1] 방향으로 gap_{line_num2}만큼 offset 한 좌표까지 후진경로 생성')
            for _j in range(j_cur, j_min - 1 - 1, -1):
                self.check_outline(opposite_outline_data, _j)
                self.add_single_route_plan(coord={'x': opposite_outline_data[_j].get('safe_x'), 'y': opposite_outline_data[_j].get('safe_y'), 'z': target_outline_data[_j].get('safe_z')}, forward=False, allocate_cell_name=outline_name, cell_name=f'df{line_num2}-No-{opposite_outline_data[_j].get("No")}')

            dd = int(math.pow(-1, line_num1))
            # BL_(i_cur+dd)_(j_min-h_num)의 후방 이동점 까지 후진경로 생성
            logging.getLogger('plan').debug(f'dd = (-1)^{line_num1} -> dd = {dd}')

            # dd = -1 * int(math.pow(-1, line_num1))
            # # BL_(i_cur+dd)_(j_min-h_num)의 후방 이동점 까지 후진경로 생성
            # logging.getLogger('plan').debug(f'dd = -1 * (-1)^{line_num1} -> dd = {dd}')

            logging.getLogger('plan').debug(f'BL_(i_cur({i_cur}) + dd({dd}))_(j_min({j_min})-h_num({h_num}))의 후방 이동점 까지 후진경로 생성')
            self.add_single_route_plan(coord={'x': self.block_items[j_min - h_num][i_cur + dd]['x_b'], 'y': self.block_items[j_min - h_num][i_cur + dd]['y_b'], 'z': self.block_items[j_min - h_num][i_cur + dd]['z_b']}, forward=False, allocate_cell_name=outline_name, cell_name=f'{self.block_items[j_min - h_num][i_cur + dd].get("block_name")}-B')

        else:
            # v1.1.0 BL_(i_cur)_(j_cur)의 전방 이동점에서부터 BL_(i_cur )_(j_min-h_num)의 후방 이동점 까지 후진경로 생성
            logging.getLogger('plan').debug(f'BL_(i_cur({i_cur}))_(j_cur({j_cur}) 전방 이동점 부터 (BL_(i_cur){i_cur}_(j_min - h_num){j_min - h_num})의 후방 이동점 까지 후진경로 생성')
            for _j in range(j_cur, j_min - h_num - 1, -1):
                self.add_route_plan(self.block_items[_j][i_cur], forward=False, allocate_cell_name=outline_name, cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')

        # df1 or df2[j_min - 1]를 df0[j_min-1] 방향으로 gap_{line_num1} 만큼 offset 한 좌표까지 전진경로 생성
        logging.getLogger('plan').debug(f'{df_name}[{j_min - 1}]를 df0[{j_min - 1}] 방향으로 gap_{line_num1}({self.gap + safety_line_df}) 만큼 offset 한 좌표까지 전진경로 생성')
        self.check_outline(target_outline_data, j_min - 1)
        self.add_single_route_plan(coord={'x': target_outline_data[j_min - 1].get('safe_x'), 'y': target_outline_data[j_min - 1].get('safe_y'), 'z': target_outline_data[j_min - 1].get('safe_z')}, forward=True, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[j_min - 1].get("No")}')

        # df1 or df2[j_max]를 df0[j_max] 방향으로 gap+safety_line_df 만큼 offset 한 좌표까지 전진경로 생성
        logging.getLogger('plan').debug(f'{df_name}[{j_max}]를 df0[{j_max}] 방향으로 gap_{line_num1}({self.gap + safety_line_df}) 만큼 offset 한 좌표까지 전진경로 생성')
        for _j in range(j_min - 1, j_max + 1):
            self.check_outline(target_outline_data, _j)
            self.add_single_route_plan(coord={'x': target_outline_data[_j].get('safe_x'), 'y': target_outline_data[_j].get('safe_y'), 'z': target_outline_data[_j].get('safe_z')}, forward=True, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

        # 반복횟수[R] 만족했는가?
        logging.getLogger('plan').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {1}, {not (repeat_count > 1)}')
        for r in range(2, repeat_count + 1):
            # df1 or df2[j_min-1] 을 df0[j_min-1] 방향으로 gap 만큼 offset 한 좌표까지
            logging.getLogger('plan').debug(f'{df_name}[j_min-1] 을 df0[j_min-1] 방향으로 gap_{line_num1}({self.gap + safety_line_df}) 만큼 offset 하여 후진경로 생성')
            for _j in range(j_max - 1, j_min - 1 - 1, -1):
                self.check_outline(target_outline_data, _j)
                self.add_single_route_plan(coord={'x': target_outline_data[_j].get('safe_x'), 'y': target_outline_data[_j].get('safe_y'), 'z': target_outline_data[_j].get('safe_z')}, forward=False, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

            # df1 or df2[j_max]를 df0[j_max] 방향으로 gap + safety_line_df 만큼 offset 한 좌표까지 전진경로 생성
            logging.getLogger('plan').debug(f'{df_name}[{j_max}]를 df0[{j_max}] 방향으로 gap_{line_num1}({self.gap + safety_line_df}) 만큼 offset 한 좌표까지 전진경로 생성')
            for _j in range(j_min - 1, j_max + 1):
                self.check_outline(target_outline_data, _j)
                self.add_single_route_plan(coord={'x': target_outline_data[_j].get('safe_x'), 'y': target_outline_data[_j].get('safe_y'), 'z': target_outline_data[_j].get('safe_z')}, forward=True, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

            # 반복횟수[R] 만족했는가?
            logging.getLogger('plan').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {r}, {not (repeat_count > r)}')
        
        return j_max
        
