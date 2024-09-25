# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import math
import csv
import logging
import json
from datetime import datetime

from route_planner_v10.block import Block
from route_planner_v10.util import log_decorator
from route_planner_v10.constants import SHOW_ALLOC_CELL_FLAG


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
        if SHOW_ALLOC_CELL_FLAG:
            coord.update({'allocate_cell_name': allocate_cell_name, 'cell_name': cell_name})
        self.route_plan.append(coord)
    # 단일 경로 추가
    def add_single_route_plan(self, coord: dict, forward: bool, allocate_cell_name: str, cell_name: str):
        # coord: {'X':0, 'Y':0, 'Z': 0}
        direction = 1 if forward else -1
        coord.update({'direction': direction})

        self.add_route(coord=coord, allocate_cell_name=allocate_cell_name, cell_name=cell_name)
        logging.getLogger('plan').debug(json.dumps(coord, ensure_ascii=False))

    # 경로 추가
    def add_route_plan(self, block: dict, forward: bool, allocate_cell_name: str, cell_name: str):
        # 'XTcoord': 'x_t', 'YTcoord': 'y_t', 'ZTcoord': 'z_t', 'XBcoord': 'x_b', 'YBcoord': 'y_b', 'ZBcoord': 'z_b', 'cutVol': 'cut_vol', 'fillVol': 'fill_vol', 'totalVol': 'total_vol', 'Y,N': 'yn',
        direction = 1 if forward else -1
        x_t, y_t, z_t, x_b, y_b, z_b = map(block.get, ['x_t', 'y_t', 'z_t', 'x_b', 'y_b', 'z_b'])
        latest_route = self.route_plan[-1] if self.route_plan else {}
        route = None
        if forward:
            # BOTTOM -> TOP
            # 마지막 경로와 좌표가 같을 경우 경로에 추가하지 않음
            if latest_route.get('x') != x_b or latest_route.get('y') != y_b  or latest_route.get('z') != z_b:
                self.add_route(coord={'x': x_b, 'y': y_b, 'z': z_b, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}B')
                logging.getLogger('plan').debug(json.dumps({'X': x_b, 'Y': y_b, 'Z': z_b, '전후진': direction}, ensure_ascii=False))
            self.add_route(coord={'x': x_t, 'y': y_t, 'z': z_t, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}T')
            logging.getLogger('plan').debug(json.dumps({'X': x_t, 'Y': y_t, 'Z': z_t, '전후진': direction}, ensure_ascii=False))
        else:
            # TOP -> BOTTOM
            if latest_route.get('x') != x_t or latest_route.get('y') != y_t  or latest_route.get('z') != z_t:
                self.add_route(coord={'x': x_t, 'y': y_t, 'z': z_t, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}T')
                logging.getLogger('plan').debug(json.dumps({'X': x_t, 'Y': y_t, 'Z': z_t, '전후진': direction}, ensure_ascii=False))
            self.add_route(coord={'x': x_b, 'y': y_b, 'z': z_b, 'direction': direction}, allocate_cell_name=allocate_cell_name, cell_name=f'{cell_name}B')
            logging.getLogger('plan').debug(json.dumps({'X': x_b, 'Y': y_b, 'Z': z_b, '전후진': direction}, ensure_ascii=False))

    @log_decorator('계획 경로 알고리즘 CSV 저장')
    def save_output_csv(self, output_file: str):
        with open(output_file, 'w', newline='\n', encoding='utf-8') as csvfile:
            headers = ['x', 'y', 'direction', 'z1', 'z2', 'allocate_cell_name', 'cell_name'] if SHOW_ALLOC_CELL_FLAG else ['x', 'y', 'direction', 'z1', 'z2']
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for i, v in enumerate(self.route_plan):
                v.pop('z',None)
                v['z1'] = 0
                v['z2'] = 0
                """
                if SHOW_ALLOC_CELL_FLAG:
                    v['할당셀'] = v.pop('allocate_cell_name')
                    v['대상셀정보'] = v.pop('cell_name')
                """
            writer.writerows(self.route_plan)


    # 도저 성토 계획 경로 산출
    # converted_block: BlName key dict
    @log_decorator('도저 계획 경로 알고리즘')
    def calc_route_plan(self, param: dict):
        e, s, h, l, obstacle_cells, h_num, s_num, gap = map(
            param.get, ['e', 's', 'h', 'l', 'obstacle_cells', 'h_num', 's_num', 'gap'])

        self.e = e
        self.gap = gap

        logging.getLogger('plan').debug(f'불도저 버켓용량(e): {e}, 최소전진거리(s): {s}, 라인변경에 필요한 거리(h): {h}, 중심선 노드간 최소거리(l): {l}')
        logging.getLogger('plan').debug(f'장애물셀: {obstacle_cells}, gap: {gap}')
        logging.getLogger('plan').debug(f'라인변경에 필요한 셀 칸수 산정(H_num): {h_num}, 최소 할당하는셀의 개수 산정(S_num): {s_num}')
        logging.getLogger('plan').debug(f'최대 할당셀 열 번호 M = {self.M}, 최대 할당셀 행 번호 N = {self.N}')

        i, j = 0, 1
        i_cur, j_cur, j_min = None, None, None
        is_first = True
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
                if i > initial_i:
                    logging.getLogger('plan').debug(f'AL_i_j(AL_{i}_{j})에 할당된 셀이 있는가? {not len(self.allocate_cell.get(j, {}).get(i, {}).get("cells", [])) == 0}')
                if len(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])) == 0:
                    logging.getLogger('plan').debug(f'다음 열 이동')
                    continue

                i_next, j_next = Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])[0])
                j_max = list(Block.get_bl_i_j(self.allocate_cell.get(j, {}).get(i, {}).get('cells')[-1]))[1]

                # AL_i_j가 전체 할당셀 중 처음 경로를 생성하는 할당셀인가?
                logging.getLogger('plan').debug(f'AL_{i}_{j}가 전체 할당셀 중 처음 경로를 생성하는 할당셀인가? {is_first}')
                if is_first:
                    # i_cur = i_next, j_cur = j_next
                    is_first, i_cur, j_cur = False, i_next, j_next
                else:
                    # i_next ==i_cur And j_next==j_cur
                    logging.getLogger('plan').debug(f'i_next({i_next}) ==i_cur({i_cur}) And j_next({j_next})==j_cur({j_cur}): {not (i_next != i_cur or j_next != j_cur)}')
                    if i_next != i_cur or j_next != j_cur:
                        # v1.0.7 직전 경로가 전진 경로인가? (가장 최근 경로의 directio이 1일경우 Y, -1일경우 N)
                        latest_route = self.route_plan[-1] if self.route_plan else {}
                        logging.getLogger('plan').debug(f'직전 경로가 전진 경로인가?: {latest_route.get("direction")}')
                        if latest_route and latest_route.get('direction') == 1:
                            # BL_(i_cur)_(j_min) 의 후방 이동점까지 후진 경로 생성
                            logging.getLogger('plan').debug(f'BL_({i_cur})_({j_min})의 후방 이동점까지 후진경로 생성')
                            for _j in range(j_cur, j_min - 1, -1):
                                self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{block.get("block_name")}')
                            j_cur = j_min

                        # BL_(i_next)_(j_next-1) 이 할당 된 적이 있는 셀인가?
                        logging.getLogger('plan').debug(f'BL_(i_next)_(j_next-1)(BL_{i_next}_{j_next-1}) 이 할당 된 적이 있는 셀인가? BL_{i_next}_{j_next-1} in {self.allocate_cell_names}')
                        buffer_j = 1 if f'BL_{i_next}_{j_next-1}' in self.allocate_cell_names else 0
                        logging.getLogger('plan').debug(f'j_cur({j_cur}) <= j_next({j_next})-H_num({h_num}) {"-1" if buffer_j > 0 else ""} : {j_cur <= j_next - h_num - buffer_j}')
                        if j_cur > j_next - h_num - buffer_j:
                            # v1.0.7 k=j_next-H_num
                            k = j_next - h_num - buffer_j
                            logging.getLogger('plan').debug(f'k({k}) = j_next({j_next})-H_num({h_num}){"-1" if buffer_j > 0 else ""}')

                            # BL_(i_cur)_(k) 이 이동 가능한 셀인가?
                            moveable = Block.check_moveable(self.block_items[k][i_cur], obstacle_cells)
                            logging.getLogger('plan').debug(f'BL_(i_cur)_(k)이 이동 가능한 셀인가?: {moveable}')
                            if moveable:
                                # BL_(i_cur)_(k) 의 후방 이동점까지 후진 경로 생성
                                logging.getLogger('plan').debug(f'BL_(i_cur)_(k) 의 후방 이동점까지 후진 경로 생성, i_cur:{i_cur}, j_cur: {j_cur}')
                                for _j in range(j_cur, k - 1, -1):
                                    self.add_route_plan(block=self.block_items[_j][i_cur], forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')
                            else:
                                if not self.block_items.get(k, {}).get(i_next, {}):
                                    logging.getLogger('plan').error(f'BL_({i_next})_({k})의 Block 정보 없음')
                                # BL_(i_next)_(k) 의 후방 이동점으로 후진 경로 생성
                                logging.getLogger('plan').debug(f'BL_(i_next)_(k) 의 후방 이동점으로 후진 경로 생성, i_cur:{i_cur}, j_cur: {j_cur}')
                            self.add_single_route_plan(coord={
                                'x': self.block_items[k][i_next].get('x_b'),
                                'y': self.block_items[k][i_next].get('y_b'),
                                'z': self.block_items[k][i_next].get('z_b')
                            }, forward=False, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[k][i_next].get("block_name")}-B')

                        # BL_(i_next )_(j_next-1 - buffer_j)의 전방 이동점으로 전진경로 생성
                        logging.getLogger('plan').debug(f'BL_({i_next})_({j_next - 1 - buffer_j})의 전방 이동점으로 전진경로 생성, i_cur:{i_cur}, j_cur: {j_cur}')
                        if not self.block_items.get(j_next - 1 - buffer_j, {}).get(i_next, {}):
                            logging.getLogger('plan').error(f'BL_({i_next})_({j_next - 1 - buffer_j})의 Block 정보 없음')
                        self.add_single_route_plan(coord={
                            'x': self.block_items[j_next - 1 - buffer_j][i_next].get('x_t'),
                            'y': self.block_items[j_next - 1 - buffer_j][i_next].get('y_t'),
                            'z': self.block_items[j_next - 1 - buffer_j][i_next].get('z_t')
                        }, forward=True, allocate_cell_name=f'AL_{i}_{j}', cell_name=f'{self.block_items[j_next - 1 - buffer_j][i_next].get("block_name")}-T')

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
                            j_cur = self.outline(j, self.allocate_outline_cell[1][j], self.outline_data['df_l'], h_num, i_cur, j_cur, 'df1')

                        else:
                            logging.getLogger('plan').debug(f'OL_1_{j} 작업 - Skip')
                    else:
                        if self.allocate_outline_cell.get(2, {}).get(j):
                            logging.getLogger('plan').debug(f'OL_2_{j} 작업')
                            j_cur = self.outline(j, self.allocate_outline_cell[2][j], self.outline_data['df_r'], h_num, i_cur, j_cur, 'df2')

                        else:
                            logging.getLogger('plan').debug(f'OL_2_{j} 작업 - Skip')
            # j%2==1
            if j % 2 == 1:
                if self.allocate_outline_cell.get(2, {}).get(j):
                    logging.getLogger('plan').debug(f'OL_2_{j} 작업')
                    j_cur = self.outline(j, self.allocate_outline_cell[2][j], self.outline_data['df_r'], h_num, i_cur, j_cur, 'df2')

                else:
                    logging.getLogger('plan').debug(f'OL_2_{j} 작업 - Skip')
            else:
                if self.allocate_outline_cell.get(1, {}).get(j):
                    logging.getLogger('plan').debug(f'OL_1_{j} 작업')
                    j_cur = self.outline(j, self.allocate_outline_cell[1][j], self.outline_data['df_l'], h_num, i_cur, j_cur, 'df1')

                else:
                    logging.getLogger('plan').debug(f'OL_1_{j} 작업 - Skip')

        # 마지막 OL 경로의 후진 경로삭제
        for idx in range(len(self.route_plan) - 1, -1, -1):
            if self.route_plan[idx].get('direction') == -1:
                logging.getLogger('plan').debug(f'마지막 OL 경로의 후진 경로삭제 - {json.dumps(self.route_plan[-1], ensure_ascii=False)}')
                del self.route_plan[-1]
            else:
                break

        return self.route_plan

    def check_outline(self, df: list, idx: int):
        logging.getLogger('plan').debug(f'df[{idx}] 경로 추가')
        if idx >= len(df):
            raise Exception(f'Not exist No.{idx + 1} in Model_line_Data table, Error: list index out of range')
    
    # block_items: {j: {i: {block}}}
    # alloc_outline_data: [{block}...]
    # center_data: [{'x': 237516.453, 'y': 425177.98, 'z': 0.0}...]
    # outline_data: [{'x': 237516.453, 'y': 425177.98, 'z': 0.0}...]
    def outline(self, j: int, alloc_outline_data: list, target_outline_data: list, h_num: int, i_cur: int, j_cur: int, df_name: str):
        # j_min = OL_2_j의 셀 중 가장 행 번호가 낮은 셀의 행 번호, j_max = OL_2_j의 셀 중 가장 행 번호가 높은 셀의 행 번호 
        j_items = [list(Block.get_bl_i_j(block))[1] for block in alloc_outline_data]
        j_min, j_max = min(j_items), max(j_items)
        repeat_count = Block.get_repeat_count(alloc_outline_data, self.e)
        outline_name = f"{'OL_1' if df_name == 'df1' else 'OL_2'}_{j}"

        logging.getLogger('plan').debug(f'j_min: {j_min},  j_max: {j_max}')
        logging.getLogger('plan').debug(f'BL_({i_cur})_({j_min}-{h_num}) BL_{i_cur}_{j_min - h_num}의 후방 이동점 까지 후진경로 생성')
        for _j in range(j_cur, j_min - h_num - 1, -1):
            # if _j <= 0:
            #     logging.getLogger('plan').warn('후진 경로 생성 시 1행 미만의 행 접근 발생')
            #     continue
            self.add_route_plan(self.block_items[_j][i_cur], forward=False, allocate_cell_name=outline_name, cell_name=f'{self.block_items[_j][i_cur].get("block_name")}')

        j_min_offset = 1 if j == 1 else 2
        
        # df1 or df2[j_min-j_min_offset] 을 df0[j_min-j_min_offset] 방향으로 gap 만큼 offset 한 좌표까지 전진경로 생성
        logging.getLogger('plan').debug(f'{df_name}[j_min-{j_min_offset}] 을 df0[j_min-{j_min_offset}] 방향으로 gap({self.gap}) 만큼 offset 하여 전진경로 생성')
        self.check_outline(target_outline_data, j_min - j_min_offset)
        start_outline = target_outline_data[j_min - j_min_offset]

        self.add_single_route_plan(coord={
            'x': start_outline.get('x'),
            'y': start_outline.get('y'),
            'z': start_outline.get('z')
        }, forward=True, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{start_outline.get("No")}')

        # df1 or df2[j_max]를 df0[j_max] 방향으로 gap 만큼 offset 한 좌표까지 전진경로 생성
        logging.getLogger('plan').debug(f'{df_name}[{j_max}]를 df0[{j_max}] 방향으로 gap({self.gap}) 만큼 offset 한 좌표까지 전진경로 생성')
        for _j in range(j_min, j_max + 1):
            self.check_outline(target_outline_data, _j)
            self.add_single_route_plan(coord={'x': target_outline_data[_j].get('x'), 'y': target_outline_data[_j].get('y'), 'z': target_outline_data[_j].get('z')}, forward=True, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

        # 반복횟수[R] 만족했는가?
        logging.getLogger('plan').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {0}, {not (repeat_count > 0)}')
        for r in range(1, repeat_count + 1):
            # df1 or df2[j_min-j_min_offset] 을 df0[j_min-j_min_offset] 방향으로 gap 만큼 offset 한 좌표까지
            logging.getLogger('plan').debug(f'{df_name}[j_min-{j_min_offset}] 을 df0[j_min-{j_min_offset}] 방향으로 gap({self.gap}) 만큼 offset 하여 후진경로 생성')
            # self.add_single_route_plan(coord={'X': start_outline.get('x'), 'Y': start_outline.get('y'), 'Z': start_outline.get('z')}, forward=False, allocate_cell_name=outline_name)
            for _j in range(j_max - 1, j_min - j_min_offset - 1, -1):
                self.check_outline(target_outline_data, _j)
                self.add_single_route_plan(coord={'x': target_outline_data[_j].get('x'), 'y': target_outline_data[_j].get('y'), 'z': target_outline_data[_j].get('z')}, forward=False, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

            # df1 or df2[j_max]를 df0[j_max] 방향으로 gap 만큼 offset 한 좌표까지 전진경로 생성
            logging.getLogger('plan').debug(f'{df_name}[{j_max}]를 df0[{j_max}] 방향으로 gap({self.gap}) 만큼 offset 한 좌표까지 전진경로 생성')
            for _j in range(j_min, j_max + 1):
                self.check_outline(target_outline_data, _j)
                self.add_single_route_plan(coord={'x': target_outline_data[_j].get('x'), 'y': target_outline_data[_j].get('y'), 'z': target_outline_data[_j].get('z')}, forward=True, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

            # 반복횟수[R] 만족했는가?
            logging.getLogger('plan').debug(f'반복횟수[R({repeat_count})] 만족했는가? current r: {r}, {not (repeat_count > r)}')
        
        # df1[j_min-1] 을 df0[j_min-1] 방향으로 gap 만큼 offset 한 좌표까지 후진경로 생성
        logging.getLogger('plan').debug(f'{df_name}[j_min-1] 을 df0[j_min-1] 방향으로 gap({self.gap}) 만큼 offset 하여 후진경로 생성')
        # self.add_single_route_plan(coord={'X': target_outline_data[j_min - 1].get('x'), 'Y': target_outline_data[j_min - 1].get('y'), 'Z': target_outline_data[j_min - 1].get('z')}, forward=False, allocate_cell_name=outline_name)
        for _j in range(j_max - 1, j_min - 1, -1):
            self.check_outline(target_outline_data, _j)
            self.add_single_route_plan(coord={'x': target_outline_data[_j].get('x'), 'y': target_outline_data[_j].get('y'), 'z': target_outline_data[_j].get('z')}, forward=False, allocate_cell_name=outline_name, cell_name=f'{df_name}-No-{target_outline_data[_j].get("No")}')

        # BL_(i_cur )_(j_cur-H_num)의 후방 이동점으로 후진경로 생성
        logging.getLogger('plan').debug(f'BL_({i_cur})_({j_min}-{h_num}) BL_{i_cur}_{j_min - h_num}의 후방 이동점으로 후진경로 생성')
        # for _j in range(alloc_j_min, alloc_j_min - h_num - 1, -1):
        #     self.add_route_plan(self.block_items[_j][i_cur], forward=False, allocate_cell_name=outline_name)
        self.add_single_route_plan(coord={
            'x': self.block_items[j_min - h_num][i_cur].get('x_b'),
            'y': self.block_items[j_min - h_num][i_cur].get('y_b'),
            'z': self.block_items[j_min - h_num][i_cur].get('z_b')
        }, forward=False, allocate_cell_name=outline_name, cell_name=f'{self.block_items[j_min - h_num][i_cur].get("block_name")}-B')

        return j_min - h_num
