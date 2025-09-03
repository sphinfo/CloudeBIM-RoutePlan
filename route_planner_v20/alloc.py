# COPYRIGHT ⓒ 2025 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import csv
import math
import logging
from os import makedirs
from datetime import datetime
from route_planner_v20.util import log_decorator
from route_planner_v20.block import Block
from route_planner_v20.constants import START_BLOCK


class Alloc():
    def __init__(self, param: dict):
        for k, v in param.items():
            setattr(self, k, v)

        self.allocate_cell = {}
        self.allocate_cell_names = {}
        self.outline_cell = {}

    # block_type: S, L, R
    @log_decorator('할당셀 알고리즘')
    def alloc(self, block_items: dict, block_type: str):
        self.allocate_cell[block_type] = {}
        self.allocate_cell_names[block_type] = []

        self.N, self.M = Block.get_n_m(block_items)
        logging.getLogger(f'alloc-{block_type}').debug(f'불도저 버켓용량(e): {self.blade_capacity}')
        logging.getLogger(f'alloc-{block_type}').debug(f'장애물셀: {self.obstacle_cell}, Start Line: {self.start_line}(1 + h_num({self.h_num}) + space({self.space}) + front_cells({self.front_cells}))')
        logging.getLogger(f'alloc-{block_type}').debug(f'라인변경에 필요한 거리(required_line_change_distance): {self.required_line_change_distance}, 장비길이(equipment_length): {self.equipment_length}')
        logging.getLogger(f'alloc-{block_type}').debug(f'최대 열 번호 M = {self.M}, 최대 행 번호 N = {self.N}')

        # current_j[]는 M개의 배열, 모든 원소 값 = start_line-1
        current_j = [self.start_line - 1 for _ in range(0, self.M)]

        k = 0
        logging.getLogger(f'alloc-{block_type}').debug(f'min(current_j): {min(current_j)}')
        while (min(current_j) < self.N):
            k += 1
            logging.getLogger(f'alloc-{block_type}').debug(f'k: {k}')

            for i in range(1, self.M + 1):
                for j in range(current_j[i - 1] + 1, self.N + 1):
                    # BL_i_j가 이동 가능한가? 0행 이하일 경우 이동 불가
                    moveable = False
                    if j > 0:
                        try:
                            moveable = Block.check_moveable(block_items[j][i], self.obstacle_cell)
                        except KeyError:
                            moveable = False
                            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_i_j({block_type}BL_{i}_{j})가 이동 가능한가?: Not exist, {moveable}')
                        else:
                            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_i_j({block_type}BL_{i}_{j})가 이동 가능한가?: {moveable}')

                    if not moveable:
                        continue

                    _h_num = self.h_num
                    _h_num += self.back_cells

                    # 현재셀(BL_i_j)의 H_num+space만큼의 셀들(BL_i_j-1, BL_i_j-2 … BL_i_j-H_num-space)은 이동가능(Y) 셀인가?
                    recent_cells = []
                    for x in range(j - 1, j - _h_num - self.space -1, -1):
                        # 0행 이하일 경우 이동 불가
                        moveable_x = False
                        if x > 0:
                            try:
                                moveable_x = Block.check_moveable(block_items[x][i], self.obstacle_cell)
                            except KeyError:
                                moveable_x = False
                            
                        recent_cells.append(f"BL_{i}_{x}): {moveable_x}")

                        if not moveable_x:
                            # 전부 이동가능하여야 하므로 False일 경우 loop 종료
                            moveable = False
                            break

                    logging.getLogger(f'alloc-{block_type}').debug(f'현재셀({block_type}BL_i_j(BL_{i}_{j}))의 H_num({_h_num}), space({self.space}) H_num+space({_h_num + self.space})만큼의 셀들(BL_i_j-1, BL_i_j-2 … BL_i_j-H_num)은 이동가능(Y) 셀인가?: {moveable}, {recent_cells}')
                    
                    if moveable:
                        target = self.allocate_cell[block_type].setdefault(k, {})
                        alloc_cell_info = target.setdefault(i, {})
                        alloc_cell_info['name'] = f'{block_type}AL_{i}_{k}'
                        alloc_cell = alloc_cell_info.setdefault('cells', [])

                        # BL_i_j 를 AL_i_k에 할당
                        alloc_cell.append(block_items[j][i])
                        logging.getLogger(f'alloc-{block_type}').debug(f'해당셀({block_type}BL_{i}_{j})) 할당, {block_type}AL_{i}_{k}: {[c.get("block_name") for c in alloc_cell]}')

                        y = 1
                        for y in range(1, self.N - j + 1):
                            # S_num만큼 할당했는가?
                            logging.getLogger(f'alloc-{block_type}').debug(f'S_num({self.s_num})만큼 할당했는가?: {len(alloc_cell) >= self.s_num}')

                            if len(alloc_cell) < self.s_num: # S_num만큼 할당했는가? -> NO
                                # j <= N
                                if j + y in block_items:
                                    # BL_i_j+y가 이동 가능(Y)한가?
                                    is_alloc = Block.check_moveable(block_items[j + y][i], self.obstacle_cell)
                                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_i_j({block_items[j + y][i]["block_name"]})가 이동 가능한가?: {is_alloc}')

                                    if is_alloc:
                                        # BL_i_j 를 AL_i_k에 할당
                                        alloc_cell.append(block_items[j + y][i])
                                        logging.getLogger(f'alloc-{block_type}').debug(f'해당셀({block_items[j + y][i]["block_name"]}) 할당, {block_type}AL_{i}_{k}: {[c.get("block_name") for c in alloc_cell]}')
                                    else:
                                        break                                        
                            else: # S_num만큼 할당했는가? -> YES
                                # 할당셀에 대한 절성토량(Vs) 산출
                                vs = sum([c.get('total_vol') for c in alloc_cell])
                                
                                if 'repeat_count' not in alloc_cell_info:
                                    logging.getLogger(f'alloc-{block_type}').debug(f'할당셀에 대한 절성토량(Vs) 산출, Vs = {vs}')
                                    # 할당셀 당 작업 반복 횟수 설정 |Vs /e|를 올림한 정수값 = 반복횟수[R]
                                    alloc_cell_info['repeat_count'] = math.ceil(abs(vs / self.blade_capacity))
                                    logging.getLogger(f'alloc-{block_type}').debug(f'할당셀 당 작업 반복 횟수 설정 |Vs /e|를 올림한 정수값 = 반복횟수[R], R = {alloc_cell_info["repeat_count"] }')
                                    vmax = alloc_cell_info['repeat_count'] * self.blade_capacity
                                    logging.getLogger(f'alloc-{block_type}').debug(f'최대 절성토량(Vmax) 계산 R(반복횟수) x e(버킷용량) = Vmax, Vmax = {vmax}')

                                if j + y in block_items:
                                    logging.getLogger(f'alloc-{block_type}').debug(f'현재 셀({block_items[j + y][i]["block_name"]})이 이동가능(Y) 셀인가?: {Block.check_moveable(block_items[j + y][i], self.obstacle_cell)}')
                                    logging.getLogger(f'alloc-{block_type}').debug(f'최대 절성토량이 넘지 않았는가? {vmax} >= {abs(vs + block_items[j + y][i]["total_vol"])} {vmax >= abs(vs + block_items[j + y][i]["total_vol"])}')
                                    if Block.check_moveable(block_items[j + y][i], self.obstacle_cell) and vmax >= abs(vs + block_items[j + y][i]["total_vol"]):
                                        # BL_i_j 를 AL_i_k에 할당
                                        alloc_cell.append(block_items[j + y][i])
                                        logging.getLogger(f'alloc-{block_type}').debug(f'해당셀({block_items[j + y][i]["block_name"]}) 할당, {block_type}AL_{i}_{k}: {[c.get("block_name") for c in alloc_cell]}')
                                    else: # NO
                                        break
                        # 할당셀에 대한 절성토량(Vs) 산출, 할당셀 당 작업 반복 횟수 설정 |Vs /e|를 올림한 정수값 = 반복횟수[R]
                        if 'repeat_count' not in alloc_cell_info:
                            vs = sum([c.get('total_vol') for c in alloc_cell])
                            alloc_cell_info['repeat_count'] = math.ceil(abs(vs / self.blade_capacity))
                        
                        alloc_cell_info['min_j'] = min([list(Block.get_bl_i_j(c))[1] for c in alloc_cell_info['cells']])
                        
                        # 루프 종료
                        j = j + y - 1
                        break
                
                current_j[i - 1] = j
                logging.getLogger(f'alloc-{block_type}').debug(f'current_j[{i - 1}] = j({j})')

            logging.getLogger(f'alloc-{block_type}').debug(f'current_j: {current_j}')

            # logging.getLogger(f'alloc-{block_type}').debug(f'################################################################################################################################################')
            # for _, v2 in self.allocate_cell[block_type][k].items():
            #     logging.getLogger(f'alloc-{block_type}').debug(f'self.allocate_cell[block_type][{v2["name"]}]: K: {k} R: {v2["repeat_count"]}, cells: {[c.get("block_name") for c in v2["cells"]]}')
            # logging.getLogger(f'alloc-{block_type}').debug(f'################################################################################################################################################')

            # 각 할당셀의 첫 셀의 행이 start_j이고, 이값들의 배열을 arr_start_j라고 했을 때, min(arr_start_j)을 기준으로 각 할당셀의 첫 셀의 행(start_j)이 2칸 초과인 경우(min(arr_start_j) +2 < start_j) 할당셀 전체 제거
            # k == 1 일 때는 min(arr_start_j)를 기준으로 start_j !=min(arr_start_j)인 할당셀 전체 제거
            if k in self.allocate_cell[block_type]:
                arr_start_j = [_alloc_cell_info['min_j'] for _alloc_cell_info in self.allocate_cell[block_type][k].values()]
                min_start_j = min(arr_start_j)
                logging.getLogger(f'alloc-{block_type}').debug(f'각 할당셀의 첫 셀의 행이 start_j이고, 이값들의 배열을 arr_start_j라고 했을 때, min(arr_start_j)을 기준으로 각 할당셀의 첫 셀의 행(start_j)이 2칸 초과인 경우(min(arr_start_j) +2 < start_j) 할당셀 전체 제거')
                logging.getLogger(f'alloc-{block_type}').debug(f'arr_start_j: {arr_start_j}, min(arr_start_j): {min_start_j}')
                _buffer_cell_count = 0 if k == 1 else 2

                # start_j > min(arr_start_j) + _buffer_cell_count 인 할당셀 전체 제거 - k == 1 일 때는 0, 
                # 각 할당셀에서 행번호가 가장 높은 셀간의 위치가 2칸 초과라면 해당 셀 할당셀 목록에서 제거 ->
                # min(current_j)를 기준으로 행 번호가 2 초과로 차이나는 셀을 해당 할당셀 목록에서 제거

                for _i in list(self.allocate_cell[block_type][k].keys()):
                    if self.allocate_cell[block_type][k][_i]['min_j'] > min_start_j + _buffer_cell_count:
                        logging.getLogger(f'alloc-{block_type}').debug(f'할당셀목록({self.allocate_cell[block_type][k][_i]["name"]}) 제거, k: {k}, start_j: {self.allocate_cell[block_type][k][_i]["min_j"]} > {min_start_j}')
                        del self.allocate_cell[block_type][k][_i]

                for _i in list(self.allocate_cell[block_type][k].keys()):
                    # 각 할당셀(AL_{}_k)의 셀 중 행번호가 min(current_j) + 2보다 클 경우 해당 할당셀 목록에서 제거
                    tmp_cell = []
                    for c in self.allocate_cell[block_type][k][_i]['cells']:
                        if list(Block.get_bl_i_j(c))[1] <= min(current_j) + 2:
                            tmp_cell.append(c)
                            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{_i}_{k}의 {c["block_name"]}셀의 행번호({list(Block.get_bl_i_j(c))[1]})가 min(current_j)({min(current_j)}) + 2보다 작거나 같으므로 해당 할당셀 목록 유지')
                        else:
                            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{_i}_{k}의 {c["block_name"]}셀의 행번호({list(Block.get_bl_i_j(c))[1]})가 min(current_j)({min(current_j)}) + 2보다 크므로 해당 할당셀 목록에서 제거')

                    self.allocate_cell[block_type][k][_i]['cells'] = tmp_cell
                    
                    # 할당셀 개수가 0인 경우 할당셀 정보 제거
                    if len(self.allocate_cell[block_type][k][_i]['cells']) == 0:
                        logging.getLogger(f'alloc-{block_type}').debug(f'할당셀목록({self.allocate_cell[block_type][k][_i]["name"]}) 제거, (할당셀 개수: 0)')
                        del self.allocate_cell[block_type][k][_i]
                        continue

            for _i in range(1, self.M + 1):
                if current_j[_i - 1] == self.N:
                    logging.getLogger(f'alloc-{block_type}').debug(f'current_j[{_i - 1}]({current_j[_i - 1]}) == N({self.N}) : {current_j[_i - 1] == self.N}')
                    continue
                for tmp in range(k, 0, -1):
                    # AL_i_tmp에 할당된 셀이 존재 하는가?
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{_i}_{tmp}에 할당된 셀이 존재 하는가?: {tmp in self.allocate_cell[block_type] and _i in self.allocate_cell[block_type][tmp]}')
                    if tmp in self.allocate_cell[block_type] and _i in self.allocate_cell[block_type][tmp]:
                        # AL_i_tmp의 마지막 할당셀이 BL_i1_j1일 때, current_j[i] = j1
                        current_j[_i - 1] = max([list(Block.get_bl_i_j(c))[1] for c in self.allocate_cell[block_type][tmp][_i]['cells']])
                        logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{_i}_{tmp}의 할당셀목록 중 행이 가장 높은 셀은 BL_i1_j1일 때, current_j[{_i - 1}] = {current_j[_i - 1]}')
                        break
                    else:
                        if tmp == 1:
                            current_j[_i - 1] = self.start_line - 1
                            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{_i}_*에 할당된 셀이 존재 하지 않음, current_j[{_i - 1}] = {self.start_line - 1}')

        logging.getLogger(f'alloc-{block_type}').debug(f'####### 할당셀 완료 #######')

        for _, v in self.allocate_cell[block_type].items():
            for _, v2 in v.items():
                v2['cells'] = Block.sort_cells(v2['cells'])
                self.allocate_cell_names[block_type].extend(Block.get_block_names(v2["cells"]))
                logging.getLogger(f'alloc-{block_type}').debug(f'{v2["name"]}: R: {v2["repeat_count"]}, cells: {[c.get("block_name") for c in v2["cells"]]}')
        logging.getLogger(f'alloc-{block_type}').debug(f'##########################')
        return self.allocate_cell, self.allocate_cell_names, self.outline(block_items, block_type) if block_type != START_BLOCK else {}


    # 할당셀 외단라인 작업
    @log_decorator('할당셀 외단라인 알고리즘')
    def outline(self, block_items: dict, block_type: str):
        outer_cells, left_outer_cells, right_outer_cells = [], [], []
        self.outline_cell[block_type] = {1: {}, 2: {}}
        # 2번 유형의 셀(진입 불가이면서 절성토량 존재, 이동 가능(N) 셀)들과 진입 가능이며 할당되지 않았으며 이동 가능(Y)한 셀 중 2번 유형 셀과 맞닿은 셀들에 대해서 
        # 좌측 외단라인에 대한 셀이면 left_outer_cells에 해당 셀 저장
        # 우측 외단라인에 대한 셀이면 right_outer_cells에 해당 셀 저장 
        # * 2번 유형셀: 진입 불가이면서 절성토량 존재, 이동 가능(Y) 셀
        # * 진입 가능 조건: 기준 셀이 BL_i_j라고 했을 때 BL_i_j-1 과 BL_i_j-2 가 이동 가능(Y) 셀이다 (j-1 혹은 j-2가 1보다 작다면 이동 불가(N)으로 취급)
        # * 맞닿은 셀 조건: 기준 셀이 BL_i_j라고 했을 때 BL_i-1_j 과 BL_i+1_j이 맞 닿은셀이다.
        # * ceil(M(최대열번호) / 2) 보다 열번호가 작거나 같을 경우 left_outer_cells, 클경우 right_outer_cells로 저장
        case_1_cells, _case_2_cells, converted_1_cells = Block.get_cell_1_2(block_items, self.allocate_cell_names[block_type], self.obstacle_cell)
        logging.getLogger(f'alloc-{block_type}').debug(f'진입 가능(Y)하고, 할당되지 않은 셀(1번 유형): {case_1_cells.keys()}')
        logging.getLogger(f'alloc-{block_type}').debug(f' 진입 불가, 할당되지 않은 셀, 이동 가능(Y) (2번 유형): {_case_2_cells.keys()}')

        case_2_cells = {}
        # 각 행에서 1번유형(진입가능, 절성토량 존재) 중 가장 i값이 작은 LBL 기준 i-1인  2번유형의 셀이 하나만 남도록 함
        # 각 행에서 1번유형(진입가능, 절성토량 존재) 중 가장 i값이 큰 LBL 기준 i+1인  2번유형의 셀이 하나만 남도록 함
        for _j, blocks in converted_1_cells.items():
            blocks_i = [int(list(Block.get_bl_i_j(_block))[0]) for _block in blocks]
            i_min, i_max = min(blocks_i), max(blocks_i)
            logging.getLogger(f'alloc-{block_type}').debug(f'{_j}행 i_min: {i_min}, i_max: {i_max}, blocks: {Block.get_block_names(blocks)}')
            min_key = f'{block_type}BL_{i_min-1}_{_j}'
            max_key = f'{block_type}BL_{i_max+1}_{_j}'
            if min_key in _case_2_cells.keys():
                case_2_cells[min_key] = _case_2_cells[min_key]
            if max_key in _case_2_cells.keys():
                case_2_cells[max_key] = _case_2_cells[max_key]

        logging.getLogger(f'alloc-{block_type}').debug(f'각 행에서 1번유형(진입가능, 절성토량 존재) 중 가장 i값이 작은 LBL 기준 i-1, 가장 i값이 큰 LBL 기준 i+1인 2번유형의 셀만 남도록 함 (2번 유형): {case_2_cells.keys()}') 

        # case_1_cells 셀 중 2번유형과 맞닿은셀 추출
        for block in case_2_cells.values():
            # block_i: 열, block_j: 행
            block_i, block_j = Block.get_bl_i_j(block)
            # outer_cells.append(block)

            # 2번 유형의 셀의 -1열의 block 중 case_1에 있을 경우 사용
            if f'{block_type}BL_{block_i - 1}_{block_j}' in case_1_cells:
                outer_cells.append(case_1_cells[f'{block_type}BL_{block_i - 1}_{block_j}']) 

            # 2번 유형의 셀의 +1열의 block 중 case_1에 있을 경우 사용
            if f'{block_type}BL_{block_i + 1}_{block_j}' in case_1_cells:
                outer_cells.append(case_1_cells[f'{block_type}BL_{block_i + 1}_{block_j}']) 

        logging.getLogger(f'alloc-{block_type}').debug(f'outer_L_cells: {Block.get_block_names(outer_cells)}')
        
        # outer_L_cells 중 j가 가장 큰 행 LBL들의 나열에서 i가 가장 작은 min_i i가 가장 큰 max_i x = (min_i + max_i) /2
        self.N, self.M = Block.get_n_m(block_items)
        tmp = {}
        for cell in outer_cells:
            # block_i: 열, block_j: 행
            block_i, block_j = Block.get_bl_i_j(cell)
            tmp.setdefault(block_j, []).append(block_i)
        
        logging.getLogger(f'alloc-{block_type}').debug(f'tmp.keys(): {tmp.keys()}')
        min_i, max_i = min(tmp[max(tmp.keys())]), max(tmp[max(tmp.keys())])
        center_x = (min_i + max_i) / 2
        logging.getLogger(f'alloc-{block_type}').debug(f'outer_L_cells 중 j가 가장 큰 행 {block_type}BL들의 나열에서 i가 가장 작은 min_i({min_i}) i가 가장 큰 max_i({max_i})')
        logging.getLogger(f'alloc-{block_type}').debug(f'x({center_x}) = (min_i({min_i}) + max_i({max_i})) /2')
        for cell in outer_cells:
            # block_i: 열, block_j: 행
            block_i, block_j = Block.get_bl_i_j(cell)
            if center_x >= block_i:
                left_outer_cells.append(cell)
            else:
                right_outer_cells.append(cell)

        # w = ceil ( required_line_change_distance / 2 / cell_size)
        w = math.ceil(self.required_line_change_distance / 2 / self.cell_size)
        logging.getLogger(f'alloc-{block_type}').debug(f'w({w}): ceil ( required_line_change_distance({self.required_line_change_distance}) / 2 / cell_size({self.cell_size}))')
        # 행 번호가 w +1 보다 작은 경우 left_outer_L_cells와 right_outer_L_cells 에서 해당 셀을 제거
        left_outer_cells = Block.sort_cells([cell for cell in left_outer_cells if list(Block.get_bl_i_j(cell))[1] > w])
        right_outer_cells = Block.sort_cells([cell for cell in right_outer_cells if list(Block.get_bl_i_j(cell))[1] > w])
        
        logging.getLogger(f'alloc-{block_type}').debug(f'left_outer_cells: {Block.get_block_names(left_outer_cells)}')
        logging.getLogger(f'alloc-{block_type}').debug(f'right_outer_cells: {Block.get_block_names(right_outer_cells)}')

        # left_outer_cell와 rtight_out_cells 모두 남은 셀이 없을 때 No, 둘중 하나라도 남은 셀이 있을 때 Yes
        j = 0
        while (len(left_outer_cells) > 0 or len(right_outer_cells) > 0):
            j += 1
            logging.getLogger(f'alloc-{block_type}').debug(f'j: {j}, len(left_outer_cells): {len(left_outer_cells)}, len(right_outer_cells): {len(right_outer_cells)}')
            if j > self.N:
                break

            left_j_outline_cell, right_j_outline_cell = [], []
            for i in range(1, self.M + 1):
                cells = self.allocate_cell[block_type].get(j, {}).get(i, {}).get('cells', [])
                # AL_i_j에 할당 된 셀이 있는가?
                logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{i}_{j}에 할당 된 셀이 있는가? : {len(cells) > 0}')

                if cells:
                    b1 = list(Block.get_bl_i_j(Block.sort_cells(cells)[-1]))[1]
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀  BL_a2_b2  = OL_1_j의 원소 일 때, b1<b2 이면 BL_a2_b2를 OL_1_j 에서 제거
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀(해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_1_j의 원소 일 때, b1>=b2 이면 BL_a2_b2를 OL_1_j 에서 추출
                    left_j_outline_cell = Block.sort_cells([cell for cell in left_outer_cells if list(Block.get_bl_i_j(cell))[1] <= b1])
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_a1_b1({b1}) = {block_type}AL_i_j의 마지막 할당셀(해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_1_j의 원소 일 때, b1>=b2 이면 BL_a2_b2를 OL_1_j 에서 추출: {Block.get_block_names(left_j_outline_cell)}')
                    break

            for i in range(self.M , 0, -1):
                cells = self.allocate_cell[block_type].get(j, {}).get(i, {}).get('cells', [])
                # AL_i_j에 할당 된 셀이 있는가?
                logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}AL_{i}_{j}에 할당 된 셀이 있는가? : {len(cells) > 0}')
                # AL_i_j에 할당 된 셀이 있는가?
                if cells:
                    b1 = list(Block.get_bl_i_j(Block.sort_cells(cells)[-1]))[1]
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀 BL_a2_b2  = OL_2_j의 원소 일 때, b1<=b2 를 만족하는 BL_a2_b2 외에 나머지 셀들을 OL_1_j 에서 제거
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀 (해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_2_j의 원소 일 때, b1>=b2 이면 BL_a2_b2를 OL_2_j 에서 추출
                    right_j_outline_cell = Block.sort_cells([cell for cell in right_outer_cells if list(Block.get_bl_i_j(cell))[1] <= b1])
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_a1_b1({b1}) = {block_type}AL_i_j의 마지막 할당셀 (해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_2_j의 원소 일 때,b1>=b2 이면 BL_a2_b2를 OL_2_j 에서 추출: {Block.get_block_names(right_j_outline_cell)}')
                    break
            
            # BL_a2_b2  = OL_1_j[k] 일 때, BL_a2_(b2+1)이 할당된 적이 있는 셀인가? (마지막행 제외)
            b21 = 0
            for k, cell in enumerate(left_j_outline_cell):
                block_i, block_j = Block.get_bl_i_j(cell)
                if f'{block_type}BL_{block_i}_{block_j + 1}' in self.allocate_cell_names[block_type]:
                    b21 = block_j + 1
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}OL_1_j[k({k})] 일 때, {block_type}BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: True')
                    break
                else:
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}OL_1_j[k({k})] 일 때,, {block_type}BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: False')

            # BL_a2_(b2+1)과 동일선상(같은 행)과 그 이후에 있는 셀을 제외한 나머지 셀을 OL_1_j에 할당
            if b21 > 0:
                logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_a2_b2({b21 - 1}) 과 동일선상(같은 행) 혹은 그 이전에 있는 할당셀 추출, {block_type}OL_1_j에 할당')
                self.outline_cell[block_type][1][j] = Block.sort_cells([cell for cell in left_j_outline_cell if list(Block.get_bl_i_j(cell))[1] < b21])
            else:
                self.outline_cell[block_type][1][j] = left_j_outline_cell

            # v1.0.7 BL_a2_b2  = OL_2_j[k] 일 때, BL_a2_(b2+1)이 할당된 적이 있는 셀인가? (마지막행 제외)
            b21 = 0
            for k, cell in enumerate(right_j_outline_cell):
                block_i, block_j = Block.get_bl_i_j(cell)
                if f'{block_type}BL_{block_i}_{block_j + 1}' in self.allocate_cell_names[block_type]:
                    b21 = block_j + 1
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_a2_b2  = {block_type}OL_2_j[k({k})] 일 때, {block_type}BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: True')
                    break
                else:
                    logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_a2_b2  = {block_type}OL_2_j[k({k})] 일 때, {block_type}BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: False')

            # BL_a2_(b2+1)과 동일선상(같은 행)과 그 이후에 있는 셀을 제외한 나머지 셀을 OL_2_j에 할당
            if b21 > 0:
                logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}BL_a2_b2({b21 - 1}) 과 동일선상(같은 행) 혹은 그 이전에 있는 할당셀 추출, {block_type}OL_2_j에 할당')
                self.outline_cell[block_type][2][j] = Block.sort_cells([cell for cell in right_j_outline_cell if list(Block.get_bl_i_j(cell))[1] < b21])
            else:
                self.outline_cell[block_type][2][j] = right_j_outline_cell

            # OL_1_j에 할당된 셀들을 left_outer_cells에서 제거
            for cell in self.outline_cell[block_type][1][j]:
                left_outer_cells.remove(cell)
            # OL_2_j에 할당된 셀들을 right_outer_cells 제거
            for cell in self.outline_cell[block_type][2][j]:
                right_outer_cells.remove(cell)
            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}OL_1_{j}: {[block.get("block_name") for block in self.outline_cell[block_type][1][j]]}')
            logging.getLogger(f'alloc-{block_type}').debug(f'{block_type}OL_2_{j}: {[block.get("block_name") for block in self.outline_cell[block_type][2][j]]}')

        return self.outline_cell
    
    # output csv
    @log_decorator('할당셀 알고리즘 결과 CSV 저장')
    def save_output_csv(self, output_path: str, block_type: str):
        makedirs(output_path, exist_ok=True)

        fieldnames = ['No', 'ALName', 'cutVol', 'fillVol', 'TotalVol', 'BLList', 'R']
        data = []
        num = 0
        for _, v in self.allocate_cell[block_type].items():
            for _, v2 in v.items():
                num += 1
                data.append({'No': num, 'ALName':v2["name"] , 'cutVol': sum([c.get("cut_vol") for c in v2["cells"]]), 'fillVol': sum([c.get("fill_vol") for c in v2["cells"]]), 'TotalVol': sum([c.get("total_vol") for c in v2["cells"]]), 'BLList': ','.join([c.get("block_name") for c in v2["cells"]]), 'R': v2["repeat_count"]})

        for out_i, _v in self.outline_cell[block_type].items():
            for out_j, blocks in _v.items():
                num += 1
                vs = sum([c.get('total_vol') for c in blocks])
                data.append({'No': num, 'ALName': f'OL_{out_i}_{out_j}' , 'cutVol': sum([c.get("cut_vol") for c in blocks]), 'fillVol': sum([c.get("fill_vol") for c in blocks]), 'TotalVol': vs, 'BLList': ','.join([c.get("block_name") for c in blocks]), 'R': math.ceil(abs(vs / self.blade_capacity))})
        
        with open(f'{output_path}/{datetime.now().strftime("%Y%m%d%H%M%S")}_{block_type}_alloc_output.csv', 'w', newline='\n', encoding='ansi') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
