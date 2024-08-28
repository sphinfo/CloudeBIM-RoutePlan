# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import csv
import math
import logging
from os import makedirs
from datetime import datetime
from route_planner_v10.util import log_decorator
from route_planner_v10.block import Block


class DozerAlloc():
    def __init__(self, block_items: dict, outline_data: dict):
        super().__init__()
        # Block 정보 {'j': {'i': {Block}}}
        self.block_items = block_items
        # 할당셀 집합의 목록 {'k': {'i': {'cells': [], 'name': 'AL_M_N', 'repeat_count': 0, }}}
        self.allocate_cell = {}
        self.allocate_cell_names = []
        self.outline_cell = {1: {}, 2: {}}
        # M 최대 열 번호, N 최대 행 번호
        self.N, self.M = Block.get_n_m(block_items)
        # 버킷 용량
        self.e = None
        # 외단라인 좌표 데이터
        self.outline_data = outline_data

    # output csv
    @log_decorator('할당셀 알고리즘 결과 CSV 저장')
    def save_output_csv(self, input_file_name: str):
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)

        fieldnames = ['No', 'ALName', 'cutVol', 'fillVol', 'TotalVol', 'BLList', 'R']
        data = []
        num = 0
        for _, v in self.allocate_cell.items():
            for _, v2 in v.items():
                num += 1
                data.append({'No': num, 'ALName':v2["name"] , 'cutVol': sum([c.get("cut_vol") for c in v2["cells"]]), 'fillVol': sum([c.get("fill_vol") for c in v2["cells"]]), 'TotalVol': sum([c.get("total_vol") for c in v2["cells"]]), 'BLList': ','.join([c.get("block_name") for c in v2["cells"]]), 'R': v2["repeat_count"]})

        for out_i, _v in self.outline_cell.items():
            for out_j, blocks in _v.items():
                num += 1
                vs = sum([c.get('total_vol') for c in blocks])
                data.append({'No': num, 'ALName': f'OL_{out_i}_{out_j}' , 'cutVol': sum([c.get("cut_vol") for c in blocks]), 'fillVol': sum([c.get("fill_vol") for c in blocks]), 'TotalVol': vs, 'BLList': ','.join([c.get("block_name") for c in blocks]), 'R': math.ceil(abs(vs / self.e))})
        
        with open(f'./output/csv/{input_file_name}/{datetime.now().strftime("%Y%m%d%H%M%S")}_dozer_v2_alloc_output.csv', 'w', newline='\n', encoding='ansi') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


    # 불도저 버켓용량: Blade Capacity -> e
    # 최소전진거리: Min Fwdist -> s
    # 라인변경에 필요한 거리: needed dist -> h
    # 중심선 노드간 최소거리: -> l
    # 장애물 셀 지정: Obstacle Cell oc
    @log_decorator('도저 할당셀 알고리즘')
    def alloc(self, param: dict):

        e, s, h, l, obstacle_cells, start_line, h_num, s_num = map(
            param.get, ['e', 's', 'h', 'l', 'obstacle_cells', 'start_line', 'h_num', 's_num'])

        # h_num이거나 두칸
        buffer_cell_count = 2

        logging.getLogger('alloc').debug(f'불도저 버켓용량(e): {e}, 최소전진거리(s): {s}, 라인변경에 필요한 거리(h): {h}, 중심선 노드간 최소거리(l): {l}')
        logging.getLogger('alloc').debug(f'장애물셀: {obstacle_cells}, Start Line: {start_line}')
        logging.getLogger('alloc').debug(f'라인변경에 필요한 셀 칸수 산정(H_num): {h_num}, 최소 할당하는셀의 개수 산정(S_num): {s_num}')
        logging.getLogger('alloc').debug(f'최대 열 번호 M = {self.M}, 최대 행 번호 N = {self.N}')

        # current_j[]는 M개의 배열, 모든 원소 값 = start_line-1
        current_j = [start_line - 1 for _ in range(0, self.M)]
        self.e = e

        k = 0
        logging.getLogger('alloc').debug(f'min(current_j): {min(current_j)}')
        while (min(current_j) < self.N):
            k += 1
            logging.getLogger('alloc').debug(f'k: {k}')
            for i in range(1, self.M + 1):
                for j in range(current_j[i - 1] + 1, self.N + 1):
                    # BL_i_j가 이동 가능한가?
                    block_name = self.block_items[j][i]['block_name']
                    moveable = Block.check_moveable(self.block_items[j][i], obstacle_cells)
                    logging.getLogger('alloc').debug(f'BL_i_j({block_name})가 이동 가능한가?: {moveable}')

                    if not moveable:
                        continue

                    # v1.0.1 - 현재셀(BL_i_j)의 직전H_num만큼의 셀들(BL_i_j-1, BL_i_j-2 … BL_i_j-H_num)은 이동가능(Y) 셀인가?
                    recent_cells = []
                    for x in range(j - 1, j - h_num - 1, -1):
                        block_x, block_name_x = self.block_items[x][i], self.block_items[x][i]['block_name']
                        moveable_x = Block.check_moveable(block_x, obstacle_cells)
                        if x > 0:
                            recent_cells.append(f"{block_name_x}: {moveable_x}")
                            if not moveable_x:
                                moveable = False
                                # break # 전부 이동가능하여야 하므로 False일 경우 다음 loop는 체크할 필요가 없으나 debugging 용도로 전부 계산( 추후 break 하면됨)
                        else:
                            moveable = False
                            # break # 전부 이동가능하여야 하므로 False일 경우 다음 loop는 체크할 필요가 없으나 debugging 용도로 전부 계산( 추후 break 하면됨)
                    logging.getLogger('alloc').debug(f'현재셀(BL_i_j)의 직전H_num({h_num})만큼의 셀들(BL_i_j-1, BL_i_j-2 … BL_i_j-H_num)은 이동가능(Y) 셀인가?: {moveable}, {recent_cells}')
                    
                    if moveable:
                        target = self.allocate_cell.setdefault(k, {})
                        alloc_cell_info = target.setdefault(i, {})
                        alloc_cell_info['name'] = f'AL_{i}_{k}'
                        alloc_cell = alloc_cell_info.setdefault('cells', [])

                        # BL_i_j 를 AL_i_k에 할당
                        alloc_cell.append(self.block_items[j][i])
                        logging.getLogger('alloc').debug(f'해당셀({block_name}) 할당, AL_{i}_{k}: {[c.get("block_name") for c in alloc_cell]}')

                        for y in range(1, self.N - j + 1):
                            # S_num만큼 할당했는가?
                            logging.getLogger('alloc').debug(f'S_num({s_num})만큼 할당했는가?: {len(alloc_cell) >= s_num}')

                            if len(alloc_cell) < s_num: # S_num만큼 할당했는가? -> NO
                                # j <= N
                                if j + y in self.block_items:
                                    # BL_i_j+y가 진입 가능(Y)한가?
                                    is_alloc = Block.check_moveable(self.block_items[j + y][i], obstacle_cells)
                                    logging.getLogger('alloc').debug(f'BL_i_j({self.block_items[j + y][i]["block_name"]})가 이동 가능한가?: {is_alloc}')

                                    if is_alloc:
                                        # BL_i_j 를 AL_i_k에 할당
                                        alloc_cell.append(self.block_items[j + y][i])
                                        logging.getLogger('alloc').debug(f'해당셀({self.block_items[j + y][i]["block_name"]}) 할당, AL_{i}_{k}: {[c.get("block_name") for c in alloc_cell]}')
                            else: # S_num만큼 할당했는가? -> YES
                                # 할당셀에 대한 절성토량(Vs) 산출
                                vs = sum([c.get('total_vol') for c in alloc_cell])
                                
                                if 'repeat_count' not in alloc_cell_info:
                                    logging.getLogger('alloc').debug(f'할당셀에 대한 절성토량(Vs) 산출, Vs = {vs}')
                                    # 할당셀 당 작업 반복 횟수 설정 |Vs /e|를 올림한 정수값 = 반복횟수[R]
                                    alloc_cell_info['repeat_count'] = math.ceil(abs(vs / e))
                                    logging.getLogger('alloc').debug(f'할당셀 당 작업 반복 횟수 설정 |Vs /e|를 올림한 정수값 = 반복횟수[R], R = {alloc_cell_info["repeat_count"] }')
                                    vmax = alloc_cell_info['repeat_count'] * e
                                    logging.getLogger('alloc').debug(f'최대 절성토량(Vmax) 계산 R(반복횟수) x e(버킷용량) = Vmax, Vmax = {vmax}')

                                if j + y in self.block_items:
                                    logging.getLogger('alloc').debug(f'현재 셀({self.block_items[j + y][i]["block_name"]})이 이동가능(Y) 셀인가?: {Block.check_moveable(self.block_items[j + y][i], obstacle_cells)}')
                                    logging.getLogger('alloc').debug(f'최대 절성토량이 넘지 않았는가? {vmax} >= {abs(vs + self.block_items[j + y][i]["total_vol"])} {vmax >= abs(vs + self.block_items[j + y][i]["total_vol"])}')
                                    if Block.check_moveable(self.block_items[j + y][i], obstacle_cells) and vmax >= abs(vs + self.block_items[j + y][i]["total_vol"]):
                                        # BL_i_j 를 AL_i_k에 할당
                                        alloc_cell.append(self.block_items[j + y][i])
                                        logging.getLogger('alloc').debug(f'해당셀({self.block_items[j + y][i]["block_name"]}) 할당, AL_{i}_{k}: {[c.get("block_name") for c in alloc_cell]}')
                                    else: # NO
                                        break
                        # 할당셀에 대한 절성토량(Vs) 산출, 할당셀 당 작업 반복 횟수 설정 |Vs /e|를 올림한 정수값 = 반복횟수[R]
                        if 'repeat_count' not in alloc_cell_info:
                            vs = sum([c.get('total_vol') for c in alloc_cell])
                            alloc_cell_info['repeat_count'] = math.ceil(abs(vs / e))
                        
                        alloc_cell_info['min_j'] = min([list(Block.get_bl_i_j(c))[1] for c in alloc_cell_info['cells']])
                        
                        # 루프 종료
                        j = j + y - 1
                        break
                
                current_j[i - 1] = j
                logging.getLogger('alloc').debug(f'current_j[{i - 1}] = j({j})')

            logging.getLogger('alloc').debug(f'current_j: {current_j}')

            # logging.getLogger('alloc').debug(f'################################################################################################################################################')
            # for _, v2 in self.allocate_cell[k].items():
            #     logging.getLogger('alloc').debug(f'self.allocate_cell[{v2["name"]}]: K: {k} R: {v2["repeat_count"]}, cells: {[c.get("block_name") for c in v2["cells"]]}')
            # logging.getLogger('alloc').debug(f'################################################################################################################################################')

            # 각 할당셀의 첫 셀의 행이 start_j이고, 이값들의 배열을 arr_start_j라고 했을 때, min(arr_start_j)을 기준으로 각 할당셀의 첫 셀의 행(start_j)이 2칸(혹은h_num) 초과인 경우(min(arr_start_j) +2(혹은h_num) < start_j) 할당셀 전체 제거
            # k == 1 일 때는 min(arr_start_j)를 기준으로 start_j !=min(arr_start_j)인 할당셀 전체 제거
            if k in self.allocate_cell:
                arr_start_j = [_alloc_cell_info['min_j'] for _alloc_cell_info in self.allocate_cell[k].values()]
                min_start_j = min(arr_start_j)
                logging.getLogger('alloc').debug(f'각 할당셀의 첫 셀의 행이 start_j이고, 이값들의 배열을 arr_start_j라고 했을 때, min(arr_start_j)을 기준으로 각 할당셀의 첫 셀의 행(start_j)이 2칸(혹은h_num) 초과인 경우(min(arr_start_j) +2(혹은h_num) < start_j) 할당셀 전체 제거')
                logging.getLogger('alloc').debug(f'arr_start_j: {arr_start_j}, min(arr_start_j): {min_start_j}')
                _buffer_cell_count = 0 if k == 1 else buffer_cell_count

                # start_j > min(arr_start_j) + _buffer_cell_count 인 할당셀 전체 제거 - k == 1 일 때는 0, 
                for _i in list(self.allocate_cell[k].keys()):
                    if self.allocate_cell[k][_i]['min_j'] > min_start_j + _buffer_cell_count:
                        logging.getLogger('alloc').debug(f'할당셀목록({self.allocate_cell[k][_i]["name"]}) 제거, k: {k}, start_j: {self.allocate_cell[k][_i]["min_j"]} > {min_start_j}')
                        del self.allocate_cell[k][_i]

                for _i in list(self.allocate_cell[k].keys()):
                    # 각 할당셀(AL_{}_k)의 셀 중 행번호가 min(current_j) + 2보다 클 경우 해당 할당셀 목록에서 제거
                    org_length = len(self.allocate_cell[k][_i]['cells'])
                    tmp_cell = []
                    for c in self.allocate_cell[k][_i]['cells']:
                        if list(Block.get_bl_i_j(c))[1] <= min(current_j) + 2:
                            tmp_cell.append(c)
                            logging.getLogger('alloc').debug(f'AL_{_i}_{k}의 {c["block_name"]}셀의 행번호({list(Block.get_bl_i_j(c))[1]})가 min(current_j)({min(current_j)}) + 2보다 작거나 같으므로 해당 할당셀 목록 유지')
                        else:
                            logging.getLogger('alloc').debug(f'AL_{_i}_{k}의 {c["block_name"]}셀의 행번호({list(Block.get_bl_i_j(c))[1]})가 min(current_j)({min(current_j)}) + 2보다 크므로 해당 할당셀 목록에서 제거')

                    self.allocate_cell[k][_i]['cells'] = tmp_cell
                    
                    # 할당셀 개수가 0인 경우 할당셀 정보 제거
                    if len(self.allocate_cell[k][_i]['cells']) == 0:
                        logging.getLogger('alloc').debug(f'할당셀목록({self.allocate_cell[k][_i]["name"]}) 제거, (할당셀 개수: 0)')
                        del self.allocate_cell[k][_i]
                        continue

                    # TODO: (최종 납기 시 삭제) 할당셀의 반복 횟수 재계산 (추가 여부 확인 필요) -알고리즘 변경됐는지 확인
                    # if org_length != len(self.allocate_cell[k][_i]['cells']):
                    #     vs = sum([c.get('total_vol') for c in alloc_cell])
                    #     self.allocate_cell[k][_i]['repeat_count'] = math.ceil(abs(vs / e))

            for _i in range(1, self.M + 1):
                if current_j[_i - 1] == self.N:
                    logging.getLogger('alloc').debug(f'current_j[{_i - 1}]({current_j[_i - 1]}) == N({self.N}) : {current_j[_i - 1] == self.N}')
                    continue
                for tmp in range(k, 0, -1):
                    # AL_i_tmp에 할당된 셀이 존재 하는가?
                    logging.getLogger('alloc').debug(f'AL_{_i}_{tmp}에 할당된 셀이 존재 하는가?: {tmp in self.allocate_cell and _i in self.allocate_cell[tmp]}')
                    if tmp in self.allocate_cell and _i in self.allocate_cell[tmp]:
                        # AL_i_tmp의 마지막 할당셀이 BL_i1_j1일 때, current_j[i] = j1
                        current_j[_i - 1] = max([list(Block.get_bl_i_j(c))[1] for c in self.allocate_cell[tmp][_i]['cells']])
                        logging.getLogger('alloc').debug(f'AL_{_i}_{tmp}의 할당셀목록 중 행이 가장 높은 셀은 BL_i1_j1일 때, current_j[{_i - 1}] = {current_j[_i - 1]}')
                        break
                    else:
                        if tmp == 1:
                            current_j[_i - 1] = start_line - 1
                            logging.getLogger('alloc').debug(f'AL_{_i}_*에 할당된 셀이 존재 하지 않음, current_j[{_i - 1}] = {start_line - 1}')

            # logging.getLogger('alloc').debug(f'################################################################################################################################################')
            # for _, v2 in self.allocate_cell[k].items():
            #     logging.getLogger('alloc').debug(f'self.allocate_cell[{v2["name"]}]: K: {k} R: {v2["repeat_count"]}, cells: {[c.get("block_name") for c in v2["cells"]]}')
            # logging.getLogger('alloc').debug(f'################################################################################################################################################')


        logging.getLogger('alloc').debug(f'####### 할당셀 완료 #######')

        for a, v in self.allocate_cell.items():
            for b, v2 in v.items():
                v2['cells'] = Block.sort_cells(v2['cells'])
                self.allocate_cell_names.extend(Block.get_block_names(v2["cells"]))
                logging.getLogger('alloc').debug(f'self.allocate_cell[{v2["name"]}]: R: {v2["repeat_count"]}, cells: {[c.get("block_name") for c in v2["cells"]]}')
        logging.getLogger('alloc').debug(f'##########################')
        return self.allocate_cell, self.allocate_cell_names, self.outline(s_num, obstacle_cells)

    # 할당셀 외단라인 작업
    def outline(self, s_num: float, obstacle_cells: list):
        left_outer_cells, right_outer_cells = [], []
        
        # 2번 유형의 셀(진입 불가이면서 절성토량 존재, 이동 가능(Y) 셀)들과 진입 가능이며 할당되지 않았으며 이동 가능(Y)한 셀 중 2번 유형 셀과 맞닿은 셀들에 대해서 
        # 좌측 외단라인에 대한 셀이면 left_outer_cells에 해당 셀 저장
        # 우측 외단라인에 대한 셀이면 right_outer_cells에 해당 셀 저장 
        # * 2번 유형셀: 진입 불가이면서 절성토량 존재, 이동 가능(Y) 셀
        # * 진입 가능 조건: 기준 셀이 BL_i_j라고 했을 때 BL_i_j-1 과 BL_i_j-2 가 이동 가능(Y) 셀이다 (j-1 혹은 j-2가 1보다 작다면 이동 불가(N)으로 취급)
        # * 맞닿은 셀 조건: 기준 셀이 BL_i_j라고 했을 때 BL_i-1_j 과 BL_i+1_j이 맞 닿은셀이다.
        # * ceil(M(최대열번호) / 2) 보다 열번호가 작거나 같을 경우 left_outer_cells, 클경우 right_outer_cells로 저장
        case_1_cells, case_2_cells = {}, {}

        for j, _v in self.block_items.items():
            for i, block in _v.items():
                # 할당된 셀이거나 절성토량이 없을 경우 skip
                block_name = block.get('block_name')
                if block_name in set(self.allocate_cell_names) or abs(block.get('total_vol'))  == 0:
                    continue

                # 접근 가능이면 case_1_cells에 추가, 아닐 경우 case_2_cells에 추가
                if Block.check_accessible(block, self.block_items, s_num, obstacle_cells):
                    case_1_cells[block_name] = block
                else:
                    case_2_cells[block_name] = block

        # 2번 유형의 셀과 1번유형의 셀 중 2번유형과 맞닿은셀 추출
        for block_name, block in case_2_cells.items():
            # block_i: 열, block_j: 행
            block_i, block_j = Block.get_bl_i_j(block)
            if math.ceil(self.M / 2) > block_i:
                left_outer_cells.append(block) 
            else:
                right_outer_cells.append(block)
            
            # 2번 유형의 셀의 -1열의 block 중 case_1에 있을 경우 사용
            if f'BL_{block_i - 1}_{block_j}' in case_1_cells:
                if math.ceil(self.M / 2) > block_i - 1:
                    left_outer_cells.append(case_1_cells[f'BL_{block_i - 1}_{block_j}']) 
                else:
                    right_outer_cells.append(case_1_cells[f'BL_{block_i - 1}_{block_j}'])
            # 2번 유형의 셀의 +1열의 block 중 case_1에 있을 경우 사용
            if f'BL_{block_i + 1}_{block_j}' in case_1_cells:
                if math.ceil(self.M / 2) > block_i + 1:
                    left_outer_cells.append(case_1_cells[f'BL_{block_i + 1}_{block_j}']) 
                else:
                    right_outer_cells.append(case_1_cells[f'BL_{block_i + 1}_{block_j}'])

        left_outer_cells, right_outer_cells = Block.sort_cells(left_outer_cells), Block.sort_cells(right_outer_cells)

        logging.getLogger('alloc').debug(f'left_outer_cells: {Block.get_block_names(left_outer_cells)}')
        logging.getLogger('alloc').debug(f'right_outer_cells: {Block.get_block_names(right_outer_cells)}')

        # left_outer_cell와 rtight_out_cells 모두 남은 셀이 없을 때 No, 둘중 하나라도 남은 셀이 있을 때 Yes
        j = 0
        while (len(left_outer_cells) > 0 or len(right_outer_cells) > 0):
            j += 1
            logging.getLogger('alloc').debug(f'j: {j}, len(left_outer_cells): {len(left_outer_cells)}, len(right_outer_cells): {len(right_outer_cells)}')
            if j > self.N:
                break

            left_j_outline_cell, right_j_outline_cell = [], []
            for i in range(1, self.M + 1):
                cells = self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])
                # AL_i_j에 할당 된 셀이 있는가?
                logging.getLogger('alloc').debug(f'AL_{i}_{j}에 할당 된 셀이 있는가? : {len(cells) > 0}')

                if cells:
                    b1 = list(Block.get_bl_i_j(Block.sort_cells(cells)[-1]))[1]
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀  BL_a2_b2  = OL_1_j의 원소 일 때, b1<b2 이면 BL_a2_b2를 OL_1_j 에서 제거
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀(해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_1_j의 원소 일 때, b1>=b2 이면 BL_a2_b2를 OL_1_j 에서 추출
                    left_j_outline_cell = Block.sort_cells([cell for cell in left_outer_cells if list(Block.get_bl_i_j(cell))[1] <= b1])
                    logging.getLogger('alloc').debug(f'BL_a1_b1 = AL_i_j의 마지막 할당셀(해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_1_j의 원소 일 때, b1>=b2 이면 BL_a2_b2를 OL_1_j 에서 추출: {Block.get_block_names(left_j_outline_cell)}')
                    break

            for i in range(self.M , 0, -1):
                cells = self.allocate_cell.get(j, {}).get(i, {}).get('cells', [])
                # AL_i_j에 할당 된 셀이 있는가?
                logging.getLogger('alloc').debug(f'AL_{i}_{j}에 할당 된 셀이 있는가? : {len(cells) > 0}')
                # AL_i_j에 할당 된 셀이 있는가?
                if cells:
                    b1 = list(Block.get_bl_i_j(Block.sort_cells(cells)[-1]))[1]
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀 BL_a2_b2  = OL_2_j의 원소 일 때, b1<=b2 를 만족하는 BL_a2_b2 외에 나머지 셀들을 OL_1_j 에서 제거
                    # BL_a1_b1 = AL_i_j의 마지막 할당셀 (해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_2_j의 원소 일 때, b1>=b2 이면 BL_a2_b2를 OL_2_j 에서 추출
                    right_j_outline_cell = Block.sort_cells([cell for cell in right_outer_cells if list(Block.get_bl_i_j(cell))[1] <= b1])
                    logging.getLogger('alloc').debug(f'BL_a1_b1 = AL_i_j의 마지막 할당셀 (해당 할당셀에서 행 번호가 가장 높은 셀) BL_a2_b2  = OL_2_j의 원소 일 때,b1>=b2 이면 BL_a2_b2를 OL_2_j 에서 추출: {Block.get_block_names(right_j_outline_cell)}')
                    break
            
            # BL_a2_b2  = OL_i_1[k] 일 때, BL_a2_(b2+1)이 할당된 적이 있는 셀인가? (마지막행 제외)
            b21 = 0
            for cell in left_j_outline_cell:
                block_i, block_j = Block.get_bl_i_j(cell)
                if f'BL_{block_i}_{block_j + 1}' in self.allocate_cell_names:
                    b21 = block_j + 1
                    logging.getLogger('alloc').debug(f'OL_i_1[k] 일 때, BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: True')
                    break
                else:
                    logging.getLogger('alloc').debug(f'OL_i_1[k] 일 때, BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: False')

            # BL_a2_(b2+1)과 동일선상(같은 행)과 그 이후에 있는 셀을 제외한 나머지 셀을 OL_1_j에 할당
            if b21 > 0:
                logging.getLogger('alloc').debug(f'BL_a2_b2({b21 - 1}) 과 동일선상(같은 행) 혹은 그 이전에 있는 할당셀 추출, OL_1_j에 할당')
                self.outline_cell[1][j] = Block.sort_cells([cell for cell in left_j_outline_cell if list(Block.get_bl_i_j(cell))[1] < b21])
            else:
                self.outline_cell[1][j] = left_j_outline_cell

            # BL_a2_b2  = OL_i_1[k] 일 때, BL_a2_(b2+1)이 할당된 적이 있는 셀인가? (마지막행 제외)
            b21 = 0
            for cell in right_j_outline_cell:
                block_i, block_j = Block.get_bl_i_j(cell)
                if f'BL_{block_i}_{block_j + 1}' in self.allocate_cell_names:
                    b21 = block_j + 1
                    logging.getLogger('alloc').debug(f'BL_a2_b2  = OL_i_1[k] 일 때, BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: True')
                    break
                else:
                    logging.getLogger('alloc').debug(f'BL_a2_b2  = OL_i_1[k] 일 때, BL_{block_i}_({block_j}+1)이 할당된 적이 있는 셀인가?: False')

            # BL_a2_(b2+1)과 동일선상(같은 행)과 그 이후에 있는 셀을 제외한 나머지 셀을 OL_2_j에 할당
            if b21 > 0:
                logging.getLogger('alloc').debug(f'BL_a2_b2({b21 - 1}) 과 동일선상(같은 행) 혹은 그 이전에 있는 할당셀 추출, OL_2_j에 할당')
                self.outline_cell[2][j] = Block.sort_cells([cell for cell in right_j_outline_cell if list(Block.get_bl_i_j(cell))[1] < b21])
            else:
                self.outline_cell[2][j] = right_j_outline_cell

            # OL_1_j에 할당된 셀들을 left_outer_cells에서 제거
            for cell in self.outline_cell[1][j]:
                left_outer_cells.remove(cell)
            # OL_2_j에 할당된 셀들을 right_outer_cells 제거
            for cell in self.outline_cell[2][j]:
                right_outer_cells.remove(cell)
            logging.getLogger('alloc').debug(f'OL_1_{j}: {[block.get("block_name") for block in self.outline_cell[1][j]]}')
            logging.getLogger('alloc').debug(f'OL_2_{j}: {[block.get("block_name") for block in self.outline_cell[2][j]]}')

        return self.outline_cell
