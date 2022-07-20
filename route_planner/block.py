# COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import logging
import numpy as np
import csv
from os import makedirs
from datetime import datetime
from route_planner.util import log_decorator


class Block(object):
    VIRTUAL_BLOCK_NAME = 'BL_V'
    VIRTUAL_BLOCK = {'BlName': VIRTUAL_BLOCK_NAME, 'YN': 'N', 'virtual': True}

    # block 변환 
    @staticmethod
    def convert_block(primary_key: str, block_items: dict):
        return {str(block_item.get(primary_key)): block_item for block_item in block_items}
    
    # indexing
    @staticmethod
    def index_block(converted_block: dict, rearranged_block: list):
        for i, i_block in enumerate(rearranged_block):
            for j, block_name in enumerate(i_block):
                target_block = converted_block.get(block_name)
                target_block['i'] = i + 1
                target_block['j'] = j + 1
        return converted_block

    # output csv
    @staticmethod
    @log_decorator('블록 재배열 알고리즘 결과 저장')
    def save_output_csv(input_file_name, rearranged_block):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)
        _rearranged_block = rearranged_block.copy()
        _rearranged_block.reverse()
        with open(f'./output/csv/{input_file_name}/{now}_1-1_output.csv', 'w', newline='\n', encoding='ansi') as csvfile:
            cw = csv.writer(csvfile)
            for i_block in _rearranged_block:
                cw.writerow(i_block)
    
   
    # 재배열
    # return 재배열된 Block 목록, 인덱스가 추가된 Block 정보(Key: BlName), 시작방향
    @log_decorator('블록 재배열 알고리즘')
    def rearrangement(self, block_items: list, start_block_name: str, _work_direction: int):
        # 1-1 셀분류
        classified_block = self.classfication_block(block_items)
        # 1-2 가상셀 생성
        _rearranged_block = self.create_virtual_cell(classified_block)
        _converted_block = self.__class__.convert_block('BlName', block_items)
        _converted_block.update({self.__class__.VIRTUAL_BLOCK_NAME: self.__class__.VIRTUAL_BLOCK})
        _converted_block = self.__class__.index_block(_converted_block, _rearranged_block)

        # 1-3 시작 방향
        rearranged_block, converted_block, work_direction = self.start_direction(_rearranged_block, _converted_block, start_block_name, _work_direction)
        return rearranged_block, converted_block, work_direction

    # 1-1 셀 분류 
    def classfication_block(self, _block_items: list):
        classified_block = [[]]
        i, first_cell = 0, False
        block_items = sorted(_block_items, key=lambda x: int(x.get('No')))
        converted_block = self.__class__.convert_block('No', block_items)
        for idx, block in enumerate(block_items):
            
            # 첫번째, 마지막 행 입력
            if idx == 0 or idx == len(block_items) - 1:
                classified_block[i].append(block)
                if idx == len(block_items) - 1:
                    logging.getLogger('block').debug(f'No{idx + 1} = n{len(block_items)} 셀{block}을 마지막 행에 추가')
            # i 행의 첫번째 셀로 입력 (No.+1)
            elif first_cell:
                classified_block[i].append(block)
                first_cell = False
                logging.getLogger('block').debug(f'No+1({block.get("BlName")})을 다음행({i + 1})의 첫번째 셀로 입력')
            else:
                block_no = int(block.get('No'))
                # 두 벡터의 내적
                prev_block, cur_block, next_block = map(lambda x: converted_block.get(str(block_no + x)), [-1, 0, 1])
                prev_x_coord, prev_y_coord, _ = self.get_coord(prev_block)
                cur_x_coord, cur_y_coord, _ = self.get_coord(cur_block)
                next_x_coord, next_y_coord, _ = self.get_coord(next_block)

                vector_a = np.array([prev_x_coord-cur_x_coord, prev_y_coord-cur_y_coord])
                vector_b = np.array([cur_x_coord-next_x_coord, cur_y_coord-next_y_coord])
                ab_dot = np.dot(vector_a, vector_b)

                # 현재 셀의 이웃 셀 중 No + 1이 있는가? & 벡터간의 내적값 >= 0
                logging.getLogger('block').debug(f'현재 셀({block.get("BlName")})의 이웃 셀({block.get("Direction").split(",")}) 중 No + 1이 있는가? {str(block_no + 1) in block.get("Direction").split(",")}')
                logging.getLogger('block').debug(f'No-1번 셀({prev_block.get("BlName")})의 Center 좌표({[prev_x_coord, prev_y_coord]}) -> No번 셀{block.get("BlName")}의 Center좌표({[cur_x_coord, cur_y_coord]})로의 벡터와 No번 셀의 Center 좌표({[cur_x_coord, cur_y_coord]}) -> No+1번셀(({next_block.get("BlName")})의 Center 좌표({[next_x_coord, next_y_coord]})로의 내적값({ab_dot}) >=0? {ab_dot >= 0}')
                if str(block_no + 1) in block.get('Direction').split(',') and ab_dot >= 0:
                    classified_block[i].append(block)
                    logging.getLogger('block').debug(f'i({i + 1})행에 {block.get("BlName")} 추가')
                else:
                    # i행에 있는 셀 입력 완료, 다음 행으로 이동
                    classified_block[i].append(block)
                    logging.getLogger('block').debug(f'i({i +1})행에 {block.get("BlName")} 추가, i행에 있는 셀 입력 완료.')
                    i += 1
                    classified_block.append([])
                    first_cell = True

        return classified_block

    # 1-2 가상셀 생성
    def create_virtual_cell(self, classified_block: list):
        gamma, gamma_list, gamma_max = 0, [(0, 0)], 0

        for i, i_block in enumerate(classified_block):
            if i == len(classified_block) - 1:
                break

            cur_block = i_block[0]
            # i행 2열의 좌표
            prev_x_coord, prev_y_coord, _ = self.get_coord(i_block[1])
            # i행 1열의 좌표
            cur_x_coord, cur_y_coord, _ = self.get_coord(cur_block)
            # i + 1 행 1열의 좌표
            next_x_coord, next_y_coord, _ = self.get_coord(classified_block[i + 1][0])

            # i행 1열의 좌표 -> i행 2열의 좌표 벡터
            vector_a = np.array([cur_x_coord-prev_x_coord, cur_y_coord-prev_y_coord])
            # i행 1열의 좌표 -> i + 1행 1열의 좌표 벡터
            vector_b = np.array([cur_x_coord-next_x_coord, cur_y_coord-next_y_coord])
            # 두 벡터의 내적값
            ab_dot = np.dot(vector_a, vector_b)

            logging.getLogger('block').debug(f'{i + 1}행 1열의 셀({cur_block.get("BlName")})의 center 좌표({[cur_x_coord, cur_y_coord]}) -> {i + 1}행 2열의 셀({i_block[1].get("BlName")})의 Center 좌표({[prev_x_coord, prev_y_coord]})의 벡터')
            logging.getLogger('block').debug(f'{i + 1}행 1열의 셀({cur_block.get("BlName")})의 center 좌표({[cur_x_coord, cur_y_coord]}) -> {i + 2}행 1열의 셀의 Center 좌표({[next_x_coord, next_y_coord]})의 벡터')
            logging.getLogger('block').debug(f'벡터간 내적값({ab_dot}) < 0? {ab_dot < 0}')
            # 벡터의 내적값 < 0
            if ab_dot < 0:
                gamma = gamma + self.get_gamma_negative_by_overlap_count(i, i_block, classified_block[i + 1])
                gamma_list.append((i + 1, gamma))
            # 내적값 = 0
            elif ab_dot == 0:
                gamma_list.append((i + 1, gamma))
            # 내적값 != 0
            else:
                gamma = gamma - self.get_gamma_positive_by_overlap_count(i, i_block, classified_block[i + 1])
                gamma_list.append((i + 1, gamma))
            logging.getLogger('block').debug(f'i: {i + 1}, gamma: {gamma}, gamma_list: {gamma_list}')

        # 가상셀 생성
        rearranged_block = []
        gamma_max = max([g[1] for g in gamma_list])
        for i, g in gamma_list:
            rearranged_block.append([])
            for j in range(0, gamma_max - g):
                rearranged_block[i].append(self.__class__.VIRTUAL_BLOCK_NAME)
            for j, block in enumerate(classified_block[i]):
                cut_vol, fill_vol, area = map(lambda x: float(block.get(x)), ['cutVol', 'fillVol', 'Area'])
                # noise data 인 경우 가상셀로 추가
                if area < 0.3 or (cut_vol == 0 and fill_vol == 0):
                    logging.getLogger('block').debug(f'{block.get("BlName")} 가상셀로 변경, area({area}) < 0.3 or (졀토({cut_vol})==0 and 성토({fill_vol})==0)')
                    rearranged_block[i].append(self.__class__.VIRTUAL_BLOCK_NAME)
                else:
                    rearranged_block[i].append(block.get('BlName'))

        epsilon = max([len(i_block) for i_block in rearranged_block])
        for i, i_block in enumerate(rearranged_block):
            if len(i_block) < epsilon:
                for j in range(0, epsilon - len(i_block)):
                    i_block.append(self.__class__.VIRTUAL_BLOCK_NAME)
        
        delete_virtual_cell_i, delete_virtual_cell_j = [], []
        # 가상셀만 있는 행 확인
        for i, i_block in enumerate(rearranged_block):
            if all([block_name == self.__class__.VIRTUAL_BLOCK_NAME for block_name in i_block]):
                delete_virtual_cell_i.append(i)
        # 가상셀만 있는 열 확인
        for j in range(0, epsilon):
            if all([rearranged_block[i][j] == self.__class__.VIRTUAL_BLOCK_NAME for i in range(0, len(rearranged_block))]):
                delete_virtual_cell_j.append(j)

        # 가상셀만 있는 행 삭제 (역순)
        for v_i in sorted(delete_virtual_cell_i, reverse=True):
            del rearranged_block[v_i]
        # 가상셀만 있는 열 삭제 (역순)
        for v_j in sorted(delete_virtual_cell_j, reverse=True):
            for b_i in range(0, len(rearranged_block)):
                del rearranged_block[b_i][v_j]
        
        return rearranged_block

    # 1-3 시작 지점에 따른 시작 방향
    def start_direction(self, rearranged_block: list, converted_block: dict, start_block_name: str, work_direction: int):
        if converted_block.get(start_block_name).get('i') is None:
            logging.getLogger('block').debug(f'rearranged_block: {rearranged_block}')
            raise Exception(f'Rearranged block {start_block_name} not found')

        omega = converted_block.get(start_block_name).get('i')
        logging.getLogger('block').debug(f'전진 Start 지점의 행 = omega일 때, a - omega <= omega - 1? {len(rearranged_block) - omega <= omega - 1}')
        # a - omega <= omega - 1 일 경우 행렬 reverse
        if len(rearranged_block) - omega <= omega - 1:
            logging.getLogger('block').debug(f'행렬 Reverse 후 a*epsilon 배열 저장 및 a = -a로 저장')
            work_direction = -work_direction
            reversed_rearranged_block = self.__class__.reverse_block(rearranged_block)
            return reversed_rearranged_block, self.__class__.index_block(converted_block, reversed_rearranged_block), work_direction
        else:
            return rearranged_block, converted_block, work_direction

    def get_coord(self, block):
        return map(lambda x: float(block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
    
    # overlap count로 gamma 계산 (양수)
    def get_gamma_positive_by_overlap_count(self, i, i_block: list, i1_block: list):
        _gamma = 0
        for j in range(0, len(i_block)):
            count = self.overlap_count(i_block[j].get('etc'), i1_block[0].get('etc'))
            logging.getLogger('block').debug(f'{i + 1}행 {j + 1}열의 셀({i_block[j].get("BlName")})과 {i + 2}행 1열의 셀({i1_block[0].get("BlName")})의 Boundary 상 겹치는 x,y,z 개수 {count}이(가) 0인가? {count == 0}')
            if count != 0:
                for _j in range(j, len(i1_block)):
                    _count = self.overlap_count(i_block[_j].get('etc'), i1_block[0].get('etc'))
                    logging.getLogger('block').debug(f'{i + 1}행 {j + 1}열의 셀({i_block[_j].get("BlName")})과 {i + 2}행 1열의 셀({i1_block[0].get("BlName")})의 Boundary 상 겹치는 x,y,z 개수 {_count}이(가) 1보다 큰가? {_count > 1}')
                    if _count > 1:
                        _gamma = _j
                        break
                break
        if _gamma is None:
            raise Exception(f'Gamma 계산 실패(벡터내적값 > 0).. i행 Block: {[b.get("BlName") for b in i_block]}, i + 1행 Block: {[b.get("BlName") for b in i1_block]}')
        return _gamma

    # overlap count로 gamma 계산 (음수)
    def get_gamma_negative_by_overlap_count(self, i, i_block: list, i1_block: list):
        _gamma = 0
        for j in range(0, len(i1_block)):
            count = self.overlap_count(i_block[0].get('etc'), i1_block[j].get('etc'))
            logging.getLogger('block').debug(f'{i + 1}행 1열의 셀({i_block[0].get("BlName")})과 {i + 2}행 {j + 1}열의 셀({i1_block[j].get("BlName")})의 Boundary 상 겹치는 x,y,z 개수 {count}이(가) 0인가? {count == 0}')
            if count != 0:
                for _j in range(j, len(i1_block)):
                    _count = self.overlap_count(i_block[0].get('etc'), i1_block[_j].get('etc'))
                    logging.getLogger('block').debug(f'{i + 1}행 1열의 셀({i_block[0].get("BlName")})과 {i + 2}행 {_j + 1}열의 셀({i1_block[_j].get("BlName")})의 Boundary 상 겹치는 x,y,z 개수 {_count}이(가) 1보다 큰가? {_count > 1}')
                    if _count > 1:
                        _gamma = j
                        break
                break
        if _gamma is None:
            logging.getLogger('block').error(f'Gamma 계산 실패(벡터내적값 < 0)..i행 Block: {[b.get("BlName") for b in i_block]}, i + 1행 Block: {[b.get("BlName") for b in i1_block]}')
            raise Exception(f'Gamma 계산 실패(벡터내적값 < 0)..i행 Block: {[b.get("BlName") for b in i_block]}, i + 1행 Block: {[b.get("BlName") for b in i1_block]}')
        return _gamma

    # 겹치는 수
    def overlap_count(self, boundary_x: str, boundary_y: str):
        return len(set([x for x in boundary_x.split(';') if x.count(',') == 2]).intersection(set([y for y in boundary_y.split(';') if y.count(',') == 2])))
    
    # reverse
    @staticmethod
    def reverse_block(rearranged_block: list):
        import copy
        _reversed_block = copy.deepcopy(rearranged_block)
        for i_block in _reversed_block:
            i_block.reverse()
        _reversed_block.reverse()
        return _reversed_block
