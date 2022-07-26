# COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import math
import csv
import logging
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pulp import *
from os import makedirs
from datetime import datetime
from route_planner.block import Block
from route_planner.util import log_decorator, to_rgba
from route_planner.constants import COLOR_LIST, FONT, RED_FONT, DPI, OUTPUT_PATH


class RoutePlan(object):
    def __init__(self, block_items: list, converted_block: dict):
        self.route_plan = []
        self.ended_block = []
        self.ended_edges = {}
        self.converted_block = converted_block
        self.block_no_name_map = {str(v.get('No')): k for k, v in self.converted_block.items()}
        self.all_edges = self.generate_edges(block_items)

    @staticmethod
    def get_distance(block_a: dict, block_b: dict):
        a_x, a_y, a_z = map(lambda x: float(block_a.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
        b_x, b_y, b_z = map(lambda x: float(block_b.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
        return math.sqrt(math.pow(a_x - b_x, 2) + math.pow(a_y - b_y, 2) + math.pow(a_z - b_z, 2))

    def set_j(self, j):
        return f'j{j}'

    def get_j(self, key_j):
        return int(key_j[1:])

    def add_route(self, target_block: dict, arrow: bool = False, compaction_count: int = 1):
        route = {k: target_block.get(k) for k in ['BlName', 'Xcoord', 'Ycoord', 'Zcoord']}
        route.update({'arrow': arrow, 'compaction_count': compaction_count})
        self.route_plan.append(route)

    # 두 선분의 교차 여부
    # https://velog.io/@jini_eun/%EB%B0%B1%EC%A4%80-20149-%EC%84%A0%EB%B6%84-%EA%B5%90%EC%B0%A8-3-Java-Python
    # a: point x1, y1
    # b: point x2, y2 (x1,y1)-> (x2,y2)
    # c: point x3, y3
    # d: point x4, y4  (x3,y3)-> (x4,y4) 두 선분이 교차하는지
    def check(self, a, b, c, d):
        ccw = lambda p1, p2, p3 : (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
        if ccw(a, b, c) * ccw(a, b, d) == 0:
            if ccw(c, d, a) * ccw(c, d, b) == 0:
                if a > b:
                    a, b = b, a
                if c > d:
                    c, d = d, c
                if b >= c and a <= d:
                    return True
                else:
                    return False
        if ccw(a, b, c) * ccw(a, b, d) <= 0:
            if ccw(c, d, a) * ccw(c, d, b) <= 0:
                return True
        return False

    # 직선이 이동가능한가? 불가능할 경우 대상에서 제외 하고 각 좌표간 최단거리를 구한다
    def can_move(self, directions, target_line):
        # 이동가능한 블럭의 전체 경계좌표 리스트를 넣는다.
        # 경계좌표 직선의 정보를 x1 < y1 < x2 < y2 순의 pair로 설정 후 중복 체크, 중복되지 않는 1의 선은 경계선으로 확인
        from collections import Counter

        target_line_x1, target_line_y1, target_line_x2, target_line_y2 = target_line[0][0], target_line[0][1], target_line[1][0], target_line[1][1]
        # target line이 경계좌표를 지나가는지(교점이 있는지) 확인한다, 교점이 있을 경우 이동 불가능한 지역으로 해당 line은 이동이 불가하다
        for (b_line_x1, b_line_y1), (b_line_x2, b_line_y2) in directions:
            if self.check((target_line_x1, target_line_y1), (target_line_x2, target_line_y2), (b_line_x1, b_line_y1), (b_line_x2, b_line_y2)):
                return False
        return True
    
    #  v 5
    #  v 4
    #  v 3
    #  1 2 
    # 3의 left, 4의 left, 5의 left, 1의 up line을 구하고 1 -> 5 직선이 해당 line을 지나가는지, 2->5는 지나가는지 체크 용도
    # 방향에 해당하는 Line
    def get_direction_line(self, block_boundary: dict, direction: str):
        if direction == 'l' or direction == 'r':
            return (block_boundary.get(f'{direction}d')[0], block_boundary.get(f'{direction}d')[1], block_boundary.get(f'{direction}u')[0], block_boundary.get(f'{direction}u')[1])
        elif direction == 'u':
            return (block_boundary.get(f'l{direction}')[0], block_boundary.get(f'l{direction}')[1], block_boundary.get(f'r{direction}')[0], block_boundary.get(f'r{direction}')[1])
        else:
            return None

    # Block의 경계선 
    def add_block_boundary(self, block: dict):
        if any([d in block for d in ['ld', 'rd', 'lu', 'ru']]):
            return {k: block.get(k) for k in ['ld', 'rd', 'lu', 'ru']}
        block_boundary = {}
        block_boundary_coord = list(set([etc for etc in block.get('etc').split(';') if etc.count(',') == 2]))
        _sorted_coord = [[float(c) for c in _coord.split(',')] for _coord in block_boundary_coord]
        a_x, a_y, a_z = map(lambda x: float(block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
        for coord in _sorted_coord:
            if coord[0] <= a_x and coord[1] <= a_y:
                block_boundary.setdefault('ld', [])
                block_boundary['ld'].append(coord)
            elif coord[0] > a_x and coord[1] <= a_y:
                block_boundary.setdefault('rd', [])
                block_boundary['rd'].append(coord)
            elif coord[0] > a_x and coord[1] > a_y:
                block_boundary.setdefault('ru', [])
                block_boundary['ru'].append(coord)
            else:
                block_boundary.setdefault('lu', [])
                block_boundary['lu'].append(coord)
        for k in list(block_boundary.keys()):
            v = block_boundary[k]
            if len(v) > 1:
                block_boundary[k] = sorted(v, key=lambda x: -(math.sqrt(math.pow(a_x - float(x[0]), 2) + math.pow(a_y - float(x[1]), 2) + math.pow(a_z - float(x[2]), 2))))[0]
            else:
                block_boundary[k] = v[0]
        block.update(block_boundary)
        return block_boundary
    
    #다각형의 내부 외부 판별
    # https://bowbowbow.tistory.com/24
    # 두선분간 https://crazyj.tistory.com/140
    # 
    # 모든 좌표간 이동, 작업 후 이동가능한 좌표 list, 
    # edges = [
    #     ("A", "B", 7),
    #     ("A", "D", 5),
    #     ("B", "C", 8),
    #     ("B", "D", 9),
    #     ("B", "E", 7),
    #     ("C", "E", 5),
    #     ("D", "E", 15),
    #     ("D", "F", 6),
    #     ("E", "F", 8),
    #     ("E", "G", 9),
    #     ("F", "G", 11)
    # ]
    # tuple (14, ('E', ('B', ('A', ()))))
    def generate_edges(self, block_items: list, _edges: dict = {}):
        edges = {} if not _edges else _edges
        for block in block_items:
            if block.get('YN') == 'Y':
                for d in block.get('Direction').split(','):
                    neighbor_block = self.converted_block.get(self.block_no_name_map.get(d))
                    if neighbor_block and neighbor_block.get('YN') == 'Y':
                        block_name, neighbor_block_name = map(lambda x: x.get('BlName'), [block, neighbor_block])
                        edge_name = '-'.join([block_name, neighbor_block_name])
                        reverse_edge_name = '-'.join([neighbor_block_name, block_name])
                        if edge_name in edges:
                            continue
                        elif reverse_edge_name in edges:
                            edges[edge_name] = (edges[reverse_edge_name][1], edges[reverse_edge_name][0], edges[reverse_edge_name][2])
                        else:
                            edges[edge_name] = (block_name, neighbor_block_name, self.__class__.get_distance(block, neighbor_block))
        return edges

    # edges = [
    #     ("A", "B", 7),
    #     ("A", "D", 5),
    #     ("B", "C", 8),
    #     ("B", "D", 9),
    #     ("B", "E", 7),
    #     ("C", "E", 5),
    #     ("D", "E", 15),
    #     ("D", "F", 6),
    #     ("E", "F", 8),
    #     ("E", "G", 9),
    #     ("F", "G", 11)
    # ]
    # f: 시작 지점
    # t: 종료 지점
    @staticmethod
    def dijkstra(edges: list, f: str, t: str):
        from collections import defaultdict
        from heapq import heappop, heappush
        g = defaultdict(list)
        for l,r,c in edges:
            g[l].append((c,r))

        q, seen, mins = [(0,f,())], set(), {f: 0}
        while q:
            (cost,v1,path) = heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == t: return (cost, path)

                for c, v2 in g.get(v1, ()):
                    if v2 in seen: continue
                    prev = mins.get(v2, None)
                    next = cost + c
                    if prev is None or next < prev:
                        mins[v2] = next
                        heappush(q, (next, v2, path))

        return float("inf"), None

    # path: ('BL_1', ('BL_11', ('BL_21', ('BL_20', ()))))
    def get_routes(self, path: tuple, route: list):
        route.insert(0, path[0])
        if path[1]:
            route = self.get_routes(path[1], route)
        return route

    # START -> END 최단경로를 계획경로에 저장
    def add_route_plan(self, e_edges, start_block_name, end_block_name, compaction_count: int = 1):
        distance, path = self.__class__.dijkstra(e_edges, start_block_name, end_block_name)
        # 이동 가능한 경로가 없을 경우 작업 전 이동 가능한 블럭 중 최단 경로 계산
        if distance == float("inf"):
            distance, path = self.__class__.dijkstra(list(self.all_edges.values()), start_block_name, end_block_name)
        
        # 경로가 존재하지 않을 경우 Error
        if distance == float("inf"):
            logging.getLogger('plan').error(f'start_block_name: {start_block_name}, end_block_name: {end_block_name} not exist')
            raise Exception('not exist path')
        else:
            # Dkj[0]의 Center 좌표에서 출발해 Amn의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장
            routes = self.get_routes(path, [])
            for i, route in enumerate(routes):
                # 현재 위치
                if i == 0: continue
                # 마지막 i는 화살표 추가
                self.add_route(self.converted_block.get(route), i == len(routes) - 1, compaction_count)

    # 위, 아래, 왼, 오 방향의 선을 구함
    def get_direction_block(self, sorted_boundary: list, a_x, a_y, a_z):
        direction_block = {}
        for coord in sorted_boundary:
            if coord[0] <= a_x and coord[1] <= a_y:
                direction_block.setdefault('ld', [])
                direction_block['ld'].append(coord)
            elif coord[0] > a_x and coord[1] <= a_y:
                direction_block.setdefault('rd', [])
                direction_block['rd'].append(coord)
            elif coord[0] > a_x and coord[1] > a_y:
                direction_block.setdefault('ru', [])
                direction_block['ru'].append(coord)
            else:
                direction_block.setdefault('lu', [])
                direction_block['lu'].append(coord)
        for k in list(direction_block.keys()):
            v = direction_block[k]
            if len(v) > 1:
                direction_block[k] = sorted(v, key=lambda x: -(math.sqrt(math.pow(a_x - float(x[0]), 2) + math.pow(a_y - float(x[1]), 2) + math.pow(a_z - float(x[2]), 2))))[0]
            else:
                direction_block[k] = v[0]
        return direction_block

    # 경계선
    def get_boundary_lines(self, block: dict):
        block_boundary = list(set([etc for etc in block.get('etc').split(';') if etc.count(',') == 2]))
        _sorted_boundary = [[float(c) for c in _coord.split(',')] for _coord in block_boundary]
        a_x, a_y, a_z = map(lambda x: float(block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
        if len(block_boundary) < 4:
            sorted_boundary = _sorted_boundary
        else:
            direction_block = self.get_direction_block(_sorted_boundary, a_x, a_y, a_z)
            sorted_boundary = [direction_block.get(k) for k in ['ld', 'rd', 'ru', 'lu'] if direction_block.get(k) is not None]

        boundary_lines = []
        for i, coord in enumerate(sorted_boundary):
            n_coord = sorted_boundary[0] if i == len(sorted_boundary) - 1 else sorted_boundary[i + 1]
            if coord[0] < n_coord[0] or (coord[0] == n_coord[0] and coord[1] < n_coord[1]):
                boundary_lines.append(((coord[0], coord[1]), (n_coord[0], n_coord[1])))
            else:
                boundary_lines.append(((n_coord[0], n_coord[1]), (coord[0], coord[1])))
        return boundary_lines

    # PNG
    @staticmethod
    @log_decorator('계획 경로 알고리즘 PNG 저장')
    def save_output_png(input_file_name: str, route_plan, rearranged_block, converted_block, allocated_cells, equip_type, compaction_count: int = 1):
        width_ratio, height_ratio = 5 / 4, 5 / 7
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'{OUTPUT_PATH}/png/{input_file_name}', exist_ok=True)
        width, width_length, height, height_length = 15, len(rearranged_block[0]), 10.5, len(rearranged_block)
        sample_colors, color_map, _converted_allocated_cells, k = COLOR_LIST.copy(), {}, {}, 1
        for k, j_block in allocated_cells.items():
            for key_j, values_j in j_block.items():
                for j in values_j:
                    _converted_allocated_cells[j] = {'k': k[1:], 'j': key_j[1:]}

        # x1, y1, +x(증감), +y (증감)
        plt.figure(figsize=(width_length * width_ratio, height_length * height_ratio))
        
        def block_text_coord(x_index, y_index, block_name):
            return x_index * width + width / 2 - (len(block_name) / 2 * 1.2), height * y_index + height / 2 - 1

        plt.axis('off')
        # [x1, x2], [y1, y2]
        for i in range(0, height_length + 1):
            plt.plot([0, width_length * width], [i * height, i * height], color="black")

        for j in range(0, width_length + 1):
            plt.plot([j * width, j * width], [0, height_length * height], color="black")
        
        for i, i_block in enumerate(rearranged_block):
            for j, block_name in enumerate(i_block):
                fontdict = FONT
                if converted_block.get(block_name).get('YN') == 'N':
                    fontdict = RED_FONT
                plt.text(j * width + width / 2 - (len(block_name) / 2 * 1.2), height * i + height / 2 - 1, block_name, fontdict=fontdict)
        k = 1
        for index, route in enumerate(route_plan):
            # 마지막 경로는 그리지 않음
            if index == len(route_plan) - 1:
                continue
            next_route = route_plan[index + 1]
            block_name, next_block_name = map(lambda x: x.get('BlName'), [route, next_route])
            i, j = map(converted_block.get(block_name).get, ['i', 'j'])
            next_i, next_j = map(converted_block.get(next_block_name).get, ['i', 'j'])
            cur_arrow, next_arrow = map(lambda x: x.get('arrow'), [route, next_route])
            if converted_block.get(block_name).get('line') is None:
                converted_block.get(block_name)['line'] = -4
            
            if converted_block.get(next_block_name).get('line') is None:
                converted_block.get(next_block_name)['line'] = -4
            
            if _converted_allocated_cells.get(block_name) and int(_converted_allocated_cells.get(block_name).get('k')) > k:
                k = int(_converted_allocated_cells.get(block_name).get('k'))
            
            if str(k) not in color_map:
                color_map[str(k)] = sample_colors.pop() if sample_colors else ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            x1 = (j - 1) * width + width / 2 + converted_block.get(block_name)['line'] * width / 9
            x2 = (next_j - 1) * width + width / 2 + converted_block.get(next_block_name)['line'] * width / 9 - x1
            y1 = (i - 1) * height + height / 2
            y2 = (next_i - 1) * height + height / 2 - y1
            converted_block.get(block_name)['line'] = converted_block.get(block_name)['line'] + 1
            converted_block.get(next_block_name)['line'] = converted_block.get(next_block_name)['line'] + (1 if next_arrow else 0)
            plt.arrow(x1, y1, x2, y2, width=0.3, head_width=1.2 if next_arrow else 0, head_length=1 if next_arrow else 0, fc=to_rgba(color_map[str(k)]), ec=to_rgba(color_map[str(k)]))
        
        
        plt.savefig(f'{OUTPUT_PATH}/png/{input_file_name}/{equip_type}_{now}.png', dpi=DPI)
        plt.clf()


# 작업 구역간 계획 경로
class MovementPlan(object):
    BORROW_PIT_NAME = 'BORROW_PIT'
    DUMPING_AREA_NAME = 'DUMPING_AREA'

    def __init__(self, block_items: list):
        self.block_items = block_items
        self.converted_block = Block.convert_block('BlName', block_items)

    def generate_edges(self, block_items: list, _edges: dict = {}):
        edges = {} if not _edges else _edges
        for c_block in block_items:
            if c_block.get('YN') == 'Y':
                for n_block in block_items:
                    c_block_name, n_block_name = map(lambda x: x.get('BlName'), [c_block, n_block])
                    if n_block.get('YN') == 'Y' and c_block_name != n_block_name:
                        edge_name = '-'.join([c_block_name, n_block_name])
                        reverse_edge_name = '-'.join([n_block_name, c_block_name])
                        if edge_name in edges:
                            continue
                        elif reverse_edge_name in edges:
                            edges[edge_name] = (edges[reverse_edge_name][1], edges[reverse_edge_name][0], edges[reverse_edge_name][2])
                        else:
                            edges[edge_name] = (c_block_name, n_block_name, RoutePlan.get_distance(c_block, n_block))
        return edges

    @staticmethod
    @log_decorator('작업 구역간 이동 경로 알고리즘 결과 저장')
    def save_output_csv(input_file_name: str, movement_plan: list):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)
        
        with open(f'./output/csv/{input_file_name}/{now}_movement_output.csv', 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Direction', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
            writer.writeheader()
            for m in movement_plan:
                p = {'Direction': f'{m.get("start_block_name")}→{m.get("end_block_name")}'}
                p.update({coord: m.get(coord) for coord in ['x1', 'y1', 'z1', 'x2', 'y2', 'z2']})
                writer.writerow(p)

    # borrow_pit: 토취장, dumping_Area: 사토장
    @log_decorator('작업 구역간 이동 경로 알고리즘')
    def calc_block(self, borrow_pit_x, borrow_pit_y, borrow_pit_z, dumping_area_x, dumping_area_y, dumping_area_z, borrow_pit_cut_vol, dumping_area_fill_vol):
        borrow_pit = {'BlName': self.__class__.BORROW_PIT_NAME, 'Xcoord': borrow_pit_x, 'Ycoord': borrow_pit_y, 'Zcoord': borrow_pit_z, 'cutVol': borrow_pit_cut_vol, 'fillVol': 0.0, 'YN': 'Y'}
        dumping_area = {'BlName': self.__class__.DUMPING_AREA_NAME, 'Xcoord': dumping_area_x, 'Ycoord': dumping_area_y, 'Zcoord': dumping_area_z, 'cutVol': 0.0, 'fillVol': dumping_area_fill_vol, 'YN': 'Y'}
        self.converted_block[self.__class__.BORROW_PIT_NAME] = borrow_pit
        self.converted_block[self.__class__.DUMPING_AREA_NAME] = dumping_area
        all_block = self.block_items.copy()
        all_block.extend([borrow_pit, dumping_area])
        all_edges = self.generate_edges(all_block).values()

        # cut_areas: 잉여구역, fill_areas: 필요구역
        cut_areas, fill_areas, sum_cut, sum_fill = [], [], 0, 0
        for block in self.block_items:
            block_name, cut_vol, fill_vol = map(block.get, ['BlName', 'cutVol', 'fillVol'])
            sum_cut += cut_vol
            sum_fill += fill_vol
            if cut_vol > fill_vol:
                cut_areas.append(block_name)
            if cut_vol < fill_vol:
                fill_areas.append(block_name)

        # 분기에 따른 토취장, 사토장 매핑을 구하고, 각 구역간 거리를 구한다
        # 각 구역간 제약조건도 구한다
        if sum_cut > sum_fill:
            # 사토장으로 매핑
            fill_areas.append(self.__class__.DUMPING_AREA_NAME)
            fill_flag = True
        else:
            # 토취장으로 매핑
            cut_areas.append(self.__class__.BORROW_PIT_NAME)
            fill_flag = False

        main_areas, target_areas, main_key, target_key = (fill_areas, cut_areas, 'fillVol', 'cutVol') if fill_flag else (cut_areas, fill_areas, 'cutVol', 'fillVol')
        _distance_metrix, _main_condition, _target_condition = [], [], []

        for m_area in main_areas:
            m_list = []
            for t_area in target_areas:
                m_list.append({'start': m_area, 'end': t_area, 'distance': RoutePlan.dijkstra(all_edges, m_area, t_area)[0]})
                # 제약 조건
                _target_condition.append(self.converted_block.get(t_area).get(target_key) - self.converted_block.get(t_area).get(main_key))
            _distance_metrix.append(m_list)
            # 제약 조건
            _main_condition.append(self.converted_block.get(m_area).get(main_key) - self.converted_block.get(m_area).get(target_key))

        # https://ichi.pro/ko/python-eul-sayonghan-seonhyeong-peulogeulaeming-38164786352096
        distance_metrix = np.array([[m.get('distance') for m in t_m] for t_m in _distance_metrix])
        main_condition = np.array(_main_condition)
        target_condition = np.array(_target_condition)
        transport_model = LpProblem("transport-model", LpMinimize)
        DV_variables = LpVariable.matrix("", list(itertools.chain.from_iterable([[f'{m.get("start")}to{m.get("end")}' for m in t_m] for t_m in _distance_metrix])), lowBound= 0 )
        
        logging.getLogger('plan').debug(f'이동 거리 Metrix: {distance_metrix}')
        logging.getLogger('plan').debug(f'조건: {str(main_condition)}, {str(target_condition)}')
        allocation = np.array(DV_variables).reshape(len(main_areas), len(target_areas))

        obj_func = lpSum(allocation*distance_metrix)

        transport_model +=  obj_func
        for i in range(len(main_areas)):
            transport_model += lpSum(allocation[i][j] for j in range(len(target_areas))) <= main_condition[i] , "cw" + str(i)
        
        for j in range(len(target_areas)):
            transport_model += lpSum(allocation[i][j] for i in range(len(main_areas))) >= target_condition[j] , "ch" + str(j)

        transport_model.solve(PULP_CBC_CMD(msg=False))

        movement_plan = []
        for v in transport_model.variables():
            logging.getLogger('plan').debug(f'{v.name}: {v.value()}')
            start_block_name, end_block_name = v.name[1:].split('to')[0], v.name[1:].split('to')[1]
            if v.value() != 0:
                x1, y1, z1 = map(lambda x: float(self.converted_block.get(start_block_name).get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                x2, y2, z2 = map(lambda x: float(self.converted_block.get(end_block_name).get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                movement_plan.append({'start_block_name': start_block_name, 'end_block_name': end_block_name, 'x1': x1, 'y1': y1, 'z1': z1, 'x2': x2, 'y2': y2, 'z2': z2})
        return movement_plan

class Grader(RoutePlan):
    def __init__(self, block_items: list, converted_block: dict, rearranged_block: list):
        super().__init__(block_items, converted_block)
        self.rearranged_block = rearranged_block
        # converted_rearrange_block 구하기 (가상셀의 근처셀이 이동가능 블럭일 경우 해당 블럭과 인접한 블럭의 라인의 경계선을 구한다, 이동불가 셀 은 etc의 경계선만 구하기)
        self.n_block_boundary = self.convert_n_block_boundary()

    @staticmethod
    @log_decorator('그레이더 계획 경로 알고리즘 결과 저장')
    def save_output_csv(input_file_name: str, _route_plan: list):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)
        route_plan = [{k: v for k, v in route.items() if k in ['Timeline', 'BlName', 'Xcoord', 'Ycoord', 'Zcoord']} for route in _route_plan]
        fieldnames = list(route_plan[0].keys())
        
        with open(f'./output/csv/{input_file_name}/{now}_grader_1-3_output.csv', 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for route in route_plan:
                writer.writerow(route)

    def convert_n_block_boundary(self):
        n_block_boundary = {}
        for i, i_block in enumerate(self.rearranged_block):
            for j, block_name in enumerate(i_block):
                target_block = self.converted_block.get(block_name)
                target_block_name = target_block.get('BlName')
                # 이동 불가일 경우
                if target_block.get('YN') == 'N':
                    # 가상셀 인 경우
                    if target_block.get('virtual'):
                        target_block_name = f'{target_block_name}-{i}-{j}'
                        # 위 아래 왼 오 Block인지 체크, Block의 경계라인 가져오기
                        virtual_block_lines = []
                        # up 'ld', 'rd', 'ru', 'lu'
                        if i < len(self.rearranged_block) - 1:
                            up_block_name = self.rearranged_block[i + 1][j]
                            up_block = self.converted_block.get(up_block_name)
                            if up_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(up_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                up_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in up_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in up_block_direction_block for x in ['ld', 'rd']]):
                                    up_ld = up_block_direction_block.get('ld') if 'ld' in up_block_direction_block else up_block_direction_block.get('lu')
                                    up_rd = up_block_direction_block.get('rd') if 'rd' in up_block_direction_block else up_block_direction_block.get('ru')
                                    virtual_block_lines.append(((up_ld[0], up_ld[1]), (up_rd[0], up_rd[1])))
                        # down
                        if i > 0:
                            down_block_name = self.rearranged_block[i - 1][j]
                            down_block = self.converted_block.get(down_block_name)
                            if down_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(down_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                down_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in down_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in down_block_direction_block for x in ['lu', 'ru']]):
                                    down_lu = down_block_direction_block.get('lu') if 'lu' in down_block_direction_block else down_block_direction_block.get('ld')
                                    down_ru = down_block_direction_block.get('ru') if 'ru' in down_block_direction_block else down_block_direction_block.get('rd')
                                    virtual_block_lines.append(((down_lu[0], down_lu[1]), (down_ru[0], down_ru[1])))
                        # right
                        if j < len(self.rearranged_block[i]) - 1:
                            right_block_name = self.rearranged_block[i][j + 1]
                            right_block = self.converted_block.get(right_block_name)
                            if right_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(right_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                right_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in right_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in right_block_direction_block for x in ['ru', 'rd']]):
                                    right_ru = right_block_direction_block.get('ru') if 'ru' in right_block_direction_block else right_block_direction_block.get('lu')
                                    right_rd = right_block_direction_block.get('rd') if 'rd' in right_block_direction_block else right_block_direction_block.get('ld')
                                    virtual_block_lines.append(((right_ru[0], right_ru[1]), (right_rd[0], right_rd[1])))
                        # left
                        if j > 0:
                            left_block_name = self.rearranged_block[i][j - 1]
                            left_block = self.converted_block.get(left_block_name)
                            if left_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(left_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                left_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in left_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in left_block_direction_block for x in ['lu', 'ld']]):
                                    left_lu = left_block_direction_block.get('lu') if 'lu' in left_block_direction_block else left_block_direction_block.get('ru')
                                    left_ld = left_block_direction_block.get('ld') if 'ld' in left_block_direction_block else left_block_direction_block.get('rd')
                                    virtual_block_lines.append(((left_lu[0], left_lu[1]), (left_ld[0], left_ld[1])))
                        if virtual_block_lines:
                            n_block_boundary[target_block_name] = virtual_block_lines
                    else:
                        n_block_boundary[target_block_name] = self.get_boundary_lines(target_block)
        return n_block_boundary

    def grader_route_plan(self, cur_block, next_block):
        cur_block_i, cur_block_j = map(cur_block.get, ['i', 'j'])
        next_block_i, next_block_j = map(next_block.get, ['i', 'j'])
        min_i, min_j, max_i, max_j = min(cur_block_i, next_block_i), min(cur_block_j, next_block_j), max(cur_block_i, next_block_i), max(cur_block_j, next_block_j)

        move_block_names = []
        dont_move_lines = []
        for c_i in range(min_i, max_i + 1):
            for c_j in range(min_j, max_j + 1):
                move_block_name = self.rearranged_block[c_i - 1][c_j - 1]
                if move_block_name == Block.VIRTUAL_BLOCK_NAME:
                    _move_block_name = f'{move_block_name}-{c_i - 1}-{c_j - 1}'
                else:
                    _move_block_name = move_block_name
                move_block_names.append(move_block_name)
                dont_move_lines.extend(self.n_block_boundary.get(_move_block_name, []))

        # 움직일 수 없는 경계선이 없을 경우 바로 감
        if not dont_move_lines:
            # 바로 이동 가능하므로 목적지 좌표 추가, 화살표 추가, 다짐횟수 추가
            self.add_route(next_block, True)
        else:
            c_x, c_y, _ = map(lambda x: float(cur_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
            n_x, n_y, _ = map(lambda x: float(next_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
            # 바로 갈 수 있음
            if self.can_move(dont_move_lines, ((c_x, c_y), (n_x, n_y))):
                # 바로 이동 가능하므로 목적지 좌표 추가, 화살표 추가, 다짐횟수 추가
                self.add_route(next_block, True)
            else:
                # 현재 블럭을 제외한 나머지 블럭에서 목적지를 바로 갈 수 있는가?, 나머지블럭과 현재블럭이 이동 가능한가? 해당되는 경우의 목록을 구하고 대상 목록의 다익스트라, 대상목록이 없을 경우 전체경로에서의 다익스트라
                sub_edges = []
                for m_v in move_block_names:
                    # 이동 불가일 경우 skip
                    m_block = self.converted_block.get(m_v)
                    m_x, m_y, _ = map(lambda x: float(cur_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                    if m_block.get('YN') == 'N':
                        continue
                    # 현재 블럭을 제외한 나머지 블럭에서 목적지를 바로 갈 수 있는가?, 나머지블럭과 현재블럭이 이동 가능한가?
                    if self.can_move(dont_move_lines, ((c_x, c_y), (m_x, m_y))) and self.can_move(dont_move_lines, ((m_x, m_y), (n_x, n_y))):
                        # 현재 블럭 <-> 대상 블럭, 대상 블럭 <-> 목적지 간 거리 정보 추가
                        sub_edges.append((m_block.get('BlName'), next_block.get('BlName'), self.__class__.get_distance(m_block, next_block)))
                        sub_edges.append((cur_block.get('BlName'), m_block.get('BlName'), self.__class__.get_distance(cur_block, m_block)))
                # 대상 목록의 다익스트라, 대상목록이 없을 경우 전체경로에서의 다익스트라
                self.add_route_plan(sub_edges, cur_block.get('BlName'), next_block.get('BlName'))


    @log_decorator('그레이더 계획 경로 알고리즘')
    def calc_route_plan(self, allocated_cells: dict, start_block_name: str, work_direction: int):
        _work_direction = work_direction
        epsilon = max([len(i_block) for i_block in self.rearranged_block])
        # 할당셀 loop
        for index, (k, v) in enumerate(allocated_cells.items()):
            # 작업 방향에 따른 시작 j, loop
            v_k_l, v_v_l = list(v.keys()), list(v.values())
            compaction_flag = False
            len_v = len(v) - 1 if _work_direction > 0 else 0
            temp_block, temp_flag = '', False
            for cur_i in range(0, len(v), _work_direction) if _work_direction > 0 else range(len(v) - 1, -1, _work_direction):
                j = self.get_j(v_k_l[cur_i])
                block_dkj = v_v_l[cur_i]
                if not block_dkj: continue
                # 목표 다짐 회수로 인한 반복은 첫번째 index pass
                if compaction_flag:
                    compaction_flag = False
                    continue
                # Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장
                logging.getLogger('plan').debug(f'Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장 {[target_block for i, target_block in enumerate(block_dkj) if len(self.route_plan) == 0 or i != 0]}')
                for i, target_block in enumerate(block_dkj):
                    # 최초 시작이 아닌 경우 첫번째 셀은 경로에서 제외(이미 포함되어 있으므로)
                    if len(self.route_plan) > 0 and i == 0:
                        continue
                    # 마지막 i는 화살표 추가
                    self.add_route(self.converted_block.get(target_block), i == len(block_dkj) - 1)
                
                if not temp_flag:
                    logging.getLogger('plan').debug(f'현재 할당셀Dkj이 포함된 할당셀 집합 내에 경로가 작성되지 않은 할당셀이 있는가? {len_v != cur_i}')

                # 같은 할당셀 집합 내 Dkj+a[0]
                if v.get(self.set_j(j + _work_direction)):
                    logging.getLogger('plan').debug(f'Dkj+a가 있는가? True')
                    # 현재 할당셀Dkj[-1]의 Center 좌표에서 출발해 Dk(j+a)[0]의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장
                    cur_block = self.converted_block.get(block_dkj[-1]) if not temp_block else self.converted_block.get(temp_block)
                    next_block = self.converted_block.get(v.get(self.set_j(j + _work_direction))[0])
                    logging.getLogger('plan').debug(f'현재 할당셀Dkj[-1]({cur_block.get("BlName")}의 Center 좌표에서 출발해 Dk(j+a)[0]({next_block.get("BlName")})의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장')
                    self.grader_route_plan(cur_block, next_block)
                    if temp_block:
                        temp_block = ''
                        temp_flag = False
                elif len_v != cur_i:
                    logging.getLogger('plan').debug(f'Dkj+a가 있는가? False')
                    # 같은 할당셀 집합 내 Dkj+a[0]에 없을 경우 pass
                    temp_block = block_dkj[-1]
                    temp_flag = True
                else:
                    # 현재 할당셀의 delta >= 목표 다짐 횟수
                    # k == c -> 다음 할당셀이 없으므로 종료
                    if index == len(allocated_cells) - 1:
                        logging.getLogger('plan').debug('k == c -> 다음 할당셀이 없으므로 종료')
                        break
                    # alpha > 0
                    logging.getLogger('plan').debug(f'alpha > 0 {_work_direction > 0}')
                    if _work_direction > 0:
                        # 현재 할당셀 집합[-1]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합 [-1]의 [0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장
                        logging.getLogger('plan').debug(f'현재 할당셀 집합[-1]의 [-1]({self.converted_block.get(v_v_l[-1][-1]).get("BlName")}) Center 좌표에서 출발해서 다음 할당셀 집합 [-1]의 [0] ({self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[-1][0])})Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                        self.grader_route_plan(self.converted_block.get(v_v_l[-1][-1]), self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[-1][0]))
                    else:
                        # 현재 할당셀 집합[0]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합 [0]의 [0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장
                        logging.getLogger('plan').debug(f'현재 할당셀 집합[0]의 [-1]({self.converted_block.get(v_v_l[0][-1])}) Center 좌표에서 출발해서 다음 할당셀 집합 [0]의 [0]({self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[0][0])}) Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                        self.grader_route_plan(self.converted_block.get(v_v_l[0][-1]), self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[0][0]))
                    _work_direction = -_work_direction
                    logging.getLogger('plan').debug(f'alpha({_work_direction}) = -alpha({-_work_direction})')

        result = []
        for i, route in enumerate(self.route_plan):
            t_route = {'Timeline': str(i + 1)}
            t_route.update(route)
            result.append(t_route)

        return result
    
    # PNG
    @staticmethod
    @log_decorator('그레이더 계획 경로 알고리즘 PNG 저장')
    def save_output_png(input_file_name: str, _route_plan, rearranged_block, converted_block, allocated_cells, equip_type):
        width_ratio, height_ratio = 5 / 4, 5 / 7
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'{OUTPUT_PATH}/png/{input_file_name}', exist_ok=True)
        width, width_length, height, height_length = 15, len(rearranged_block[0]), 10.5, len(rearranged_block)
        sample_colors, color_map, _converted_allocated_cells, k = COLOR_LIST.copy(), {}, {}, 1
        for k, j_block in allocated_cells.items():
            for key_j, values_j in j_block.items():
                for j in values_j:
                    _converted_allocated_cells[j] = {'k': k[1:], 'j': key_j[1:]}

        route_plan = [r_p for r_p in _route_plan]
        # x1, y1, +x(증감), +y (증감)
        plt.figure(figsize=(width_length * width_ratio, height_length * height_ratio))
        
        def block_text_coord(x_index, y_index, block_name):
            return x_index * width + width / 2 - (len(block_name) / 2 * 1.2), height * y_index + height / 2 - 1

        plt.axis('off')
        # [x1, x2], [y1, y2]
        for i in range(0, height_length + 1):
            plt.plot([0, width_length * width], [i * height, i * height], color="black")

        for j in range(0, width_length + 1):
            plt.plot([j * width, j * width], [0, height_length * height], color="black")
        
        for i, i_block in enumerate(rearranged_block):
            for j, block_name in enumerate(i_block):
                fontdict = FONT
                if converted_block.get(block_name).get('YN') == 'N':
                    fontdict = RED_FONT
                plt.text(j * width + width / 2 - (len(block_name) / 2 * 1.2), height * i + height / 2 - 1, block_name, fontdict=fontdict)
        k = 1
        for index, route in enumerate(route_plan):
            # 마지막 경로는 그리지 않음
            if index == len(route_plan) - 1:
                continue
            next_route = route_plan[index + 1]
            block_name, next_block_name = map(lambda x: x.get('BlName'), [route, next_route])
            i, j = map(converted_block.get(block_name).get, ['i', 'j'])
            next_i, next_j = map(converted_block.get(next_block_name).get, ['i', 'j'])
            cur_arrow, next_arrow = map(lambda x: x.get('arrow'), [route, next_route])
            # k가 다를 경우 그리지 않음 (다짐횟수가 마지막이 아닐경우만)
            if int(_converted_allocated_cells.get(block_name).get('k')) != int(_converted_allocated_cells.get(next_block_name).get('k')):
                continue

            if converted_block.get(block_name).get('line') is None:
                converted_block.get(block_name)['line'] = -4
            
            if converted_block.get(next_block_name).get('line') is None:
                converted_block.get(next_block_name)['line'] = -4
            
            if _converted_allocated_cells.get(block_name) and int(_converted_allocated_cells.get(block_name).get('k')) > k:
                k = int(_converted_allocated_cells.get(block_name).get('k'))
            
            if str(k) not in color_map:
                color_map[str(k)] = sample_colors.pop() if sample_colors else ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            x1 = (j - 1) * width + width / 2 + converted_block.get(block_name)['line'] * width / 9
            x2 = (next_j - 1) * width + width / 2 + converted_block.get(next_block_name)['line'] * width / 9 - x1
            y1 = (i - 1) * height + height / 2
            y2 = (next_i - 1) * height + height / 2 - y1
            converted_block.get(block_name)['line'] = converted_block.get(block_name)['line'] + 1
            converted_block.get(next_block_name)['line'] = converted_block.get(next_block_name)['line'] + (1 if next_arrow else 0)
            plt.arrow(x1, y1, x2, y2, width=0.3, head_width=1.2 if next_arrow else 0, head_length=1 if next_arrow else 0, fc=to_rgba(color_map[str(k)]), ec=to_rgba(color_map[str(k)]))
        
        
        plt.savefig(f'{OUTPUT_PATH}/png/{input_file_name}/{equip_type}_{now}.png', dpi=DPI)
        plt.clf()

class Paver(RoutePlan):
    def __init__(self, block_items: list, converted_block: dict, rearranged_block: list):
        super().__init__(block_items, converted_block)
        self.rearranged_block = rearranged_block
        # converted_rearrange_block 구하기 (가상셀의 근처셀이 이동가능 블럭일 경우 해당 블럭과 인접한 블럭의 라인의 경계선을 구한다, 이동불가 셀 은 etc의 경계선만 구하기)
        self.n_block_boundary = self.convert_n_block_boundary()

    @staticmethod
    @log_decorator('페이버 계획 경로 알고리즘 결과 저장')
    def save_output_csv(input_file_name: str, _route_plan: list):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)
        route_plan = [{k: v for k, v in route.items() if k in ['Timeline', 'BlName', 'Xcoord', 'Ycoord', 'Zcoord']} for route in _route_plan]
        fieldnames = list(route_plan[0].keys())
        
        with open(f'./output/csv/{input_file_name}/{now}_paver_1-3_output.csv', 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for route in route_plan:
                writer.writerow(route)

    def convert_n_block_boundary(self):
        n_block_boundary = {}
        for i, i_block in enumerate(self.rearranged_block):
            for j, block_name in enumerate(i_block):
                target_block = self.converted_block.get(block_name)
                target_block_name = target_block.get('BlName')
                # 이동 불가일 경우
                if target_block.get('YN') == 'N':
                    # 가상셀 인 경우
                    if target_block.get('virtual'):
                        target_block_name = f'{target_block_name}-{i}-{j}'
                        # 위 아래 왼 오 Block인지 체크, Block의 경계라인 가져오기
                        virtual_block_lines = []
                        # up 'ld', 'rd', 'ru', 'lu'
                        if i < len(self.rearranged_block) - 1:
                            up_block_name = self.rearranged_block[i + 1][j]
                            up_block = self.converted_block.get(up_block_name)
                            if up_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(up_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                up_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in up_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in up_block_direction_block for x in ['ld', 'rd']]):
                                    up_ld = up_block_direction_block.get('ld') if 'ld' in up_block_direction_block else up_block_direction_block.get('lu')
                                    up_rd = up_block_direction_block.get('rd') if 'rd' in up_block_direction_block else up_block_direction_block.get('ru')
                                    virtual_block_lines.append(((up_ld[0], up_ld[1]), (up_rd[0], up_rd[1])))
                        # down
                        if i > 0:
                            down_block_name = self.rearranged_block[i - 1][j]
                            down_block = self.converted_block.get(down_block_name)
                            if down_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(down_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                down_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in down_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in down_block_direction_block for x in ['lu', 'ru']]):
                                    down_lu = down_block_direction_block.get('lu') if 'lu' in down_block_direction_block else down_block_direction_block.get('ld')
                                    down_ru = down_block_direction_block.get('ru') if 'ru' in down_block_direction_block else down_block_direction_block.get('rd')
                                    virtual_block_lines.append(((down_lu[0], down_lu[1]), (down_ru[0], down_ru[1])))
                        # right
                        if j < len(self.rearranged_block[i]) - 1:
                            right_block_name = self.rearranged_block[i][j + 1]
                            right_block = self.converted_block.get(right_block_name)
                            if right_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(right_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                right_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in right_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in right_block_direction_block for x in ['ru', 'rd']]):
                                    right_ru = right_block_direction_block.get('ru') if 'ru' in right_block_direction_block else right_block_direction_block.get('lu')
                                    right_rd = right_block_direction_block.get('rd') if 'rd' in right_block_direction_block else right_block_direction_block.get('ld')
                                    virtual_block_lines.append(((right_ru[0], right_ru[1]), (right_rd[0], right_rd[1])))
                        # left
                        if j > 0:
                            left_block_name = self.rearranged_block[i][j - 1]
                            left_block = self.converted_block.get(left_block_name)
                            if left_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(left_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                left_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in left_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in left_block_direction_block for x in ['lu', 'ld']]):
                                    left_lu = left_block_direction_block.get('lu') if 'lu' in left_block_direction_block else left_block_direction_block.get('ru')
                                    left_ld = left_block_direction_block.get('ld') if 'ld' in left_block_direction_block else left_block_direction_block.get('rd')
                                    virtual_block_lines.append(((left_lu[0], left_lu[1]), (left_ld[0], left_ld[1])))
                        if virtual_block_lines:
                            n_block_boundary[target_block_name] = virtual_block_lines
                    else:
                        n_block_boundary[target_block_name] = self.get_boundary_lines(target_block)
        return n_block_boundary

    def paver_route_plan(self, cur_block, next_block):
        cur_block_i, cur_block_j = map(cur_block.get, ['i', 'j'])
        next_block_i, next_block_j = map(next_block.get, ['i', 'j'])
        min_i, min_j, max_i, max_j = min(cur_block_i, next_block_i), min(cur_block_j, next_block_j), max(cur_block_i, next_block_i), max(cur_block_j, next_block_j)

        move_block_names = []
        dont_move_lines = []
        for c_i in range(min_i, max_i + 1):
            for c_j in range(min_j, max_j + 1):
                move_block_name = self.rearranged_block[c_i - 1][c_j - 1]
                if move_block_name == Block.VIRTUAL_BLOCK_NAME:
                    _move_block_name = f'{move_block_name}-{c_i - 1}-{c_j - 1}'
                else:
                    _move_block_name = move_block_name
                move_block_names.append(move_block_name)
                dont_move_lines.extend(self.n_block_boundary.get(_move_block_name, []))

        # 움직일 수 없는 경계선이 없을 경우 바로 감
        if not dont_move_lines:
            # 바로 이동 가능하므로 목적지 좌표 추가, 화살표 추가, 다짐횟수 추가
            self.add_route(next_block, True)
        else:
            c_x, c_y, _ = map(lambda x: float(cur_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
            n_x, n_y, _ = map(lambda x: float(next_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
            # 바로 갈 수 있음
            if self.can_move(dont_move_lines, ((c_x, c_y), (n_x, n_y))):
                # 바로 이동 가능하므로 목적지 좌표 추가, 화살표 추가, 다짐횟수 추가
                self.add_route(next_block, True)
            else:
                # 현재 블럭을 제외한 나머지 블럭에서 목적지를 바로 갈 수 있는가?, 나머지블럭과 현재블럭이 이동 가능한가? 해당되는 경우의 목록을 구하고 대상 목록의 다익스트라, 대상목록이 없을 경우 전체경로에서의 다익스트라
                sub_edges = []
                for m_v in move_block_names:
                    # 이동 불가일 경우 skip
                    m_block = self.converted_block.get(m_v)
                    m_x, m_y, _ = map(lambda x: float(cur_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                    if m_block.get('YN') == 'N':
                        continue
                    # 현재 블럭을 제외한 나머지 블럭에서 목적지를 바로 갈 수 있는가?, 나머지블럭과 현재블럭이 이동 가능한가?
                    if self.can_move(dont_move_lines, ((c_x, c_y), (m_x, m_y))) and self.can_move(dont_move_lines, ((m_x, m_y), (n_x, n_y))):
                        # 현재 블럭 <-> 대상 블럭, 대상 블럭 <-> 목적지 간 거리 정보 추가
                        sub_edges.append((m_block.get('BlName'), next_block.get('BlName'), self.__class__.get_distance(m_block, next_block)))
                        sub_edges.append((cur_block.get('BlName'), m_block.get('BlName'), self.__class__.get_distance(cur_block, m_block)))
                # 대상 목록의 다익스트라, 대상목록이 없을 경우 전체경로에서의 다익스트라
                self.add_route_plan(sub_edges, cur_block.get('BlName'), next_block.get('BlName'))

    @log_decorator('페이버 계획 경로 알고리즘')
    def calc_route_plan(self, allocated_cells: dict, start_block_name: str, work_direction: int):
        _work_direction = work_direction
        epsilon = max([len(i_block) for i_block in self.rearranged_block])
        # 할당셀 loop
        for index, (k, v) in enumerate(allocated_cells.items()):
            # 작업 방향에 따른 시작 j, loop
            v_k_l, v_v_l = list(v.keys()), list(v.values())
            compaction_flag = False
            len_v = len(v) - 1 if _work_direction > 0 else 0
            temp_block, temp_flag = '', False
            for cur_i in range(0, len(v), _work_direction) if _work_direction > 0 else range(len(v) - 1, -1, _work_direction):
                j = self.get_j(v_k_l[cur_i])
                block_dkj = v_v_l[cur_i]
                if not block_dkj: continue
                # 목표 다짐 회수로 인한 반복은 첫번째 index pass
                if compaction_flag:
                    compaction_flag = False
                    continue
                # Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장
                logging.getLogger('plan').debug(f'Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장 {[target_block for i, target_block in enumerate(block_dkj) if len(self.route_plan) == 0 or i != 0]}')
                for i, target_block in enumerate(block_dkj):
                    # 최초 시작이 아닌 경우 첫번째 셀은 경로에서 제외(이미 포함되어 있으므로)
                    if len(self.route_plan) > 0 and i == 0:
                        continue
                    # 마지막 i는 화살표 추가
                    self.add_route(self.converted_block.get(target_block), i == len(block_dkj) - 1)
                
                if not temp_flag:
                    logging.getLogger('plan').debug(f'현재 할당셀Dkj이 포함된 할당셀 집합 내에 경로가 작성되지 않은 할당셀이 있는가? {len_v != cur_i}')

                # 같은 할당셀 집합 내 Dkj+a[0]
                if v.get(self.set_j(j + _work_direction)):
                    logging.getLogger('plan').debug(f'Dkj+a가 있는가? True')
                    # 현재 할당셀Dkj[-1]의 Center 좌표에서 출발해 Dk(j+a)[0]의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장
                    cur_block = self.converted_block.get(block_dkj[-1]) if not temp_block else self.converted_block.get(temp_block)
                    next_block = self.converted_block.get(v.get(self.set_j(j + _work_direction))[0])
                    logging.getLogger('plan').debug(f'현재 할당셀Dkj[-1]({cur_block.get("BlName")}의 Center 좌표에서 출발해 Dk(j+a)[0]({next_block.get("BlName")})의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장')
                    self.paver_route_plan(cur_block, next_block)
                    if temp_block:
                        temp_block = ''
                        temp_flag = False
                elif len_v != cur_i:
                    logging.getLogger('plan').debug(f'Dkj+a가 있는가? False')
                    # 같은 할당셀 집합 내 Dkj+a[0]에 없을 경우 pass
                    temp_block = block_dkj[-1]
                    temp_flag = True
                else:
                    # 현재 할당셀의 delta >= 목표 다짐 횟수
                    # k == c -> 다음 할당셀이 없으므로 종료
                    if index == len(allocated_cells) - 1:
                        logging.getLogger('plan').debug('k == c -> 다음 할당셀이 없으므로 종료')
                        break
                    # alpha > 0
                    logging.getLogger('plan').debug(f'alpha > 0 {_work_direction > 0}')
                    if _work_direction > 0:
                        # 현재 할당셀 집합[-1]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합 [-1]의 [0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장
                        logging.getLogger('plan').debug(f'현재 할당셀 집합[-1]의 [-1]({self.converted_block.get(v_v_l[-1][-1]).get("BlName")}) Center 좌표에서 출발해서 다음 할당셀 집합 [-1]의 [0] ({self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[-1][0])})Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                        self.paver_route_plan(self.converted_block.get(v_v_l[-1][-1]), self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[-1][0]), c_c_i + 1)
                    else:
                        # 현재 할당셀 집합[0]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합 [0]의 [0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장
                        logging.getLogger('plan').debug(f'현재 할당셀 집합[0]의 [-1]({self.converted_block.get(v_v_l[0][-1])}) Center 좌표에서 출발해서 다음 할당셀 집합 [0]의 [0]({self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[0][0])}) Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                        self.paver_route_plan(self.converted_block.get(v_v_l[0][-1]), self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[0][0]), c_c_i + 1)
                    _work_direction = -_work_direction
                    logging.getLogger('plan').debug(f'alpha({_work_direction}) = -alpha({-_work_direction})')

        result = []
        for i, route in enumerate(self.route_plan):
            t_route = {'Timeline': str(i + 1)}
            t_route.update(route)
            result.append(t_route)

        return result
   
    # PNG
    @staticmethod
    @log_decorator('페이버 계획 경로 알고리즘 PNG 저장')
    def save_output_png(input_file_name: str, _route_plan, rearranged_block, converted_block, allocated_cells, equip_type):
        width_ratio, height_ratio = 5 / 4, 5 / 7
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'{OUTPUT_PATH}/png/{input_file_name}', exist_ok=True)
        width, width_length, height, height_length = 15, len(rearranged_block[0]), 10.5, len(rearranged_block)
        sample_colors, color_map, _converted_allocated_cells, k = COLOR_LIST.copy(), {}, {}, 1
        for k, j_block in allocated_cells.items():
            for key_j, values_j in j_block.items():
                for j in values_j:
                    _converted_allocated_cells[j] = {'k': k[1:], 'j': key_j[1:]}

        route_plan = [r_p for r_p in _route_plan]
        # x1, y1, +x(증감), +y (증감)
        plt.figure(figsize=(width_length * width_ratio, height_length * height_ratio))
        
        def block_text_coord(x_index, y_index, block_name):
            return x_index * width + width / 2 - (len(block_name) / 2 * 1.2), height * y_index + height / 2 - 1

        plt.axis('off')
        # [x1, x2], [y1, y2]
        for i in range(0, height_length + 1):
            plt.plot([0, width_length * width], [i * height, i * height], color="black")

        for j in range(0, width_length + 1):
            plt.plot([j * width, j * width], [0, height_length * height], color="black")
        
        for i, i_block in enumerate(rearranged_block):
            for j, block_name in enumerate(i_block):
                fontdict = FONT
                if converted_block.get(block_name).get('YN') == 'N':
                    fontdict = RED_FONT
                plt.text(j * width + width / 2 - (len(block_name) / 2 * 1.2), height * i + height / 2 - 1, block_name, fontdict=fontdict)
        k = 1
        for index, route in enumerate(route_plan):
            # 마지막 경로는 그리지 않음
            if index == len(route_plan) - 1:
                continue
            next_route = route_plan[index + 1]
            block_name, next_block_name = map(lambda x: x.get('BlName'), [route, next_route])
            i, j = map(converted_block.get(block_name).get, ['i', 'j'])
            next_i, next_j = map(converted_block.get(next_block_name).get, ['i', 'j'])
            cur_arrow, next_arrow = map(lambda x: x.get('arrow'), [route, next_route])
            # k가 다를 경우 그리지 않음 (다짐횟수가 마지막이 아닐경우만)
            if int(_converted_allocated_cells.get(block_name).get('k')) != int(_converted_allocated_cells.get(next_block_name).get('k')):
                continue

            if converted_block.get(block_name).get('line') is None:
                converted_block.get(block_name)['line'] = -4
            
            if converted_block.get(next_block_name).get('line') is None:
                converted_block.get(next_block_name)['line'] = -4
            
            if _converted_allocated_cells.get(block_name) and int(_converted_allocated_cells.get(block_name).get('k')) > k:
                k = int(_converted_allocated_cells.get(block_name).get('k'))
            
            if str(k) not in color_map:
                color_map[str(k)] = sample_colors.pop() if sample_colors else ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            x1 = (j - 1) * width + width / 2 + converted_block.get(block_name)['line'] * width / 9
            x2 = (next_j - 1) * width + width / 2 + converted_block.get(next_block_name)['line'] * width / 9 - x1
            y1 = (i - 1) * height + height / 2
            y2 = (next_i - 1) * height + height / 2 - y1
            converted_block.get(block_name)['line'] = converted_block.get(block_name)['line'] + 1
            converted_block.get(next_block_name)['line'] = converted_block.get(next_block_name)['line'] + (1 if next_arrow else 0)
            plt.arrow(x1, y1, x2, y2, width=0.3, head_width=1.2 if next_arrow else 0, head_length=1 if next_arrow else 0, fc=to_rgba(color_map[str(k)]), ec=to_rgba(color_map[str(k)]))
        
        
        plt.savefig(f'{OUTPUT_PATH}/png/{input_file_name}/{equip_type}_{now}.png', dpi=DPI)
        plt.clf()

class Roller(RoutePlan):
    def __init__(self, block_items: list, converted_block: dict, rearranged_block: list):
        super().__init__(block_items, converted_block)
        self.rearranged_block = rearranged_block
        # converted_rearrange_block 구하기 (가상셀의 근처셀이 이동가능 블럭일 경우 해당 블럭과 인접한 블럭의 라인의 경계선을 구한다, 이동불가 셀 은 etc의 경계선만 구하기)
        self.n_block_boundary = self.convert_n_block_boundary()

    @staticmethod
    @log_decorator('롤러 계획 경로 알고리즘 결과 저장')
    def save_output_csv(input_file_name: str, _route_plan: list):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)
        route_plan = [{k: v for k, v in route.items() if k in ['Timeline', 'BlName', 'Xcoord', 'Ycoord', 'Zcoord']} for route in _route_plan]
        fieldnames = list(route_plan[0].keys())
        
        with open(f'./output/csv/{input_file_name}/{now}_roller_1-3_output.csv', 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for route in route_plan:
                writer.writerow(route)

    def convert_n_block_boundary(self):
        n_block_boundary = {}
        for i, i_block in enumerate(self.rearranged_block):
            for j, block_name in enumerate(i_block):
                target_block = self.converted_block.get(block_name)
                target_block_name = target_block.get('BlName')
                # 이동 불가일 경우
                if target_block.get('YN') == 'N':
                    # 가상셀 인 경우
                    if target_block.get('virtual'):
                        target_block_name = f'{target_block_name}-{i}-{j}'
                        # 위 아래 왼 오 Block인지 체크, Block의 경계라인 가져오기
                        virtual_block_lines = []
                        # up 'ld', 'rd', 'ru', 'lu'
                        if i < len(self.rearranged_block) - 1:
                            up_block_name = self.rearranged_block[i + 1][j]
                            up_block = self.converted_block.get(up_block_name)
                            if up_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(up_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                up_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in up_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in up_block_direction_block for x in ['ld', 'rd']]):
                                    up_ld = up_block_direction_block.get('ld') if 'ld' in up_block_direction_block else up_block_direction_block.get('lu')
                                    up_rd = up_block_direction_block.get('rd') if 'rd' in up_block_direction_block else up_block_direction_block.get('ru')
                                    virtual_block_lines.append(((up_ld[0], up_ld[1]), (up_rd[0], up_rd[1])))
                        # down
                        if i > 0:
                            down_block_name = self.rearranged_block[i - 1][j]
                            down_block = self.converted_block.get(down_block_name)
                            if down_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(down_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                down_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in down_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in down_block_direction_block for x in ['lu', 'ru']]):
                                    down_lu = down_block_direction_block.get('lu') if 'lu' in down_block_direction_block else down_block_direction_block.get('ld')
                                    down_ru = down_block_direction_block.get('ru') if 'ru' in down_block_direction_block else down_block_direction_block.get('rd')
                                    virtual_block_lines.append(((down_lu[0], down_lu[1]), (down_ru[0], down_ru[1])))
                        # right
                        if j < len(self.rearranged_block[i]) - 1:
                            right_block_name = self.rearranged_block[i][j + 1]
                            right_block = self.converted_block.get(right_block_name)
                            if right_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(right_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                right_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in right_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in right_block_direction_block for x in ['ru', 'rd']]):
                                    right_ru = right_block_direction_block.get('ru') if 'ru' in right_block_direction_block else right_block_direction_block.get('lu')
                                    right_rd = right_block_direction_block.get('rd') if 'rd' in right_block_direction_block else right_block_direction_block.get('ld')
                                    virtual_block_lines.append(((right_ru[0], right_ru[1]), (right_rd[0], right_rd[1])))
                        # left
                        if j > 0:
                            left_block_name = self.rearranged_block[i][j - 1]
                            left_block = self.converted_block.get(left_block_name)
                            if left_block.get('YN') == 'Y':
                                a_x, a_y, a_z = map(lambda x: float(left_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                                left_block_direction_block = self.get_direction_block([[float(c) for c in _coord.split(',')] for _coord in list(set([etc for etc in left_block.get('etc').split(';') if etc.count(',') == 2]))], a_x, a_y, a_z)
                                if any([x in left_block_direction_block for x in ['lu', 'ld']]):
                                    left_lu = left_block_direction_block.get('lu') if 'lu' in left_block_direction_block else left_block_direction_block.get('ru')
                                    left_ld = left_block_direction_block.get('ld') if 'ld' in left_block_direction_block else left_block_direction_block.get('rd')
                                    virtual_block_lines.append(((left_lu[0], left_lu[1]), (left_ld[0], left_ld[1])))
                        if virtual_block_lines:
                            n_block_boundary[target_block_name] = virtual_block_lines
                    else:
                        n_block_boundary[target_block_name] = self.get_boundary_lines(target_block)
        return n_block_boundary

    def roller_route_plan(self, cur_block, next_block, compaction_count):
        cur_block_i, cur_block_j = map(cur_block.get, ['i', 'j'])
        next_block_i, next_block_j = map(next_block.get, ['i', 'j'])
        min_i, min_j, max_i, max_j = min(cur_block_i, next_block_i), min(cur_block_j, next_block_j), max(cur_block_i, next_block_i), max(cur_block_j, next_block_j)

        move_block_names = []
        dont_move_lines = []
        for c_i in range(min_i, max_i + 1):
            for c_j in range(min_j, max_j + 1):
                move_block_name = self.rearranged_block[c_i - 1][c_j - 1]
                if move_block_name == Block.VIRTUAL_BLOCK_NAME:
                    _move_block_name = f'{move_block_name}-{c_i - 1}-{c_j - 1}'
                else:
                    _move_block_name = move_block_name
                move_block_names.append(move_block_name)
                dont_move_lines.extend(self.n_block_boundary.get(_move_block_name, []))

        # 움직일 수 없는 경계선이 없을 경우 바로 감
        if not dont_move_lines:
            # 바로 이동 가능하므로 목적지 좌표 추가, 화살표 추가, 다짐횟수 추가
            self.add_route(next_block, True, compaction_count)
        else:
            c_x, c_y, _ = map(lambda x: float(cur_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
            n_x, n_y, _ = map(lambda x: float(next_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
            # 바로 갈 수 있음
            if self.can_move(dont_move_lines, ((c_x, c_y), (n_x, n_y))):
                # 바로 이동 가능하므로 목적지 좌표 추가, 화살표 추가, 다짐횟수 추가
                self.add_route(next_block, True, compaction_count)
            else:
                # 현재 블럭을 제외한 나머지 블럭에서 목적지를 바로 갈 수 있는가?, 나머지블럭과 현재블럭이 이동 가능한가? 해당되는 경우의 목록을 구하고 대상 목록의 다익스트라, 대상목록이 없을 경우 전체경로에서의 다익스트라
                sub_edges = []
                for m_v in move_block_names:
                    # 이동 불가일 경우 skip
                    m_block = self.converted_block.get(m_v)
                    m_x, m_y, _ = map(lambda x: float(cur_block.get(x)), ['Xcoord', 'Ycoord', 'Zcoord'])
                    if m_block.get('YN') == 'N':
                        continue
                    # 현재 블럭을 제외한 나머지 블럭에서 목적지를 바로 갈 수 있는가?, 나머지블럭과 현재블럭이 이동 가능한가?
                    if self.can_move(dont_move_lines, ((c_x, c_y), (m_x, m_y))) and self.can_move(dont_move_lines, ((m_x, m_y), (n_x, n_y))):
                        # 현재 블럭 <-> 대상 블럭, 대상 블럭 <-> 목적지 간 거리 정보 추가
                        sub_edges.append((m_block.get('BlName'), next_block.get('BlName'), self.__class__.get_distance(m_block, next_block)))
                        sub_edges.append((cur_block.get('BlName'), m_block.get('BlName'), self.__class__.get_distance(cur_block, m_block)))
                # 대상 목록의 다익스트라, 대상목록이 없을 경우 전체경로에서의 다익스트라
                self.add_route_plan(sub_edges, cur_block.get('BlName'), next_block.get('BlName'), compaction_count)


    @log_decorator('롤러 계획 경로 알고리즘')
    def calc_route_plan(self, allocated_cells: dict, start_block_name: str, work_direction: int, compaction_count: int):
        _work_direction = work_direction
        epsilon = max([len(i_block) for i_block in self.rearranged_block])
        # 할당셀 loop
        for index, (k, v) in enumerate(allocated_cells.items()):
            # 작업 방향에 따른 시작 j, loop
            v_k_l, v_v_l = list(v.keys()), list(v.values())
            compaction_flag = False
            for c_c_i in range(0, compaction_count):
                len_v = len(v) - 1 if _work_direction > 0 else 0
                temp_block, temp_flag = '', False
                for cur_i in range(0, len(v), _work_direction) if _work_direction > 0 else range(len(v) - 1, -1, _work_direction):
                    j = self.get_j(v_k_l[cur_i])
                    block_dkj = v_v_l[cur_i]
                    if not block_dkj: continue
                    # 목표 다짐 회수로 인한 반복은 첫번째 index pass
                    if compaction_flag:
                        compaction_flag = False
                        continue
                    # Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장
                    logging.getLogger('plan').debug(f'Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장 {[target_block for i, target_block in enumerate(block_dkj) if len(self.route_plan) == 0 or i != 0]}')
                    for i, target_block in enumerate(block_dkj):
                        # 최초 시작이 아닌 경우 첫번째 셀은 경로에서 제외(이미 포함되어 있으므로)
                        if len(self.route_plan) > 0 and i == 0:
                            continue
                        # 마지막 i는 화살표 추가
                        self.add_route(self.converted_block.get(target_block), i == len(block_dkj) - 1, c_c_i + 1)
                    
                    if not temp_flag:
                        logging.getLogger('plan').debug(f'현재 할당셀Dkj이 포함된 할당셀 집합 내에 경로가 작성되지 않은 할당셀이 있는가? {len_v != cur_i}')

                    # 같은 할당셀 집합 내 Dkj+a[0]
                    if v.get(self.set_j(j + _work_direction)):
                        logging.getLogger('plan').debug(f'Dkj+a가 있는가? True')
                        # 현재 할당셀Dkj[-1]의 Center 좌표에서 출발해 Dk(j+a)[0]의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장
                        cur_block = self.converted_block.get(block_dkj[-1]) if not temp_block else self.converted_block.get(temp_block)
                        next_block = self.converted_block.get(v.get(self.set_j(j + _work_direction))[0])
                        logging.getLogger('plan').debug(f'현재 할당셀Dkj[-1]({cur_block.get("BlName")}의 Center 좌표에서 출발해 Dk(j+a)[0]({next_block.get("BlName")})의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장')
                        self.roller_route_plan(cur_block, next_block, c_c_i + 1)
                        if temp_block:
                            temp_block = ''
                            temp_flag = False
                    elif len_v != cur_i:
                        logging.getLogger('plan').debug(f'Dkj+a가 있는가? False')
                        # 같은 할당셀 집합 내 Dkj+a[0]에 없을 경우 pass
                        temp_block = block_dkj[-1]
                        temp_flag = True
                    else:
                        # 현재 할당셀의 delta >= 목표 다짐 횟수
                        logging.getLogger('plan').debug(f'현재 할당셀의 delta({c_c_i + 1}) >= 목표 다짐 횟수({compaction_count}) {c_c_i == compaction_count - 1}')
                        if c_c_i == compaction_count - 1:
                            # k == c -> 다음 할당셀이 없으므로 종료
                            if index == len(allocated_cells) - 1:
                                logging.getLogger('plan').debug('k == c -> 다음 할당셀이 없으므로 종료')
                                break
                            # alpha > 0
                            logging.getLogger('plan').debug(f'alpha > 0 {_work_direction > 0}')
                            if _work_direction > 0:
                                # 현재 할당셀 집합[-1]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합 [-1]의 [0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장
                                logging.getLogger('plan').debug(f'현재 할당셀 집합[-1]의 [-1]({self.converted_block.get(v_v_l[-1][-1]).get("BlName")}) Center 좌표에서 출발해서 다음 할당셀 집합 [-1]의 [0] ({self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[-1][0])})Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                                self.roller_route_plan(self.converted_block.get(v_v_l[-1][-1]), self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[-1][0]), c_c_i + 1)
                            else:
                                # 현재 할당셀 집합[0]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합 [0]의 [0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장
                                logging.getLogger('plan').debug(f'현재 할당셀 집합[0]의 [-1]({self.converted_block.get(v_v_l[0][-1])}) Center 좌표에서 출발해서 다음 할당셀 집합 [0]의 [0]({self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[0][0])}) Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                                self.roller_route_plan(self.converted_block.get(v_v_l[0][-1]), self.converted_block.get(list(allocated_cells.get(list(allocated_cells.keys())[index + 1]).values())[0][0]), c_c_i + 1)
                            _work_direction = -_work_direction
                            logging.getLogger('plan').debug(f'alpha({_work_direction}) = -alpha({-_work_direction})')
                        else:
                            logging.getLogger('plan').debug(f'alpha({_work_direction}) = -alpha({-_work_direction})')
                            _work_direction = -_work_direction
                            compaction_flag = True
                            # Dkj[-1] Center 좌표에서 출발해서 Dk(j+a)[0] Center 좌표에 도착하는 최단 경로를 계획 경로에 저장                         
                            cur_block = self.converted_block.get(block_dkj[-1])
                            if v.get(self.set_j(j + _work_direction)):
                                next_block = self.converted_block.get(v.get(self.set_j(j + _work_direction))[0])
                                self.roller_route_plan(cur_block, next_block, c_c_i + 1)
                                logging.getLogger('plan').debug(f'Dkj[-1]({block_dkj[-1]}) Center 좌표에서 출발해서 Dk(j+a)[0]({next_block.get("BlName")}) Center 좌표에 도착하는 최단 경로를 계획 경로에 저장')
                            else:
                                logging.getLogger('plan').debug(f'Dkj+a가 있는가? False')
                                temp_block = block_dkj[-1]
                                temp_flag = True

        result = []
        for i, route in enumerate(self.route_plan):
            t_route = {'Timeline': str(i + 1)}
            t_route.update(route)
            result.append(t_route)

        return result
    
    # PNG
    @staticmethod
    @log_decorator('롤러 계획 경로 알고리즘 PNG 저장')
    def save_output_png(input_file_name: str, _route_plan, rearranged_block, converted_block, allocated_cells, equip_type, compaction_count: int = 1):
        width_ratio, height_ratio = 5 / 4, 5 / 7
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'{OUTPUT_PATH}/png/{input_file_name}', exist_ok=True)
        width, width_length, height, height_length = 15, len(rearranged_block[0]), 10.5, len(rearranged_block)
        sample_colors, color_map, _converted_allocated_cells, k = COLOR_LIST.copy(), {}, {}, 1
        for k, j_block in allocated_cells.items():
            for key_j, values_j in j_block.items():
                for j in values_j:
                    _converted_allocated_cells[j] = {'k': k[1:], 'j': key_j[1:]}

        for c_c_i in range(1, compaction_count + 1):
            route_plan = [r_p for r_p in _route_plan if r_p.get('compaction_count') == c_c_i]
            # x1, y1, +x(증감), +y (증감)
            plt.figure(figsize=(width_length * width_ratio, height_length * height_ratio))
            
            def block_text_coord(x_index, y_index, block_name):
                return x_index * width + width / 2 - (len(block_name) / 2 * 1.2), height * y_index + height / 2 - 1

            plt.axis('off')
            # [x1, x2], [y1, y2]
            for i in range(0, height_length + 1):
                plt.plot([0, width_length * width], [i * height, i * height], color="black")

            for j in range(0, width_length + 1):
                plt.plot([j * width, j * width], [0, height_length * height], color="black")
            
            for i, i_block in enumerate(rearranged_block):
                for j, block_name in enumerate(i_block):
                    fontdict = FONT
                    if converted_block.get(block_name).get('YN') == 'N':
                        fontdict = RED_FONT
                    plt.text(j * width + width / 2 - (len(block_name) / 2 * 1.2), height * i + height / 2 - 1, block_name, fontdict=fontdict)
            k = 1
            for index, route in enumerate(route_plan):
                # 마지막 경로는 그리지 않음
                if index == len(route_plan) - 1:
                    continue
                next_route = route_plan[index + 1]
                block_name, next_block_name = map(lambda x: x.get('BlName'), [route, next_route])
                i, j = map(converted_block.get(block_name).get, ['i', 'j'])
                next_i, next_j = map(converted_block.get(next_block_name).get, ['i', 'j'])
                cur_arrow, next_arrow = map(lambda x: x.get('arrow'), [route, next_route])
                # k가 다를 경우 그리지 않음 (다짐횟수가 마지막이 아닐경우만)
                if int(_converted_allocated_cells.get(block_name).get('k')) != int(_converted_allocated_cells.get(next_block_name).get('k')) and c_c_i != compaction_count:
                    continue

                if converted_block.get(block_name).get('line') is None:
                    converted_block.get(block_name)['line'] = -4
                
                if converted_block.get(next_block_name).get('line') is None:
                    converted_block.get(next_block_name)['line'] = -4
                
                if _converted_allocated_cells.get(block_name) and int(_converted_allocated_cells.get(block_name).get('k')) > k:
                    k = int(_converted_allocated_cells.get(block_name).get('k'))
                
                if str(k) not in color_map:
                    color_map[str(k)] = sample_colors.pop() if sample_colors else ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

                x1 = (j - 1) * width + width / 2 + converted_block.get(block_name)['line'] * width / 9
                x2 = (next_j - 1) * width + width / 2 + converted_block.get(next_block_name)['line'] * width / 9 - x1
                y1 = (i - 1) * height + height / 2
                y2 = (next_i - 1) * height + height / 2 - y1
                converted_block.get(block_name)['line'] = converted_block.get(block_name)['line'] + 1
                converted_block.get(next_block_name)['line'] = converted_block.get(next_block_name)['line'] + (1 if next_arrow else 0)
                plt.arrow(x1, y1, x2, y2, width=0.3, head_width=1.2 if next_arrow else 0, head_length=1 if next_arrow else 0, fc=to_rgba(color_map[str(k)]), ec=to_rgba(color_map[str(k)]))
            
            
            plt.savefig(f'{OUTPUT_PATH}/png/{input_file_name}/{equip_type}_{now}_{c_c_i}.png', dpi=DPI)
            plt.clf()

class Dozer(RoutePlan):
    def __init__(self, block_items: list, converted_block: dict):
        super().__init__(block_items, converted_block)

    @staticmethod
    @log_decorator('도저 계획 경로 알고리즘 결과 저장')
    def save_output_csv(input_file_name: str, _route_plan: list):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        makedirs(f'./output/csv/{input_file_name}', exist_ok=True)
        route_plan = [{k: v for k, v in route.items() if k in ['Timeline', 'BlName', 'Xcoord', 'Ycoord', 'Zcoord']} for route in _route_plan]
        fieldnames = list(route_plan[0].keys())
        
        with open(f'./output/csv/{input_file_name}/{now}_dozer_1-3_output.csv', 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for route in route_plan:
                writer.writerow(route)

    # 도저 성토 계획 경로 산출
    # converted_block: BlName key dict
    @log_decorator('도저 성토 계획 경로 알고리즘')
    def calc_fill_route_plan(self, rearranged_block: list, allocated_cells: dict, start_block_name: str, work_direction: int, allowable_error_height: float, s: int):
        _work_direction = work_direction
        epsilon = max([len(i_block) for i_block in rearranged_block])
        temp_cell_index = 0
        j_direction_flag = False
        # 할당셀 반복
        for index, (k, v) in enumerate(allocated_cells.items()):
            # 작업 방향에 따른 시작 j, loop
            v_k_l, v_v_l = list(v.keys()), list(v.values())

            for cur_i in range(0, len(v), _work_direction) if _work_direction > 0 else range(len(v) - 1, -1, _work_direction):
                j = self.get_j(v_k_l[cur_i])
                block_dkj = v_v_l[cur_i]
                if not block_dkj: continue

                # 1020 j < epsilon loop
                if not j_direction_flag:
                    # Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장
                    logging.getLogger('plan').debug(f'Dkj[0] 부터 Dkj[-1]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장 {[target_block for i, target_block in enumerate(v_v_l[cur_i]) if len(self.route_plan) == 0 or i != 0]}')
                    for i, target_block in enumerate(v_v_l[cur_i]):
                        self.ended_block.append(target_block)
                        self.ended_edges = self.generate_edges([self.converted_block.get(b) for b in self.ended_block], self.ended_edges)
                        # 최초 시작이 아닌 경우 첫번째 셀은 경로에서 제외(이미 포함되어 있으므로)
                        if len(self.route_plan) > 0 and i == 0:
                            continue
                        # 마지막 i는 화살표 추가
                        self.add_route(self.converted_block.get(target_block), i == len(v_v_l[cur_i]) - 1)
                    
                    # Dkj[0]의 i가 1일 경우
                    if self.converted_block.get(block_dkj[0]).get('i') - temp_cell_index == 1:
                        # 이동 가능한 임시 가상셀 생성
                        dkj_f_x, dkj_f_y, dkj_f_z = map(self.converted_block.get(block_dkj[0]).get, ['Xcoord', 'Ycoord', 'Zcoord'])
                        block_t_name = f'BL_T0{j}'
                        block_t = {'BlName': block_t_name, 'YN': 'Y', 'Xcoord': dkj_f_x, 'Ycoord': str(float(dkj_f_y) - s), 'Zcoord': dkj_f_z, 'i': 1, 'j': j}
                        logging.getLogger('plan').debug(f'이동 가능한 임시 가상셀 생성: {block_t}')
                        self.converted_block[block_t_name] = block_t
                        self.converted_block['BL_-'] = {'BlName': 'BL_-', 'YN': 'N'}
                        # 임시 가상셀 라인이 없을 경우 추가
                        if rearranged_block[0][0] != 'BL_-' and rearranged_block[0][0] != 'BL_T':
                            temp_cell_index += 1
                            rearranged_block.insert(0, ['BL_-' for _ in range(0, epsilon)])
                            # j열 i = i + 1 (한칸씩 위로 이동)
                            for t_k in allocated_cells.values():
                                for t_k_b in t_k.values():
                                    for t_k_b_l in t_k_b:
                                        self.converted_block.get(t_k_b_l)['i'] = self.converted_block.get(t_k_b_l)['i'] + 1
                        # 임시 가상셀 추가
                        rearranged_block[0][j - 1] = block_t_name
                        # 임시가상셀 edge 추가
                        t_block_i, t_block_j = map(self.converted_block.get(block_dkj[0]).get, ['i', 'j'])
                        # 임시가상셀 상단 Block의 좌우 Block 
                        t_block_l = rearranged_block[t_block_i - 1][t_block_j - 2] if t_block_j - 2 >= 0 else Block.VIRTUAL_BLOCK_NAME
                        t_block_r = rearranged_block[t_block_i - 1][t_block_j] if t_block_j < len(rearranged_block[t_block_i - 1]) else Block.VIRTUAL_BLOCK_NAME
                        # 이동 가능한 지역의 경우 경로 정보를 추가한다
                        for t_block_d in [t_block_l, block_dkj[0], t_block_r]:
                            if self.converted_block.get(t_block_d).get('YN') == 'Y':
                                t_distance = self.__class__.get_distance(self.converted_block.get(block_t_name), self.converted_block.get(t_block_d))
                                edge_name = '-'.join([block_t_name, t_block_d])
                                reverse_edge_name = '-'.join([t_block_d, block_t_name])
                                self.all_edges[edge_name] = (block_t_name, t_block_d, t_distance)
                                self.all_edges[reverse_edge_name] = (t_block_d, block_t_name, t_distance)
                                self.ended_edges[edge_name] = (block_t_name, t_block_d, t_distance)
                                self.ended_edges[reverse_edge_name] = (t_block_d, block_t_name, t_distance)

                # 같은 할당셀 집합 내 Dkj+a[0]
                logging.getLogger('plan').debug(f'현재 할당셀Dkj이 포함된 할당셀 집합 내에 Dkj+a이 있는가? {v.get(self.set_j(j + _work_direction)) is not None}')
                if v.get(self.set_j(j + _work_direction)):
                    j_direction_flag = False
                    block_amn = self.converted_block.get(v.get(self.set_j(j + _work_direction))[0])
                    logging.getLogger('plan').debug(f'같은 할당셀 집합 내 Dk(j+a)[0] = Amn : {block_amn.get("BlName")}')
                    # A(m-1)n
                    if block_amn.get('i') - 1 - 1 == -1 or rearranged_block[block_amn.get('i') - 1 - 1][block_amn.get('j') - 1] == 'BL_-':
                        block_amn_d = self.converted_block.get('BL_V')
                        block_amn_d_cut_vol, block_amn_d_fill_vol, block_amn_d_area = 0, 0, 0
                    else:
                        block_amn_d = self.converted_block.get(rearranged_block[block_amn.get('i') - 1 - 1][block_amn.get('j') - 1])
                        block_amn_d_cut_vol, block_amn_d_fill_vol, block_amn_d_area = map(lambda x: float(block_amn_d.get(x, 0)), ['cutVol', 'fillVol', 'Area'])

                    # A(m-1)n이 이동불가지역(YN) = Y or A(m-1)n이 가상셀이 아닌가? or A(m-1)n의 성토-절토 < A(m-1)n의 면적 * 허용오차높이
                    logging.getLogger('plan').debug(f'A(m-1)n ({block_amn_d.get("BlName")})이 이동불가지역(YN) = Y {block_amn_d.get("YN") == "Y"}')
                    logging.getLogger('plan').debug(f'A(m-1)n ({block_amn_d.get("BlName")})이 가상셀이 아닌가? {not block_amn_d.get("virtual", False)}')
                    logging.getLogger('plan').debug(f'A(m-1)n의 성토({block_amn_d_fill_vol}-절토({block_amn_d_cut_vol}) < A(m-1)n의 면적({block_amn_d_area}) * 허용오차높이({allowable_error_height}) {(block_amn_d_fill_vol - block_amn_d_cut_vol) < block_amn_d_area * allowable_error_height}')
                    if block_amn_d.get('YN') == 'Y' and not block_amn_d.get('virtual', False) or (block_amn_d_fill_vol - block_amn_d_cut_vol) < block_amn_d_area * allowable_error_height:
                        # 현재 할당셀[Dkj][-1]에서 출발해 A(m-1)n의 Center 좌표에 도착하는 경로 중 최단 경로를 계획 경로에 저장
                        logging.getLogger('plan').debug(f'현재 할당셀[Dkj][-1]({block_dkj[-1]})에서 출발해 A(m-1)n({block_amn_d.get("BlName")})의 Center 좌표에 도착하는 경로 중 최단 경로를 계획 경로에 저장')
                        self.add_route_plan(list(self.ended_edges.values()), block_dkj[-1], block_amn_d.get('BlName'))
                        # A(m-1)n과 Amn의 Center 좌표를 이어 계획 경로에 저장
                        logging.getLogger('plan').debug(f'A(m-1)n({block_amn_d.get("BlName")})과 Amn({block_amn.get("BlName")})의 Center 좌표를 이어 계획 경로에 저장')
                        self.add_route_plan(list(self.ended_edges.values()), block_amn_d.get('BlName'), block_amn.get('BlName'))
                    else:
                        # Dkj[-1]부터 Dkj[0]까지 -i행 방향으로 각각의 셀좌표를 이어 계획경로에 저장
                        logging.getLogger('plan').debug(f'Dkj[-1]부터 Dkj[0]까지 -i행 방향으로 각각의 셀좌표를 이어 계획경로에 저장: {[target_block for i, target_block in enumerate(sorted(v_v_l[cur_i], key=lambda x: int(x.split("_")[1]), reverse=True)) if i != 0]}')
                        for i, target_block in enumerate(sorted(v_v_l[cur_i], key=lambda x: int(x.split('_')[1]), reverse=True)):
                            # Dkj[-1]는 현재 경로이므로 제외
                            if i == 0:
                                continue
                            # 마지막 i는 화살표 추가
                            self.add_route(self.converted_block.get(target_block), i == len(v_v_l[cur_i]) - 1)

                        
                        block_aop = self.converted_block.get(block_dkj[0])
                        # A(o-1)p
                        if block_aop.get('i') - 1 - 1 == -1 or rearranged_block[block_aop.get('i') - 1 - 1][block_aop.get('j') - 1] == '-':
                            block_aop_d = self.converted_block.get('BL_V')
                        else:
                            block_aop_d = self.converted_block.get(rearranged_block[block_aop.get('i') - 1 - 1][block_aop.get('j') - 1])
                            
                        # A(o-1)p가 이동 가능한가?
                        logging.getLogger('plan').debug(f'A(o-1)p ({block_aop_d.get("BlName")})가 이동 가능한가? {block_aop_d.get("YN") == "Y"}')
                        if block_aop_d.get('YN') == 'Y':
                            # Dkj[0] -> A(o-1)p, Amn의 Center 좌표를 경유하는 경로중 최단 경로를 계획 경로에 저장
                            logging.getLogger('plan').debug(f'Dkj[0]({block_dkj[0]}) -> A(o-1)p ({block_aop_d.get("BlName")}), Amn({block_amn.get("BlName")})의 Center 좌표를 경유하는 경로중 최단 경로를 계획 경로에 저장')
                            self.add_route_plan(list(self.ended_edges.values()), block_dkj[0], block_aop_d.get('BlName'))
                            self.add_route_plan(list(self.ended_edges.values()), block_aop_d.get('BlName'), block_amn.get('BlName'))
                        else:
                            # Dkj[0]의 Center 좌표에서 출발해 Amn의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장
                            logging.getLogger('plan').debug(f'Dkj[0]({block_dkj[0]}) -> Amn({block_amn.get("BlName")})의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장')
                            self.add_route_plan(list(self.ended_edges.values()), block_dkj[0], block_amn.get('BlName'))

                    # Dkj 내의 각 셀의 절토, 성토 = 0
                    for block_name in block_dkj:
                        block = self.converted_block.get(block_name)
                        block['cutVol'] = 0
                        block['fillVol'] = 0

                # 현재 할당셀 Dkj이 포함된 할당셀 집합내에 Dk(j+a)이 없을 경우
                else:
                    logging.getLogger('plan').debug(f'a > 0 ? {_work_direction > 0}, j == epsilon ? {cur_i == len(v) - 1}, j == 1 ? {cur_i == 0}')
                    if (_work_direction > 0 and cur_i == len(v) - 1) or (_work_direction < 0 and cur_i == 0):
                        j_direction_flag = False
                        # 마지막 할당셀이 아닌 경우 다음 할당셀 최단 경로 입력
                        logging.getLogger('plan').debug(f'현재 k + 1의 k가 존재하는가? {index != len(allocated_cells) - 1}')
                        logging.getLogger('plan').debug(f'work direction({_work_direction}) > 0  {_work_direction > 0}')
                        if index != len(allocated_cells) - 1:
                            if _work_direction > 0:
                                # 현재 할당셀 집합[-1]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합[-1] [0] Center 좌표에 도착하는 최단경로를 계획경로에 저장
                                next_key = list(allocated_cells.keys())[index + 1]
                                next_allocate_cell0 = list(allocated_cells[list(allocated_cells.keys())[index + 1]].values())[-1][0]
                                logging.getLogger('plan').debug(f'현재 할당셀 집합[-1]의 [-1]({v_v_l[-1][-1]}) Center 좌표에서 출발해서 다음 할당셀 집합[-1] [0]({next_allocate_cell0}) Center 좌표에 도착하는 최단경로를 계획경로에 저장')
                                self.add_route_plan(list(self.ended_edges.values()), v_v_l[-1][-1], next_allocate_cell0)
                            else:
                                # 현재 할당셀 집합[0]의 [-1] Center 좌표에서 출발해서 다음 할당셀 집합[0] [0] Center 좌표에 도착하는 최단경로를 계획경로에 저장
                                next_allocate_cell0 = list(allocated_cells[list(allocated_cells.keys())[index + 1]].values())[0][0]
                                logging.getLogger('plan').debug(f'현재 할당셀 집합[0]의 [-1]({v_v_l[0][-1]}) Center 좌표에서 출발해서 다음 할당셀 집합[0] [0]({next_allocate_cell0}) Center 좌표에 도착하는 최단경로를 계획경로에 저장')
                                self.add_route_plan(list(self.ended_edges.values()), v_v_l[0][-1], next_allocate_cell0)
                    else:
                        j_direction_flag = True

            # alpha = -alpha
            _work_direction = -_work_direction        
        
        result = []
        for i, route in enumerate(self.route_plan):
            t_route = {'Timeline': str(i + 1)}
            t_route.update(route)
            result.append(t_route)

        return result


    # 도저 절토 계획 경로 산출
    @log_decorator('도저 절토 계획 경로 알고리즘')
    def calc_cut_route_plan(self, rearranged_block: list, allocated_cells: dict, start_block_name: str, work_direction: int, allowable_error_height: float, s: int):
        _work_direction = work_direction
        epsilon = max([len(i_block) for i_block in rearranged_block])
        temp_cell_index = 0
        j_direction_flag = False
        last_block_dkj = []
        # 할당셀 반복
        for index, (k, v) in enumerate(allocated_cells.items()):
            # 작업 방향에 따른 시작 j, loop
            v_k_l, v_v_l = list(v.keys()), list(v.values())
            for cur_i in range(0, len(v), _work_direction) if _work_direction > 0 else range(len(v) - 1, -1, _work_direction):
                j = self.get_j(v_k_l[cur_i])
                block_dkj = v_v_l[cur_i]
                if not block_dkj: continue
                
                # j < epsilon loop
                if not j_direction_flag:
                    last_block_dkj = block_dkj
                    # Dkj[-1] 부터 Dkj[0]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장
                    logging.getLogger('plan').debug(f'Dkj[-1] 부터 Dkj[0]까지 +i행의 방향으로 각각의 Center 좌표를 이어 계획 경로에 저장 {[target_block for i, target_block in enumerate(reversed(block_dkj)) if len(self.route_plan) == 0 or i != 0]}')
                    for i, target_block in enumerate(reversed(block_dkj)):
                        self.ended_block.append(target_block)
                        self.ended_edges = self.generate_edges([self.converted_block.get(b) for b in self.ended_block], self.ended_edges)
                        # 최초 시작이 아닌 경우 첫번째 셀은 경로에서 제외(이미 포함되어 있으므로)
                        if len(self.route_plan) > 0 and i == 0:
                            continue
                        # 마지막 i는 화살표 추가
                        self.add_route(self.converted_block.get(target_block), i == len(block_dkj) - 1)
                        
                    # k == 1?
                    logging.getLogger('plan').debug(f'k = 1 ? {k == "k1"}')
                    if k != 'k1':
                        block_agh = self.converted_block.get(block_dkj[0])
                        # Dkj[0] = Agh, A(g+1)h가 이동 가능한가?
                        block_agh_u = self.converted_block.get(rearranged_block[block_agh.get('i') - 1 + 1][block_agh.get('j') - 1]) if len(rearranged_block) > block_agh.get('i') else {}
                        logging.getLogger('plan').debug(f'Dkj[0]({block_dkj[0]}) = Agh, A(g+1)h가 이동 가능한가? {block_agh_u.get("YN") == "Y"}')
                        if block_agh_u.get('YN') == 'Y':
                            self.add_route_plan(list(self.ended_edges.values()), block_dkj[0], block_agh_u.get('BlName'))
                            logging.getLogger('plan').debug(f'Dkj[0]({block_dkj[0]} 부터 A(g+1)h({block_agh_u.get("BlName")}) 까지 center 좌표를 이어 계획 경로에 저장')
                    
                # 같은 할당셀 집합 내 Dkj+a[0]
                logging.getLogger('plan').debug(f'현재 할당셀Dkj이 포함된 할당셀 집합 내에 Dkj+a이 있는가? {v.get(self.set_j(j + _work_direction)) is not None}')
                if v.get(self.set_j(j + _work_direction)):
                    j_direction_flag = False
                    block_amn = self.converted_block.get(v.get(self.set_j(j + _work_direction))[-1])
                    logging.getLogger('plan').debug(f'같은 할당셀 집합 내 Dk(j+a)[-1] = Amn : {block_amn.get("BlName")}')
                    # A(m-1)n
                    if block_amn.get('i') - 1 - 1 == -1 or rearranged_block[block_amn.get('i') - 1 - 1][block_amn.get('j') - 1] == 'BL_-':
                        block_amn_d = self.converted_block.get('BL_V')
                        block_amn_d_cut_vol, block_amn_d_fill_vol, block_amn_d_area = 0, 0, 0
                    else:
                        block_amn_d = self.converted_block.get(rearranged_block[block_amn.get('i') - 1 - 1][block_amn.get('j') - 1])
                        block_amn_d_cut_vol, block_amn_d_fill_vol, block_amn_d_area = map(lambda x: float(block_amn_d.get(x, 0)), ['cutVol', 'fillVol', 'Area'])

                    # A(m-1)n이 이동불가지역(YN) = Y or A(m-1)n이 가상셀이 아닌가? or A(m-1)n의 성토-절토 < A(m-1)n의 면적 * 허용오차높이
                    logging.getLogger('plan').debug(f'A(m-1)n ({block_amn_d.get("BlName")})이 이동불가지역(YN) = Y {block_amn_d.get("YN") == "Y"}')
                    logging.getLogger('plan').debug(f'A(m-1)n ({block_amn_d.get("BlName")})이 가상셀이 아닌가? {not block_amn_d.get("virtual", False)}')
                    logging.getLogger('plan').debug(f'A(m-1)n의 성토({block_amn_d_fill_vol}-절토({block_amn_d_cut_vol}) < A(m-1)n의 면적({block_amn_d_area}) * 허용오차높이({allowable_error_height}) {(block_amn_d_fill_vol - block_amn_d_cut_vol) < block_amn_d_area * allowable_error_height}')
                    if block_amn_d.get('YN') == 'Y' and not block_amn_d.get('virtual', False) or (block_amn_d_fill_vol - block_amn_d_cut_vol) < block_amn_d_area * allowable_error_height:
                        # 현재 할당셀[Dkj][0]에서 출발해 A(m-1)n의 Center 좌표에 도착하는 경로 중 최단 경로를 계획 경로에 저장
                        logging.getLogger('plan').debug(f'현재 할당셀[Dkj][0]({block_dkj[0]})에서 출발해 A(m-1)n({block_amn_d.get("BlName")})의 Center 좌표에 도착하는 경로 중 최단 경로를 계획 경로에 저장')
                        self.add_route_plan(list(self.ended_edges.values()), block_dkj[0], block_amn_d.get('BlName'))
                        # A(m-1)n과 Amn의 Center 좌표를 이어 계획 경로에 저장
                        logging.getLogger('plan').debug(f'A(m-1)n({block_amn_d.get("BlName")})과 Amn({block_amn.get("BlName")})의 Center 좌표를 이어 계획 경로에 저장')
                        self.add_route_plan(list(self.ended_edges.values()), block_amn_d.get('BlName'), block_amn.get('BlName'))
                    else:
                        # 현재 계획경로에 저장된 마지막 셀부터 Dkj[-1]까지 -i행 방향으로 각각의 셀좌표를 이어 계획경로에 저장
                        line_route_plan = [self.route_plan[-1].get("BlName")]
                        line_route_plan.extend(block_dkj)
                        logging.getLogger('plan').debug(f'현재 계획경로에 저장된 마지막 셀({self.route_plan[-1].get("BlName")})부터 Dkj[-1]까지 -i행 방향으로 각각의 셀좌표를 이어 계획경로에 저장: {[p for p in line_route_plan if p != self.route_plan[-1].get("BlName")]}')
                        for i, target_block in enumerate(line_route_plan):
                            # 계획경로에 저장된 마지막 셀은 경로에 포함되어 있으므로 제외
                            if target_block == self.route_plan[-1].get("BlName"):
                                continue
                            # 마지막 i는 화살표 추가
                            self.add_route(self.converted_block.get(target_block), i == len(line_route_plan) - 1)

                        block_aop = self.converted_block.get(block_dkj[-1])
                        # A(o-1)p
                        if block_aop.get('i') - 1 - 1 == -1 or rearranged_block[block_aop.get('i') - 1 - 1][block_aop.get('j') - 1] == '-':
                            block_aop_d = self.converted_block.get('BL_V')
                        else:
                            block_aop_d = self.converted_block.get(rearranged_block[block_aop.get('i') - 1 - 1][block_aop.get('j') - 1])
                            
                        # A(o-1)p가 이동 가능한가?
                        logging.getLogger('plan').debug(f'A(o-1)p ({block_aop_d.get("BlName")})가 이동 가능한가? {block_aop_d.get("YN") == "Y"}')
                        if block_aop_d.get('YN') == 'Y':
                            # Dkj[-1] -> A(o-1)p, Amn의 Center 좌표를 경유하는 경로중 최단 경로를 계획 경로에 저장
                            logging.getLogger('plan').debug(f'Dkj[-1]({block_dkj[-1]}) -> A(o-1)p ({block_aop_d.get("BlName")}), Amn({block_amn.get("BlName")})의 Center 좌표를 경유하는 경로중 최단 경로를 계획 경로에 저장')
                            self.add_route_plan(list(self.ended_edges.values()), block_dkj[-1], block_aop_d.get('BlName'))
                            self.add_route_plan(list(self.ended_edges.values()), block_aop_d.get('BlName'), block_amn.get('BlName'))
                        else:
                            # Dkj[-1]의 Center 좌표에서 출발해 Amn의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장
                            logging.getLogger('plan').debug(f'Dkj[-1]({block_dkj[-1]}) -> Amn({block_amn.get("BlName")})의 Center 좌표로 도착하는 최단경로를 계획 경로에 저장')
                            self.add_route_plan(list(self.ended_edges.values()), block_dkj[-1], block_amn.get('BlName'))

                    # Dkj 내의 각 셀의 절토, 성토 = 0
                    for block_name in block_dkj:
                        block = self.converted_block.get(block_name)
                        block['cutVol'] = 0
                        block['fillVol'] = 0

                # 현재 할당셀 Dkj이 포함된 할당셀 집합내에 Dk(j+a)이 없을 경우
                else:
                    logging.getLogger('plan').debug(f'a > 0 ? {_work_direction > 0}, j == epsilon ? {cur_i == len(v) - 1}, j == 1 ? {cur_i == 0}')
                    if (_work_direction > 0 and cur_i == len(v) - 1) or (_work_direction < 0 and cur_i == 0):
                        j_direction_flag = False
                        # 마지막 할당셀이 아닌 경우 다음 할당셀 최단 경로 입력
                        logging.getLogger('plan').debug(f'현재 k + 1의 k가 존재하는가? {index != len(allocated_cells) - 1}')
                        logging.getLogger('plan').debug(f'work direction({_work_direction}) > 0  {_work_direction > 0}')
                        if index != len(allocated_cells) - 1:
                            if _work_direction > 0:
                                # 현재 할당셀 집합[-1]의 [0] Center 좌표에서 출발해서 다음 할당셀 집합[-1] [-1] Center 좌표에 도착하는 최단경로를 계획경로에 저장
                                next_key = list(allocated_cells.keys())[index + 1]
                                next_allocate_cell0 = list(allocated_cells[list(allocated_cells.keys())[index + 1]].values())[-1][-1]
                                logging.getLogger('plan').debug(f'현재 할당셀 집합[-1]의 [0]({v_v_l[-1][0]}) Center 좌표에서 출발해서 다음 할당셀 집합[-1] [-1]({next_allocate_cell0}) Center 좌표에 도착하는 최단경로를 계획경로에 저장')
                                self.add_route_plan(list(self.ended_edges.values()), v_v_l[-1][0], next_allocate_cell0)
                            else:
                                # 현재 할당셀 집합[0]의 [0] Center 좌표에서 출발해서 다음 할당셀 집합[0] [-1] Center 좌표에 도착하는 최단경로를 계획경로에 저장
                                next_allocate_cell0 = list(allocated_cells[list(allocated_cells.keys())[index + 1]].values())[0][-1]
                                logging.getLogger('plan').debug(f'현재 할당셀 집합[0]의 [0]({v_v_l[0][0]}) Center 좌표에서 출발해서 다음 할당셀 집합[0] [-1]({next_allocate_cell0}) Center 좌표에 도착하는 최단경로를 계획경로에 저장')
                                self.add_route_plan(list(self.ended_edges.values()), v_v_l[0][0], next_allocate_cell0)
                    else:
                        j_direction_flag = True

            # alpha = -alpha
            _work_direction = -_work_direction        
        
        result = []
        for i, route in enumerate(self.route_plan):
            t_route = {'Timeline': str(i + 1)}
            t_route.update(route)
            result.append(t_route)

        return result
