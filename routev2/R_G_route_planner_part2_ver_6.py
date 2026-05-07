import pandas as pd
import numpy as np
import math
import json
from R_G_route_arguments import args
import geopandas as gpd
import geojson

from shapely.geometry import Polygon

# csv 파일 불러오기
def read_csv_files(file):
    """Read CSV files into DataFrames."""
    df = pd.read_csv(file)

    df1 = df[['x1', 'y1']].rename(columns={'x1': 'x', 'y1': 'y'})
    df2 = df[['x2', 'y2']].rename(columns={'x2': 'x', 'y2': 'y'})

    mid_x = (df1['x'] + df2['x']) / 2
    mid_y = (df1['y'] + df2['y']) / 2

    # 중간 점을 이용한 df0 생성
    df0 = pd.DataFrame({'x': mid_x, 'y': mid_y})

    return df1, df2, df0
# 두 점 거리 구하는 함수(도로 폭)
def calculate_distances(df1, df2):
    """Calculate distances between points in df1 and df2."""
    distances = []
    for i in range(len(df1)):
        x1, y1 = df1.iloc[i]['x'], df1.iloc[i]['y'] 
        x2, y2 = df2.iloc[i]['x'], df2.iloc[i]['y']  
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(distance)
    return distances

# 두 점 거리 구하는 함수(중심 선)
def calculate_min_distances_center_node(df0):
    distances = []
    for i in range(len(df0)-1):
        x1, y1 = df0.iloc[i]['x'], df0.iloc[i]['y']
        x2, y2 = df0.iloc[i+1]['x'], df0.iloc[i+1]['y']
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(distance)
    return min(distances)

# 두 점 사이 거리 구하기
def dist(pt1, pt2) :
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

# df0 노드 사이의 거리 구하기
def dist_each_node(df0):
    distances = []

    for i in range(len(df0) - 1):
        # df0에서 연속된 두 노드 간의 거리
        x00, y00 = df0.iloc[i]['x'], df0.iloc[i]['y']
        x01, y01 = df0.iloc[i + 1]['x'], df0.iloc[i + 1]['y']
        
        # df0에서의 연속된 두 노드 간 거리
        dist0 = dist([x00, y00], [x01, y01])

        # 각 노드 간 거리 저장
        distances.append({
            'df0_dist': dist0
        })

    return distances

# 추가할 노드 개수 계산
def calculate_node_num(node_dist, effective_width) :
    node_num = math.floor(node_dist/effective_width)
    
    if node_num == 0 :
        return 0
    else :
        node_dist1 = float(node_dist/(node_num+1)) # 노드 개수 1개 추가
        node_dist2 = float(node_dist/node_num) # 노드 간격 증가

        node_gap1 = abs(effective_width-node_dist1) # 노드 개수 1개 추가
        node_gap2 = abs(effective_width-node_dist2) # 노드 간격 증가

        node_gap = min(node_gap1,node_gap2)

        if node_gap == node_gap1 :
                node_num += 1
    
        return max(node_num - 1, 0)  # 음수가 나오지 않도록 0보다 작을 경우 0으로 고정

# 노드 추가하기
def add_node(df0, df1, df2, distances_each_line_node, effective_width):
    """df0 기준으로 노드 개수를 계산하고, df1 및 df2에도 동일한 개수의 등간격 점을 추가"""
    new_df0 = []
    new_df1 = []
    new_df2 = []

    # df0를 기준으로 노드를 추가
    for i in range(len(df0) - 1):
        # 현재 노드 추가
        new_df0.append([df0.iloc[i]['x'], df0.iloc[i]['y']])
        new_df1.append([df1.iloc[i]['x'], df1.iloc[i]['y']])
        new_df2.append([df2.iloc[i]['x'], df2.iloc[i]['y']])
        
        # df0의 노드 사이 거리
        node_dist0 = distances_each_line_node[i]['df0_dist']

        # df0를 기준으로 추가할 노드 개수 계산
        node_num = calculate_node_num(node_dist0, effective_width)

        # df0에 대해 노드 추가
        for j in range(1, node_num + 1):
            new_x0 = df0.iloc[i]['x'] + j * (df0.iloc[i + 1]['x'] - df0.iloc[i]['x']) / (node_num + 1)
            new_y0 = df0.iloc[i]['y'] + j * (df0.iloc[i + 1]['y'] - df0.iloc[i]['y']) / (node_num + 1)
            new_df0.append([new_x0, new_y0])

            # df1에 동일한 개수로 등간격 점 추가
            new_x1 = df1.iloc[i]['x'] + j * (df1.iloc[i + 1]['x'] - df1.iloc[i]['x']) / (node_num + 1)
            new_y1 = df1.iloc[i]['y'] + j * (df1.iloc[i + 1]['y'] - df1.iloc[i]['y']) / (node_num + 1)
            new_df1.append([new_x1, new_y1])

            # df2에 동일한 개수로 등간격 점 추가
            new_x2 = df2.iloc[i]['x'] + j * (df2.iloc[i + 1]['x'] - df2.iloc[i]['x']) / (node_num + 1)
            new_y2 = df2.iloc[i]['y'] + j * (df2.iloc[i + 1]['y'] - df2.iloc[i]['y']) / (node_num + 1)
            new_df2.append([new_x2, new_y2])

    # 마지막 노드 추가
    new_df0.append([df0.iloc[-1]['x'], df0.iloc[-1]['y']])
    new_df1.append([df1.iloc[-1]['x'], df1.iloc[-1]['y']])
    new_df2.append([df2.iloc[-1]['x'], df2.iloc[-1]['y']])

    # 새로운 좌표를 DataFrame으로 변환
    new_df0 = pd.DataFrame(new_df0, columns=['x', 'y'])
    new_df1 = pd.DataFrame(new_df1, columns=['x', 'y'])
    new_df2 = pd.DataFrame(new_df2, columns=['x', 'y'])

    return new_df0, new_df1, new_df2

# 점 사이 간격이 min_distance 미만일 때 뒤쪽의 점을 삭제
def remove_close_points(df0, df1, df2, min_distance):
    """점 사이의 간격이 min_distance 미만일 때 뒤쪽의 점을 삭제"""
    i = 0  # 인덱스 초기화
    while i < len(df0) - 1:  # 매 반복마다 최신 df0의 길이를 확인
        # df0의 현재 점과 그 다음 점의 거리 계산
        dist0 = dist([df0.iloc[i]['x'], df0.iloc[i]['y']], [df0.iloc[i + 1]['x'], df0.iloc[i + 1]['y']])

        # 간격이 최소 간격(min_distance)보다 작으면 뒤쪽의 점을 삭제
        if dist0 < min_distance:
            # 뒤쪽 점 삭제
            df0 = df0.drop(i + 1).reset_index(drop=True)
            df1 = df1.drop(i + 1).reset_index(drop=True)
            df2 = df2.drop(i + 1).reset_index(drop=True)
            # i 값을 증가하지 않음: 현재 점과 다음 점을 다시 검사
        else:
            i += 1  # 간격이 충분하면 다음 점으로 이동
    
    # 마지막 점에서는 앞의 점을 삭제 (필요한 경우)
    if len(df0) > 1:
        dist_last = dist([df0.iloc[-2]['x'], df0.iloc[-2]['y']], [df0.iloc[-1]['x'], df0.iloc[-1]['y']])
        if dist_last < min_distance:
            # 마지막 점은 앞쪽 점을 삭제
            df0 = df0.drop(len(df0) - 2).reset_index(drop=True)
            df1 = df1.drop(len(df1) - 2).reset_index(drop=True)
            df2 = df2.drop(len(df2) - 2).reset_index(drop=True)

    return df0, df1, df2

# 도로 폭이 가장 클 때, 중복도 구하기
def first_repeated_rate(model_width, attachment_width, equipment_width, safety_line, x_min):
    """Calculate repeated rate at the widest point of the road."""
    model_width = model_width # 도로 폭
    attachment_width = attachment_width # 어테치먼트 폭
    equipment_width = equipment_width # 장비 폭
    safety_line = safety_line # 안전 거리
    x_min = x_min
    
    # 첫 평행이동 거리는 어테치먼트 폭과 장비 폭중 더 큰것
    first_gap = max(attachment_width, equipment_width)

    # 중복도 범위 
    # max 값은 반드시 설정 안해도 되지만 오류 방지 위해 넣음
    x_max = 0.99

    # 라인수 범위
    y_min = 2
    y_max = 100

    # 중복도 구하는 식
    for y in range(y_min, y_max):
        x = 1 - (model_width - first_gap - 2 * safety_line) / (attachment_width * (y - 1))
        if x > x_min and x < x_max:
            exact_y = y
            break

    # 유효폭
    effective_width = (1 - x) * attachment_width

    return exact_y, x, effective_width, first_gap

# 도로 폭이 가장 큰지점이 아닌 곳에서의 중복도 구하기
def second_repeated_rate(y, model_width, attachment_width, equipment_width, safety_line):
    """Calculate repeated rate at points other than the widest point of the road."""
    model_width = model_width # 도로 폭은 양 최외단 라인의 사이
    y = y # 라인 수는 도로폭이 가장 큰 지점에서 구한 값으로 고정
    
    first_gap = max(attachment_width, equipment_width)
    x = 1 - (model_width - first_gap - 2 * safety_line) / (attachment_width * (y - 1))

    effective_width = (1 - x) * attachment_width

    return effective_width


# 평행이동
def move_point_parallel(point, direction_vector, distance):
    """Move a point parallel to a given direction vector."""
    x, y = point
    new_x = x + distance * direction_vector[0]
    new_y = y + distance * direction_vector[1]
    return new_x, new_y

# 방향벡터 구하기
def calculate_direction_vector(point1, point2):
    """Calculate direction vector between two points."""

    # 점으로 방향벡터 구하기
    x1, y1 = point1
    x2, y2 = point2
    direction_vector = (x2 - x1, y2 - y1)
    magnitude = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2) 
    direction_vector_normalized = (direction_vector[0] / magnitude, direction_vector[1] / magnitude) # 정규화 방향 벡터
    return direction_vector_normalized

# 첫번쨰 평행이동 함수
def first_move_points_in_parallel(df1, df2, dist, safety_line):
    """Perform first parallel movement of points."""
    new_points = []
    for i in range(len(df1)):
        point1 = (df1.iloc[i]['x'], df1.iloc[i]['y'])
        point2 = (df2.iloc[i]['x'], df2.iloc[i]['y'])
        direction_vector = calculate_direction_vector(point1, point2)
        new_point = move_point_parallel(point1, direction_vector, dist+safety_line)
        new_points.append(new_point)
    return pd.DataFrame(new_points, columns=['x', 'y'])

# 두번째 이상의 평행이동 함수
def second_move_points_in_parallel(df1, df2, second_gap):
    """Perform subsequent parallel movement of points."""
    new_points = []
    for i in range(len(df1)):
        point1 = (df1.iloc[i]['x'], df1.iloc[i]['y'])  # Corrected column labels
        point2 = (df2.iloc[i]['x'], df2.iloc[i]['y'])
        direction_vector = calculate_direction_vector(point1, point2)
        new_point = move_point_parallel(point1, direction_vector, second_gap[i])
        new_points.append(new_point)
    return pd.DataFrame(new_points, columns=['x', 'y'])

# 장애물 검사 함수    
def check_obstacle(obstacles, line_num, df1_length):
    """Check if there is an obstacle on the specified line."""
    for obstacle in obstacles:
        if obstacle[0] == line_num:
            if obstacle[1] <= df1_length:  # 장애물의 인덱스가 df1의 개수보다 작거나 같으면 장애물이 있는 것으로 간주
                return True  # 장애물이 있는 경우
            else:
                return False  # 장애물의 인덱스가 df1의 개수보다 크면 장애물이 없는 것으로 간주
    return False  # 장애물이 없는 경우

# 장비가 들어갈 공간
def set_equipment_space(equipment_length, min_node_dist) :
    space = math.ceil(equipment_length/min_node_dist)
    return space

# 라인 변경에 필요한 최소 노드 개수
def get_requierd_dist_for_line_change_num(line_change_way, min_node_dist, turning_radius, repeated_rate):
    
    required_dist_for_line_change = 0
    if line_change_way == 1 : # 1:후진 후 변경
        required_dist_for_line_change = turning_radius * 1.3 * (1 - repeated_rate) / 7.5 * 10
    elif line_change_way == 2 : # 2:후진 중 변경
        required_dist_for_line_change = 0
    else : # 3: 삼점 회전법
        required_dist_for_line_change = turning_radius
    node_num = math.ceil(required_dist_for_line_change/min_node_dist)
    
    return node_num+1

# 작업구역 분할할
def split_work_areas(line, start_line, end_line):
    """Split each line into work and non-work areas based on the given start_line and end_line."""
    work_area = line[start_line:end_line+1]
    non_work_before = line[:start_line]
    non_work_after = line[end_line+1:]
    
    return work_area, non_work_before, non_work_after

# 장애물 위치 인덱스 조정
def adjust_obstacle_index(obstacles, df_length, start_line, starting_direction):
    adjusted_obstacles = []

    for line_number, index in obstacles:
        if starting_direction == 'B':
            # Reverse the index if the starting direction is 'B'
            reversed_index = df_length - 1 - index
            # Adjust the reversed index to account for the start_line
            if reversed_index >= start_line:
                adjusted_index = reversed_index - start_line
                adjusted_obstacles.append([line_number, adjusted_index])
        else:
            # Adjust the index directly based on the start_line
            if index >= start_line:
                adjusted_index = index - start_line
                adjusted_obstacles.append([line_number, adjusted_index])

    return adjusted_obstacles

# 라인 인덱스 구하기기
def get_index(i, total_lines, starting_position, cycle):
    if starting_position == "1":
        if cycle % 2 == 0:  # 짝수 싸이클은 정순  (0부터 1번 싸이클 시작작)
            return i
        else:  # 짝수 싸이클은 역순
            return total_lines - i - 1
    elif starting_position == "2":
        if cycle % 2 == 0:  # 짝수 싸이클은 역순 (0부터 1번 싸이클 시작작)
            return total_lines - i - 1
        else:  # 짝수 싸이클은 정순
            return i
    return i  # 기본값은 정순

def get_next_line_index(starting_position, current_cycle, index, exact_y):
    if starting_position=="1" :
        if current_cycle%2==0 :
            next_line_index = index + 1
            previous_line_index = index - 1
            first_line_index = 0
            last_line_index = exact_y - 1
        else :
            next_line_index = index - 1
            previous_line_index = index + 1
            first_line_index = exact_y - 1
            last_line_index = 0
    else :
        if current_cycle%2==0 :
            next_line_index = index - 1
            previous_line_index = index + 1
            first_line_index = exact_y - 1
            last_line_index = 0
        else :
            next_line_index = index + 1
            previous_line_index = index - 1
            first_line_index = 0
            last_line_index = exact_y - 1
    return next_line_index, previous_line_index, first_line_index, last_line_index

# 라인 변경 함수
def line_change(line1, line2):
    num_points = min(len(line1), len(line2))
    points = {'x': [], 'y': []}

    # Add the first point from line1
    points['x'].append(line1['x'][0])
    points['y'].append(line1['y'][0])

    # Calculate the intermediate points
    for i in range(1, num_points - 1):
        x = line1['x'][i] + ((line2['x'][i] - line1['x'][i]) / (num_points - 1)) * i
        y = line1['y'][i] + ((line2['y'][i] - line1['y'][i]) / (num_points - 1)) * i
        points['x'].append(x)
        points['y'].append(y)

    # Add the last point from line2
    points['x'].append(line2['x'][num_points - 1])
    points['y'].append(line2['y'][num_points - 1])

    return pd.DataFrame(points)

def get_blade_front_distance_num(blade_front_distance, min_node_dist) : 

    blade_front_distance_num = math.ceil(blade_front_distance/min_node_dist)
    
    return blade_front_distance_num



def curve(x0,y0,x1,y1, turning_radius, starting_position, starting_direction,each_line_dist,j,index,exact_y,reverse = False):

    r = turning_radius
    d = np.sqrt((r**2)*2)
    A = np.arccos((2*r*r-d*d)/(2*r*r))

    print(np.degrees(A))
    rad = A

    
    is_latter = index > exact_y // 2
    is_standard = (starting_position == "1" and starting_direction == "A") or \
                (starting_position == "2" and starting_direction == "B")
    is_even = (j % 2 == 0)

    if starting_position == "1":
        if (is_latter and is_standard) or (not is_latter and not is_standard):
            rad = -rad if is_even else rad
        else:
            rad = rad if is_even else -rad

    elif starting_position == "2":
        if (is_latter and is_standard) or (not is_latter and not is_standard):
            rad = -rad if is_even else rad
        else:
            rad = rad if is_even else -rad

    
    num = 10
    num = num-1
    angle_increment = rad / (num)
    points_x = []
    points_y = []
    for i in range(0, num+1) :
        cur_rad = i * angle_increment
        x2 = float(np.cos(cur_rad)*(x1-x0) - np.sin(cur_rad)*(y1-y0) + x0)
        y2 = float(np.sin(cur_rad)*(x1-x0) + np.cos(cur_rad)*(y1-y0) + y0) 
        points_x.append(x2)
        points_y.append(y2)
    

        print(f"Point {i}: ({x2}, {y2})")

    total_points = pd.DataFrame({
        'x': points_x,
        'y': points_y
    })

    if reverse:
        total_points = total_points[::-1].reset_index(drop=True)
    return total_points

# csv 통합
def combine_and_save_waypoints(forward_waypoints, backward_waypoints, output_file):
    combined = []
    for cycle in range(len(forward_waypoints)):
        for line in range(len(forward_waypoints[cycle])):
            if line in forward_waypoints[cycle]:
                combined.append(forward_waypoints[cycle][line])
                #combined.append(pd.DataFrame({'x': [np.nan], 'y': [np.nan], 'direction': [np.nan]}))  # Add separator
            if line in backward_waypoints[cycle]:
                combined.append(backward_waypoints[cycle][line])
                #combined.append(pd.DataFrame({'x': [np.nan], 'y': [np.nan], 'direction': [np.nan]}))  # Add separator
    combined_waypoints = pd.concat(combined, ignore_index=True)    
    combined_waypoints['z1'] = 0
    combined_waypoints['z2'] = 0
    combined_waypoints.to_csv(output_file, index=False)
    # # csv 따로 생성하는 부분
    # for cycle in forward_waypoints.keys():
    #     for i, (forward_line, backward_line) in enumerate(zip(forward_waypoints[cycle].values(), backward_waypoints[cycle].values())):
    #         forward_output_file = f"forward_waypoint_{cycle}_{i+1}.csv"
    #         backward_output_file = f"backward_waypoint_{cycle}_{i+1}.csv"
    #         forward_line.to_csv(forward_output_file, index=False)
    #         backward_line.to_csv(backward_output_file, index=False)
def main():
    # input 값 입력하기
    equipment_width = args['equipment_width']
    attachment_width = args['attachment_width']
    equipment_length = args["equipment_length"]
    safety_line = args['safety_line']
    x_min = args['x_min']
    turning_radius = args['turning_radius']
    starting_position = args['starting_position']
    starting_direction = args['starting_direction']
    input_file = args['input_file']
    output_file = args['output_file']
    start_line = args['start_line']
    end_line = args['end_line']
    
    df1, df2, df0 = read_csv_files(input_file)
    
    cycle_num = args['cycle_num'] # 싸이클 횟수
    line_change_way = args['line_change_way'] # 1:후진 후 변경, 2:후진 중 변경, 3:3점회전법
    obstacles = args['obstacles']
    blade_front_distance = args["blade_front_distance"]
    # df0 노드 사이간 거리 집합
    distances_each_line_node = dist_each_node(df0)
    # 노드 추가
    df0, df1, df2 = add_node(df0, df1, df2, distances_each_line_node, (1-x_min)*attachment_width)
    # 노드 제거
    df0, df1, df2 = remove_close_points(df0,df1,df2, min_distance=((1-x_min)*attachment_width)/2)
    df0, df1, df2 = df0.reset_index(drop=True), df1.reset_index(drop=True), df2.reset_index(drop=True) 
    # 시작 위치가 B->A 일 경우 df1과 df2, df0 역순으로 정렬
    if starting_direction == "B" :
        df1 = df1.iloc[::-1].reset_index(drop=True)
        df2 = df2.iloc[::-1].reset_index(drop=True)
        df0 = df0.iloc[::-1].reset_index(drop=True)

    min_node_dist = calculate_min_distances_center_node(df0) # 도로 중심선 노드 사이 거리 최소값

    space = set_equipment_space(equipment_length, min_node_dist) # 장비가 들어갈 노드 개수
    
    required_dist_for_line_change_num = get_requierd_dist_for_line_change_num(line_change_way, min_node_dist, turning_radius, x_min) # 라인변경에 필요한 최소 노드 수
    min_start_line = required_dist_for_line_change_num + space # start_line 최소값
    
    max_start_line = len(df1)-1
    if blade_front_distance > 0 :
        blade_front_distance_num = get_blade_front_distance_num(blade_front_distance, min_node_dist)
        if line_change_way == 1 or line_change_way == 2 :
            max_end_line = len(df1) - blade_front_distance_num
        elif line_change_way == 3 :
            max_end_line = min(len(df1)-required_dist_for_line_change_num - space, len(df1) - blade_front_distance_num) 
    else:
        if line_change_way == 1 or line_change_way == 2 :
            max_end_line = len(df1) - 1
        elif line_change_way == 3:
            max_end_line = len(df1)- required_dist_for_line_change_num - space

    # starting_direction == B 일 경우
    if starting_direction == "B" :
        start_line = len(df1)-start_line-1 # start_line 인덱스 전환
        end_line = len(df1)-end_line-1 # end_line 인덱스 전환


    if int(start_line) > int(end_line) : # start_line보다 end_line이 후방에 있을 경우
        start_line, end_line = end_line, start_line # start_line과 end_line을 서로 바꿈

    if int(start_line) < int(min_start_line) :  # start_line이 min_start_line 보다 전방에 있을 경우
        start_line = int(min_start_line) # start_line 값을 min_start_line으로 설정
    
    if int(start_line) > int(max_start_line) : # start_line이 max_start_line보다 클 경우
        start_line = int(max_start_line) # start_line을 max_start_line으로 설정

    # end_line 범위 초과 시 범위 내로 수정
    if int(end_line) > int(max_end_line) : # end_line이 max_end_line보다 클 경우
        end_line = int(max_end_line) # end_line을 max_end_line으로 설정

    # working_type = "rolling" # rolling : 다짐, grading : 평탄화, fill : 성토 등등
    equipment = 'roller' # roller & grader    

    # df1과 df2 사이의 모든 노드간 거리
    distances = calculate_distances(df1, df2)

    max_distance = max(distances) # 도로 폭이 최대인 곳
    min_distance = min(distances) # 도로 폭이 최소인 곳

    fg = max(attachment_width, equipment_width)
    # 최대 안전라인 거리
    max_safety_line = (min_distance - fg) / 2

    # 설정한 안전라인이 최대값을 넘어가면 최대값으로 설정
    if safety_line > max_safety_line:
        safety_line = max_safety_line

    # 라인 수, 중복도, 유효폭, 첫 평행이동 거리 구하기 - 도로 폭은 최대거리 사용
    exact_y, x, effective_width, first_gap = first_repeated_rate(max_distance, attachment_width, equipment_width, safety_line, x_min)
    
    # 두번째 offset 부터의 offset 거리
    second_gap = [second_repeated_rate(exact_y, distance, attachment_width, equipment_width, safety_line) for distance in distances]
    tmp_obstacles = []
    # 장애물 위치 설정
    if obstacles != '-':
        obstacles = obstacles.split(',')
        tmp_obstacles = []
        for obstacle in obstacles:
            tmp_obstacles.append([int(obstacle)//len(df1),int(obstacle)%len(df1)])
        
    obstacles = tmp_obstacles
    forward_lines = {}
    work_backward_line = {}
    non_work_forward_line_1 = {}
    non_work_backward_line_1 = {}
    work_forward_line = {}
    non_work_forward_line_2 = {}
    non_work_backward_line_2 = {}

    segmented_work_zones = [[None] * 4 for _ in range(exact_y)]  # 한 번만 초기화

    for i in range(exact_y):
        if i == 0:  # 첫 번째 라인
            forward_line = first_move_points_in_parallel(df1, df2, first_gap / 2, safety_line)
        else:  # 이후의 라인들
            forward_line = second_move_points_in_parallel(forward_line, df2, second_gap)

        forward_lines[i] = forward_line

        # 각 라인을 지정된 start_line과 end_line에 따라 나눔
        non_work_forward_line_1[i] = forward_line.iloc[:start_line].reset_index(drop=True)
        work_forward_line[i] = forward_line.iloc[start_line:end_line + 1].reset_index(drop=True)
        non_work_forward_line_2[i] = forward_line.iloc[end_line + 1:].reset_index(drop=True)

        # 작업구역에서의 후진 노드
        work_backward_line[i] = work_forward_line[i].iloc[::-1].reset_index(drop=True)
        # 비작업구역에서의 전진 노드
        required_backward_section_start = len(non_work_forward_line_1[i]) - required_dist_for_line_change_num
        non_work_forward_line_1[i] = non_work_forward_line_1[i].iloc[required_backward_section_start:].reset_index(drop=True)

        # 비작업구역에서의 후진 노드
        non_work_backward_line_1[i] = non_work_forward_line_1[i].iloc[::-1].reset_index(drop=True)
        non_work_backward_line_2[i] = non_work_forward_line_2[i].reset_index(drop=True)

        # 장애물을 고려하여 work_forward_line[i]를 다시 네 부분으로 나눔
        obstacle_info = next((obs for obs in obstacles if obs[0] == i), None)
        if obstacle_info:
            obs_index = obstacle_info[1]-start_line  # 장애물 인덱스 조정
            print("obs_index: ", obs_index)
            print("required_dist_for_line_change_num : ", required_dist_for_line_change_num)
            if obs_index - required_dist_for_line_change_num <= 0 :
                segmented_work_zones[i][0] = work_forward_line[i].iloc[:obs_index]
                segmented_work_zones[i][1] = pd.DataFrame()  # 빈 데이터프레임
                segmented_work_zones[i][2] = pd.DataFrame()  # 빈 데이터프레임
                segmented_work_zones[i][3] = pd.DataFrame()  # 빈 데이터프레임                
            
            elif obs_index + required_dist_for_line_change_num * 2 + 2 > len(work_forward_line[i]):
                segmented_work_zones[i][0] = work_forward_line[i].iloc[:obs_index]
                segmented_work_zones[i][1] = pd.DataFrame()  # 빈 데이터프레임
                segmented_work_zones[i][2] = pd.DataFrame()  # 빈 데이터프레임
                segmented_work_zones[i][3] = pd.DataFrame()  # 빈 데이터프레임     
            
            else :
                # 전진 구역: 장애물로부터 라인 변경에 필요한 거리 전까지
                segmented_work_zones[i][0] = work_forward_line[i].iloc[:obs_index - required_dist_for_line_change_num]
                print(f"Segment 1 (Before Obstacle Adjustment) for Line {i}: {segmented_work_zones[i][0]}")

                # 라인 변경 구역: 장애물 위치에서 필요한 변경 거리
                segmented_work_zones[i][1] = work_forward_line[i].iloc[obs_index - required_dist_for_line_change_num:obs_index]
                print(f"Segment 2 (Line Change) for Line {i}: {segmented_work_zones[i][1]}")

                # 라인 변경 후 구역: 변경 후 다시 원래 라인으로 돌아오기
                segmented_work_zones[i][2] = work_forward_line[i].iloc[obs_index:obs_index + required_dist_for_line_change_num]
                print(f"Segment 3 (Return to Line) for Line {i}: {segmented_work_zones[i][2]}")

                # 전진 구역: 라인 변경 후 전진
                segmented_work_zones[i][3] = work_forward_line[i].iloc[obs_index + required_dist_for_line_change_num:]
                print(f"Segment 4 (After Obstacle) for Line {i}: {segmented_work_zones[i][3]}")
        else:
            # 장애물이 없는 경우 전체 라인을 전진 구역으로 처리
            segmented_work_zones[i][0] = work_forward_line[i]
            segmented_work_zones[i][1] = pd.DataFrame()  # 빈 데이터프레임
            segmented_work_zones[i][2] = pd.DataFrame()  # 빈 데이터프레임
            segmented_work_zones[i][3] = pd.DataFrame()  # 빈 데이터프레임
            print(f"Line {i} has no obstacles. Entire line set to Segment 1.")

    # 초기화: 전진과 후진 경로를 저장할 딕셔너리
    forward_waypoints = {}   # 전진 경로를 저장할 딕셔너리: (cycle → line → DataFrame)
    backward_waypoints = {}  # 후진 경로를 저장할 딕셔너리: (cycle → line → DataFrame)

    # ---------------------------
    # line_change_way == 1 (후진 후 변경)
    # ---------------------------
    if line_change_way == 1:
        for i in range(cycle_num):
            current_cycle = i  # 0부터 시작하도록 수정

            # 싸이클 키 초기화
            forward_waypoints.setdefault(current_cycle, {})
            backward_waypoints.setdefault(current_cycle, {})

            for j in range(exact_y):
                index = get_index(j, exact_y, starting_position, current_cycle)  # 현재 라인 인덱스 가져오기

                # 장애물 회피 및 경로 설정
                segment_1 = segmented_work_zones[index][0]
                segment_2 = segmented_work_zones[index][1]
                segment_3 = segmented_work_zones[index][2]
                segment_4 = segmented_work_zones[index][3]

                if segment_1 is not None and not segment_1.empty:
                    final_path = segment_1.copy()
                else:
                    final_path = pd.DataFrame(columns=['x', 'y'])
                    
                next_line_index, previous_line_index, first_line_index, last_line_index = get_next_line_index(starting_position, current_cycle, index, exact_y)          

                if segment_2 is not None and not segment_2.empty:
                    start_index_2 = segment_2.index[0]
                    end_index_2 = segment_2.index[-1]
                    start_index_3 = segment_3.index[0]
                    end_index_3 = segment_3.index[-1]    
                    print("required_dist_for_line_change_num : ", required_dist_for_line_change_num)
                    print("start_index_2 : ", start_index_2)
                    print("end-index_2 : ", end_index_2)

                    if index == first_line_index :
                        next_segment_2 = work_forward_line[next_line_index].iloc[start_index_2:end_index_2].reset_index(drop=True)
                        next_segment_3 = work_forward_line[next_line_index].iloc[start_index_3:end_index_3].reset_index(drop=True)
                    else :
                        next_segment_2 = work_forward_line[previous_line_index].iloc[start_index_2:end_index_2].reset_index(drop=True)
                        next_segment_3 = work_forward_line[previous_line_index].iloc[start_index_3:end_index_3].reset_index(drop=True)
                segment_2 = segment_2.reset_index(drop=True)
                segment_3 = segment_3.reset_index(drop=True)

                if segment_2 is not None and not segment_2.empty:
                    transition_to_next = line_change(segment_2, next_segment_2)
                    final_path = pd.concat([final_path, transition_to_next], ignore_index=True)

                if segment_3 is not None and not segment_3.empty:
                    # 현재 라인으로 복귀
                    transition_to_current = line_change(next_segment_3, segment_3)
                    final_path = pd.concat([final_path, transition_to_current], ignore_index=True)

                if segment_4 is not None and not segment_4.empty:
                    final_path = pd.concat([final_path, segment_4], ignore_index=True)

                # 후진 경로 생성
                backward_path = final_path.iloc[::-1].reset_index(drop=True)


                # 다음 라인 인덱스 결정
                print("index : ", index)
                print("next_line_index: ", next_line_index)
                print("j : ", j)
                # 다음 라인의 non_work_backward_line_1 가져오기
                if index != last_line_index : 
                    next_non_work_backward = non_work_backward_line_1[next_line_index]
                    transition_backward = line_change(non_work_backward_line_1[index], next_non_work_backward)
                    backward_path = pd.concat([backward_path, transition_backward], ignore_index=True)
                if index != first_line_index : 
                    final_path = pd.concat([non_work_forward_line_1[index], final_path], ignore_index=True)
            
                forward_waypoints[current_cycle][j] = final_path
                backward_waypoints[current_cycle][j] = backward_path

                print(f"Cycle {current_cycle}, Line {index}: Final path length: {len(final_path)}")

        # line_change_way == 1 블록 끝

    elif line_change_way == 2:

        print("---------------")
        print(forward_lines[0])
        for i in range(cycle_num):
            current_cycle = i
            forward_waypoints.setdefault(current_cycle, {})
            backward_waypoints.setdefault(current_cycle, {})
            
            for j in range(exact_y):
                index = get_index(j, exact_y, starting_position, current_cycle)
                next_line_index, previous_line_index, first_line_index, last_line_index = get_next_line_index(starting_position, current_cycle, index, exact_y)    
                forward_waypoints[current_cycle][j] = work_forward_line[index]
                print("index : ", index)
                if index!=last_line_index :
                    backward_waypoints[current_cycle][j] = work_backward_line[index]
                    current_line_waypoints = backward_waypoints[current_cycle][j]
                    next_line_waypoints = work_backward_line[next_line_index]

                    new_backward_line = line_change(current_line_waypoints, next_line_waypoints)
                    backward_waypoints[current_cycle][j] = new_backward_line

    elif line_change_way == 3:
        print("-------------------------------")
        print(" endline/startline : ", max_end_line / min_start_line)
        print("max_end_line:", max_end_line)

        if min_distance  <= turning_radius * 2:
            print("3점 회전 불가") 
            line_change_way = input(int())
        else:
            for i in range(cycle_num):
                
                current_cycle = i
                forward_waypoints.setdefault(current_cycle, {})
                backward_waypoints.setdefault(current_cycle, {})      
                
                        
                for j in range(exact_y):
                    
                    index = get_index(j, exact_y, starting_position, current_cycle)  # 현재 라인 인덱스 가져오기
                    non_work_forward_line_1[j] = non_work_forward_line_1[j][::-1].reset_index(drop=True) 
                    # 장애물 회피 및 경로 설정
                    segment_1 = segmented_work_zones[index][0]
                    segment_2 = segmented_work_zones[index][1]
                    segment_3 = segmented_work_zones[index][2]
                    segment_4 = segmented_work_zones[index][3]

                    if segment_1 is not None and not segment_1.empty:
                        final_path = segment_1.copy()
                    else:
                        final_path = pd.DataFrame(columns=['x', 'y'])
                        
                    next_line_index, previous_line_index, first_line_index, last_line_index = get_next_line_index(starting_position, current_cycle, index, exact_y)          

                    if segment_2 is not None and not segment_2.empty:
                        start_index_2 = segment_2.index[0]
                        end_index_2 = segment_2.index[-1]
                        start_index_3 = segment_3.index[0]
                        end_index_3 = segment_3.index[-1] 
                        print("required_dist_for_line_change_num : ", required_dist_for_line_change_num)
                        print("start_index_2 : ", start_index_2)
                        print("end-index_2 : ", end_index_2)

                        if index == first_line_index :
                            next_segment_2 = work_forward_line[next_line_index].iloc[start_index_2:end_index_2].reset_index(drop=True)
                            next_segment_3 = work_forward_line[next_line_index].iloc[start_index_3:end_index_3].reset_index(drop=True)
                        else :
                            next_segment_2 = work_forward_line[previous_line_index].iloc[start_index_2:end_index_2].reset_index(drop=True)
                            next_segment_3 = work_forward_line[previous_line_index].iloc[start_index_3:end_index_3].reset_index(drop=True)
                    segment_2 = segment_2.reset_index(drop=True)
                    segment_3 = segment_3.reset_index(drop=True)

                    if segment_2 is not None and not segment_2.empty:
                        transition_to_next = line_change(segment_2, next_segment_2)
                        final_path = pd.concat([final_path, transition_to_next], ignore_index=True)

                    if segment_3 is not None and not segment_3.empty:
                        # 현재 라인으로 복귀
                        transition_to_current = line_change(next_segment_3, segment_3)
                        final_path = pd.concat([final_path, transition_to_current], ignore_index=True)

                    if segment_4 is not None and not segment_4.empty:
                        final_path = pd.concat([final_path, segment_4], ignore_index=True)
                    
                    work_forward_line[index] = final_path
                    current_line_waypoints = work_forward_line[index]
                    
                    
                    if index != last_line_index:
                        current_line_waypoints = work_forward_line[index]
                        next_line_waypoints = work_forward_line[next_line_index]
                        nindex = next_line_index
                    
                        if index <= exact_y // 2:
                            for k in range(1,exact_y):
                                each_line_dist =dist(work_forward_line[index].iloc[-1],work_forward_line[index+k].iloc[-1])
                                print("each_line_dist:",each_line_dist)
                                if each_line_dist >= turning_radius:
                                    nindex = index+k
                                    if index + k == next_line_index:
                                        nindex = next_line_index
                                    print(f"nindex: {nindex}, next_line_index: {next_line_index}")
                                    break
                                else:
                                    continue
                        else:
                            for k in range(1,exact_y):
                                each_line_dist =dist(work_forward_line[index].iloc[-1],work_forward_line[index-k].iloc[-1])
                                if each_line_dist >= turning_radius:
                                    nindex = index-k
                                    if index - k == next_line_index:
                                        nindex = next_line_index
                                    print(f"nindex: {nindex}, next_line_index: {next_line_index}")
                                    break
                                else:
                                    continue
                        
                        if j%2 == 0:
                            print("index:",index)
                            print("next_line_index:",next_line_index)
                            x0 = work_forward_line[nindex]["x"].iloc[-1] #첫 번째 회전(전진 경로)
                            y0 = work_forward_line[nindex]["y"].iloc[-1]
                            x1 = work_forward_line[index]["x"].iloc[-1]
                            y1 = work_forward_line[index]["y"].iloc[-1]
                            curve1 = curve(x0,y0,x1,y1, turning_radius, starting_position, starting_direction,each_line_dist, j,index,exact_y,reverse = False)
                            print("curve1:", curve1)
                            dx = work_forward_line[nindex]["x"].iloc[-1] - curve1["x"].iloc[-1]
                            dy = work_forward_line[nindex]["y"].iloc[-1] - curve1["y"].iloc[-1]
                            print("dx:",dx)
                            print("dy:",dy)
                            x0 = curve1["x"].iloc[-1] - dx #두 번째 회전(후진 경로)
                            y0 = curve1["y"].iloc[-1] - dy
                            x1 = curve1["x"].iloc[-1]
                            y1 = curve1["y"].iloc[-1]
                            curve2 = curve(x0,y0,x1,y1, turning_radius, starting_position, starting_direction,each_line_dist, j,index,exact_y,reverse = False)
                            print("curve2:", curve2)
                            print("x0:",x0)
                            print("y0:",y0)                    
                    
                            new_forward_line = pd.concat([current_line_waypoints,curve1], ignore_index=True)
                            new_backward_line = curve2
                            

                        else:
                            print("index:",index)
                            print("next_line_index:", next_line_index)
                            x0 = work_forward_line[nindex]["x"].iloc[0] #첫 번째 회전(전진 경로)
                            y0 = work_forward_line[nindex]["y"].iloc[0]  
                            x1 = work_forward_line[index]["x"].iloc[0] 
                            y1 = work_forward_line[index]["y"].iloc[0]
                            
                            print("xxxx0:", x0)
                            print("yyyy0:", y0)
                            print("xxxx1:", x1)
                            print("yyyy1:", y1)
                            curve3 = curve(x0,y0,x1,y1, turning_radius, starting_position, starting_direction,each_line_dist, j,index,exact_y,reverse = True)
                        
                            print("curve3:",curve3)
                            dx = work_forward_line[nindex]["x"].iloc[0] - curve3["x"].iloc[0]
                            dy = work_forward_line[nindex]["y"].iloc[0] - curve3["y"].iloc[0]
                            print("dx:",dx)
                            print("dy:",dy)
                            x0 = curve3["x"].iloc[0] - dx #두 번째 회전(후진 경로)
                            y0 = curve3["y"].iloc[0] - dy
                            x1 = curve3["x"].iloc[0]
                            y1 = curve3["y"].iloc[0]
                            curve4 = curve(x0,y0,x1,y1, turning_radius, starting_position, starting_direction, each_line_dist,j,index,exact_y,reverse = True)
                            
                            print("curve4:",curve4)
                            
                            new_forward_line = pd.concat([curve3,current_line_waypoints], ignore_index=True)
                            new_backward_line = curve4
                            
                    else:
                        new_forward_line = current_line_waypoints
                        new_backward_line = pd.DataFrame(columns=['x', 'y'])   # 마지막 후진 경로 빈 데이터 처리
                

                    if j % 2 == 0:
                        forward_waypoints[current_cycle][j] = new_forward_line
                        backward_waypoints[current_cycle][j] = new_backward_line
                        print(f"forward_waypoints{current_cycle}{j}", forward_waypoints[current_cycle][j])
                    else:
                        forward_waypoints[current_cycle][j] = new_forward_line[::-1].reset_index(drop=True)
                        backward_waypoints[current_cycle][j] = new_backward_line[::-1].reset_index(drop=True)
                        print(f"forward_waypoints{current_cycle}{j}", forward_waypoints[current_cycle][j])                                    
    # ---------------------------
    # direction 컬럼을 블록 외부에서 일괄 설정
    # ---------------------------
    # 전진 경로에 direction = 1 추가
    for cycle in forward_waypoints:
        for line_idx in forward_waypoints[cycle]:
            df_f = forward_waypoints[cycle][line_idx]
            if not df_f.empty:
                df_f = df_f.copy()  # 경고 방지
                df_f['direction'] = 1
                forward_waypoints[cycle][line_idx] = df_f
            else:
                forward_waypoints[cycle][line_idx] = pd.DataFrame(columns=['x','y','direction'])

    # 후진 경로에 direction = -1 추가
    for cycle in backward_waypoints:
        for line_idx in backward_waypoints[cycle]:
            df_b = backward_waypoints[cycle][line_idx]
            if not df_b.empty:
                df_b = df_b.copy()  # 경고 방지
                df_b['direction'] = -1
                backward_waypoints[cycle][line_idx] = df_b
            else:
                backward_waypoints[cycle][line_idx] = pd.DataFrame(columns=['x','y','direction'])

    # ---------------------------
    # CSV 통합
    # ---------------------------
    combine_and_save_waypoints(forward_waypoints, backward_waypoints, output_file)

if __name__ == "__main__":
    main()
