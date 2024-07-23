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

    return df1, df2

# 두 점 거리 구하는 함수
def calculate_distances(df1, df2):
    """Calculate distances between points in df1 and df2."""
    distances = []
    for i in range(len(df1)):
        x1, y1 = df1.iloc[i]['x'], df1.iloc[i]['y'] 
        x2, y2 = df2.iloc[i]['x'], df2.iloc[i]['y']  
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(distance)
    return distances

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

# 장비 시작지점 설정 함수
# True : 1 -> 2, False : 2 -> 1
def set_starting_point(num1):
    if num1=="1" :
        return True
    else :
        return False

# 장비 작업 방향 설정 함수
# True : A -> B, False : B -> A
def set_starting_direction(A):
    if A=='A' :
        return True
    else :
        return False
    

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


# csv 통합
def combine_and_save_waypoints(forward_waypoints, backward_waypoints, output_file):
    combined = []
    for cycle in range(len(forward_waypoints)):
        for line in range(len(forward_waypoints[cycle])):
            if line in forward_waypoints[cycle]:
                combined.append(forward_waypoints[cycle][line])
                combined.append(pd.DataFrame({'x': [np.nan], 'y': [np.nan], 'direction': [np.nan]}))  # Add separator
            if line in backward_waypoints[cycle]:
                combined.append(backward_waypoints[cycle][line])
                combined.append(pd.DataFrame({'x': [np.nan], 'y': [np.nan], 'direction': [np.nan]}))  # Add separator
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
    safety_line = args['safety_line']
    x_min = args['x_min']
    turning_radius = args['turning_radius']
    starting_position = args['starting_position']
    starting_direction = args['starting_direction']
    input_file = args['input_file']
    output_file = args['output_file']
    df1, df2 = read_csv_files(input_file)
    
    cycle_num = args['cycle_num'] # 싸이클 횟수
    line_change_way = args['line_change_way'] # 1:후진 후 변경, 2:후진 중 변경, 3:3점회전법

    # working_type = "rolling" # rolling : 다짐, grading : 평탄화, fill : 성토 등등
    
    equipment = 'roller' # roller & grader
    
    # 필요한 최소 후진거리
    required_distance_for_backward = 0
    if line_change_way == 1 :
        required_distance_for_backward = turning_radius * 1.5
    elif line_change_way == 2 :
        required_distance_for_backward = 0
    else : 
        required_distance_for_backward = turning_radius

    node_distance = 2 # 노드 사이 거리
    
    # 후진에 필요한 노드 수
    required_points = math.ceil(required_distance_for_backward / node_distance)

    # 장애물 위치 설정 (원한다면 추가도 가능)
    obstacles = [
    [2, 20],
    [3, 10]
    ]

    # 시작 위치, 방향 설정
    start_option = set_starting_point(starting_position)
    direction_option = set_starting_direction(starting_direction)

    # 시작 위치가 B->A 일 경우 df1과 df2 역순으로 정렬
    if direction_option == False :
        df1 = df1.iloc[::-1].reset_index(drop=True)
        df2 = df2.iloc[::-1].reset_index(drop=True)

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

    forward_lines = {}
    backward_lines = {}

    # 각 라인에 대한 평행 이동
    for i in range(exact_y):
        if i == 0:  # 첫번째 라인
            forward_line = first_move_points_in_parallel(df1, df2, first_gap/2, safety_line)
        else:  # 이후의 라인들
            forward_line = second_move_points_in_parallel(forward_line, df2, second_gap)

        forward_lines[i] = forward_line

        backward_line = forward_line[::-1].reset_index(drop=True)  # 후진 경로 생성
        backward_lines[i] = backward_line

    forward_waypoints = {}  # 전진경로 초기화
    backward_waypoints = {}  # 후진경로 초기화

    # 후진 라인 저장
    for i in range(cycle_num):
        current_cycle = i  # 현재 싸이클 횟수
        forward_waypoints[current_cycle] = {}  # 전진경로 초기화
        backward_waypoints[current_cycle] = {} # 후진경로 초기화

        # 홀수번째 싸이클
        if current_cycle % 2 == 0:
            # 라인 번호 할당
            for j in range(exact_y):
                if start_option == True: # 라인 방향 1->2
                    forward_waypoints[current_cycle][j] = forward_lines[j]
                    backward_waypoints[current_cycle][j] = backward_lines[j]
                else : # 라인 방향 2->1
                    forward_waypoints[current_cycle][j] = forward_lines[exact_y - j - 1]
                    backward_waypoints[current_cycle][j] = backward_lines[exact_y - j - 1]
        
        else : # 짝수번째 싸이클
            for j in range(exact_y):
                if start_option == True: # 라인 방향 1->2
                    forward_waypoints[current_cycle][j] = forward_lines[exact_y - j - 1]
                    backward_waypoints[current_cycle][j] = backward_lines[exact_y - j - 1]
                else : # 라인 방향 2->1
                    forward_waypoints[current_cycle][j] = forward_lines[j]
                    backward_waypoints[current_cycle][j] = backward_lines[j]

    updated_forward_waypoints = {} 
    
    # 후진 후 변경
    if line_change_way == 1:
        tmp = {}
        
        for i in range(cycle_num):
            current_cycle = i

            tmp[current_cycle] = [None] * exact_y
            for j in range(1, exact_y):
                # Access the DataFrame for the current and previous lines
                
                previous_line_df = forward_waypoints[current_cycle][j - 1]
                current_line_df = forward_waypoints[current_cycle][j]
                

                # Slice the DataFrames
                first_part_forward_line = current_line_df.iloc[:required_points]
                second_part_forward_line = current_line_df.iloc[required_points:]
                first_part_previous_forward_line = previous_line_df.iloc[:required_points]

                # Apply the line_change function between the previous and current lines
                new_forward_line = line_change(first_part_previous_forward_line, first_part_forward_line)
                
                updated_forward_waypoints = pd.concat([new_forward_line, second_part_forward_line], ignore_index=True)
                
                tmp[current_cycle][j] = updated_forward_waypoints
                # forward_waypoints[current_cycle][j] = updated_forward_waypoints

            for j in range(1, exact_y) :
                forward_waypoints[current_cycle][j] = tmp[current_cycle][j]

    # 후진 중 변경
    elif line_change_way == 2 :
        for i in range(cycle_num):
            current_cycle = i

            for j in range(exact_y):
                if j < exact_y - 1:
                    current_line_waypoints = backward_waypoints[current_cycle][j]
                    next_line_waypoints = backward_waypoints[current_cycle][j+1]

                    new_backward_line = line_change(current_line_waypoints, next_line_waypoints)
                    backward_waypoints[current_cycle][j] = new_backward_line
    


    # 마지막 사이클의 마지막 후진 경로 삭제
    last_cycle = cycle_num - 1
    if last_cycle in backward_waypoints and len(backward_waypoints[last_cycle]) > 0:
        del backward_waypoints[last_cycle][len(backward_waypoints[last_cycle]) - 1]

    # 전진 경로에 1 추가
    for cycle, waypoints in forward_waypoints.items():
        for line_num, df in waypoints.items():
            df['direction'] = 1

    # 후진 경로에 -1 추가
    for cycle, waypoints in backward_waypoints.items():
        for line_num, df in waypoints.items():
            df['direction'] = -1


    # csv 통합
    combine_and_save_waypoints(forward_waypoints, backward_waypoints, output_file)

if __name__ == "__main__":
    main()
