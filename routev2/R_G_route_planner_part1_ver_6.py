import pandas as pd
import numpy as np
import math
import json
from R_G_route_arguments import args
from shapely.geometry import Point, MultiPoint,LineString
import geopandas as gpd
import geojson

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

    return df0, df1, df2

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
    df0, df1, df2 = read_csv_files(input_file)
    
    # df0 노드 사이간 거리 집합
    distances_each_line_node = dist_each_node(df0)
    
    # 노드 추가
    df0, df1, df2 = add_node(df0, df1, df2, distances_each_line_node, (1-x_min)*attachment_width)

    # 노드 제거 및 df0, df1, df2 업데이트
    df0, df1, df2 = remove_close_points(df0,df1,df2, min_distance=((1-x_min)*attachment_width)/2)

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

    forward_lines = []
    # 각 라인에 대한 평행 이동 및 CSV파일 저장
    for i in range(exact_y):
        if i == 0:  # 첫번째 라인
            forward_line = first_move_points_in_parallel(df1, df2, first_gap/2, safety_line)
        else:  # 이후의 라인들
            forward_line = second_move_points_in_parallel(forward_line, df2, second_gap)
        forward_lines.append(forward_line)

    df_lines = pd.concat(forward_lines)
    gdf_lines = gpd.GeoDataFrame(df_lines, geometry=gpd.points_from_xy(df_lines.x, df_lines.y))
    multipoints = MultiPoint(np.asarray(gdf_lines['geometry']))
    #df_lines = pd.DataFrame({'exact_y':exact_y, 'x':x, 'effective_width':effective_width, 'geometry': multipoints})
    gdf_lines = gpd.GeoDataFrame({'exact_y':[exact_y], 'x':[x], 'effective_width':[effective_width], 'geometry': [multipoints]})
    gdf_lines.to_file(output_file,driver='GeoJSON')
    
if __name__ == "__main__":
    main()