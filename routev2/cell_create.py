import pandas as pd
import numpy as np
import math
import csv
import os
from shapely.geometry import Polygon, LineString, GeometryCollection, Point
from shapely.ops import split
import matplotlib.pyplot as plt
import argparse


# csv 파일 불러오기
def read_csv_files(file):
    """Read CSV files into DataFrames."""
    df = pd.read_csv(file)

    df0 = df[['x0', 'y0']].rename(columns={'x0': 'x', 'y0': 'y'}) # 도로 중심선
    df1 = df[['x1', 'y1']].rename(columns={'x1': 'x', 'y1': 'y'}) # 도로 진북방향 우측 외단선
    df2 = df[['x2', 'y2']].rename(columns={'x2': 'x', 'y2': 'y'}) # 도로 진북방향 외측 외단선

    return df0, df1, df2

# 경로 생성 방향 설정
def direction(df1, df2,start_point, start_direction):
    """Set path direction."""
    if start_direction == 'B' :
        df1 = df1.iloc[::-1].reset_index(drop=True)
        df2 = df2.iloc[::-1].reset_index(drop=True)
    if start_point == '2' :
        df1, df2 = df2, df1
    return df1, df2

# 중앙선 새로 그리기
def create_new_center_line(df1, df2):
    new_df0 = pd.DataFrame(columns=['x', 'y'])  # 새로운 DataFrame 생성
    for i in range(len(df1)):
        x1, y1 = df1.iloc[i]['x'], df1.iloc[i]['y']
        x2, y2 = df2.iloc[i]['x'], df2.iloc[i]['y']
        x0, y0 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        new_df0.loc[len(new_df0)] = {'x': x0, 'y': y0}  # DataFrame에 행 추가
    return new_df0

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
def calculate_node_num(node_dist, cell_size) :
    node_num = math.floor(node_dist/cell_size)
    
    if node_num == 0 :
        return 0
    else :
        node_dist1 = float(node_dist/(node_num+1)) # 노드 개수 1개 추가
        node_dist2 = float(node_dist/node_num) # 노드 간격 증가

        node_gap1 = abs(cell_size-node_dist1) # 노드 개수 1개 추가
        node_gap2 = abs(cell_size-node_dist2) # 노드 간격 증가

        node_gap = min(node_gap1,node_gap2)

        if node_gap == node_gap1 :
                node_num += 1
    
        return max(node_num - 1, 0)  # 음수가 나오지 않도록 0보다 작을 경우 0으로 고정

# 노드 추가하기
def add_node(df0, df1, df2, distances_each_line_node, cell_size):
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
        node_num = calculate_node_num(node_dist0, cell_size)

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
# 외단라인 df csv 업데이트 하기
def df_update_csv(df0, df1, df2, filename='output/new_df.csv'):
    """Update CSV file with new DataFrames."""
    
    # DataFrames를 하나로 병합
    merged_df = pd.DataFrame({
        'x0': df0['x'], 'y0': df0['y'], 'z0' : 0,
        'x1': df1['x'], 'y1': df1['y'], 'z1' : 0,
        'x2': df2['x'], 'y2': df2['y'], 'z2' : 0
    })
    
    # CSV 파일로 저장
    merged_df.to_csv(filename, index=False)

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

# 평행이동
def move_point_parallel(point, direction_vector, distance):
    """Move a point parallel to a given direction vector."""
    x, y = point
    new_x = x + distance * direction_vector[0]
    new_y = y + distance * direction_vector[1]
    return new_x, new_y

# 횡축 그리기
def vertical_line_create(df0, df1, df2, cell_size, max_distance):
    """ Create vertical line between side lines"""
    
    # 셀 몇개 생성할지 계산
    cell_num = math.ceil(max_distance / cell_size)
    line_len = (cell_num * cell_size)/2

    points = []
    point_num = math.floor(cell_num/2)
    
            
    # print("line_len : ", line_len)
    # print("cell_num : ", cell_num )
    # print("point_num : ", point_num)

    # 방향벡터 설정
    for i in range(len(df0)):
        direction_vector_0_1 = calculate_direction_vector((df1.iloc[i]['x'], df1.iloc[i]['y']), (df0.iloc[i]['x'], df0.iloc[i]['y']))
        direction_vector_0_2 = calculate_direction_vector((df0.iloc[i]['x'], df0.iloc[i]['y']), (df2.iloc[i]['x'], df2.iloc[i]['y']))
        
        mid_point = (df0.iloc[i]['x'], df0.iloc[i]['y'])
        vector = calculate_direction_vector((df0.iloc[i]['x'], df0.iloc[i]['y']), (df1.iloc[i]['x'], df1.iloc[i]['y']))
        sub_points = []  # 새로운 리스트 생성
        
        point = move_point_parallel(mid_point, vector, line_len)
        sub_points.append(point)
        if cell_num%2 == 0 : 
            for j in range(point_num) :
                point = move_point_parallel(point, direction_vector_0_1, cell_size)
                sub_points.append(point)  # 각 점을 튜플로 추가
            for j in range(point_num)  :
                point = move_point_parallel(point, direction_vector_0_2, cell_size)
                sub_points.append(point) 
        
        else :
            for j in range(point_num) :
                point = move_point_parallel(point, direction_vector_0_1, cell_size)
                sub_points.append(point)  # 각 점을 튜플로 추가

            for j in range(point_num+1)  :
                point = move_point_parallel(point, direction_vector_0_2, cell_size)
                sub_points.append(point) 

        points.append(sub_points)  # 각 i에 대한 점 리스트를 points에 추가
                
    return points

# 면적 계산
def calculate_area(vertices):
    polygon = Polygon(vertices).convex_hull
    return polygon.area

# 점이 꼬였는지 확인
def check_convex(vertices):
    polygon = Polygon(vertices)
    if not polygon.is_valid:
        convex_polygon = polygon.convex_hull
        new_vertices = list(convex_polygon.exterior.coords)[:-1]
        if new_vertices != vertices:
            return True, new_vertices
    return False, vertices

# 점 뽑는 순서 정렬
def update_arrangement(arrangement, i, j, new_vertices):
    arrangement[i][j] = new_vertices[0]
    arrangement[i+1][j] = new_vertices[1]
    arrangement[i+1][j+1] = new_vertices[2]
    arrangement[i][j+1] = new_vertices[3]

# 셀 생성 코드
def create_grid_cells(arrangement):
    grid_cells = []
    convex_cells = []
    stop_creation = False

    for i in range(len(arrangement) - 1):
        if stop_creation:
            break
        for j in range(len(arrangement[i]) - 1):
            if stop_creation:
                break
            vertices = [
                arrangement[i][j],
                arrangement[i][j+1],
                arrangement[i+1][j+1],
                arrangement[i+1][j]
            ]
            is_convex, new_vertices = check_convex(vertices) # 점이 꼬였는지 계산
            if is_convex: # 점이 꼬였으면 풀기
                # print(f"Convex hull used for BL_{j+1}_{i+1}")
                convex_cells.append((i+1, j+1))
                update_arrangement(arrangement, i, j, new_vertices)
                vertices = new_vertices
                area = calculate_area(vertices)
                cell_name = f"BL_{j+1}_{i+1}"
                grid_cell = {'cell_name': cell_name, 'vertices': vertices, 'area': area}
                grid_cells.append(grid_cell)
                stop_creation = True
                break
            area = calculate_area(vertices) # 면적 계산
            cell_name = f"BL_{j+1}_{i+1}" # 셀 이름 설정
            grid_cell = {'cell_name': cell_name, 'vertices': vertices, 'area': area}
            grid_cells.append(grid_cell) # 셀 이어 붙이기
    return grid_cells, convex_cells

# 중앙선 재생성
def calculate_midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2.0, (point1[1] + point2[1]) / 2.0)

# 셀 시각화 코드
def visualize_grid_cells(grid_cells,df0, df1, df2):
    # for grid_cell in grid_cells:
    #     print(f"Cell Name: {grid_cell['cell_name']}")
    #     print("Vertices:")
    #     for vertex in grid_cell['vertices']:
    #         print(f"({vertex[0]:.1f}, {vertex[1]:.1f})")
    #     print(f"Area: {grid_cell['area']:.1f}\n")

    fig, ax = plt.subplots()
    for grid_cell in grid_cells:
        vertices = grid_cell['vertices']
        polygon = Polygon(vertices).convex_hull
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='black')
        centroid = polygon.centroid
        ax.text(centroid.x, centroid.y, grid_cell['cell_name'], fontsize=5, ha='center', va='center')

    ax.plot(df1['x'], df1['y'], color='red', marker='o', linestyle='-', label='df1')
    ax.plot(df2['x'], df2['y'], color='red', marker='o', linestyle='-', label='df2')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.legend()
    plt.title('Grid Cells Visualization')
    plt.show()

# 외단라인 연장
def extend_coordinates(df, extension_length=0.1):
    # 첫 두점 추출
    x1, y1 = df.iloc[0]
    x2, y2 = df.iloc[1]

    # 첫 두점 방향벡터
    direction_vector_first = np.array([x1 - x2, y1 - y2])
    unit_direction_vector_first = direction_vector_first / np.linalg.norm(direction_vector_first)

    # 첫점 다시 찍기
    new_first_point = np.array([x1, y1]) + unit_direction_vector_first * extension_length

    # 마지막 두 점 추출
    x_last, y_last = df.iloc[-1]
    x_second_last, y_second_last = df.iloc[-2]

    # 마지막 두 점 방향 벡터
    direction_vector_last = np.array([x_last - x_second_last, y_last - y_second_last])
    unit_direction_vector_last = direction_vector_last / np.linalg.norm(direction_vector_last)

    # 마지막 점 다시 찍기
    new_last_point = np.array([x_last, y_last]) + unit_direction_vector_last * extension_length

    return new_first_point, new_last_point

def create_new_df(df0,df1,df2):
    # 외단라인 연장
    new_first_point_df0, new_last_point_df0 = extend_coordinates(df0)
    new_first_point_df1, new_last_point_df1 = extend_coordinates(df1)
    new_first_point_df2, new_last_point_df2 = extend_coordinates(df2)

    # df0, df1, df2 재생성
    df0 = pd.DataFrame({
        'x': [new_first_point_df0[0]] + df0['x'].tolist()[1:-1] + [new_last_point_df0[0]],
        'y': [new_first_point_df0[1]] + df0['y'].tolist()[1:-1] + [new_last_point_df0[1]]
    })

    df1 = pd.DataFrame({
        'x': [new_first_point_df1[0]] + df1['x'].tolist()[1:-1] + [new_last_point_df1[0]],
        'y': [new_first_point_df1[1]] + df1['y'].tolist()[1:-1] + [new_last_point_df1[1]]
    })

    df2 = pd.DataFrame({
        'x': [new_first_point_df2[0]] + df2['x'].tolist()[1:-1] + [new_last_point_df2[0]],
        'y': [new_first_point_df2[1]] + df2['y'].tolist()[1:-1] + [new_last_point_df2[1]]
    })
    return df0, df1, df2

# 외단라인 이용하여 외곽선 polygon 형성
def create_combined_boundary_polygon(df1, df2):
    """Create a combined polygon from boundary lines of df1 and df2."""
    coordinates1 = list(zip(df1['x'], df1['y']))
    coordinates2 = list(zip(df2['x'], df2['y']))
    combined_coordinates = coordinates1 + coordinates2[::-1]  # Reverse df2 coordinates for correct boundary
    return Polygon(combined_coordinates)

# 외단라인으로 셀 분할 및 내부에 있는 셀 선택 (뒤에 있는 코드에서 사용하기 위한 부분)
def split_and_filter(poly, lines, boundary):
    result = [poly]
    for line in lines:
        new_result = []
        for geom in result:
            split_result = split(geom, line)
            if isinstance(split_result, GeometryCollection):
                new_result.extend([p for p in split_result.geoms if isinstance(p, Polygon)])
            elif isinstance(split_result, Polygon):
                new_result.append(split_result)
        result = new_result
    filtered = [p for p in result if p.intersects(boundary)]
    return filtered

# 폴리곤의 대부분 면적이 셀 내부에 있는지 판별(셀이 외단라인 내부에 있어도 제거되는 오류 방지 위해서 만듬)
def most_area_inside(poly, boundary):
    intersection_area = poly.intersection(boundary).area
    return intersection_area > poly.area / 2

# 외단라인과 교차하는 셀 찾기
def find_intersecting_cells(grid_cells, line):
    intersecting_cells = []
    for grid_cell in grid_cells:
        polygon = Polygon(grid_cell['vertices'])
        if polygon.intersects(line):
            intersecting_cells.append(grid_cell['cell_name'])
    return intersecting_cells

# 셀 나누고 내부에 있는 셀 선택
def split_cells_and_filter_inside(grid_cells, intersecting_cells, lines, boundary_polygon):
    inside_polygons = {}
    for bl_name in intersecting_cells:
        # 해당 셀을 grid_cells 리스트에서 검색
        grid_cell = next(cell for cell in grid_cells if cell['cell_name'] == bl_name)
        polygon = Polygon(grid_cell['vertices'])
        split_polys = split_and_filter(polygon, lines, boundary_polygon)
        filtered_polys = [poly for poly in split_polys if most_area_inside(poly, boundary_polygon)]
        if filtered_polys:
            inside_polygons[bl_name] = filtered_polys
    return inside_polygons

# 좌표값 유효성 확인
def is_valid_coordinate(coord):
    """Check if the coordinate is valid (not None and finite)."""
    return coord is not None and not pd.isnull(coord)


# 중앙값 계산
def calculate_midpoint(point1, point2):
    """Calculate the midpoint between two points."""
    x1, y1 = point1
    x2, y2 = point2
    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0
    return midpoint_x, midpoint_y

# 중복점 삭제
def remove_duplicate_points(vertices, tolerance=1e-4):
    """Remove later instances of duplicate points by comparing each point with every other point."""
    unique_vertices = []
    seen = set()
    
    for i in range(len(vertices)):
        is_duplicate = False
        for j in range(len(unique_vertices)):
            if np.linalg.norm(np.array(vertices[i]) - np.array(unique_vertices[j])) < tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_vertices.append(vertices[i])
    
    return unique_vertices
    

# csv 파일 만들기
def write_cells_to_csv(grid_cells_dict, filename='output/grid_cells.csv', boundary_polygon=None, inside_polygons=None, intersecting_cells=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    rows = []

    # intersecting_cells가 제공되지 않은 경우 빈 리스트로 초기화
    if intersecting_cells is None:
        intersecting_cells = []

    for idx, (bl_name, grid_cells) in enumerate(grid_cells_dict.items(), start=1):
        for grid_cell in grid_cells:
            vertices = list(grid_cell.exterior.coords)
            vertices = remove_duplicate_points(vertices)
            num_vertices = len(vertices)

            # 삼각형, 사각형, 오각형만 처리 (vertices가 3보다 작으면 무시)
            if num_vertices < 3:
                continue

            # vertices의 좌표를 coords에 저장
            coords = [(coord[0], coord[1]) for coord in vertices]

            # 기본 YN 값을 'Y'로 설정
            yn_value = 'Y'

            # intersecting_cells에 있는 셀이라면 YN을 'N'으로 설정
            if bl_name in intersecting_cells:
                yn_value = 'N'
            else:
                # Boundary_polygon과의 교차 여부 및 내부 포함 여부 확인
                polygon = Polygon(coords)
                if boundary_polygon:
                    if polygon.intersects(boundary_polygon):
                        # 모든 점이 boundary_polygon 내부에 있거나 경계 위에 있는지 확인
                        if not all(boundary_polygon.contains(Point(coord)) or boundary_polygon.touches(Point(coord)) for coord in coords):
                            yn_value = 'N'
                    else:
                        yn_value = 'N'

            # 중점 좌표 계산 (YN이 Y일 때만 계산)
            if yn_value == 'Y':
                xb, yb = calculate_midpoint(coords[0], coords[1])
                xt, yt = calculate_midpoint(coords[-1], coords[-2])
                # 반올림 제거
                xb, yb = str(xb), str(yb)
                xt, yt = str(xt), str(yt)
            else:
                xb, yb, xt, yt = "", "", "", ""

            # inside_polygons에 포함되지 않은 YN이 'N'인 셀들은 좌표 데이터를 빈칸으로 설정
            if yn_value == 'N' and bl_name not in inside_polygons:
                row = [
                    idx,
                    bl_name,
                    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "","","","", "", "", "", "", yn_value, ""
                ]
            else:
                # vertices의 개수에 따른 좌표 처리
                row = [
                    idx,
                    bl_name,
                    f"{grid_cell.area}",
                    str(coords[0][0]), str(coords[0][1]), "0.0",  # X1, Y1, Z1coord
                    str(coords[1][0]), str(coords[1][1]), "0.0",  # X2, Y2, Z2coord
                    str(coords[2][0]), str(coords[2][1]), "0.0",  # X3, Y3, Z3coord
                    str(coords[3][0]) if num_vertices > 3 and len(coords) > 3 else "", 
                    str(coords[3][1]) if num_vertices > 3 and len(coords) > 3 else "", 
                    "0.0" if num_vertices > 3 and len(coords) > 3 else "",  # X4, Y4, Z4coord
                    str(coords[4][0]) if num_vertices > 4 and len(coords) > 4 else "", 
                    str(coords[4][1]) if num_vertices > 4 and len(coords) > 4 else "", 
                    "0.0" if num_vertices > 4 and len(coords) > 4 else "",  # X5, Y5, Z5coord
                    xt, yt, "0.0" if yn_value == 'Y' else "",  # XTcoord, YTcoord, ZTcoord
                    xb, yb, "0.0" if yn_value == 'Y' else "",  # XBcoord, YBcoord, ZBcoord
                    "", "", yn_value, ""  # YN, 기타 빈칸 처리
                ]

            rows.append(row)

    # CSV로 저장
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['No', 'BLName', 'Area', 'X1coord', 'Y1coord', 'Z1coord', 'X2coord', 'Y2coord', 'Z2coord', 'X3coord', 'Y3coord', 'Z3coord', 'X4coord', 'Y4coord', 'Z4coord','X5coord', 'Y5coord', 'Z5coord', 'XTcoord', 'YTcoord', 'ZTcoord', 'XBcoord', 'YBcoord', 'ZBcoord', 'cutVol', 'fillVol', 'YN', 'ClusterName'])
        writer.writerows(rows)

def inputParam():
    parser = argparse.ArgumentParser(description='Cell Create', allow_abbrev=False)
    parser.add_argument('--input_file', type=str, required=True, help='입력 파일 경로')
    parser.add_argument('--output_file', type=str, required=True, help='출력 파일 경로')
    parser.add_argument('--equipment_width', type=float,   required=True, help='장비 폭')
    parser.add_argument('--attachment_width', type=float,   required=True, help='어테치먼트 폭')
    parser.add_argument('--equipment_length', type=float,   required=False, help='장비 길이')
    parser.add_argument('--starting_position', type=str,   required=True, choices=['1', '2'], help='작업 진행 방향')
    parser.add_argument('--starting_direction', type=str,   required=True, choices=['A', 'B'], help='작업 시작 방향')
    parser.add_argument('--repeated_rate', type=float, required=False, default=5, help="중복도")
    parser.add_argument('--output_newline_file', type=str, required=True, help="출력 외곽선 라인(sorted)")
    args = {k: v for k, v in parser.parse_args().__dict__.items() if v is not None}
    return args

def main():
    args = inputParam()
    file = args['input_file']
    df0, df1, df2 = read_csv_files(file)
    start_point = args['starting_position']
    start_direction = args['starting_direction']
    repeated_rate = args['repeated_rate']
    cell_size = args['attachment_width']*(1-repeated_rate)
    df1, df2 = direction(df1,df2,start_point, start_direction)
    # print("origin df = ", len(df0))
    
    df0 = create_new_center_line(df1, df2)
    # 노드 사이의 거리 계산
    distances_each_line_node = dist_each_node(df0)
    # 노드 추가 및 df0, df1, df2 업데이트
    df0, df1, df2 = add_node(df0, df1, df2, distances_each_line_node, cell_size)

    # 노드 제거 및 df0, df1, df2 업데이트
    df0, df1, df2 = remove_close_points(df0,df1,df2, min_distance=1.15)

    distances = calculate_distances(df1, df2)
    max_distance = max(distances) # 도로 폭이 최대인 곳

    # print("new df = ", len(df0))
    df_update_csv(df0, df1, df2, args['output_newline_file'])  # 새 CSV 파일로 저장

    arrangement = vertical_line_create(df0, df1, df2, cell_size, max_distance)

    ## 좌표 리스트 출력
    # for row in arrangement:
    #     print(row)

    stop_creation = False

    while not stop_creation:
        grid_cells, convex_cells = create_grid_cells(arrangement)
        if convex_cells:
            # print("\nConvex hull adjustments made. Updating arrangement...\n")
            pass
        else:
            stop_creation = True

    # 연장한 df0, df1, df2
    df0,df1,df2 = create_new_df(df0,df1,df2)

    # 외단라인으로 분할 선 만들기
    line1 = LineString(list(zip(df1['x'], df1['y'])))
    line2 = LineString(list(zip(df2['x'], df2['y'])))
    
    # 외단라인과 겹치는 셀 찾기
    intersecting_cells_1 = find_intersecting_cells(grid_cells, line1)
    intersecting_cells_2 = find_intersecting_cells(grid_cells, line2)
    intersecting_cells = set(intersecting_cells_1).union(set(intersecting_cells_2))

    # cell 경계선 분할을 위한 boundary_polygon 형성
    boundary_polygon = create_combined_boundary_polygon(df1, df2)

    # 외단라인으로 겹치는 셀들에 대해 분할 및 내부에 있는 셀들 추출
    lines = [line1, line2]
    inside_polygons_1 = split_cells_and_filter_inside(grid_cells, intersecting_cells_1, lines, boundary_polygon)
    inside_polygons_2 = split_cells_and_filter_inside(grid_cells, intersecting_cells_2, lines, boundary_polygon)
    
    # inside_polygons_1과 inside_polygons_2를 합치기
    inside_polygons = {**inside_polygons_1, **inside_polygons_2}

    # grid_cells를 딕셔너리로 변환
    grid_cells_dict = {cell['cell_name']: [Polygon(cell['vertices'])] for cell in grid_cells}

    # combined_inside_polygons로 grid_cells_dict 덮어쓰기
    grid_cells_dict.update(inside_polygons_1)
    grid_cells_dict.update(inside_polygons_2)

    # 덮어쓴 grid_cells_dict를 사용하여 CSV 파일로 저장
    write_cells_to_csv(grid_cells_dict, args['output_file'],boundary_polygon, inside_polygons, intersecting_cells)
 
    # # print(grid_cells)
    # visualize_grid_cells(grid_cells,df0,df1,df2)

if __name__ == "__main__":
    main()