import pandas as pd
import numpy as np
import math
import csv
import os
from shapely.geometry import Polygon, LineString, Point, MultiPoint, GeometryCollection, MultiLineString
from shapely.ops import split
import matplotlib.pyplot as plt
import argparse


def read_csv_files(file):
    """CSV 파일에서 도로 데이터 읽기"""
    df = pd.read_csv(file)
    
    df0 = df[['x0', 'y0']].rename(columns={'x0': 'x', 'y0': 'y'})  # 중심선
    df1 = df[['x1', 'y1']].rename(columns={'x1': 'x', 'y1': 'y'})  # 좌측 경계
    df2 = df[['x2', 'y2']].rename(columns={'x2': 'x', 'y2': 'y'})  # 우측 경계
    
    # A, B 정보 읽기
    ab_file = file.replace('.csv', '_AB.csv')
    ab_info = None
    if os.path.exists(ab_file):
        ab_info = pd.read_csv(ab_file)
    
    # 중점 정보 읽기
    midpoint_file = file.replace('.csv', '_midpoint.csv')
    midpoint_info = None
    if os.path.exists(midpoint_file):
        midpoint_info = pd.read_csv(midpoint_file)
    
    return df0, df1, df2, ab_info, midpoint_info


def align_data_to_ab_bottom(df0, df1, df2, ab_info):
    """AB 정보를 기준으로 데이터를 정렬하여 AB가 바닥(첫 번째 행)이 되도록 함"""
    if ab_info is None or len(ab_info) < 2:
        print("⚠️ AB 정보가 없어 기본 순서를 유지합니다.")
        return df0, df1, df2
    
    # A, B 점 좌표
    point_A = np.array([ab_info.iloc[0]['x'], ab_info.iloc[0]['y']])
    point_B = np.array([ab_info.iloc[1]['x'], ab_info.iloc[1]['y']])
    
    print(f"Point A: ({point_A[0]:.2f}, {point_A[1]:.2f})")
    print(f"Point B: ({point_B[0]:.2f}, {point_B[1]:.2f})")
    
    # df1의 시작점과 끝점
    df1_start = np.array([df1.iloc[0]['x'], df1.iloc[0]['y']])
    df1_end = np.array([df1.iloc[-1]['x'], df1.iloc[-1]['y']])
    
    # A, B와 df1 시작/끝점 사이의 거리 계산
    dist_start_A = np.linalg.norm(df1_start - point_A)
    dist_start_B = np.linalg.norm(df1_start - point_B)
    dist_end_A = np.linalg.norm(df1_end - point_A)
    dist_end_B = np.linalg.norm(df1_end - point_B)
    
    print(f"df1 시작점과 A 거리: {dist_start_A:.2f}")
    print(f"df1 시작점과 B 거리: {dist_start_B:.2f}")
    print(f"df1 끝점과 A 거리: {dist_end_A:.2f}")
    print(f"df1 끝점과 B 거리: {dist_end_B:.2f}")
    
    # AB가 시작점에 더 가까운지 끝점에 더 가까운지 판단
    ab_near_start = (min(dist_start_A, dist_start_B) < min(dist_end_A, dist_end_B))
    
    if ab_near_start:
        print("✓ AB가 이미 시작점(바닥) 근처에 있습니다. 순서를 유지합니다.")
        return df0, df1, df2
    else:
        print("✓ AB가 끝점 근처에 있습니다. 데이터를 역순으로 정렬합니다.")
        df0_new = df0.iloc[::-1].reset_index(drop=True)
        df1_new = df1.iloc[::-1].reset_index(drop=True)
        df2_new = df2.iloc[::-1].reset_index(drop=True)
        return df0_new, df1_new, df2_new


def create_new_center_line(df1, df2):
    """좌우 경계선의 중점을 연결하여 새로운 중심선 생성"""
    new_df0 = pd.DataFrame(columns=['x', 'y'])
    for i in range(len(df1)):
        x0 = (df1.iloc[i]['x'] + df2.iloc[i]['x']) / 2.0
        y0 = (df1.iloc[i]['y'] + df2.iloc[i]['y']) / 2.0
        new_df0.loc[len(new_df0)] = {'x': x0, 'y': y0}
    return new_df0


def calculate_distance(pt1, pt2):
    """두 점 사이의 거리 계산"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def calculate_node_distances(df0):
    """중심선 노드 간 거리 계산"""
    distances = []
    for i in range(len(df0) - 1):
        pt1 = [df0.iloc[i]['x'], df0.iloc[i]['y']]
        pt2 = [df0.iloc[i + 1]['x'], df0.iloc[i + 1]['y']]
        distances.append({'df0_dist': calculate_distance(pt1, pt2)})
    return distances


def calculate_additional_nodes(node_dist, cell_size):
    """셀 크기에 맞춰 추가할 노드 개수 계산"""
    node_num = math.floor(node_dist / cell_size)
    
    if node_num == 0:
        return 0
    
    # 노드 개수 최적화: 셀 크기에 가장 가까운 간격 선택
    dist_with_extra = node_dist / (node_num + 1)
    dist_without_extra = node_dist / node_num
    
    gap_with_extra = abs(cell_size - dist_with_extra)
    gap_without_extra = abs(cell_size - dist_without_extra)
    
    if gap_with_extra < gap_without_extra:
        node_num += 1
    
    return max(node_num - 1, 0)


def add_nodes(df0, df1, df2, distances, cell_size):
    """모든 라인에 동일한 개수의 노드 추가"""
    new_dfs = [[], [], []]
    dfs = [df0, df1, df2]
    
    for i in range(len(df0) - 1):
        # 현재 노드 추가
        for j, df in enumerate(dfs):
            new_dfs[j].append([df.iloc[i]['x'], df.iloc[i]['y']])
        
        # 추가할 노드 개수 계산
        node_num = calculate_additional_nodes(distances[i]['df0_dist'], cell_size)
        
        # 노드 추가
        for k in range(1, node_num + 1):
            t = k / (node_num + 1)
            for j, df in enumerate(dfs):
                x = df.iloc[i]['x'] + t * (df.iloc[i + 1]['x'] - df.iloc[i]['x'])
                y = df.iloc[i]['y'] + t * (df.iloc[i + 1]['y'] - df.iloc[i]['y'])
                new_dfs[j].append([x, y])
    
    # 마지막 노드 추가
    for j, df in enumerate(dfs):
        new_dfs[j].append([df.iloc[-1]['x'], df.iloc[-1]['y']])
    
    # DataFrame으로 변환
    return [pd.DataFrame(pts, columns=['x', 'y']) for pts in new_dfs]


def create_boundary_polygon(df1, df2):
    """경계선으로부터 폴리곤 생성"""
    coords1 = list(zip(df1['x'], df1['y']))
    coords2 = list(zip(df2['x'], df2['y']))
    return Polygon(coords1 + coords2[::-1])


def create_grid_cells_from_midpoint(df1, df2, cell_size, df0=None, basename='', ab_info=None, midpoint_info=None, prefix='BL'):
    """중점 기준으로 회전된 격자 생성"""
    # 1. 경계 폴리곤 생성
    boundary_polygon = create_boundary_polygon(df1, df2)
    
    # 2. 중점 정보 사용
    if midpoint_info is not None and len(midpoint_info) > 0:
        # 중점 정보가 있으면 사용
        midpoint = np.array([midpoint_info.iloc[0]['midpoint_x'], midpoint_info.iloc[0]['midpoint_y']])
        pt_A = np.array([midpoint_info.iloc[0]['A_x'], midpoint_info.iloc[0]['A_y']])
        pt_B = np.array([midpoint_info.iloc[0]['B_x'], midpoint_info.iloc[0]['B_y']])
    else:
        # 중점 정보가 없으면 기존 방식으로 계산
        if ab_info is not None and len(ab_info) >= 2:
            pt_A = np.array([ab_info.iloc[0]['x'], ab_info.iloc[0]['y']])
            pt_B = np.array([ab_info.iloc[1]['x'], ab_info.iloc[1]['y']])
        else:
            pt_A = np.array([df1.iloc[0]['x'], df1.iloc[0]['y']])
            pt_B = np.array([df2.iloc[0]['x'], df2.iloc[0]['y']])
        midpoint = (pt_A + pt_B) / 2
    
    # 3. A-B 벡터를 기준으로 회전 각도 계산
    AB_vec = pt_B - pt_A
    angle = np.arctan2(AB_vec[1], AB_vec[0])
    
    # 4. 회전 행렬
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    inv_rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    # 5. 모든 점을 회전시켜 바운딩 박스 계산 (중점 기준)
    all_points = np.vstack([
        df1[['x', 'y']].values,
        df2[['x', 'y']].values
    ])
    
    # 중점을 원점으로 하여 회전
    rotated_points = np.array([rotation_matrix @ (pt - midpoint) for pt in all_points])
    
    xmin, xmax = rotated_points[:, 0].min(), rotated_points[:, 0].max()
    ymin, ymax = rotated_points[:, 1].min(), rotated_points[:, 1].max()
    
    # 6. 중점 기준으로 격자 시작점 계산 (cell_size의 배수로 정렬)
    x_start = np.floor(xmin / cell_size) * cell_size
    x_end = np.ceil(xmax / cell_size) * cell_size
    y_start = np.floor(ymin / cell_size) * cell_size
    y_end = np.ceil(ymax / cell_size) * cell_size
    
    x_coords = np.arange(x_start, x_end + cell_size/2, cell_size)
    y_coords = np.arange(y_start, y_end + cell_size/2, cell_size)
    
    # 격자 정보를 리스트로 저장 (정렬을 위해)
    grid_list = []
    
    # 7. 각 격자 셀 처리
    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
            # 회전된 좌표계에서의 꼭짓점 (중점 기준)
            vertices_rot = [
                [x, y],                          # 0번: 좌측 하단
                [x + cell_size, y],              # 1번: 우측 하단
                [x + cell_size, y + cell_size],  # 2번: 우측 상단
                [x, y + cell_size]               # 3번: 좌측 상단
            ]
            
            # 원래 좌표계로 역변환 (중점을 다시 더함)
            vertices = [tuple(inv_rotation_matrix @ v + midpoint) for v in vertices_rot]
            poly = Polygon(vertices)
            
            # 경계와 교차 확인
            if poly.intersects(boundary_polygon) and poly.intersection(boundary_polygon).area > 1e-10:
                centroid = poly.centroid
                # 회전된 좌표계에서의 중심점 위치
                centroid_rot = rotation_matrix @ (np.array([centroid.x, centroid.y]) - midpoint)
                
                grid_list.append({
                    'x_idx': i,  # 격자 인덱스
                    'y_idx': j,
                    'x_rot': centroid_rot[0],
                    'y_rot': centroid_rot[1],
                    'vertices': vertices,
                    'area': cell_size * cell_size,
                    'is_boundary': not boundary_polygon.contains(centroid)  # 경계 셀 표시
                })
    
    # 8. BL 번호 매기기 - AB가 바닥이 되도록 y 기준 정렬
    if not grid_list:
        return []
    
    # AB 선분이 바닥(y값이 작은 쪽)에 있도록 확인
    ab_point_rot = rotation_matrix @ (np.array([df1.iloc[0]['x'], df1.iloc[0]['y']]) - midpoint)
    ab_line_rot_y = ab_point_rot[1]
    print(f"AB 라인의 회전된 Y 좌표: {ab_line_rot_y:.2f}")
    
    # y_idx 기준으로 정렬하여 가로줄 그룹 만들기 (AB가 바닥이므로 작은 y부터)
    grid_list.sort(key=lambda g: (g['y_idx'], g['x_idx']))
    
    # 가로줄별로 그룹화
    rows = {}
    for cell in grid_list:
        y_idx = cell['y_idx']
        if y_idx not in rows:
            rows[y_idx] = []
        rows[y_idx].append(cell)
    
    # BL 생성
    grid_cells = []
    
    # y_idx 기준으로 정렬 (작은 값부터 = AB쪽부터)
    sorted_y_indices = sorted(rows.keys())
    print(f"Y 인덱스 범위: {min(sorted_y_indices)} ~ {max(sorted_y_indices)}")
    
    # 세로줄별로 처리
    max_x_idx = max(max(cell['x_idx'] for cell in row_cells) 
                    for row_cells in rows.values())
    
    # 세로줄별로 번호 매기기 (왼쪽에서 오른쪽으로)
    for x_idx in range(max_x_idx + 1):
        bl_i = 1  # 각 세로줄에서 AB쪽(아래)부터 시작
        
    for _y, y_idx in enumerate(sorted_y_indices):
        for x_idx in range(max_x_idx + 1):
            # 현재 y_idx 행에서 x_idx에 해당하는 셀 찾기
            cell_found = None
            for cell in rows[y_idx]:
                if cell['x_idx'] == x_idx:
                    cell_found = cell
                    break
            if cell_found:
                grid_cells.append({
                    'cell_name': f"{prefix}_{x_idx + 1}_{_y + 1}",
                    'vertices': cell_found['vertices'],
                    'area': cell_found['area'],
                    'is_boundary': cell_found.get('is_boundary', False)
                })
    
    return grid_cells


def find_intersecting_cells(grid_cells, line):
    """경계선과 교차하는 셀 찾기"""
    intersecting = []
    for cell in grid_cells:
        if Polygon(cell['vertices']).intersects(line):
            intersecting.append(cell['cell_name'])
    return intersecting


# ====== 보조 유틸: 교차점 추출/정렬/중복제거/시계방향 교정 ======

def _extract_points(geom):
    """Shapely 교차 결과에서 Point만 뽑아 리스트로 반환"""
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, MultiPoint):
        return list(geom.geoms)
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        if len(coords) >= 2:
            return [Point(coords[0]), Point(coords[-1])]
        return []
    if isinstance(geom, MultiLineString):
        pts = []
        for g in geom.geoms:
            pts.extend(_extract_points(g))
        return pts
    if isinstance(geom, GeometryCollection):
        pts = []
        for g in geom.geoms:
            pts.extend(_extract_points(g))
        return pts
    return []


def _dedupe_xy(points, ndigits=6):
    """좌표 중복 제거 (라운딩 기반)"""
    seen = set()
    out = []
    for x, y in points:
        key = (round(x, ndigits), round(y, ndigits))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    return out


def _order_ccw(points):
    """점들을 중심각(atan2) 기준으로 CCW 정렬"""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _polygon_area_signed(points):
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _bottom_left_index_by_AB(points, midpoint_info=None):
    """AB 기준 지역좌표계에서 좌하단 점의 인덱스를 찾음"""
    eps = 1e-9
    if midpoint_info is not None and len(midpoint_info) > 0:
        A = np.array([midpoint_info.iloc[0]['A_x'], midpoint_info.iloc[0]['A_y']])
        B = np.array([midpoint_info.iloc[0]['B_x'], midpoint_info.iloc[0]['B_y']])
        O = (A + B) / 2.0
        theta = math.atan2(B[1] - A[1], B[0] - A[0])
        c, s = math.cos(-theta), math.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        rot = [R @ (np.array(p) - O) for p in points]
    else:
        rot = [np.array(p) for p in points]
    
    # x 최소, x 동률이면 y 최소
    xy = [(i, rp[0], rp[1]) for i, rp in enumerate(rot)]
    xy.sort(key=lambda t: (round(t[1], 8), round(t[2], 8)))
    return xy[0][0]


def _reindex_start_bottom_left_ccw(points, midpoint_info=None):
    """좌하단부터 시작하고 CCW가 되도록 재인덱싱"""
    if len(points) < 3:
        return points
    pts = _order_ccw(points)
    if _polygon_area_signed(pts) < 0:  # 시계방향이면 뒤집기
        pts = pts[:1] + list(reversed(pts[1:]))
    bl_idx = _bottom_left_index_by_AB(pts, midpoint_info)
    return pts[bl_idx:] + pts[:bl_idx]


def _internal_angle_deg(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _cap_vertices_to_five(points, max_vertices=5):
    """거의 일직선인 꼭짓점을 우선 제거하여 최대 5점으로 제한"""
    P = points[:]
    if len(P) <= max_vertices:
        return P

    def prev_i(i, n): return (i - 1) % n
    def next_i(i, n): return (i + 1) % n

    while len(P) > max_vertices:
        n = len(P)
        angles = []
        for i in range(n):
            a = P[prev_i(i, n)]; b = P[i]; c = P[next_i(i, n)]
            angles.append((i, abs(180.0 - _internal_angle_deg(a, b, c))))
        # 180°와 차이가 가장 작은(= 가장 평평한) 점부터 제거 시도
        angles.sort(key=lambda t: t[1])
        removed = False
        for i, _ in angles:
            cand = P[:i] + P[i+1:]
            poly = Polygon(cand)
            if isinstance(poly, Polygon) and (not poly.is_empty) and poly.is_valid:
                P = cand
                removed = True
                break
        if not removed:
            break
    return P


def _clip_cell_minimal(cell_vertices, df_lines, boundary_polygon, midpoint_info=None, eps=1e-9, cap_to_five=True):
    """
    원래 사각형 꼭짓점 중 내부/경계 위 점 + 각 변과 df1/df2 교차점만 사용해 다각형 구성.
    반환되는 좌표는 좌하단부터 CCW.
    """
    rect = Polygon(cell_vertices)

    # 완전 내부 → 원형 유지(정렬만 보정)
    if boundary_polygon.contains(rect):
        ordered = _reindex_start_bottom_left_ccw(list(rect.exterior.coords)[:-1], midpoint_info)
        return Polygon(ordered)

    # 완전 외부 → 없음
    if not rect.intersects(boundary_polygon):
        return None

    # 1) 원래 사각형 꼭짓점 중 내부/경계 점 유지
    pts = []
    buf = boundary_polygon.buffer(eps)
    for (x, y) in rect.exterior.coords[:-1]:
        p = Point(x, y)
        if buf.covers(p):
            pts.append((x, y))

    # 2) 각 변과 df1/df2 교차점 추가
    verts = list(rect.exterior.coords)[:-1]
    for i in range(len(verts)):
        a = verts[i]
        b = verts[(i + 1) % len(verts)]
        edge = LineString([a, b])
        for df in df_lines:
            # 경계선의 끝점들도 명시적으로 확인
            df_coords = list(df.coords)
            
            # 끝점이 셀 변 위에 있는지 확인
            for endpoint in [df_coords[0], df_coords[-1]]:  # A점, B점
                pt = Point(endpoint)
                if edge.distance(pt) < eps:  # 변 위에 있으면
                    pts.append(endpoint)
            
            # 일반 교차점도 추가
            inter = edge.intersection(df)
            for q in _extract_points(inter):
                pts.append((q.x, q.y))

    # 3) 중복 제거 & 최소 점수 확인
    pts = _dedupe_xy(pts, ndigits=6)
    if len(pts) < 3:
        res = rect.intersection(boundary_polygon)
        if isinstance(res, Polygon) and not res.is_empty:
            ordered = _reindex_start_bottom_left_ccw(list(res.exterior.coords)[:-1], midpoint_info)
            return Polygon(ordered)
        return None

    # 4) CCW 정렬 + 좌하단 시작
    ordered = _reindex_start_bottom_left_ccw(pts, midpoint_info)

    # 5) 최대 5점 제한(옵션)
    if cap_to_five and len(ordered) > 5:
        ordered = _cap_vertices_to_five(ordered, max_vertices=5)
        ordered = _reindex_start_bottom_left_ccw(ordered, midpoint_info)

    poly = Polygon(ordered)
    if (not poly.is_valid) or poly.is_empty:
        poly = poly.buffer(0)
    return poly if (isinstance(poly, Polygon) and not poly.is_empty) else None


def simplify_polygon_by_angle(coords, angle_tolerance=5.0):
    """
    (이전 로직) 거의 일직선 구간 단순화 - 사용하지 않지만 보존
    """
    if len(coords) <= 3:
        return coords
    angle_threshold = np.radians(angle_tolerance)
    simplified = []
    i = 0
    while i < len(coords):
        simplified.append(coords[i])
        if i >= len(coords) - 2:
            i += 1
            continue
        j = i + 1
        while j < len(coords) - 1:
            if not is_nearly_straight(coords[i:j+2], angle_threshold):
                break
            j += 1
        if j > i + 1:
            i = j
        else:
            i += 1
    if len(simplified) == 0 or simplified[-1] != coords[-1]:
        simplified.append(coords[-1])
    return simplified


def is_nearly_straight(points, angle_threshold):
    if len(points) < 3:
        return True
    start = np.array(points[0])
    end = np.array(points[-1])
    base_vector = end - start
    base_length = np.linalg.norm(base_vector)
    if base_length < 1e-10:
        return False
    base_unit = base_vector / base_length
    for i in range(1, len(points) - 1):
        point = np.array(points[i])
        point_vector = point - start
        projection = np.dot(point_vector, base_unit)
        projected_point = start + projection * base_unit
        distance = np.linalg.norm(point - projected_point)
        if projection > 0 and projection < base_length:
            angle_approx = np.arctan(distance / min(projection, base_length - projection))
            if angle_approx > angle_threshold:
                return False
    return True


def split_intersecting_cells(grid_cells, boundary_lines, boundary_polygon, midpoint_info=None):
    """교차하는 셀 분할 (최소 꼭짓점 방식) — 나머지 파이프라인/출력은 그대로 유지"""
    # 경계선과 교차하는 셀 찾기 (기존 방식 유지)
    intersecting_cells = set()
    for line in boundary_lines:
        intersecting_cells.update(find_intersecting_cells(grid_cells, line))
    
    inside_polygons = {}
    for cell in grid_cells:
        name = cell['cell_name']
        if name in intersecting_cells:
            clipped = _clip_cell_minimal(cell['vertices'], boundary_lines, boundary_polygon, midpoint_info)
            if clipped and clipped.area > 1e-10:
                # 이미 _clip_cell_minimal에서 좌하단 시작 CCW로 정렬 완료
                inside_polygons[name] = [clipped]
    return intersecting_cells, inside_polygons


def calculate_midpoints(coords):
    """사각형의 각 변 중점 계산"""
    if len(coords) != 4:
        return [""] * 8
    xb, yb = (coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2
    xt, yt = (coords[3][0] + coords[2][0]) / 2, (coords[3][1] + coords[2][1]) / 2
    xl, yl = (coords[0][0] + coords[3][0]) / 2, (coords[0][1] + coords[3][1]) / 2
    xr, yr = (coords[1][0] + coords[2][0]) / 2, (coords[1][1] + coords[2][1]) / 2
    return [str(v) for v in [xb, yb, xt, yt, xl, yl, xr, yr]]


def write_cells_to_csv(grid_cells, inside_polygons, intersecting_cells, boundary_polygon, filename, midpoint_info=None):
    """셀 정보를 CSV로 저장"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 헤더
    headers = ['No', 'BLName', 'Area', 
               'X1coord', 'Y1coord', 'Z1coord',
               'X2coord', 'Y2coord', 'Z2coord', 
               'X3coord', 'Y3coord', 'Z3coord',
               'X4coord', 'Y4coord', 'Z4coord',
               'X5coord', 'Y5coord', 'Z5coord',
               'XTcoord', 'YTcoord', 'ZTcoord',
               'XBcoord', 'YBcoord', 'ZBcoord',
               'XLcoord', 'YLcoord', 'ZLcoord',
               'XRcoord', 'YRcoord', 'ZRcoord',
               'cutVol', 'fillVol', 'YN', 'ClusterName']
    
    rows = []
    idx = 1
    
    # 모든 셀을 딕셔너리로 변환 (교차 셀은 inside_polygons가 대체)
    all_cells = {}
    for cell in grid_cells:
        all_cells[cell['cell_name']] = [Polygon(cell['vertices'])]
    all_cells.update(inside_polygons)

    # 항상 LBL_i_j / RBL_i_j / BL_i_j의 숫자 기준으로 정렬 (1,2,3... 순)
    def _parse_name(nm: str):
        parts = nm.replace('LBL_', '').replace('RBL_', '').replace('BL_', '').split('_')
        try:
            if len(parts) >= 2:
                return (int(parts[-2]), int(parts[-1]))
        except Exception:
            pass
    
        return (10**9, 10**9)

    items_sorted = sorted(all_cells.items(), key=lambda kv: _parse_name(kv[0]))
    
    # CSV 작성
    for bl_name, polygons in items_sorted:
        for poly in polygons:
            coords = list(poly.exterior.coords)[:-1]  # 마지막 중복점 제거
            # 좌표를 좌하단(X1)부터 CCW가 되도록 재정렬 (AB 기준)
            coords = _reindex_start_bottom_left_ccw(coords, midpoint_info)
            
            # YN 판단
            yn_value = 'Y'
            if bl_name in intersecting_cells:
                yn_value = 'N'
            
            # 중점 계산 (정사각형 셀만)
            midpoints = calculate_midpoints(coords) if yn_value == 'Y' else [""] * 8
            
            # 행 데이터 구성
            row = [idx, bl_name, f"{poly.area}"]
            
            # 꼭짓점 좌표 (최대 5개)
            for i in range(5):
                if i < len(coords):
                    row.extend([str(coords[i][0]), str(coords[i][1]), "0.0"])
                else:
                    row.extend(["", "", ""])
            
            # 중점 좌표 (XT, XB, XL, XR)
            row.extend(midpoints[2:4] + ["0.0" if yn_value == 'Y' else ""])  # Top
            row.extend(midpoints[:2] + ["0.0" if yn_value == 'Y' else ""])   # Bottom
            row.extend(midpoints[4:6] + ["0.0" if yn_value == 'Y' else ""])  # Left
            row.extend(midpoints[6:8] + ["0.0" if yn_value == 'Y' else ""])  # Right
            
            # 기타 필드
            row.extend(["", "", yn_value, ""])
            
            rows.append(row)
            idx += 1
    
    # CSV 저장
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def visualize_grid_cells(grid_cells, df0, df1, df2, intersecting_cells=None):
    """격자 셀 시각화"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 셀 그리기
    for cell in grid_cells:
        poly = Polygon(cell['vertices'])
        x, y = poly.exterior.xy
        
        # YN 값에 따라 색상 결정
        if intersecting_cells and cell['cell_name'] in intersecting_cells:
            ax.plot(x, y, 'r-', linewidth=1.5)  # N인 셀은 빨간색
        else:
            ax.plot(x, y, 'k-', linewidth=0.5)  # Y인 셀은 검은색
        
        # 셀 이름 표시
        centroid = poly.centroid
        ax.text(centroid.x, centroid.y, cell['cell_name'], 
                fontsize=6, ha='center', va='center')
    
    # 경계선 그리기
    ax.plot(df1['x'], df1['y'], 'r-', linewidth=2, label='df1 (좌측)')
    ax.plot(df2['x'], df2['y'], 'b-', linewidth=2, label='df2 (우측)')
    
    if df0 is not None and not df0.empty:
        ax.plot(df0['x'], df0['y'], 'g--', linewidth=1, label='df0 (중심선)')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('BL Grid Cells (빨간 테두리 = YN이 N인 셀)')
    plt.tight_layout()
    plt.show()


def reorder_polygon_vertices_to_bottom_left(poly, midpoint_info):
    """폴리곤 꼭짓점을 AB 기준 좌표계에서 좌하단부터 시작하도록 재정렬(CCW 보장)"""
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) < 3:
        return coords
    # 좌하단 시작 + CCW
    ordered = _reindex_start_bottom_left_ccw(coords, midpoint_info)
    return ordered


def main():
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description='BL 셀 생성 프로그램')
    parser.add_argument('-i', '--input', required=True, help='입력 CSV 파일')
    parser.add_argument('-o', '--output', default='output/grid_cells.csv', 
                       help='출력 CSV 파일')
    parser.add_argument('-b', '--blade_width', type=float, default=2.7,
                       help='블레이드 폭 (m) (기본값: 2.7)')
    parser.add_argument('-r', '--overlap_rate', type=float, default=0.2,
                       help='중복도 (0.0 ~ 1.0) (기본값: 0.2)')
    args = parser.parse_args()
    
    # 설정값
    BLADE_WIDTH = args.blade_width  # 블레이드 폭
    OVERLAP_RATE = args.overlap_rate  # 중복도
    CELL_SIZE = (1 - OVERLAP_RATE) * BLADE_WIDTH  # 셀 크기 계산
    
    print(f"블레이드 폭: {BLADE_WIDTH}m")
    print(f"중복도: {OVERLAP_RATE} ({OVERLAP_RATE*100}%)")
    print(f"계산된 셀 크기: {CELL_SIZE}m")
    
    # 1. 데이터 읽기
    df0, df1, df2, ab_info, midpoint_info = read_csv_files(args.input)
    
    # 2. BL prefix 결정 (region 정보 기반)
    basename = os.path.basename(args.input)
    if 'region1' in basename:
        prefix = 'LBL'
    elif 'region2' in basename:
        prefix = 'RBL'
    else:
        prefix = 'BL'  # 기본값
    
    print(f"\n처리 중인 파일: {basename}")
    print(f"BL prefix: {prefix}")
    
    # 3. AB 기준으로 데이터 정렬 (region별 특별 처리 제거)
    print(f"\n▶ AB 기준 데이터 정렬 중...")
    df0, df1, df2 = align_data_to_ab_bottom(df0, df1, df2, ab_info)
    
    # 4. 중심선 재생성 및 노드 처리
    print(f"\n▶ 중심선 재생성 및 노드 처리 중...")
    df0 = create_new_center_line(df1, df2)
    distances = calculate_node_distances(df0)
    df0, df1, df2 = add_nodes(df0, df1, df2, distances, CELL_SIZE)
    
    # 5. 격자 셀 생성 (중점 기준)
    print(f"\n▶ 격자 셀 생성 중...")
    grid_cells = create_grid_cells_from_midpoint(df1, df2, CELL_SIZE, df0, basename, ab_info, midpoint_info, prefix)
    
    # 6. 경계 처리 - midpoint_info 추가 전달
    print(f"\n▶ 경계 처리 중...")
    boundary_polygon = create_boundary_polygon(df1, df2)
    boundary_lines = [
        LineString(list(zip(df1['x'], df1['y']))),
        LineString(list(zip(df2['x'], df2['y'])))
    ]
    
    # midpoint_info 전달
    intersecting_cells, inside_polygons = split_intersecting_cells(
        grid_cells, boundary_lines, boundary_polygon, midpoint_info
    )
    
    # 7. CSV 저장
    print(f"\n▶ CSV 저장 중...")
    write_cells_to_csv(grid_cells, inside_polygons, intersecting_cells,
                      boundary_polygon, args.output, midpoint_info)
    
    # 8. 시각화
    #print(f"\n▶ 시각화 중...")
    #visualize_grid_cells(grid_cells, df0, df1, df2, intersecting_cells)
    
    print(f"\n✅ 처리 완료: {args.output}")
    print(f"   총 {len(grid_cells)}개의 격자 셀 생성")
    print(f"   경계 처리 대상: {len(intersecting_cells)}개 셀")


if __name__ == "__main__":
    main()
