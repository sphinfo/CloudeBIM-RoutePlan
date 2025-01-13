import ifcopenshell
import time
import uuid
from dataclasses import dataclass
from typing import List,Tuple, Optional
import json
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
import math
import open3d as o3d
import numpy as np
import sys
import pyvista as pv
from shapely.geometry import LineString

model_json_file = sys.argv[1]
angle_ratio_input = sys.argv[2]
slope_distance = float(sys.argv[3])
ply_file_input = sys.argv[4]
output_ifcfile = sys.argv[5]

@dataclass
class Point3D:
    x: float
    y: float
    z: float
    
    @classmethod
    def from_list(cls, coords: List[float]):
        return cls(coords[0], coords[1], coords[2])
    def __hash__(self):
        """Point3D 객체를 해시 가능하도록 설정"""
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        """Point3D 객체 간 동등성 비교"""
        if not isinstance(other, Point3D):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y) and math.isclose(self.z, other.z)
@dataclass
class EdgePoints:
    base_point: Point3D
    base_point_index: int
    start_point: Point3D  # 모서리의 시작점
    end_point: Point3D    # 모서리의 끝점
@dataclass
class Polygon:
    points: List[Point3D]
    
    @classmethod
    def from_coordinates(cls, coordinates: List[List[float]]):
        return cls([Point3D.from_list(coord) for coord in coordinates])

@dataclass
class Feature:
    order: int
    polygon: Polygon
    
    @classmethod
    def from_dict(cls, feature_dict: dict):
        order = feature_dict['properties']['order']
        coordinates = feature_dict['geometry']['coordinates'][0]
        
        coordinates = ensure_counter_clockwise(coordinates)
        
        return cls(order, Polygon.from_coordinates(coordinates))
    
@dataclass
class SiteData:
    features: List[Feature]
    slope: dict  # 추가

    @classmethod
    def from_json(cls, json_data: str):
        data = json.loads(json_data)
        features = [Feature.from_dict(feature) for feature in data['features']]
        # 정렬 제거: 데이터가 이미 올바른 순서로 주어졌다고 가정
        slope = data.get('slope', {})  # 추가
        return cls(features, slope)

    def get_highest_order_feature(self) -> Feature:
        """가장 높은 Order를 가진 Feature 반환"""
        return max(self.features, key=lambda x: x.order)


@dataclass
class SlopeSurface:
    base_polygon: Polygon
    distance: float  # 사면 거리 (예: 36.056m)
    angle_ratio: tuple  # 경사 비율 (1, 1.5)
    resolution_factor: float = 1.0  # 가중치 조절 factor (기본값 1.0)
    edge_points: List[EdgePoints] = None  # EdgePoints 리스트 추가
    def calculate_points_count(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """두 벡터 사이의 각도에 따른 점 개수 계산 (가중치 적용)"""
        # 두 벡터 사이의 각도 계산
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_degrees = np.degrees(angle)
        
        # 기본 개수 계산 (30개/90도 기준)
        base_count = (angle_degrees / 90.0) * 30
        
        # 가중치 적용
        points_count = int(base_count * self.resolution_factor)
        points_count = max(points_count, 2)  # 최소 2개 보장
        
        # print(f"\n각도 기반 점 개수 계산:")
        # print(f"벡터 사이 각도: {angle_degrees:.2f}도")
        # print(f"가중치: {self.resolution_factor}")
        # print(f"기본 개수: {base_count:.1f}")
        # print(f"최종 점 개수: {points_count}")
        
        return points_count
    # 사면 방향 계산
    def calculate_triangle_dimensions(self) -> tuple:
        """A-B-C 삼각형의 실제 치수 계산, 아래쪽 방향 지원"""
        horizontal_ratio = self.angle_ratio[1]  
        vertical_ratio = self.angle_ratio[0]    

        # 비율의 총 길이를 정규화
        total_ratio_square = horizontal_ratio**2 + vertical_ratio**2
        scaling_factor = self.distance / math.sqrt(total_ratio_square)

        # 수평 및 수직 거리 계산
        horizontal_length = horizontal_ratio * scaling_factor
        vertical_length = vertical_ratio * scaling_factor

   
        
        return horizontal_length, vertical_length

    def get_outward_vectors(self, point_idx: int, total_points: int) -> tuple:
        """현재 점에서의 두 바깥쪽 방향 벡터 계산 (폐합점 고려 및 오목 구간 처리)"""
        points = self.base_polygon.points
        actual_points = total_points - 1

        if point_idx == actual_points:
            return self.get_outward_vectors(0, total_points)

        # 현재, 이전, 다음 점
        current = points[point_idx]
        next_point = points[(point_idx + 1) % actual_points]
        prev_point = points[point_idx - 1 if point_idx > 0 else actual_points - 1]

        # 벡터 계산
        v1 = np.array([current.x - prev_point.x, current.y - prev_point.y])
        v2 = np.array([next_point.x - current.x, next_point.y - current.y])

        dx_next, dy_next = v2
        direction_next = np.array([-dy_next, dx_next]) / np.linalg.norm(v2)

        dx_prev, dy_prev = v1
        direction_prev = np.array([-dy_prev, dx_prev]) / np.linalg.norm(v1)
        return direction_next, direction_prev
        
    def generate_slope(self, terrain_points: List[Point3D]) -> dict:
        """지형 데이터를 기반으로 경사면 생성"""
        
        terrain_array = np.array([[p.x, p.y, p.z] for p in terrain_points])
        terrain_2d = terrain_array[:, :2]
        delaunay = Delaunay(terrain_2d)
        horizontal_dist, vertical_dist = self.calculate_triangle_dimensions()
        base_to_slope = {}
        self.edge_points = []
        total_points = len(self.base_polygon.points)
        for i in range(total_points):
            base_point = self.base_polygon.points[i]
            dir_next, dir_prev = self.get_outward_vectors(i, total_points)
            points_count = self.calculate_points_count(dir_next, dir_prev)
            
            # 시작점 계산과 방향 조정
            _, next_reversed = self.adjust_point_with_ray_casting(
                base_point, dir_next, delaunay, terrain_array, horizontal_dist, vertical_dist
            )
            start_x = base_point.x + dir_next[0] * horizontal_dist
            start_y = base_point.y + dir_next[1] * horizontal_dist
            
            start_z = base_point.z + (vertical_dist if next_reversed else vertical_dist *-1)
            start_point = Point3D(start_x, start_y, start_z)


            # 끝점 계산과 방향 조정
            _, prev_reversed = self.adjust_point_with_ray_casting(
                base_point, dir_prev, delaunay, terrain_array, horizontal_dist, vertical_dist
            )

            end_x = base_point.x + dir_prev[0] * horizontal_dist
            end_y = base_point.y + dir_prev[1] * horizontal_dist
            end_z = base_point.z + (vertical_dist if prev_reversed else vertical_dist *-1)
            end_point = Point3D(end_x, end_y, end_z)
            slope_points = [start_point]
            for t in np.linspace(0, 1, points_count):
                
                mid_dir = dir_next * (1 - t) + dir_prev * t
                # 보간된 중간 방향 계산
                mid_dir /= np.linalg.norm(mid_dir)
                
                # 레이캐스팅 수행
                _, is_down = self.adjust_point_with_ray_casting(
                    base_point, mid_dir, delaunay, terrain_array, horizontal_dist, vertical_dist
                )
                mid_x = base_point.x + mid_dir[0] * horizontal_dist
                mid_y = base_point.y + mid_dir[1] * horizontal_dist
                mid_z = base_point.z + (vertical_dist if is_down else vertical_dist *-1)
                
                slope_points.append(Point3D(mid_x, mid_y, mid_z))
            
            if end_point not in slope_points:
                slope_points.append(end_point)
            
            # Edge 정보 저장
            if True:
                edge_info = EdgePoints(
                    base_point=base_point,
                    base_point_index=i,
                    start_point=start_point,
                    end_point=end_point
                )
                self.edge_points.append(edge_info)
            base_to_slope[base_point] = slope_points
        return base_to_slope
    def adjust_point_with_ray_casting(self, base_point, direction, delaunay, terrain_array, horizontal_dist, vertical_dist):
        """
        레이캐스팅을 통해 주어진 방향으로 확장된 점을 지형 포인트와 맞닿게 보정.
        교차점이 없을 경우 위/아래 방향으로 재시도.
        Args:
            base_point: 기준점(Point3D).
            direction: 확장 방향 벡터.
            delaunay: Delaunay 삼각망.
            terrain_array: 지형 데이터를 나타내는 numpy 배열.
            horizontal_dist: 수평 거리.
            vertical_dist: 수직 거리.
        Returns:
            tuple[Point3D, bool]: 보정된 점과 방향 반전 여부 (False: 위쪽 성공, True: 아래쪽 성공)
        """
        # 경사각 기반 Z 방향 계산
        
        slope_magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
        z_slope_ratio = vertical_dist / horizontal_dist  # Z 방향 경사 비율
        slope_z = -z_slope_ratio * slope_magnitude  # 음수는 아래 방향
        
        # 레이 시작점 설정
        ray_origin = np.array([base_point.x, base_point.y, base_point.z])
        
        down_direction = np.array([direction[0], direction[1], slope_z])
        down_direction /= np.linalg.norm(down_direction)  # 단위 벡터화
        
        
        intersection = ray_cast_delaunay(ray_origin, down_direction, delaunay, terrain_array)
        
        if intersection is not None:
            return Point3D(*intersection), False
        else:
            up_direction = np.array([direction[0], direction[1], -slope_z])
            up_direction /= np.linalg.norm(up_direction)
            
            intersection = ray_cast_delaunay(ray_origin, up_direction, delaunay, terrain_array)
            
            if intersection is not None:
                return Point3D(*intersection), True
            else:
                # 양방향 모두 실패한 경우 예외 발생
                raise ValueError("No intersection found in either up or down direction with terrain")
            
def ray_cast_delaunay(ray_origin, ray_direction, delaunay, terrain_array):
    """
    Delaunay 삼각망과 레이의 교차점을 계산.
    Args:
        ray_origin: 레이의 시작점 (x, y, z).
        ray_direction: 레이의 방향 벡터.
        delaunay: Delaunay 삼각망.
        terrain_array: 지형 데이터를 나타내는 numpy 배열.
    Returns:
        np.ndarray or None: 교차점 좌표 (x, y, z) 또는 None.
    """
    for simplex in delaunay.simplices:
        # 삼각형의 세 점 가져오기
        triangle = terrain_array[simplex]
        intersection = ray_triangle_intersection(ray_origin, ray_direction, triangle)
        if intersection is not None:
            return intersection
    return None
def ray_triangle_intersection(ray_origin, ray_direction, triangle):
    """
    레이와 삼각형 간의 교차점을 계산.
    Args:
        ray_origin: 레이의 시작점.
        ray_direction: 레이의 방향 벡터.
        triangle: 삼각형의 세 점.
    Returns:
        np.ndarray or None: 교차점 좌표 또는 None.
    """
    p1, p2, p3 = triangle
    edge1 = p2 - p1
    edge2 = p3 - p1
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -1e-10 < a < 1e-10:
        return None  # 평행

    f = 1.0 / a
    s = ray_origin - p1
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)
    if t > 1e-10:  # 레이가 삼각형과 교차
        return ray_origin + t * ray_direction
    return None

def slerp(v1, v2, t):
    # 두 벡터 사이의 각도 계산
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)  # numerical stability
    theta = np.arccos(dot)
    
    if theta < 1e-7:  # 각도가 매우 작으면 선형 보간
        return v1 * (1 - t) + v2 * t
    
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) * v1 + np.sin(t * theta) * v2) / sin_theta

def create_guid():
    """22자리 GlobalId 생성"""
    return str(uuid.uuid4()).replace('-', '')[:22]

def create_surface_faces(ifc_file, points_list):
    """점들의 리스트로부터 IFC 면을 생성"""
    cartesian_points = [
        ifc_file.create_entity("IfcCartesianPoint", Coordinates=[
            float(point.x), float(point.y), float(point.z)
        ]) for point in points_list
    ]
    
    poly_loop = ifc_file.create_entity("IfcPolyLoop", Polygon=cartesian_points)
    face_bound = ifc_file.create_entity("IfcFaceOuterBound", Bound=poly_loop, Orientation=True)
    return ifc_file.create_entity("IfcFace", Bounds=[face_bound])

def ensure_counter_clockwise(coordinates):
    """
    Ensure the given coordinates are in counter-clockwise order.
    If not, reverse the order to make them counter-clockwise.
    """
    polygon = ShapelyPolygon(coordinates)
    
    if polygon.exterior.is_ccw:
        # Reverse the coordinates if they are not counter-clockwise
        return list(reversed(coordinates))
    return coordinates


def reorder_points_by_start(points, reference_point):
    """기준점에 가장 가까운 점을 시작점으로 배열 재정렬"""
    # 기준점과 가장 가까운 점의 인덱스 찾기
    min_dist = float('inf')
    start_idx = 0
    
    for i, point in enumerate(points):
        dist = math.sqrt((point.x - reference_point.x)**2 + 
                        (point.y - reference_point.y)**2 + 
                        (point.z - reference_point.z)**2)
        if dist < min_dist:
            min_dist = dist
            start_idx = i
    
    # 찾은 시작점을 기준으로 배열 재정렬
    return points[start_idx:] + points[:start_idx]


def calculate_distance_2d(p1, p2):
    """3D 거리 계산."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def generate_side_faces_dynamic(current_points, next_points):
    """
    동적으로 위, 아래 인덱스를 선택하며 삼각망 생성.
    - current_points: 현재 층의 점 리스트.
    - next_points: 다음 층의 점 리스트.
    """
    side_faces = []
    i, j = 0, 0

    while i < len(current_points) - 1 or j < len(next_points) - 1:
        if i < len(current_points) - 1 and j < len(next_points) - 1:
            # 양쪽에 다음 점이 있는 경우 거리 비교
            dist1 = calculate_distance_2d(current_points[i + 1], next_points[j])  # 위쪽 다음 점
            dist2 = calculate_distance_2d(current_points[i], next_points[j + 1])  # 아래쪽 다음 점
            if dist1 < dist2:
                # 위쪽 점을 선택
                side_faces.append([current_points[i], next_points[j], current_points[i + 1]])
                i += 1
            else:
                # 아래쪽 점을 선택
                side_faces.append([current_points[i], next_points[j], next_points[j + 1]])
                j += 1
        elif i < len(current_points) - 1:
            # 아래쪽이 끝난 경우 위쪽 점만 추가
            side_faces.append([current_points[i], next_points[j], current_points[i + 1]])
            i += 1
        elif j < len(next_points) - 1:
            # 위쪽이 끝난 경우 아래쪽 점만 추가
            side_faces.append([current_points[i], next_points[j], next_points[j + 1]])
            j += 1

    return side_faces
def sort_points_clockwise(points: List[Point3D]) -> List[Point3D]:
    """
    주어진 Point3D 객체들을 시계 방향으로 정렬합니다.
    - points: Point3D 객체의 리스트.
    """
    # NumPy 배열로 변환
    points_array = np.array([[point.x, point.y, point.z] for point in points])

    # 중심 계산
    centroid = np.mean(points_array, axis=0)

    # 중심을 기준으로 각도 계산
    angles = np.arctan2(points_array[:, 1] - centroid[1], points_array[:, 0] - centroid[0])

    # 각도를 기준으로 정렬
    sorted_indices = np.argsort(angles)
    sorted_points_array = points_array[sorted_indices]

    # 다시 Point3D 객체로 변환
    sorted_points = [Point3D(x, y, z) for x, y, z in sorted_points_array]

    return sorted_points


def create_side_faces(site_data: SiteData, ifc_file) -> List:
    """층 간 연결 규칙에 따라 삼각분할 생성"""
    all_faces = []

    # `features`가 1개 이하일 경우 처리
    if len(site_data.features) < 2:
        if len(site_data.features) == 1:
            # print("features가 1개입니다. 처리할 삼각형이 없습니다.")
            return []
        else:
            raise ValueError("features가 비어 있습니다. 처리할 데이터가 없습니다.")

    # 모든 층 사이 처리 (마지막 층까지 포함)
    for i in range(len(site_data.features) - 1):
        current_feature = site_data.features[i]
        next_feature = site_data.features[i + 1]
        
        current_points = current_feature.polygon.points
        next_points = next_feature.polygon.points

        side_faces = generate_side_faces_dynamic(current_points, next_points)

        # 생성된 삼각형을 IFC 면으로 변환
        for face in side_faces:
            surface_face = create_surface_faces(ifc_file, face)
            all_faces.append(surface_face)
            
    return all_faces

def find_closest_point(p1, p2, second_last_feature):
    def distance_with_intersection_check(p):
        # p1과 p까지의 선, p2와 p까지의 선 모두 체크
        test_line1 = LineString([(p1.x, p1.y), (p.x, p.y)])
        test_line2 = LineString([(p2.x, p2.y), (p.x, p.y)])
        
        # second_last_feature의 연속된 두 점 사이의 선분들과 교차 검사
        edges = zip(second_last_feature[:-1], second_last_feature[1:])
        for e1, e2 in edges:
            edge_line = LineString([(e1.x, e1.y), (e2.x, e2.y)])
            # 둘 중 하나라도 교차하면 제외
            if test_line1.crosses(edge_line) or test_line2.crosses(edge_line):
                return float('inf')
        
        return math.sqrt((p.x - p1.x)**2 + (p.y - p1.y)**2)
    
    return min(second_last_feature, key=distance_with_intersection_check)

def create_slope_faces(ifc_file, base_to_slope: dict) -> List:
    """base_point와 확장된 slope_points만을 사용하여 삼각망 생성"""
    all_faces = []
    
    for idx, (base_point, slope_points) in enumerate(base_to_slope.items()):
        slp =  slope_points
        # face 생성
        for x in range(len(slp) - 1):
            face = create_surface_faces(ifc_file, [slp[x], slp[x + 1], base_point])
            all_faces.append(face)

    return all_faces


def divide_line(start, end, interval):
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # 분할 개수 계산
    segment_count = max(1, int(distance / interval))
    points = []
    
    # 모든 점 생성 (시작점, 중간점들, 끝점)
    for j in range(segment_count + 1):
        t = j / segment_count
        x = start.x + dx * t
        y = start.y + dy * t
        z = start.z + dz * t
        points.append(Point3D(x, y, z))
               
    return points


def create_section_faces(ifc_file, highest_feature: Feature, slope_surface: SlopeSurface, interval: float = 10.0):
    """섹션 단위로 Delaunay 삼각망을 생성하여 모서리 끝점과 시작점을 연결하고 최상층 부지 외곽선에 초록색 스타일 적용"""
    all_faces = []
    base_points = highest_feature.polygon.points
    edge_points = slope_surface.edge_points

    # 섹션 처리
    for i in range(len(base_points)):
        # 현재와 다음 base points
        b1 = base_points[i]
        b2 = base_points[(i + 1) % len(base_points)]

        # 현재 모서리의 끝점과 다음 모서리의 시작점
        p1 = edge_points[i].start_point
        p2 = edge_points[(i + 1) % len(edge_points)].end_point

        # 두 선을 10m 간격으로 분할
        top_points = divide_line(p1, p2, interval)
        bottom_points = divide_line(b1, b2, interval)

        # 섹션 내 모든 점
        section_points = top_points + bottom_points

        # Delaunay 삼각망 생성
        points_2d = np.array([[p.x, p.y] for p in section_points])
        delaunay_tri = Delaunay(points_2d[::-1])

        # 삼각형 생성
        for simplex in delaunay_tri.simplices:
            p0, p1, p2 = [section_points[idx] for idx in simplex]
            face = create_surface_faces(ifc_file, [p2, p0, p1])
            all_faces.append(face)

 
    return all_faces


def filter_points_within_distance(
    polygon_points: List[Point3D],
    terrain_points: List[Point3D],
    horizontal_ratio: float
) -> Tuple[List[Point3D], List[float]]:
    """
    폴리곤으로부터 특정 거리 이내에 있는 terrain_points를 필터링.
    거리 기준은 horizontal_ratio와 vertical_ratio를 기반으로 자동 계산하며,
    선분 범위 내부의 포인트도 포함합니다.

    Args:
        polygon_points: 폴리곤을 구성하는 Point3D 리스트.
        terrain_points: 필터링할 지형 Point3D 리스트.
        horizontal_ratio: 수평 거리 비율 (horizontal:vertical = x:1).
        vertical_ratio: 수직 거리 비율 (1로 고정).

    Returns:
        filtered_points: 필터링된 Point3D 리스트.
    """
    # 1. base_points와 terrain_points의 z값 차이 계산
    base_min_z = min(p.z for p in polygon_points)
    base_max_z = max(p.z for p in polygon_points)
    terrain_min_z = min(p.z for p in terrain_points)
    terrain_max_z = max(p.z for p in terrain_points)

    # 최대 높이 차 계산
    max_height_diff = max(abs(base_min_z - terrain_min_z), abs(base_max_z - terrain_max_z))

    # 수평 거리 계산 (horizontal_ratio : vertical_ratio)
    max_distance = max_height_diff * horizontal_ratio*2
    
    # 2. 폴리곤의 모든 선분 구성
    segments = []
    for i in range(len(polygon_points)):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % len(polygon_points)]
        segments.append((p1, p2))

    # 3. terrain_points 필터링
    filtered_points = []
    for terrain_point in terrain_points:
        # 각 선분에 대해 점과의 최소 거리 계산
        min_distance = float('inf')

        for p1, p2 in segments:
            # 선분의 벡터
            segment_vector = Point3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
            segment_length_squared = segment_vector.x**2 + segment_vector.y**2 + segment_vector.z**2

            if segment_length_squared == 0:
                # p1과 p2가 같은 점인 경우
                distance = math.sqrt(
                    (terrain_point.x - p1.x)**2 +
                    (terrain_point.y - p1.y)**2 +
                    (terrain_point.z - p1.z)**2
                )
            else:
                # 점에서 선분까지의 수직 거리 계산
                t = max(0, min(1, (
                    (terrain_point.x - p1.x) * segment_vector.x +
                    (terrain_point.y - p1.y) * segment_vector.y +
                    (terrain_point.z - p1.z) * segment_vector.z
                ) / segment_length_squared))

                # 선분 위의 가장 가까운 점 계산
                closest_x = p1.x + t * segment_vector.x
                closest_y = p1.y + t * segment_vector.y
                closest_z = p1.z + t * segment_vector.z

                # 거리 계산
                distance = math.sqrt(
                    (terrain_point.x - closest_x)**2 +
                    (terrain_point.y - closest_y)**2 +
                    (terrain_point.z - closest_z)**2
                )

            min_distance = min(min_distance, distance)
        
        # 최대 거리와 선분 내 거리 조건을 모두 만족하는 경우 필터링 리스트에 추가
        if min_distance <= max_distance:
            filtered_points.append(terrain_point)

    return filtered_points,max_height_diff

def read_ply_with_open3d(file_path):
    """Open3D로 PLY 파일 읽어서 Point3D 객체 리스트로 변환"""
    pcd = o3d.io.read_point_cloud(file_path)
    np_points = np.asarray(pcd.points)
    
    # numpy 배열을 Point3D 객체 리스트로 변환
    point3d_list = [Point3D(x=float(point[0]), 
                           y=float(point[1]), 
                           z=float(point[2])) for point in np_points]
    
    return point3d_list


def remove_unused_polyloops(ifc_file):
    # 모든 PolyLoop 수집
    all_polyloops = set(ifc_file.by_type("IfcPolyLoop"))
    
    # IFCCLOSEDSHELL에서 사용 중인 PolyLoop 수집
    used_polyloops = set()
    for shell in ifc_file.by_type("IfcClosedShell"):
        for face in shell.CfsFaces:
            for bound in face.Bounds:
                if bound.Bound.is_a("IfcPolyLoop"):
                    used_polyloops.add(bound.Bound)
    
    # 사용되지 않은 PolyLoop 찾기
    unused_polyloops = all_polyloops - used_polyloops
    
    # 사용되지 않은 PolyLoop와 관련 점들 제거
    for polyloop in unused_polyloops:
        # PolyLoop에 연결된 점들 제거 (다른 곳에서 사용되지 않는 경우만)
        for point in polyloop.Polygon:
            if len(ifc_file.get_inverse(point)) <= 1:
                ifc_file.remove(point)
        # PolyLoop 제거
        ifc_file.remove(polyloop)
    return len(unused_polyloops)

def create_civil3d_style_ifc_with_slope(site_data: SiteData, slope_info: dict, file_path="./phase2/p1.ifc"):
    # [IFC 파일 생성 및 기본 설정]
    ifc_file = ifcopenshell.file(schema="IFC2X3")
    
    # 기본 엔티티 생성
    person = ifc_file.create_entity("IfcPerson", GivenName="User", FamilyName="Sample")
    organization = ifc_file.create_entity("IfcOrganization", Name="Sample Organization")
    person_and_org = ifc_file.create_entity("IfcPersonAndOrganization", ThePerson=person, TheOrganization=organization)
    application = ifc_file.create_entity(
        "IfcApplication",
        ApplicationDeveloper=organization,
        Version="1.0",
        ApplicationFullName="Civil3D Style Application",
        ApplicationIdentifier="CA",
    )
    owner_history = ifc_file.create_entity(
        "IfcOwnerHistory",
        OwningUser=person_and_org,
        OwningApplication=application,
        State="READWRITE",
        ChangeAction="ADDED",
        LastModifiedDate=int(time.time()),
    )

    # 프로젝트 생성
    project = ifc_file.create_entity(
        "IfcProject",
        GlobalId=create_guid(),
        OwnerHistory=owner_history,
        Name="Civil3DProject"
    )

    # 단위 설정
    length_unit = ifc_file.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    area_unit = ifc_file.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
    volume_unit = ifc_file.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
    plane_angle_unit = ifc_file.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
    
    unit_assignment = ifc_file.create_entity(
        "IfcUnitAssignment", 
        Units=[length_unit, area_unit, volume_unit, plane_angle_unit]
    )
    project.UnitsInContext = unit_assignment

    # 좌표계 설정
    axis_placement = ifc_file.create_entity(
        "IfcAxis2Placement3D",
        Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=[0., 0., 0.]),
        Axis=ifc_file.create_entity("IfcDirection", DirectionRatios=[0., 0., 1.]),
        RefDirection=ifc_file.create_entity("IfcDirection", DirectionRatios=[1., 0., 0.])
    )
    
    context = ifc_file.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model",
        ContextIdentifier="Body",
        CoordinateSpaceDimension=3,
        Precision=1.0E-5,
        WorldCoordinateSystem=axis_placement
    )

    body_subcontext = ifc_file.create_entity(
        "IfcGeometricRepresentationSubContext",
        ContextIdentifier="Body",
        ContextType="Model",
        ParentContext=context,
        TargetView="MODEL_VIEW"
    )
    project.RepresentationContexts = [context]

    building_placement = ifc_file.create_entity(
        "IfcLocalPlacement",
        RelativePlacement=axis_placement
    )
    
    building = ifc_file.create_entity(
        "IfcBuilding",
        GlobalId=create_guid(),
        OwnerHistory=owner_history,
        Name="CivilModel",
        ObjectPlacement=building_placement,
        CompositionType="ELEMENT",
        ElevationOfRefHeight=0.0,
        ElevationOfTerrain=0.0
    )

    ifc_file.create_entity(
        "IfcRelAggregates",
        GlobalId=create_guid(),
        OwnerHistory=owner_history,
        Name="ProjectContainer",
        RelatingObject=project,
        RelatedObjects=[building]
    )
    
    # 바닥면 생성 (Order 1)
    base_feature = site_data.features[0]
    base_points = base_feature.polygon.points

    t_points = read_ply_with_open3d(ply_file_input)
    ratio = slope_info["angle"].split(":")
    terrain_points, _ = filter_points_within_distance(base_points, t_points, float(ratio[1]))

    try:
        polydata = pv.PolyData(np.array([[p.x, p.y, p.z] for p in base_points]))
        triangulated = polydata.delaunay_2d()

        # 외곽선 정의
        polygon_coords = [(p.x, p.y) for p in base_points]
        polygon = ShapelyPolygon(polygon_coords)

        # 삼각형 정보 추출
        faces = triangulated.faces.reshape((-1, 4))  # [3, idx0, idx1, idx2]

        # 삼각형 생성
        base_faces = []
        for face in faces:
            idx0, idx1, idx2 = face[1:]  # 첫 번째 값은 점의 개수(3)
            p0 = base_points[idx0]
            p1 = base_points[idx1]
            p2 = base_points[idx2]

            # 삼각형 중심점 계산
            centroid_x = (p0.x + p1.x + p2.x) / 3
            centroid_y = (p0.y + p1.y + p2.y) / 3
            centroid = ShapelyPoint(centroid_x, centroid_y)

            # 삼각형 중심이 폴리곤 내부에 있는지 확인
            if polygon.contains(centroid):
                base_faces.append(create_surface_faces(ifc_file, [p0, p1, p2]))

        base_closed_shell = ifc_file.create_entity("IfcClosedShell", CfsFaces=base_faces)
        base_surface_model = ifc_file.create_entity("IfcShellBasedSurfaceModel", SbsmBoundary=[base_closed_shell])

        base_element = ifc_file.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=create_guid(),
            OwnerHistory=owner_history,
            Name=f"Base_Layer",
            ObjectPlacement=building_placement,
            Representation=ifc_file.create_entity(
                "IfcProductDefinitionShape",
                Representations=[
                    ifc_file.create_entity(
                        "IfcShapeRepresentation",
                        ContextOfItems=body_subcontext,
                        RepresentationIdentifier="Body",
                        RepresentationType="SurfaceModel",
                        Items=[base_surface_model]
                    )
                ]
            )
        )

        ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=create_guid(),
            OwnerHistory=owner_history,
            RelatingStructure=building,
            RelatedElements=[base_element]
        )
    except Exception as e:
        print(f"바닥층 삼각망 생성 오류: {e}")

    

    try:
        if slope_distance != 0:
            highest_order_feature = site_data.get_highest_order_feature()
            # print(f"Generating slope for Feature Order: {highest_order_feature.order}")
            angle_data  = slope_info["angle"].split(":")

            slope_surface = SlopeSurface(
                base_polygon=highest_order_feature.polygon,
                distance=slope_info["distance"],
                angle_ratio=(float(angle_data[0]), float(angle_data[1]))
            )   
            
            base_to_slope = slope_surface.generate_slope(terrain_points)
            
            # Slope Faces 생성
            slope_faces = create_slope_faces(ifc_file, base_to_slope)
            slope_closed_shell = ifc_file.create_entity("IfcClosedShell", CfsFaces=slope_faces)
            slope_surface_model = ifc_file.create_entity("IfcShellBasedSurfaceModel", SbsmBoundary=[slope_closed_shell])

            slope_element = ifc_file.create_entity(
                "IfcBuildingElementProxy",
                GlobalId=create_guid(),
                OwnerHistory=owner_history,
                Name="Slope_Surface",
                ObjectPlacement=building_placement,
                Representation=ifc_file.create_entity(
                    "IfcProductDefinitionShape",
                    Representations=[
                        ifc_file.create_entity(
                            "IfcShapeRepresentation",
                            ContextOfItems=body_subcontext,
                            RepresentationIdentifier="Body",
                            RepresentationType="SurfaceModel",
                            Items=[slope_surface_model]
                        )
                    ]
                )
            )

            ifc_file.create_entity(
                "IfcRelContainedInSpatialStructure",
                GlobalId=create_guid(),
                OwnerHistory=owner_history,
                RelatingStructure=building,
                RelatedElements=[slope_element]
            )
    except Exception as e:
        print(f"사면 삼각망 생성 오류: {e}")

    try:
        if slope_distance != 0:            
            # 단면 생성
            section_faces = create_section_faces(ifc_file, highest_order_feature, slope_surface)
            section_closed_shell = ifc_file.create_entity("IfcClosedShell", CfsFaces=section_faces)
            section_surface_model = ifc_file.create_entity("IfcShellBasedSurfaceModel", SbsmBoundary=[section_closed_shell])


            section_element = ifc_file.create_entity(
                "IfcBuildingElementProxy",
                GlobalId=create_guid(),
                OwnerHistory=owner_history,
                Name="Section_Lines",
                ObjectPlacement=building_placement,
                Representation=ifc_file.create_entity(
                    "IfcProductDefinitionShape",
                    Representations=[
                        ifc_file.create_entity(
                            "IfcShapeRepresentation",
                            ContextOfItems=body_subcontext,
                            RepresentationIdentifier="Body",
                            RepresentationType="SurfaceModel",
                            Items=[section_surface_model]
                        )
                    ]
                )
            )

            ifc_file.create_entity(
                "IfcRelContainedInSpatialStructure",
                GlobalId=create_guid(),
                OwnerHistory=owner_history,
                RelatingStructure=building,
                RelatedElements=[section_element]
            )
    except Exception as e:
        print(f"단면 생성 오류: {e}")

    
    try:
        
        side_faces = create_side_faces(site_data, ifc_file)
        
        side_closed_shell = ifc_file.create_entity("IfcClosedShell", CfsFaces=side_faces)
        side_surface_model = ifc_file.create_entity("IfcShellBasedSurfaceModel", SbsmBoundary=[side_closed_shell])
        
        side_element = ifc_file.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=create_guid(),
            OwnerHistory=owner_history,
            Name="Side_Layers",
            ObjectPlacement=building_placement,
            Representation=ifc_file.create_entity(
                "IfcProductDefinitionShape",
                Representations=[
                    ifc_file.create_entity(
                        "IfcShapeRepresentation",
                        ContextOfItems=body_subcontext,
                        RepresentationIdentifier="Body",
                        RepresentationType="SurfaceModel",
                        Items=[side_surface_model]
                    )
                ]
            )
        )
        
        ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=create_guid(),
            OwnerHistory=owner_history,
            RelatingStructure=building,
            RelatedElements=[side_element]
            )
    except Exception as e:
        print(f"측면부 삼각망 생성 오류: {e}")
        
    remove_unused_polyloops(ifc_file) 
    ifc_file.write(file_path)
    

if __name__ == "__main__":
    # JSON 데이터 읽기
    with open(model_json_file, 'r') as file:
        json_data = json.load(file)
    
    # SiteData 객체 생성
    site_data = SiteData.from_json(json.dumps(json_data))  # JSON 데이터를 문자열로 변환하여 처리

    # 경사 정보 추출
    slope_info = {
        "distance": slope_distance,
        "angle": angle_ratio_input
    }

    # IFC 파일 생성 함수 호출
    create_civil3d_style_ifc_with_slope(site_data, slope_info,output_ifcfile)
