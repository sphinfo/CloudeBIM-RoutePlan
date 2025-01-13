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
ply_file_input = sys.argv[3]
output_file = sys.argv[4]

@dataclass
class Point3D:
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x) ** 2 + 
                    (self.y - other.y) ** 2 + 
                    (self.z - other.z) ** 2)

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
    is_reverse: bool

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
    def adjust_z_direction(self, z_value: float, terrain_z: float) -> float:
        return abs(z_value - terrain_z) < abs(-z_value - terrain_z)
            
    def calculate_points_count(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """두 벡터 사이의 각도에 따른 점 개수 계산 (가중치 적용)"""
        # 두 벡터 사이의 각도 계산
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_degrees = np.degrees(angle)
        
        # 기본 개수 계산 (30개/90도 기준)
        base_count = (angle_degrees / 90.0) * 30
        
        # 가중치 적용
        # print(f"base count : {base_count}")
        points_count =  math.ceil(base_count * self.resolution_factor)
        
        
        #print(f"\n각도 기반 점 개수 계산:")
        #print(f"벡터 사이 각도: {angle_degrees:.2f}도")
        #print(f"가중치: {self.resolution_factor}")
        #print(f"기본 개수: {base_count:.1f}")
        #print(f"최종 점 개수: {points_count}")
        
        return points_count
    # 사면 방향 계산
    def calculate_triangle_dimensions(self) -> tuple:
        """A-B-C 삼각형의 실제 치수 계산, 아래쪽 방향 지원"""
        horizontal_ratio = self.angle_ratio[1]  # 예: 1
        vertical_ratio = self.angle_ratio[0]    # 예: 1.5

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

        # 외적 계산 (볼록/오목 확인)
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        is_reverse = cross_product >= 0
        # 기본 법선 벡터 계산 (볼록 구간)
        dx_next, dy_next = v2
        direction_next = np.array([-dy_next, dx_next]) / np.linalg.norm(v2)

        dx_prev, dy_prev = v1
        direction_prev = np.array([-dy_prev, dx_prev]) / np.linalg.norm(v1)
        
        return direction_next, direction_prev, is_reverse


    
    
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
            tuple[Point3D, bool]: 보정된 점과 방향 반전 여부 (True: 위쪽 성공, False: 아래쪽 성공)
        """
        # 경사각 기반 Z 방향 계산
        slope_magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
        z_slope_ratio = vertical_dist / horizontal_dist  # Z 방향 경사 비율
        slope_z = -z_slope_ratio * slope_magnitude  # 음수는 아래 방향
        
        # 레이 시작점 설정
        ray_origin = np.array([base_point.x, base_point.y, base_point.z])
        
        # 아래 방향으로 첫 시도
        down_direction = np.array([direction[0], direction[1], slope_z])
        down_direction /= np.linalg.norm(down_direction)  # 단위 벡터화
        
        # 아래 방향 레이캐스팅 시도
        
        intersection = ray_cast_delaunay(ray_origin, down_direction, delaunay, terrain_array)
        
        if intersection is not None:
            return Point3D(*intersection), False
        else:
            
            # 위 방향으로 재시도 (z만 반전)
            up_direction = np.array([direction[0], direction[1], -slope_z])
            up_direction /= np.linalg.norm(up_direction)
            
            intersection = ray_cast_delaunay(ray_origin, up_direction, delaunay, terrain_array)
            
            if intersection is not None:
                return Point3D(*intersection), True
            else:
                # 양방향 모두 실패한 경우 예외 발생
                raise ValueError("No intersection found in either up or down direction with terrain")
            
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

            # 방향 벡터 계산 
            dir_next, dir_prev, is_reverse = self.get_outward_vectors(i, total_points)
            points_count = self.calculate_points_count(dir_next, dir_prev)

            # 시작점 계산과 방향 조정
            start_point, next_reversed = self.adjust_point_with_ray_casting(
                base_point, dir_next, delaunay, terrain_array, horizontal_dist, vertical_dist
            )

            # 끝점 계산과 방향 조정
            end_point, prev_reversed = self.adjust_point_with_ray_casting(
                base_point, dir_prev, delaunay, terrain_array, horizontal_dist, vertical_dist
            )

            slope_points = [start_point]
            prev_mid_point = start_point  # 이전 mid point 초기화
            for t in np.linspace(0, 1, points_count):
                
                # next_reversed와 prev_reversed를 고려하여 방향 벡터 계산
                # 보간된 중간 방향 계산
                mid_dir = slerp(dir_next, dir_prev, t)

                # 레이캐스팅 수행
                mid_point, mid_reversed = self.adjust_point_with_ray_casting(
                    base_point, mid_dir, delaunay, terrain_array, horizontal_dist, vertical_dist
                )
                # 이전 미드포인트와 현재 미드포인트 사이의 지형 점들 추가
                search_ter = self.interpolate_terrain_points(
                    prev_mid_point, mid_point, delaunay, terrain_array
                )
                slope_points.extend(search_ter)

                slope_points.append(mid_point)
                prev_mid_point = mid_point
            # 마지막 미드포인트와 end_point 사이의 지형 점들도 확인
            final_search_ter = self.interpolate_terrain_points(
                prev_mid_point, end_point, delaunay, terrain_array
            )
            slope_points.extend(final_search_ter)

            if end_point not in slope_points:
                slope_points.append(end_point)

            # Edge 정보 저장
            if True:
                edge_info = EdgePoints(
                    base_point=base_point,
                    base_point_index=i,
                    start_point=start_point,
                    end_point=end_point,
                    is_reverse=is_reverse
                )
                self.edge_points.append(edge_info)
            if not is_reverse:
                base_to_slope[base_point] = slope_points
        return base_to_slope
    
    def get_triangle_index(self, point: Point3D, delaunay: Delaunay) -> int:
        """점이 속한 삼각형의 인덱스를 반환"""
        point_2d = np.array([point.x, point.y])
        return delaunay.find_simplex(point_2d)

    def interpolate_terrain_points(self, point1: Point3D, point2: Point3D, delaunay: Delaunay, terrain_array: np.ndarray) -> List[Point3D]:
        # 디버깅
        
        triangle1_idx = self.get_triangle_index(point1, delaunay)
        triangle2_idx = self.get_triangle_index(point2, delaunay)
        
        
        if triangle1_idx == triangle2_idx:
            return []
        
        points = []
        point1_2d = np.array([point1.x, point1.y])
        point2_2d = np.array([point2.x, point2.y])
        direction_2d = point2_2d - point1_2d
        
        current_triangle = int(triangle1_idx)
        visited_triangles = set()  # 시작 삼각형을 visited에 넣지 않음

        
        while current_triangle != int(triangle2_idx):  # 종료 조건을 목표 삼각형에 도달할 때까지로만 변경
            if current_triangle in visited_triangles:  # 이미 방문한 삼각형이면 종료
                break
            
            visited_triangles.add(current_triangle)
            # 디버깅
            
            triangle_vertices = delaunay.simplices[current_triangle]
            triangle_points_3d = terrain_array[triangle_vertices]
            triangle_points_2d = triangle_points_3d[:, :2]
            
            for i in range(3):
                edge_start_2d = triangle_points_2d[i]
                edge_end_2d = triangle_points_2d[(i + 1) % 3]
                edge_start_3d = triangle_points_3d[i]
                edge_end_3d = triangle_points_3d[(i + 1) % 3]
                
                # 디버깅
                
                denominator = (
                    (edge_end_2d[1] - edge_start_2d[1]) * direction_2d[0] -
                    (edge_end_2d[0] - edge_start_2d[0]) * direction_2d[1]
                )
                
                if abs(denominator) < 1e-10:
                    # print("Edges are parallel")
                    continue
                
                t = (
                    (edge_end_2d[0] - edge_start_2d[0]) * (point1_2d[1] - edge_start_2d[1]) -
                    (edge_end_2d[1] - edge_start_2d[1]) * (point1_2d[0] - edge_start_2d[0])
                ) / denominator
                
                s = (
                    direction_2d[0] * (point1_2d[1] - edge_start_2d[1]) -
                    direction_2d[1] * (point1_2d[0] - edge_start_2d[0])
                ) / denominator
                
                
                if 0 <= s <= 1 and 0 <= t <= 1:
                    intersection_2d = point1_2d + t * direction_2d
                    z = edge_start_3d[2] + s * (edge_end_3d[2] - edge_start_3d[2])
                    intersection_point = Point3D(intersection_2d[0], intersection_2d[1], z)
                    
                    # 디버깅
                    points.append(intersection_point)
            
            neighbors = delaunay.neighbors[current_triangle]
            next_triangle = -1
            
            
            for neighbor in neighbors:
                if neighbor >= 0 and int(neighbor) not in visited_triangles:
                    next_triangle = int(neighbor)
                    visited_triangles.add(next_triangle)
                    break
            
            if next_triangle == -1:
                break
                
            current_triangle = next_triangle
        
        # 디버깅
        return sorted(points, key=lambda p: (p.x - point1.x)**2 + (p.y - point1.y)**2)

def slerp(v1, v2, t):
    # 두 벡터 사이의 각도 계산
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)  # numerical stability
    theta = np.arccos(dot)
    
    if theta < 1e-7:  # 각도가 매우 작으면 선형 보간
        return v1 * (1 - t) + v2 * t
    
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) * v1 + np.sin(t * theta) * v2) / sin_theta

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


def read_ply_with_open3d(file_path):
    """Open3D로 PLY 파일 읽어서 Point3D 객체 리스트로 변환"""
    pcd = o3d.io.read_point_cloud(file_path)
    np_points = np.asarray(pcd.points)
    
    # numpy 배열을 Point3D 객체 리스트로 변환
    point3d_list = [Point3D(x=float(point[0]), 
                           y=float(point[1]), 
                           z=float(point[2])) for point in np_points]
    
    return point3d_list

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
def sort_points_clockwise(points):
    """
    주어진 점들을 시계 방향으로 정렬합니다.
    - points: 점들의 리스트 [[x1, y1, z1], [x2, y2, z2], ...].
    """
    # 점들의 중심 계산
    centroid = np.mean(points, axis=0)

    # 중심을 기준으로 각도 계산 (atan2를 사용하여 시계 방향 정렬)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # 각도를 기준으로 정렬 (오름차순 -> 시계 방향)
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

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
    for i in range(len(site_data.features) - 2):
        current_feature = site_data.features[i]
        next_feature = site_data.features[i + 1]
        
        current_points = current_feature.polygon.points
        next_points = next_feature.polygon.points

        if current_points[0] != current_points[-1]:
            current_points.append(current_points[0])
        if next_points[0] != next_points[-1]:
            next_points.append(next_points[0])
        # 동적으로 삼각망 생성
        side_faces = generate_side_faces_dynamic(current_points, next_points)

        # 생성된 삼각형을 IFC 면으로 변환
        for face in side_faces:
            surface_face = create_surface_faces(ifc_file, face)  
            all_faces.append(surface_face)
    
    if len(site_data.features) >= 2:
        last_feature = site_data.features[-1].polygon.points[:-1]
        second_last_feature = site_data.features[-2].polygon.points[:-1]
        
        # last_feature의 첫 점을 기준으로 second_last_feature 재정렬
        second_last_feature = reorder_points_by_start(second_last_feature, last_feature[0])
        
        pre_point = second_last_feature[0]
        create_cnt = 0    
        
        for i in range(len(last_feature) - 1):
            p1 = last_feature[i]
            p2 = last_feature[i + 1]
            if p1 == p2:
                continue
                
            closest_point = find_closest_point(p1, p2, second_last_feature)
            if closest_point != pre_point:
                face_reverse = create_surface_faces(ifc_file, [p1, pre_point, closest_point])
                if create_cnt > 0:
                    all_faces.append(face_reverse)
                create_cnt = 0
                pre_point = closest_point
                
            face = create_surface_faces(ifc_file, [p1, p2, closest_point])
            all_faces.append(face)
            create_cnt += 1
        
        # 마지막 연결 처리
        if closest_point != second_last_feature[0]:
            face_final = create_surface_faces(ifc_file, [
                last_feature[0],
                closest_point,
                second_last_feature[0]
            ])
            all_faces.append(face_final)
        
    return all_faces

def create_slope_faces(ifc_file, base_to_slope: dict, slope_surface: SlopeSurface) -> List:
    """base_point와 확장된 slope_points만을 사용하여 삼각망 생성"""
    all_faces = []
    
    for idx, (base_point, slope_points) in enumerate(base_to_slope.items()):
        # print(f"Index: {idx}, Current base_point: {base_point}")
        slp =  slope_points
        # face 생성
        for x in range(len(slp) - 1):
            face = create_surface_faces(ifc_file, [slp[x], slp[x + 1], base_point])
            all_faces.append(face)

    return all_faces

def generate_search_slope_polygon_with_base_points(
    base_start: Point3D, 
    base_end: Point3D, 
    terrain_points: List[Point3D], 
    horizontal_ratio: float, 
    vertical_ratio: float
) -> List[Point3D]:
    """
    주어진 두 점(base_start, base_end)을 기준으로 경사면 확장 좌표 생성
    :param base_start: 시작점 Point3D 객체
    :param base_end: 끝점 Point3D 객체
    :param terrain_points: 지형 데이터 (Point3D 리스트)
    :param horizontal_ratio: 경사의 밑변 비율
    :param vertical_ratio: 경사의 높이 비율
    :return: 경사면 확장된 두 점의 Point3D 리스트
    """
    # Terrain z 좌표에서 최저값 계산
    terrain_min_z = min(p.z for p in terrain_points)
    top_height = base_start.z  # 두 점의 높이는 동일하다고 가정
    lower_z = terrain_min_z - 2
    height_difference = abs(top_height - lower_z)

    # 충분히 큰 대각선 거리 설정
    target_distance = height_difference * 1000  # 매우 큰 값으로 설정

    # 정규화와 스케일링
    total_ratio_square = horizontal_ratio**2 + vertical_ratio**2
    scaling_factor = target_distance / math.sqrt(total_ratio_square)

    # 실제 거리 계산
    horizontal_length = horizontal_ratio * scaling_factor
    vertical_length = vertical_ratio * scaling_factor

    # 수직 거리 반전 (아래 방향)
    vertical_length = -vertical_length

    slope_points = []
    for base_point in [base_start, base_end]:  # 시작점과 끝점 각각 처리
        # 각 base_point에 대해 방향 벡터 계산
        dx = base_end.x - base_start.x
        dy = base_end.y - base_start.y

        # 법선 벡터 계산 (외곽 방향)
        normal_dx = -dy
        normal_dy = dx

        # 방향 벡터 정규화
        length = math.sqrt(normal_dx**2 + normal_dy**2)
        unit_dx = normal_dx / length
        unit_dy = normal_dy / length

        # 수평 거리만큼 확장
        new_x = base_point.x + unit_dx * horizontal_length
        new_y = base_point.y + unit_dy * horizontal_length
        new_z = terrain_min_z + vertical_length  # Z 방향 경사 반영

        slope_points.append(Point3D(new_x, new_y, new_z))

    return slope_points


def line_triangle_intersection(edge, triangle):
    """선분과 삼각형의 교차점을 계산 (평행한 경우도 포함)"""
    # Point3D 객체를 numpy 배열로 변환
    start = np.array([edge[0].x, edge[0].y, edge[0].z])
    end = np.array([edge[1].x, edge[1].y, edge[1].z])
    v1 = np.array([triangle[0].x, triangle[0].y, triangle[0].z])
    v2 = np.array([triangle[1].x, triangle[1].y, triangle[1].z])
    v3 = np.array([triangle[2].x, triangle[2].y, triangle[2].z])

    # 선벡터 및 삼각형의 두 변 벡터 계산
    direction = end - start
    edge1 = v2 - v1
    edge2 = v3 - v1

    # 평면 법선 벡터 계산
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)

    EPSILON = 1e-8  # 수치 안정성을 위한 임계값

    # 평행한 경우에도 계산 진행
    inv_det = 1.0 / (det if abs(det) > EPSILON else EPSILON)  # 0으로 나누는 것 방지
    
    tvec = start - v1
    u = np.dot(tvec, pvec) * inv_det

    # u 범위 확인 (삼각형 외부 제외)
    if u < 0 or u > 1:
        return None

    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det

    # v 범위 및 u + v 확인 (삼각형 외부 제외)
    if v < 0 or u + v > 1:
        return None

    # 교차점 t 계산
    t = np.dot(edge2, qvec) * inv_det

    # t가 [0, 1] 범위 내에 있어야 교차
    if t < 0 or t > 1:
        return None

    # 교차점 좌표 반환
    intersection_point = start + t * direction
    return Point3D(intersection_point[0], intersection_point[1], intersection_point[2])


def find_rectangle_intersections(
    rectangle: List[Point3D],
    terrain: List[Point3D],
    terrain_tri: Delaunay
) -> List[Point3D]:
    """
    경사면 사각형에 닿는 지형의 점 또는 선 교차점 계산
    사각형과 교차점이 없을 경우 위/아래로 반전하여 재시도
    :param rectangle: 경사면 사각형 (4개의 Point3D로 구성된 리스트)
    :param terrain: 지형 좌표 (Point3D 리스트)
    :param terrain_tri: 지형의 들로네 삼각망
    :return: tuple[List[Point3D], bool] - 교차점 좌표 리스트와 반전 여부
    """
    # terrain 데이터를 numpy 배열로 변환
    terrain_array = np.array([[p.x, p.y, p.z] for p in terrain])
    
    def check_intersections(rect_points):
        intersections = []
        # 사각형을 두 삼각형으로 분할
        tri1 = [rect_points[0], rect_points[1], rect_points[2]]
        tri2 = [rect_points[0], rect_points[2], rect_points[3]]
        
        for simplex in terrain_tri.simplices:
            # simplex를 통해 terrain 삼각형 생성
            terrain_triangle = [Point3D(*terrain_array[vertex]) for vertex in simplex]
            # terrain 삼각형의 각 edge 계산
            edges = [
                (terrain_triangle[i], terrain_triangle[(i + 1) % 3])
                for i in range(3)
            ]
            
            for edge in edges:
                # 각 선분과 사각형(두 삼각형)의 교차점 계산
                intersection1 = line_triangle_intersection(edge, tri1)
                intersection2 = line_triangle_intersection(edge, tri2)
                
                if intersection1 is not None:
                    intersections.append(intersection1)
                if intersection2 is not None:
                    intersections.append(intersection2)
        
        return intersections
    
    # 원래 방향으로 시도
    intersections = check_intersections(rectangle)
    if intersections:
        return intersections
        
    # 교차점이 없으면 위/아래 방향 반전하여 재시도
    # z 좌표만 반전
    flipped_rectangle = [
        Point3D(p.x, p.y, (2 * rectangle[0].z - p.z)) 
        for p in rectangle
    ]
    
    flipped_intersections = check_intersections(flipped_rectangle)
    if flipped_intersections:
        return flipped_intersections
        
    # 양쪽 다 실패하면 빈 리스트 반환
    #print("No intersections found in either direction")
    return []

def project_points_to_line(
    b1: Point3D,
    b2: Point3D,
    sp1: Point3D,
    sp2: Point3D,
    intersection_points: List[Point3D]
) -> List[Tuple[Point3D, Point3D]]:
    """
    sp1~sp2 사이의 교차점들을 b1~b2 직선상에 투영
    :param b1: 직선 시작점
    :param b2: 직선 끝점
    :param sp1: 경사면 외곽선 시작점
    :param sp2: 경사면 외곽선 끝점
    :param intersection_points: sp1~sp2 사이에서 발견된 교차점들
    :return: b1~b2 직선상의 새로운 점들과 투영된 교차점의 매핑
    """
    # sp1~sp2 선분의 길이
    slope_dx = sp2.x - sp1.x
    slope_dy = sp2.y - sp1.y
    slope_length = math.sqrt(slope_dx**2 + slope_dy**2)

    # b1~b2 선분의 벡터와 길이
    base_dx = b2.x - b1.x
    base_dy = b2.y - b1.y
    base_dz = b2.z - b1.z

    projected_points = []
    for point in intersection_points:
        # sp1에서 교차점까지의 비율 계산
        vec_x = point.x - sp1.x
        vec_y = point.y - sp1.y
        
        # 경사면에서의 상대적 위치 계산 (0~1 사이 값)
        ratio = (vec_x * slope_dx + vec_y * slope_dy) / (slope_length * slope_length)
        
        if 0 <= ratio <= 1:  # 범위 내에 있는 점만 처리
            # 같은 비율로 b1~b2 직선상에 점 생성
            new_x = b1.x + ratio * base_dx
            new_y = b1.y + ratio * base_dy
            new_z = b1.z + ratio * base_dz  # 직선 상의 높이 보간
            
            projected_points.append((Point3D(new_x, new_y, new_z), point))
    
    return projected_points

def integrate_line_points(
    b1: Point3D,
    b2: Point3D,
    divided_points: List[Point3D],
    projected_points: List[Tuple[Point3D, Point3D]]
) -> List[Point3D]:
    """
    기준 분할점과 투영된 점들을 직선상에서 통합하고 정렬
    투영된 점과 원래 점의 매핑을 포함
    """
    all_points = divided_points.copy()

    # b1~b2 직선의 단위 벡터
    base_dx = b2.x - b1.x
    base_dy = b2.y - b1.y
    base_length = math.sqrt(base_dx**2 + base_dy**2)
    unit_dx = base_dx / base_length
    unit_dy = base_dy / base_length

    # b1으로부터의 거리를 기준으로 정렬
    def get_distance_from_start(point):
        vec_x = point.x - b1.x
        vec_y = point.y - b1.y
        return vec_x * unit_dx + vec_y * unit_dy

    # 투영된 점들 추가
    all_points.extend([p[0] for p in projected_points])
    all_points = [b1] + all_points + [b2]
    # 거리 기준으로 정렬하고 중복 제거
    sorted_points = sorted(all_points, key=get_distance_from_start)

    return sorted_points
def calculate_intersection_point(p1: Point3D, p2: Point3D, p3: Point3D, p4: Point3D) -> Point3D:
   """
   두 선분의 교차점을 계산합니다.
   Args:
       p1, p2: 첫 번째 선분의 시작점과 끝점
       p3, p4: 두 번째 선분의 시작점과 끝점
   Returns:
       교차점(Point3D) 또는 None
   """
   x1, y1 = p1.x, p1.y
   x2, y2 = p2.x, p2.y
   x3, y3 = p3.x, p3.y
   x4, y4 = p4.x, p4.y
   
   denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
   
   if abs(denominator) < 1e-10:
       return None

   t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
   u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
   
   # 끝점 교차를 제외하기 위해 strict inequality 사용
   if not (0 < t < 1 and 0 < u < 1):
       return None
   
   x = x1 + t * (x2 - x1)
   y = y1 + t * (y2 - y1)
   z = p1.z + t * (p2.z - p1.z)
   intersection_point = Point3D(x, y, z)

   # 선분 1의 시작점과 끝점에서 교차점까지의 길이 계산
   length1_start = np.sqrt((x - x1)**2 + (y - y1)**2)
   length1_end = np.sqrt((x2 - x)**2 + (y2 - y)**2)
   
   # 선분 2의 시작점과 끝점에서 교차점까지의 길이 계산
   length2_start = np.sqrt((x - x3)**2 + (y - y3)**2)
   length2_end = np.sqrt((x4 - x)**2 + (y4 - y)**2)
   
   # 각 선분에서 더 짧은 길이 찾기
   min_length1 = min(length1_start, length1_end)
   min_length2 = min(length2_start, length2_end)
   if min_length2 < 0.01 or min_length1 < 0.01:
       return None
   
   return intersection_point
def calculate_line_distance(p: Point3D, line_start: Point3D, line_end: Point3D) -> float:
    """
    점과 선분 사이의 상대적 거리를 계산
    선분을 따라 진행하는 방향으로의 거리를 반환
    """
    line_vector = (line_end.x - line_start.x, line_end.y - line_start.y)
    point_vector = (p.x - line_start.x, p.y - line_start.y)
    
    # 선분 방향으로의 투영 거리 계산
    line_length = (line_vector[0]**2 + line_vector[1]**2)**0.5
    if line_length == 0:
        return 0
        
    projection = (point_vector[0]*line_vector[0] + point_vector[1]*line_vector[1]) / line_length
    return projection


from collections import OrderedDict


def preprocess_section_faces(
    highest_feature: Feature,
    slope_surface: SlopeSurface,
    site_data: SiteData,
    terrain_points: List[Point3D] = None
):
    preprocessed_faces = []
    base_points = highest_feature.polygon.points[:-1]  # 폐곡선의 마지막 점 제외
    edge_points = slope_surface.edge_points

    # 지형 데이터 준비
    terrain_array = np.array([[p.x, p.y, p.z] for p in terrain_points]) if terrain_points else np.array([])
    terrain_2d = terrain_array[:, :2] if terrain_points else np.array([])
    terrain_tin = Delaunay(terrain_2d) if terrain_points is not None and len(terrain_2d) > 0 else None

    last_level_points = []
    all_bot_lists = []
    asd = set()

    # 1. 모든 세트 수집
    for i in range(len(base_points)):
        b1 = base_points[i]
        b2 = base_points[(i + 1) % len(base_points)]
        e1 = edge_points[i].start_point
        e2 = edge_points[(i + 1) % len(base_points)].end_point

        # search polygon 생성
        sp1, sp2 = generate_search_slope_polygon_with_base_points(
            b1, b2, 
            terrain_points,
            slope_surface.angle_ratio[1],
            slope_surface.angle_ratio[0]
        )
        search_polygon = [b1, b2, sp2, sp1]
        
        # 각 영역의 교차점 찾기
        inter_points = find_rectangle_intersections(search_polygon, terrain_points, terrain_tin)
        result_points = [e2] + inter_points + [e1]

        # 선분 위의 점들 처리
        divided_points = [b1, b2]
        projected_points = project_points_to_line(b1, b2, sp1, sp2, result_points)
        final_base_points = integrate_line_points(b1, b2, divided_points, projected_points)

        # 투영된 점들 매핑 처리
        mapped_projected_points = {proj[0]: proj[1] for proj in projected_points}

        # 하단 점들 정렬
        sorted_result_points = sorted(
            result_points,
            key=lambda p: calculate_line_distance(p, b1, b2)
        )
        bot_list = sorted_result_points
        all_bot_lists.append((bot_list, final_base_points, mapped_projected_points))

        # last_level_points 업데이트
        if i == 0:
            last_level_points.extend(final_base_points)
        else:
            last_level_points.extend(final_base_points[1:])

    # 2. 인접한 세트들 간의 교차점 처리
    for i in range(len(all_bot_lists)-1):
        current_bot, current_base, current_mapping = all_bot_lists[i]
        next_bot, next_base, next_mapping = all_bot_lists[i+1]

        # 교차점 찾기
        all_intersections = []
        for k in range(len(current_bot) - 1):
            for m in range(len(next_bot) - 1):
                intersection = calculate_intersection_point(
                    current_bot[k], current_bot[k + 1],
                    next_bot[m], next_bot[m + 1]
                )
                if intersection:
                    all_intersections.append({
                        'point': intersection,
                        'current_segment': (k, k+1),
                        'next_segment': (m, m+1),
                    })

        if all_intersections:
            b1, b2 = [base_points[i], base_points[i+1]]
            intersection_info = all_intersections[0]
            intersection = intersection_info['point']

            # 현재 세트 수정
            current_result = []
            for cb in current_bot:
                if not compare_angles(b1, b2, intersection, cb):
                    current_result.append(cb)
            current_result.append(intersection)

            # 투영된 점 제거
            for key, value in current_mapping.items():
                if value not in current_result:
                    if key in current_base:
                        current_base.remove(key)
            all_bot_lists[i] = (current_result, current_base, current_mapping)

            # 다음 세트 수정
            next_result = []
            for cb in next_bot:
                if compare_angles(b1, b2, intersection, cb):
                    next_result.append(cb)
            next_result.append(intersection)

            # 투영된 점 제거
            for key, value in next_mapping.items():
                if value not in next_result:
                    if key in next_base:
                        next_base.remove(key)

            all_bot_lists[i+1] = (next_result, next_base, next_mapping)


    for bot_list, final_base_points, _ in all_bot_lists:
        # section_points 생성
        combined_points = final_base_points + bot_list
        section_points = list(OrderedDict.fromkeys(combined_points))

        preprocessed_faces.append({
            "final_base_points": final_base_points,
            "bot_list": bot_list,
            "section_points": section_points
        })

    # 폐곡선 완성
    if last_level_points and last_level_points[0] != last_level_points[-1]:
        last_level_points.append(last_level_points[0])

    # 마지막 feature 업데이트
    if last_level_points:
        updated_last_feature = Feature(
            polygon=Polygon(points=last_level_points),
            order=highest_feature.order
        )
        site_data.features[-1] = updated_last_feature

    return preprocessed_faces


def create_ifc_section_faces(ifc_file, preprocessed_faces):
    """
    전처리된 데이터를 사용하여 IFC 섹션 모델 생성.
    """
    all_faces = []
    
    for pi,face_data in enumerate(preprocessed_faces):
        bot_list = face_data["bot_list"]
        section_points = face_data["section_points"] 
        bot_set = set(bot_list)
        points_2d = np.array([[p.x, p.y] for p in section_points ])
        delaunay_tri = Delaunay(points_2d)

        for simplex in delaunay_tri.simplices:
            p0, p1, p2 = [section_points[idx] for idx in simplex]

            # bot_list로만 이루어진 삼각형인지 확인
            if not (p0 in bot_set and p1 in bot_set and p2 in bot_set):
                face = create_surface_faces(ifc_file, [p2, p0, p1])
                all_faces.append(face)

    try:
        section_closed_shell = ifc_file.create_entity("IfcClosedShell", CfsFaces=all_faces)
        section_surface_model = ifc_file.create_entity("IfcShellBasedSurfaceModel", SbsmBoundary=[section_closed_shell])
    except Exception as e:
        raise Exception(f"Error creating section surface model: {e}")

    return section_surface_model



def create_section_faces(ifc_file, highest_feature: Feature, slope_surface: SlopeSurface,site_data:SiteData, terrain_points: List[Point3D] = None):
    # 1. 전처리 단계: 연산 결과 저장
    preprocessed_faces = preprocess_section_faces(highest_feature, slope_surface, site_data, terrain_points)

    # 2. 필요 시 IFC 생성
    return create_ifc_section_faces(ifc_file, preprocessed_faces)

   
def calculate_angle(base1: Point3D, base2: Point3D, point: Point3D) -> float:
    """
    Calculate the angle formed by the vector from base1 to base2 and base1 to point (ignoring z).
    """
    # Vector components (ignoring z)
    vector1 = (base2.x - base1.x, base2.y - base1.y)
    vector2 = (point.x - base1.x, point.y - base1.y)

    # Dot product and magnitudes
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Angle in radians
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # Prevent division by zero

    cos_theta = dot_product / (magnitude1 * magnitude2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to avoid numerical issues
    return math.acos(cos_theta)  # Return angle in radians

def compare_angles(base1: Point3D, base2: Point3D, point1: Point3D, point2: Point3D) -> bool:
    """
    Compare the angles formed by base1-base2 and base1-point1/base1-point2.
    Return "point1" if angle with point1 is larger, "point2" if angle with point2 is larger, or "equal" if they are the same.
    """
    angle1 = calculate_angle(base1, base2, point1)
    angle2 = calculate_angle(base1, base2, point2)

    return angle1 > angle2

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

def is_within_polygon(polygon, p0, p1, p2):
    triangle_edges = [
        LineString([(p0.x, p0.y), (p1.x, p1.y)]),
        LineString([(p1.x, p1.y), (p2.x, p2.y)]),
        LineString([(p2.x, p2.y), (p0.x, p0.y)])
    ]
    for edge in triangle_edges:
        # 변이 폴리곤 내부에 포함되거나 외곽선과 접하는 경우만 유효
        if not (polygon.contains(edge) or polygon.touches(edge)):
            return False
    return True

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

def create_civil3d_style_ifc_with_slope(site_data: SiteData, slope_info: dict, file_path):
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
    
    terrain_points,slope_distance = filter_points_within_distance(base_points, t_points, float(ratio[1]))
    slope_info['distance'] = slope_distance*1000

        
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

    #사면 생성
    try:
        highest_order_feature = site_data.get_highest_order_feature()
        #print(f"Generating slope for Feature Order: {highest_order_feature.order}")
        angle_data  = slope_info["angle"].split(":")

        slope_surface = SlopeSurface(
            base_polygon=highest_order_feature.polygon,
            distance=slope_info["distance"],
            angle_ratio=(float(angle_data[0]), float(angle_data[1]))
        )
        base_to_slope = slope_surface.generate_slope(terrain_points)

        # Slope Faces 생성
        slope_faces = create_slope_faces(ifc_file, base_to_slope,slope_surface)
        
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
        
        # 단면 생성
        section_surface_model = create_section_faces(ifc_file, highest_order_feature, slope_surface,site_data,terrain_points)

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
        
# 측면 생성
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
    #print(f"Civil 3D 스타일 IFC 파일이 생성되었습니다: {file_path}")

# def main():
if __name__ == "__main__":
    with open(model_json_file, 'r') as file:
        json_data = json.load(file)
    
    site_data = SiteData.from_json(json.dumps(json_data))  # JSON 데이터를 문자열로 변환하여 처리

    # 경사 정보 추출
    slope_info = {
        "angle": angle_ratio_input
    }

    create_civil3d_style_ifc_with_slope(site_data, slope_info,output_file)
