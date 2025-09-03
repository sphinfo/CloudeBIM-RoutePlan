import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


class PolygonSplitter:
    """부지 폴리곤을 두 개의 영역으로 분할하는 클래스"""
    
    def __init__(self, input_file, idx_A, idx_B, n_divisions, output_dir='output'):
        self.input_file = input_file
        self.idx_A = idx_A
        self.idx_B = idx_B
        self.n_divisions = n_divisions
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def cumulative_length(self, coords):
        """경로의 누적 길이 계산 (XY 평면만 고려)"""
        if len(coords) < 2:
            return np.array([0])
        
        diffs = coords[1:, :2] - coords[:-1, :2]
        dists = np.linalg.norm(diffs, axis=1)
        return np.concatenate(([0], np.cumsum(dists)))
    
    def interpolate_point(self, p1, p2, t):
        """두 점 사이의 보간점 계산"""
        return p1 + t * (p2 - p1)
    
    def compute_half_point(self, path):
        """경로의 중간 지점 찾기"""
        cum = self.cumulative_length(path)
        total_length = cum[-1]
        
        if total_length == 0:
            return path[0]
        
        half_length = total_length / 2
        idx = np.searchsorted(cum, half_length)
        
        if idx == 0:
            return path[0]
        
        idx = min(idx, len(path) - 1)
        
        if cum[idx] == cum[idx-1]:
            return path[idx-1]
        
        t = (half_length - cum[idx-1]) / (cum[idx] - cum[idx-1])
        return self.interpolate_point(path[idx-1], path[idx], t)
    
    def divide_path(self, path, n_divisions):
        """경로를 n개로 균등 분할"""
        cum = self.cumulative_length(path)
        total = cum[-1]
        
        if total == 0:
            return np.array([path[0]] * (n_divisions + 1))
        
        targets = np.linspace(0, total, n_divisions + 1)
        points = []
        
        for target in targets:
            idx = np.searchsorted(cum, target)
            
            if idx == 0:
                points.append(path[0])
            else:
                idx = min(idx, len(path) - 1)
                prev, nxt = path[idx-1], path[idx]
                seg_len = cum[idx] - cum[idx-1]
                
                if seg_len > 0:
                    t = (target - cum[idx-1]) / seg_len
                else:
                    t = 0
                
                points.append(self.interpolate_point(prev, nxt, t))
        
        return np.array(points)
    
    def extract_path(self, coords_all, start_idx, end_idx):
        """전체 좌표에서 특정 경로 추출"""
        if start_idx <= end_idx:
            return coords_all[start_idx : end_idx+1]
        # 순환 경로 처리
        return np.vstack((coords_all[start_idx:], coords_all[:end_idx+1]))
    
    def determine_left_right(self, df0_coords, candidate1_coords, candidate2_coords, is_region2=False):
        """df0의 진행 방향을 기준으로 좌측/우측 결정"""
        if len(df0_coords) < 2:
            return candidate1_coords, candidate2_coords
        
        # df0의 진행 방향 벡터
        df0_start = df0_coords[0][:2]
        df0_end = df0_coords[-1][:2]
        df0_direction = df0_end - df0_start
        
        if np.allclose(df0_direction, [0, 0]):
            return candidate1_coords, candidate2_coords
        
        # candidate1이 df0의 좌측인지 확인
        vec_to_candidate1 = candidate1_coords[0][:2] - df0_start
        cross_product = df0_direction[0] * vec_to_candidate1[1] - df0_direction[1] * vec_to_candidate1[0]
        
        if cross_product > 1e-9:  # candidate1이 좌측
            return candidate1_coords, candidate2_coords
        else:  # candidate1이 우측
            return candidate2_coords, candidate1_coords
    
    def process_single_boundary(self, boundary_path):
        """단일 경계 경로 처리"""
        # 경로의 중간점 찾기
        mid_point = self.compute_half_point(boundary_path)
        
        # 중간점 인덱스 찾기
        cum = self.cumulative_length(boundary_path)
        half_length = cum[-1] / 2
        split_idx = np.searchsorted(cum, half_length)
        
        # Side 1: 시작점에서 중간점까지
        if split_idx == 0 or len(boundary_path[:split_idx]) == 0:
            side1_points = np.array([boundary_path[0], mid_point])
        else:
            side1_points = np.vstack((boundary_path[:split_idx], mid_point))
        
        # Side 2: 끝점에서 중간점까지 (역방향)
        rev_boundary = boundary_path[::-1]
        cum_rev = self.cumulative_length(rev_boundary)
        split_idx_rev = np.searchsorted(cum_rev, half_length)
        
        if split_idx_rev == 0 or len(rev_boundary[:split_idx_rev]) == 0:
            side2_points = np.array([rev_boundary[0], mid_point])
        else:
            side2_points = np.vstack((rev_boundary[:split_idx_rev], mid_point))
        
        # 각 side를 n개로 분할
        divided_side1 = self.divide_path(side1_points, self.n_divisions)
        divided_side2 = self.divide_path(side2_points, self.n_divisions)
        
        # 중심선 계산 (대응점들의 중점)
        center_points = (divided_side1 + divided_side2) / 2
        
        # DataFrame으로 변환
        df0 = pd.DataFrame(center_points, columns=['x', 'y', 'z'])
        df_candidate1 = pd.DataFrame(divided_side1, columns=['x', 'y', 'z'])
        df_candidate2 = pd.DataFrame(divided_side2, columns=['x', 'y', 'z'])
        
        # 좌/우 결정
        df1, df2 = self.determine_left_right(
            center_points, divided_side1, divided_side2
        )
        
        return {
            'df0': df0,
            'df1': pd.DataFrame(df1, columns=['x', 'y', 'z']),
            'df2': pd.DataFrame(df2, columns=['x', 'y', 'z'])
        }
    
    def determine_regions(self, data1, data2, P_A, P_B):
        """B→A 기준선을 기준으로 좌/우 영역 결정"""
        # B에서 A를 바라보는 벡터
        look_vec_B_to_A = P_A[:2] - P_B[:2]
        
        # path1의 중간점
        path1_mid = self.compute_half_point(
            np.vstack([data1['df1'].values, data1['df2'].values])
        )
        
        # B에서 path1 중간점으로의 벡터
        vec_B_to_mid = path1_mid[:2] - P_B[:2]
        
        # 외적으로 좌/우 판단
        cross = look_vec_B_to_A[0] * vec_B_to_mid[1] - look_vec_B_to_A[1] * vec_B_to_mid[0]
        
        if cross > 1e-9:  # path1이 좌측
            return {'region1': data1, 'region2': data2}
        else:  # path1이 우측
            return {'region1': data2, 'region2': data1}
    
    def split_polygon(self):
        """메인 처리 함수"""
        # 1. 데이터 읽기
        try:
            df = pd.read_csv(self.input_file)
        except Exception as e:
            raise Exception(f"파일 읽기 오류: {e}")
        
        # 2. A, B 점 확인
        if not (df['No'].isin([self.idx_A]).any() and df['No'].isin([self.idx_B]).any()):
            raise ValueError(f"지정된 점 A({self.idx_A}) 또는 B({self.idx_B})를 찾을 수 없습니다.")
        
        # 3. 좌표 추출
        coords_all = df[['x', 'y', 'z']].values
        A_idx = np.where(df['No'] == self.idx_A)[0][0]
        B_idx = np.where(df['No'] == self.idx_B)[0][0]
        
        P_A = coords_all[A_idx]
        P_B = coords_all[B_idx]
        
        # 4. 두 경로 추출
        path1 = self.extract_path(coords_all, A_idx, B_idx)  # A → B
        path2 = self.extract_path(coords_all, B_idx, A_idx)  # B → A
        
        # 5. 각 경로 처리
        data1 = self.process_single_boundary(path1)
        data2 = self.process_single_boundary(path2)
        
        # 6. 좌/우 영역 결정
        regions = self.determine_regions(data1, data2, P_A, P_B)
        
        # 7. CSV 파일 저장
        for region_name, data in regions.items():
            self.save_region_csv(region_name, data)
        
        return regions, df
    
    def save_region_csv(self, region_name, data):
        """영역 데이터를 CSV로 저장"""
        df0, df1, df2 = data['df0'], data['df1'], data['df2']
        max_len = len(df0)
        
        # 데이터 병합
        df_region = pd.DataFrame({
            'No.': list(range(1, max_len + 1)),
            'x0': df0['x'], 'y0': df0['y'], 'z0': df0['z'],
            'x1': df1['x'], 'y1': df1['y'], 'z1': df1['z'],
            'x2': df2['x'], 'y2': df2['y'], 'z2': df2['z']
        })
        
        # CSV 저장 (필터링 없이)
        output_file = os.path.join(self.output_dir, f'{region_name}.csv')
        df_region.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {output_file}")
        
        # A, B 정보를 별도 파일로 저장
        ab_info = pd.DataFrame({
            'point': ['A', 'B'],
            'x': [df1.iloc[0]['x'], df2.iloc[0]['x']],  # A는 df1의 첫점, B는 df2의 첫점
            'y': [df1.iloc[0]['y'], df2.iloc[0]['y']],
            'z': [df1.iloc[0]['z'], df2.iloc[0]['z']],
            'original_idx': [self.idx_A, self.idx_B]
        })
        
        ab_file = os.path.join(self.output_dir, f'{region_name}_AB.csv')
        ab_info.to_csv(ab_file, index=False, encoding='utf-8-sig')
        print(f"A/B 정보 저장: {ab_file}")
        
        # A, B 중점 정보 저장 (격자 생성을 위해)
        A_point = np.array([df1.iloc[0]['x'], df1.iloc[0]['y'], df1.iloc[0]['z']])
        B_point = np.array([df2.iloc[0]['x'], df2.iloc[0]['y'], df2.iloc[0]['z']])
        midpoint = (A_point + B_point) / 2
        
        midpoint_info = pd.DataFrame({
            'midpoint_x': [midpoint[0]],
            'midpoint_y': [midpoint[1]],
            'midpoint_z': [midpoint[2]],
            'A_x': [A_point[0]],
            'A_y': [A_point[1]],
            'A_z': [A_point[2]],
            'B_x': [B_point[0]],
            'B_y': [B_point[1]],
            'B_z': [B_point[2]]
        })
        
        midpoint_file = os.path.join(self.output_dir, f'{region_name}_midpoint.csv')
        midpoint_info.to_csv(midpoint_file, index=False, encoding='utf-8-sig')
        print(f"중점 정보 저장: {midpoint_file}")
    
    def visualize(self, regions, original_df):
        """결과 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # A, B 점 좌표
        P_A = original_df[original_df['No'] == self.idx_A][['x', 'y']].iloc[0]
        P_B = original_df[original_df['No'] == self.idx_B][['x', 'y']].iloc[0]
        
        for idx, (region_name, ax) in enumerate(zip(['region1', 'region2'], axes)):
            data = regions[region_name]
            
            # 경계선 그리기
            ax.plot(data['df1']['x'], data['df1']['y'], 'b-', marker='.', 
                   markersize=4, label='df1 (좌측)', linewidth=1.5)
            ax.plot(data['df2']['x'], data['df2']['y'], 'r-', marker='.', 
                   markersize=4, label='df2 (우측)', linewidth=1.5)
            ax.plot(data['df0']['x'], data['df0']['y'], 'g--', marker='s', 
                   markersize=3, label='df0 (중심선)', linewidth=1)
            
            # A, B 점 표시
            ax.plot(P_A['x'], P_A['y'], 'kX', markersize=10, 
                   label=f'점 A (No.{self.idx_A})')
            ax.plot(P_B['x'], P_B['y'], 'kP', markersize=10, 
                   label=f'점 B (No.{self.idx_B})')
            
            # B→A 화살표
            ax.annotate('', xy=(P_A['x'], P_A['y']), xytext=(P_B['x'], P_B['y']),
                       arrowprops=dict(arrowstyle='->', color='cyan', lw=2, alpha=0.7))
            
            region_label = "Region 1 (좌측)" if region_name == 'region1' else "Region 2 (우측)"
            ax.set_title(f'{region_label}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.suptitle(f'부지 분할 결과 (A:{self.idx_A}, B:{self.idx_B}, n={self.n_divisions})', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description='부지 폴리곤을 두 영역으로 분할')
    parser.add_argument('-i', '--input', default='input/광명부지모델2.csv', 
                       help='입력 CSV 파일')
    parser.add_argument('-a', '--idx_a', type=int, default=3, 
                       help='시작점 A의 번호')
    parser.add_argument('-b', '--idx_b', type=int, default=7,
                       help='시작점 B의 번호')
    parser.add_argument('-n', '--divisions', type=int, default=100, 
                       help='경로 분할 개수')
    parser.add_argument('-o', '--output', default='output', 
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # PolygonSplitter 인스턴스 생성 및 실행
    splitter = PolygonSplitter(
        input_file=args.input,
        idx_A=args.idx_a,
        idx_B=args.idx_b,
        n_divisions=args.divisions,
        output_dir=args.output
    )
    
    try:
        regions, original_df = splitter.split_polygon()
        #splitter.visualize(regions, original_df)
        print("✅ 처리 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
