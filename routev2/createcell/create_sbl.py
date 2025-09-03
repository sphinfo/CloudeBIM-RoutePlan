import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import argparse
import os
import re
import math

def parse_bl_name(bl_name):
    """BL 이름에서 i, j 인덱스 추출 (예: LBL_2_3 -> (2, 3))"""
    match = re.match(r'[LRS]BL_(\d+)_(\d+)', bl_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def get_polygon(block: dict):
    # # 꼭짓점 좌표 추출 (최대 5개)
    vertices = []
    for i in range(1, 6):
        x_col = f'X{i}coord'
        y_col = f'Y{i}coord'
        if x_col in block and y_col in block:
            if pd.notna(block[x_col]) and pd.notna(block[y_col]) and str(block[x_col]) != '':
                vertices.append((float(block[x_col]), float(block[y_col])))
    return Polygon(vertices)

# X1,X2,X3 coord 순서 의미 있는지 확인필요
def read_bl_data(csv_file):
    """CSV 파일에서 BL 데이터 읽기"""
    df = pd.read_csv(csv_file)
    
    bl_data = {}
    for _, row in df.iterrows():
        bl_name = row['BLName']
        bl_data[bl_name] = dict(row)
   
    return bl_data

def draw_block(block_data: dict, ax1, n_rows: int, is_sbl: bool):
    for bl_name, bl in block_data.items():
        i, j = parse_bl_name(bl_name)
        if i is not None and j is not None and j > n_rows:

            # 꼭짓점 좌표 추출 (최대 5개)
            vertices = []
            for i in range(1, 6):
                x_col = f'X{i}coord'
                y_col = f'Y{i}coord'
                if x_col in bl and y_col in bl:
                    if pd.notna(bl[x_col]) and pd.notna(bl[y_col]) and str(bl[x_col]) != '':
                        vertices.append((float(bl[x_col]), float(bl[y_col])))
            poly = Polygon(vertices)

            x, y = poly.exterior.xy

            if is_sbl:
                ax1.fill(x, y, alpha=0.3, color='blue', edgecolor='blue', linewidth=1)
                ax1.plot(x, y, color='red', linewidth=1, alpha=0.8)
                if bl['YN'] == 'N':
                    ax1.fill(x, y, alpha=0.5, color='gold', edgecolor='gold', linewidth=1)
            else:
                ax1.plot(x, y, color='black', linewidth=0.5, alpha=0.3)
            
            centroid = poly.centroid
            ax1.text(centroid.x, centroid.y, bl_name, fontsize=6, ha='center', va='center')


def merge_blocks_to_sbl(lbl_data, rbl_data, n_rows):
    """LBL과 RBL의 n행까지를 합쳐서 SBL 생성"""
    sbl_data = {}

    # n_rows가 0일 수가 있을지 확인 필요
    target_row_nums = set(list(range(1, n_rows + 1)))
    
    if n_rows < 1:
        raise Exception('n_rows is zero')

    # 대상 행(j값)의 셀만 추출
    s_lbl_data, s_rbl_data = map(lambda x: {k: v for k, v in x.items() if parse_bl_name(k)[1] in target_row_nums}, [lbl_data, rbl_data])
    
    # LBL은 역방향 열 부터, RBL은 1열 부터
    # 열 개수가 차이가 발생할 경우 처리 필요
    max_l_i = max([parse_bl_name(k)[0] for k in s_lbl_data.keys()])
    max_r_i = max([parse_bl_name(k)[0] for k in s_rbl_data.keys()])

    # if max_l_i != max_r_i:
    #     raise Exception(f'LBL and RBL have different numbers of columns, LBL: {max_l_i}, , LBL: {max_r_i}')

    for block_name, block in s_lbl_data.items():
        i, j = parse_bl_name(block_name)
        sbl_name = f'SBL_{n_rows + j}_{max_l_i - i + 1}'
        block['BLName'] = sbl_name
        block['OriginalBlock'] = block_name
        
        # LBL L -> T, B -> L, R -> B, T -> R
        for xyz in ['X', 'Y', 'Z']:
            temp = block[f'{xyz}Tcoord'] # T -> temp
            block[f'{xyz}Tcoord'] = block[f'{xyz}Lcoord'] # L -> T
            block[f'{xyz}Lcoord'] = block[f'{xyz}Bcoord'] # B -> L
            block[f'{xyz}Bcoord'] = block[f'{xyz}Rcoord'] # R -> B
            block[f'{xyz}Rcoord'] = temp # T(temp) -> R
        sbl_data[sbl_name] = block

    for block_name, block in s_rbl_data.items():
        i, j = parse_bl_name(block_name)
        sbl_name = f'SBL_{n_rows - j + 1}_{i}' 
        block['BLName'] = sbl_name
        block['OriginalBlock'] = block_name
        
        # RBL R -> T, B -> R, L -> B, T -> L
        for xyz in ['X', 'Y', 'Z']:
            temp = block[f'{xyz}Tcoord'] # T -> temp
            block[f'{xyz}Tcoord'] = block[f'{xyz}Rcoord'] # R -> T
            block[f'{xyz}Rcoord'] = block[f'{xyz}Bcoord'] # B -> R
            block[f'{xyz}Bcoord'] = block[f'{xyz}Lcoord'] # L -> B
            block[f'{xyz}Lcoord'] = temp # T(temp) -> L
        sbl_data[sbl_name] = block

    sorted_sbl = dict(sorted(sbl_data.items(), key=lambda item: (parse_bl_name(item[0])[1], parse_bl_name(item[0])[0])))

    for _i, v in enumerate(sorted_sbl.items()):
        # No 다시 설정
        v[1]['No'] = _i
        #print(f"생성된 SBL: {v[0]}")
        #print(f"{v[1]['OriginalBlock']} -> {v[0]}")
    
    return sorted_sbl


def calculate_midpoints(coords):
    """사각형의 각 변 중점 계산"""
    if len(coords) != 4:
        return [""] * 8
    
    # 하단, 상단, 좌측, 우측 중점
    xb, yb = (coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2
    xt, yt = (coords[3][0] + coords[2][0]) / 2, (coords[3][1] + coords[2][1]) / 2
    xl, yl = (coords[0][0] + coords[3][0]) / 2, (coords[0][1] + coords[3][1]) / 2
    xr, yr = (coords[1][0] + coords[2][0]) / 2, (coords[1][1] + coords[2][1]) / 2
    
    return [str(v) for v in [xb, yb, xt, yt, xl, yl, xr, yr]]


def write_sbl_to_csv(sbl_data, output_file):
    """SBL 데이터를 CSV로 저장"""
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
               'cutVol', 'fillVol', 'YN', 'ClusterName',
               'OriginalBlock']  # 추가 컬럼

    # CSV 저장
    df = pd.DataFrame(sbl_data.values())
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"SBL 데이터 저장 완료: {output_file}")


def visualize_sbl(lbl_data, rbl_data, sbl_data, n_rows, save_path=None):
    """SBL 시각화"""
    import matplotlib
    # matplotlib.use('Agg')  # 백엔드 설정 (파일 저장용)
    import matplotlib.pyplot as plt
    
    # fig, ax = plt.subplots(figsize=(12, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    
    # 첫 번째 플롯: 원본 LBL/RBL
    ax1.set_title(f'원본 BL (n={n_rows}행까지 표시)', fontsize=14)
    
    # LBL 그리기
    draw_block(lbl_data, ax1, n_rows, is_sbl=False)
    
    # RBL 그리기
    draw_block(rbl_data, ax1, n_rows, is_sbl=False)

    # SBL 그리기
    draw_block(sbl_data, ax1, 0, is_sbl=True)

    # 두 번째 플롯: SBL
    ax2.set_title('생성된 SBL', fontsize=14)
    
    for sbl_name, sbl in sbl_data.items():
        poly = get_polygon(sbl)

        x, y = poly.exterior.xy
        
        # YN에 따라 색상 구분
        if sbl['YN'] == 'Y':
            ax2.fill(x, y, alpha=0.3, color='green', edgecolor='green', linewidth=2)
        else:
            ax2.fill(x, y, alpha=0.3, color='orange', edgecolor='orange', linewidth=2)
        
        ax2.plot(x, y, 'k-', linewidth=2)
        
        centroid = poly.centroid
        ax2.text(centroid.x, centroid.y, sbl_name, fontsize=4, ha='center', va='center', 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 축 설정
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

def calculate_w(required_line_change_distance, blade_width, overlap_rate):
    """w 값 계산: w = ceil(required_line_change_distance / 2 / cell_size)"""
    cell_size = (1 - overlap_rate) * blade_width
    w = math.ceil(required_line_change_distance / 2 / cell_size)
    print(f"  - Required line change distance: {required_line_change_distance}m")
    print(f"  - Cell size: {cell_size}m")
    print(f"  - Calculated w: {w}")
    return w

def main():
    parser = argparse.ArgumentParser(description='LBL과 RBL을 합쳐서 SBL 생성')
    parser.add_argument('-l', '--lbl', default='output/grid_cells_LBL.csv',
                       help='LBL CSV 파일 경로')
    parser.add_argument('-r', '--rbl', default='output/grid_cells_RBL.csv',
                       help='RBL CSV 파일 경로')
    parser.add_argument('-d', '--line-change-distance', type=float, default=2.5,
                       help='Required line change distance (기본값: 2.5m)')
    parser.add_argument('-b', '--blade-width', type=float, default=2.7,
                       help='블레이드 폭 (기본값: 2.7m)')
    parser.add_argument('-w', '--overlap-rate', type=float, default=0.2,
                       help='중복도 (기본값: 0.2 = 20%%)')
    parser.add_argument('-o', '--output', default='output/grid_cells_SBL.csv',
                       help='출력 SBL CSV 파일 경로')
    parser.add_argument('-v', '--visualize', type=bool,default=False,
                       help='결과 시각화 표시')
    
    args = parser.parse_args()
    w = calculate_w(args.line_change_distance, args.blade_width, args.overlap_rate)
    print(f"LBL 파일: {args.lbl}")
    print(f"RBL 파일: {args.rbl}")
    print(f"합칠 행 개수: {w}")
    
    # 데이터 읽기
    print("▶ LBL 데이터 읽는 중...")
    lbl_data = read_bl_data(args.lbl)
    print(f"  - 총 {len(lbl_data)}개의 LBL 블록 로드")
    
    print("▶ RBL 데이터 읽는 중...")
    rbl_data = read_bl_data(args.rbl)
    print(f"  - 총 {len(rbl_data)}개의 RBL 블록 로드")
    print()
    
    # SBL 생성
    print(f"▶ SBL 생성 중 (1~{w}행)...")
    sbl_data = merge_blocks_to_sbl(lbl_data, rbl_data, w)
    print(f"  - 총 {len(sbl_data)}개의 SBL 생성 완료")
    
    # CSV 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_sbl_to_csv(sbl_data, args.output)
    
    # 시각화 (옵션에 따라)
    if args.visualize:
        # 시각화 파일 경로
        viz_path = args.output.replace('.csv', '_visualization.png')
        visualize_sbl(lbl_data, rbl_data, sbl_data, w, save_path=viz_path)


if __name__ == '__main__':
    main()
