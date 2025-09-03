# process/run.py

import subprocess
import sys
import os
import argparse
import math
from cell_split_visualize import PolygonSplitter

# 이 파일(process/run.py) 기준으로 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
def run_cell_split():
    print(args)
    """1번 코드: 부지를 두 영역으로 분할"""
    #script = os.path.join(PROJECT_ROOT, 'process', 'cell_split_visualize.py')
    #print(f"▶ Running cell split: {script}")
    #subprocess.run([sys.executable, script], check=True, cwd=PROJECT_ROOT)
    splitter = PolygonSplitter(args.input, args.idx_a, args.idx_b, args.divisions,args.output)
    regions, original_df = splitter.split_polygon()
    #args.no_viz
    #splitter.visualize(regions, original_df)

def run_node_pipeline(region_csv, label, blade_width=2.7, overlap_rate=0.2):
    """2번 코드: BL 생성"""
    script = os.path.join(PROJECT_ROOT, 'process', 'cell_create.py')
    out_csv = os.path.join(PROJECT_ROOT, 'output', f'grid_cells_{label}.csv')
    print(f"▶ Running node pipeline on {region_csv} → {out_csv}")
    subprocess.run(
        [sys.executable, script, 
         "-i" , region_csv, 
         "-o", out_csv,
         "-b", str(blade_width),
         "-r", str(overlap_rate)],
        check=True,
        cwd=PROJECT_ROOT
    )
    
def calculate_w(required_line_change_distance, blade_width, overlap_rate):
    """w 값 계산: w = ceil(required_line_change_distance / 2 / cell_size)"""
    cell_size = (1 - overlap_rate) * blade_width
    w = math.ceil(required_line_change_distance / 2 / cell_size)
    print(f"  - Required line change distance: {required_line_change_distance}m")
    print(f"  - Cell size: {cell_size}m")
    print(f"  - Calculated w: {w}")
    return w

def run_create_sbl(w_rows, visualize=True):
    """SBL 생성: LBL과 RBL의 w행까지를 합침"""
    script = os.path.join(PROJECT_ROOT, 'process', 'create_sbl.py')
    lbl_csv = os.path.join(PROJECT_ROOT, 'output', 'grid_cells_LBL.csv')
    rbl_csv = os.path.join(PROJECT_ROOT, 'output', 'grid_cells_RBL.csv')
    sbl_csv = os.path.join(PROJECT_ROOT, 'output', 'grid_cells_SBL.csv')
    
    print(f"▶ Creating SBL (merging {w_rows} rows)")
    
    # 명령 구성
    cmd = [sys.executable, script, 
           "-l", lbl_csv, 
           "-r", rbl_csv, 
           "-w", str(w_rows),
           "-o", sbl_csv]
    
    # 시각화 옵션 추가
    if visualize:
        cmd.append("-v")
    
    # 한 번만 실행
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

args = None
def main():
    global args
    parser = argparse.ArgumentParser(description='부지 분할 및 BL 생성 통합 파이프라인')
    #cell_split argument
    parser.add_argument('-i', '--input', default='input/부지모델test2.csv', 
                       help='입력 CSV 파일')
    parser.add_argument('-a', '--idx_a', type=int, default=1, 
                       help='시작점 A의 번호')
    parser.add_argument('-b', '--idx_b', type=int, default=9, 
                       help='시작점 B의 번호')
    parser.add_argument('-n', '--divisions', type=int, default=100, 
                       help='경로 분할 개수')
    parser.add_argument('-o', '--output', default='output', 
                       help='출력 디렉토리')
    
    parser.add_argument('--skip-split', action='store_true',
                       help='부지 분할 단계 건너뛰기 (region1.csv, region2.csv가 이미 있는 경우)')
    """
    parser.add_argument('-d', '--line-change-distance', type=float, default=2.5,
                       help='Required line change distance (기본값: 2.5m)')
    parser.add_argument('-b', '--blade-width', type=float, default=2.7,
                       help='블레이드 폭 (기본값: 2.7m)')
    parser.add_argument('-r', '--overlap-rate', type=float, default=0.2,
                       help='중복도 (기본값: 0.2 = 20%%)')
    parser.add_argument('--skip-bl', action='store_true',
                       help='BL 생성 단계 건너뛰기 (grid_cells_LBL.csv, grid_cells_RBL.csv가 이미 있는 경우)')
    parser.add_argument('--skip-sbl', action='store_true',
                       help='SBL 생성 건너뛰기')
    parser.add_argument('--no-viz', action='store_true',
                       help='시각화 표시 안함')
    """
    args = parser.parse_args()
    
    try:
        # 1. 부지 분할 (선택적)
        if not args.skip_split:
            print("="*50)
            print("STEP 1: 부지 분할")
            print("="*50)
            run_cell_split()
        else:
            print("▶ 부지 분할 단계 건너뜀")
        """
        # 2. BL 생성 (선택적)
        if not args.skip_bl:
            print("\n" + "="*50)
            print("STEP 2: BL 생성")
            print("="*50)
            
            regions = [
                (os.path.join(PROJECT_ROOT, 'output', 'region1.csv'), 'LBL'),
                (os.path.join(PROJECT_ROOT, 'output', 'region2.csv'), 'RBL'),
            ]
            
            for region_path, label in regions:
                run_node_pipeline(region_path, label, args.blade_width, args.overlap_rate)
        else:
            print("▶ BL 생성 단계 건너뜀")
        
        # 3. SBL 생성 (선택적)
        if not args.skip_sbl:
            print("\n" + "="*50)
            print("STEP 3: SBL 생성")
            print("="*50)
            
            # w 값 자동 계산
            w = calculate_w(args.line_change_distance, args.blade_width, args.overlap_rate)
            
            run_create_sbl(w, visualize=not args.no_viz)
            
            if not args.no_viz:
                print("\n📊 SBL 시각화가 이미지 파일로 저장되었습니다.")
                print("   output/grid_cells_SBL_visualization.png")
        else:
            print("\n▶ SBL 생성 건너뜀 (--skip-sbl 옵션 지정)")
        
        print("\n" + "="*50)
        print("✅ 전체 파이프라인 완료!")
        print("="*50)
        
        # 생성된 파일 목록 출력
        print("\n📁 생성된 파일:")
        output_files = [
            ('output/region1.csv', '좌측 영역'),
            ('output/region2.csv', '우측 영역'),
            ('output/grid_cells_LBL.csv', '좌측 BL'),
            ('output/grid_cells_RBL.csv', '우측 BL'),
        ]
        
        if not args.skip_sbl:
            w = calculate_w(args.line_change_distance, args.blade_width, args.overlap_rate)
            output_files.append(('output/grid_cells_SBL.csv', f'통합 BL (1~{w}행)'))
        
        for file_path, desc in output_files:
            full_path = os.path.join(PROJECT_ROOT, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path} - {desc}")
            else:
                print(f"  ✗ {file_path} - 파일 없음")
        """  
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 오류 발생: {e}")
        raise e
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        raise e
        sys.exit(1)

if __name__ == '__main__':
    main()