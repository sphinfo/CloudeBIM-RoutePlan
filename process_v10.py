
# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.
import time
import logging
from route_planner_v10.arguments import args
from route_planner_v10.constants import TABLE_NAMES
from route_planner_v10.util import read_mdb, read_csv, read_geojson, file_name, calculate_min_dist_center_node
from route_planner_v10.block import Block
from route_planner_v10.alloc import DozerAlloc
from route_planner_v10.plan import DozerRoutePlan


def execute_dozer(data: dict, input_file_name: str, execute_type: str):
    param, block_items, outline_data = Block.valid_parameter(*list(map(data.get, TABLE_NAMES)))
    
    # 할당셀
    dozer = DozerAlloc(block_items, outline_data)
    allocate_cell, allocate_cell_names, allocate_outline_cell = dozer.alloc(param)

    # 할당셀 결과 저장
    #dozer.save_output_csv(input_file_name)
    
    if execute_type == 'alloc': return
    # 계획경로
    dozer_route = DozerRoutePlan(block_items, allocate_cell, allocate_outline_cell, outline_data, allocate_cell_names)
    route_plan = dozer_route.calc_route_plan(param)

    # 계획 경로 결과 저장
    dozer_route.save_output_csv(args['output_file'])

# python3.12 process_v10.py --input_path "./input/20240715_test.MDB"
# python3.12 process_v10.py --input_path "./input/20240715_test.MDB" --execute_type alloc
if __name__ == "__main__":
    logging.info(f'arguments: {args}')
    logging.info(f'cmd: python process.py --{" --".join([k + " " + str(v) for k, v in args.items()])}')

    start = time.time()
    # Input_Parameter, Cell_Data
    #data = read_mdb(args.get('input_path'), TABLE_NAMES)
    #TO SPH 입력 Parameter model_line = csv file
    #                     cell_data = json file
    #                     input_parameter = argument
    data = {}
    data["Model_line_Data"] = read_csv(args['input_line_file'])
    Min_Cendist = calculate_min_dist_center_node(data["Model_line_Data"])
    args["Min_Cendist"] = Min_Cendist
    data["Input_Parameter"] = [args]
    data["Cell_Data"] = read_geojson(args['input_cell_file'])

    logging.info(f'Read {args.get("input_path")}.. duration: {time.time() - start} sec')

    execute_dozer(data, file_name(args.get("input_path")), args.get("execute_type"))

    logging.info(f'total duration: {time.time() - start} sec')