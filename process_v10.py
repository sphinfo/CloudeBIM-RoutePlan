
# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.


import time
import logging
from route_planner_v10.arguments import args
from route_planner_v10.util import read_geojson, read_csv
from route_planner_v10.block import Block
from route_planner_v10.alloc import DozerAlloc
from route_planner_v10.plan import DozerRoutePlan
from route_planner_v10.plot import draw, draw_obstacle
from route_planner_v10.exception import RouteCreationError

def execute_dozer(args: dict):
    output_file, visual_mode, input_line_file, input_cell_file = map(
        args.get, ['output_file', 'visual_mode',
                   'input_line_file', 'input_cell_file'])

    param, block_items, outline_data = Block.valid_parameter(
        args=args, cell_data=read_geojson(input_cell_file), model_line_data=read_csv(input_line_file))

    # 할당셀
    dozer = DozerAlloc(block_items, outline_data)
    allocate_cell, allocate_cell_names, allocate_outline_cell = dozer.alloc(param)

    # 계획경로
    dozer_route = DozerRoutePlan(block_items, allocate_cell, allocate_outline_cell, outline_data, allocate_cell_names)
    route_plan, intersected_blocks, intersected_block_names = dozer_route.calc_route_plan(param)
    
    # 계획 경로 결과 저장
    dozer_route.save_output_csv(output_file)

    # 장애물 교차
    if intersected_block_names:
        if visual_mode:
            draw_obstacle(block_items, route_plan, intersected_blocks, param)
        raise RouteCreationError('Buffer lines must not overlap with other obstacle cells, Block: ' + ','.join(intersected_block_names))

    # visual
    if visual_mode:
        draw(block_items, route_plan)



# python3.12 process_v10.py --input_path "./input/20240715_test.MDB"
# python3.12 process_v10.py --input_path "./input/20240715_test.MDB" --execute_type alloc
if __name__ == "__main__":
    logging.info(f'arguments: {args}')
    logging.info(f'cmd: python process.py --{" --".join([k + " " + str(v) for k, v in args.items()])}')

    start = time.time()

    data = {
        'Model_line_Data': read_csv(args['input_line_file']),
        'Input_Parameter': [args],
        'Cell_Data': read_geojson(args['input_cell_file'])
    }

    execute_dozer(args)

    logging.info(f'total duration: {time.time() - start} sec')
    
