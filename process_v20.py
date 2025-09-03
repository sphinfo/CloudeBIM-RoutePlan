
# COPYRIGHT ⓒ 2024 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.


import time
import logging
from route_planner_v20.arguments import args
from route_planner_v20.util import read_input_files
from route_planner_v20.block import Block
from route_planner_v20.alloc import Alloc
from route_planner_v20.plan import RoutePlan
from route_planner_v20.plot import draw, draw_obstacle
from route_planner_v20.constants import START_BLOCK, LEFT_BLOCK, RIGHT_BLOCK


def execute(args: dict):
    cell_data, outline_data = read_input_files(args.get('input_path'))

    params, block_items, outline_items, org_blocks = Block.valid_parameter(args, cell_data, outline_data)

    # 할당셀
    alloc_obj = Alloc(params)
    _ = alloc_obj.alloc(block_items[START_BLOCK], block_type=START_BLOCK)
    _ = alloc_obj.alloc(block_items[LEFT_BLOCK], block_type=LEFT_BLOCK)
    allocate_cell, allocate_cell_names, allocate_outline_cell = alloc_obj.alloc(block_items[RIGHT_BLOCK], block_type=RIGHT_BLOCK)

    # 0, -1, -2 행 set
    Block.set_bottom_by_opposite(block_items[LEFT_BLOCK], block_items[RIGHT_BLOCK])

    route_plan_obj = RoutePlan(params)
    _ = route_plan_obj.calc_start_route_plan(
        block_items=block_items[START_BLOCK], allocate_cell=allocate_cell[START_BLOCK], 
        allocate_cell_names=allocate_cell_names[START_BLOCK], block_type=START_BLOCK)

    _ = route_plan_obj.calc_route_plan(
        block_items=block_items[LEFT_BLOCK],
        allocate_cell=allocate_cell[LEFT_BLOCK], 
        allocate_cell_names=allocate_cell_names[LEFT_BLOCK],
        allocate_outline_cell=allocate_outline_cell[LEFT_BLOCK],
        outline_items=outline_items[LEFT_BLOCK],
        block_type=LEFT_BLOCK)
    route_plan = route_plan_obj.calc_route_plan(
        block_items=block_items[RIGHT_BLOCK],
        allocate_cell=allocate_cell[RIGHT_BLOCK], 
        allocate_cell_names=allocate_cell_names[RIGHT_BLOCK],
        allocate_outline_cell=allocate_outline_cell[RIGHT_BLOCK],
        outline_items=outline_items[RIGHT_BLOCK],
        block_type=RIGHT_BLOCK)

    route_plan_obj.save_output_csv(args.get('output_file'))
    
    # # visual
    if args.get('visual_mode'):
        draw(Block.merge_bl(block_items, set(org_blocks)), route_plan)



# python3.12 process_v20.py --input_path ./input/광명2_3_7/ --blade_capacity 2.6 --blade_width 2.7 --equipment_width 6 --output_path ./output/광명2_3_7/ --required_line_change_distance 5 --equipment_length 4.75 --repeated_rate 0.5 --min_fwdist 5
# python3.12 process_v20.py --input_path ./input/광명2_3_7/ --blade_capacity 2.6 --blade_width 2.7 --equipment_width 6 --output_path ./output/광명2_3_7/ --required_line_change_distance 5 --equipment_length 4.75 --repeated_rate 0.5 --min_fwdist 5 --visual_mode
if __name__ == "__main__":
    logging.info(f'arguments: {args}')
    logging.info(f'cmd: python process.py --{" --".join([k + " " + str(v) for k, v in args.items()])}')

    start = time.time()

    execute(args)

    logging.info(f'total duration: {time.time() - start} sec')
    