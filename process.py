
# COPYRIGHT ⓒ 2021 HANYANG UNIVERSITY. ALL RIGHTS RESERVED.

import sys
import json
import time
import logging
from route_planner.arguments import args
from route_planner import VERSION
from route_planner.constants import TABLE_NAME
from route_planner.util import read_mdb, read_csv, file_name
from route_planner.block import Block
from route_planner.alloc import Dozer, Roller, Grader, Paver
from route_planner.plan import Dozer as DozerRoutePlan, Roller as RollerRoutePlan, MovementPlan, Grader as GraderRoutePlan, Paver as PaverRoutePlan


def execute_movement(block_items, **kwargs):
    input_path, borrow_pit_x, borrow_pit_y, borrow_pit_z, dumping_area_x, dumping_area_y, dumping_area_z, borrow_pit_cut_vol, dumping_area_fill_vol = map(
        kwargs.get, ['input_path', 'borrow_pit_x', 'borrow_pit_y', 'borrow_pit_z', 'dumping_area_x', 'dumping_area_y', 'dumping_area_z', 'borrow_pit_cut_vol', 'dumping_area_fill_vol']
    )
    movement_plan = MovementPlan(block_items).calc_block(borrow_pit_x, borrow_pit_y, borrow_pit_z, dumping_area_x, dumping_area_y, dumping_area_z, borrow_pit_cut_vol, dumping_area_fill_vol)
    MovementPlan.save_output_csv(file_name(input_path), movement_plan)

def execute_grader(block_items, execute_type, equip_type, input_path, input_file_type, output_file_type, start_block_name,
                  work_direction, e=None, s=None, l=None):
    logging.info(f'execute grader')
    # 1-1 output
    rearranged_block, converted_block, work_direction = Block().rearrangement(
        block_items, start_block_name, work_direction)
    # 1-1 output
    Block.save_output_csv(file_name(input_path), rearranged_block)
    if execute_type == 'block': return

    # 1-2
    allocated_cells = Grader().group(rearranged_block=rearranged_block, converted_block=converted_block, s=s, l=l)
    # 1-2 output
    Grader.save_output_csv(file_name(input_path), 'grader', rearranged_block, allocated_cells, converted_block)
    if execute_type == 'alloc': return

    # 1-3
    route_plan = GraderRoutePlan(block_items, converted_block, rearranged_block).calc_route_plan(allocated_cells, start_block_name, work_direction)
    GraderRoutePlan.save_output_csv(file_name(input_path), route_plan)

    if output_file_type == 'all':
        GraderRoutePlan.save_output_png(file_name(input_path), route_plan, rearranged_block, converted_block, allocated_cells, 'grader')

def execute_paver(block_items, execute_type, equip_type, input_path, input_file_type, output_file_type, start_block_name,
                  work_direction, e=None, s=None, l=None):
    logging.info(f'execute paver')
    # 1-1 output
    rearranged_block, converted_block, work_direction = Block().rearrangement(
        block_items, start_block_name, work_direction)
    # 1-1 output
    Block.save_output_csv(file_name(input_path), rearranged_block)
    if execute_type == 'block': return

    # 1-2
    allocated_cells = Paver().group(rearranged_block=rearranged_block, converted_block=converted_block, s=s, l=l)
    # 1-2 output
    Paver.save_output_csv(file_name(input_path), 'paver', rearranged_block, allocated_cells, converted_block)
    if execute_type == 'alloc': return

    # 1-3
    route_plan = PaverRoutePlan(block_items, converted_block, rearranged_block).calc_route_plan(allocated_cells, start_block_name, work_direction)
    PaverRoutePlan.save_output_csv(file_name(input_path), route_plan)

    if output_file_type == 'all':
        PaverRoutePlan.save_output_png(file_name(input_path), route_plan, rearranged_block, converted_block, allocated_cells, 'paver')

def execute_roller(block_items, execute_type, equip_type, input_path, input_file_type, output_file_type, start_block_name,
                  work_direction, e=None, s=None, l=None, compaction_count=None):
    logging.info(f'execute roller')
    # 1-1 output
    rearranged_block, converted_block, work_direction = Block().rearrangement(
        block_items, start_block_name, work_direction)
    # 1-1 output
    Block.save_output_csv(file_name(input_path), rearranged_block)
    if execute_type == 'block': return

    # 1-2
    allocated_cells = Roller().group(rearranged_block=rearranged_block, converted_block=converted_block, s=s, l=l)
    # 1-2 output
    Roller.save_output_csv(file_name(input_path), 'roller', rearranged_block, allocated_cells, converted_block)
    if execute_type == 'alloc': return

    # 1-3
    route_plan = RollerRoutePlan(block_items, converted_block, rearranged_block).calc_route_plan(allocated_cells, start_block_name, work_direction, compaction_count)
    RollerRoutePlan.save_output_csv(file_name(input_path), route_plan)

    if output_file_type == 'all':
        RollerRoutePlan.save_output_png(file_name(input_path), route_plan, rearranged_block, converted_block, allocated_cells, 'roller', compaction_count)


def execute_dozer(block_items, execute_type, equip_type, input_path, input_file_type, output_file_type, start_block_name,
                  work_direction, sub_type=None, e=None, s=None, l=None, allowable_error_height=None):
    logging.info(f'execute dozer')
    # 1-1 output
    rearranged_block, converted_block, work_direction = Block().rearrangement(
        block_items, start_block_name, work_direction)
    # 1-1 output
    Block.save_output_csv(file_name(input_path), rearranged_block)
    
    if execute_type == 'block': return
    
    # delete
    # rearranged_block = [
    #     ['BL_3', 'BL_4', 'BL_5', 'BL_6', 'BL_7', 'BL_8', 'BL_9', 'BL_10', 'BL_11', 'BL_12', 'BL_13'],
    #     ['BL_14', 'BL_15', 'BL_16', 'BL_17', 'BL_18', 'BL_19', 'BL_20', 'BL_21', 'BL_22', 'BL_23', 'BL_24'],
    #     ['BL_25', 'BL_26', 'BL_27', 'BL_28', 'BL_29', 'BL_30', 'BL_31', 'BL_32', 'BL_33', 'BL_34', 'BL_35'],
    #     ['BL_37', 'BL_38', 'BL_39', 'BL_40', 'BL_41', 'BL_42', 'BL_43', 'BL_44', 'BL_45', 'BL_46', 'BL_V'],
    #     ['BL_47', 'BL_48', 'BL_49', 'BL_50', 'BL_51', 'BL_52', 'BL_53', 'BL_54', 'BL_55', 'BL_56',  'BL_V'],
    #     ['BL_67', 'BL_68', 'BL_69', 'BL_70', 'BL_71', 'BL_72', 'BL_73', 'BL_74', 'BL_75', 'BL_76', 'BL_V'],
    #     ['BL_V', 'BL_78', 'BL_79', 'BL_80', 'BL_81', 'BL_82', 'BL_83', 'BL_84', 'BL_85', 'BL_86', 'BL_V'],
    #     ['BL_V', 'BL_87', 'BL_88', 'BL_89', 'BL_90', 'BL_91', 'BL_92', 'BL_93', 'BL_94', 'BL_95', 'BL_V'],
    #     ['BL_V', 'BL_96', 'BL_97', 'BL_98', 'BL_99', 'BL_100', 'BL_101', 'BL_102', 'BL_103', 'BL_104', 'BL_V']]

    dozer_group = getattr(Dozer(), f'{sub_type}_group')
    # 1-2
    allocated_cells = dozer_group(
        rearranged_block=rearranged_block, converted_block=converted_block, e=e, s=s, l=l,
        allowable_error_height=allowable_error_height)
    # 1-2 output
    Dozer.save_output_csv(file_name(input_path), f'dozer_{sub_type}', rearranged_block, allocated_cells, converted_block)

    if execute_type == 'alloc': return
    
    # delete
    # allocated_cells = {
    #     'k1': {'j2': ['BL_96', 'BL_87', 'BL_78'], 'j3': ['BL_97', 'BL_88'], 'j4': ['BL_98', 'BL_89'], 'j5': ['BL_99', 'BL_90', 'BL_81'], 'j6': ['BL_100', 'BL_91', 'BL_82'], 'j7': ['BL_101', 'BL_92', 'BL_83'], 'j8': ['BL_102', 'BL_93', 'BL_84'], 'j9': ['BL_103', 'BL_94', 'BL_85'], 'j10': ['BL_104', 'BL_95', 'BL_86']} ,
    #     'k2': {'j1': ['BL_67', 'BL_57', 'BL_47'], 'j2': ['BL_68', 'BL_58'], 'j3': ['BL_79'], 'j4': ['BL_80', 'BL_70', 'BL_60'], 'j5': ['BL_71', 'BL_61', 'BL_51'], 'j6': ['BL_72', 'BL_62', 'BL_52'], 'j7': ['BL_73', 'BL_63', 'BL_53'], 'j8': ['BL_74', 'BL_64', 'BL_54'], 'j9': ['BL_75', 'BL_65', 'BL_55'], 'j10': ['BL_76', 'BL_66', 'BL_56']},
    #     'k3': {'j1': ['BL_37', 'BL_25', 'BL_14'], 'j2': ['BL_48', 'BL_38'], 'j3': ['BL_69', 'BL_59'], 'j4': ['BL_50', 'BL_40'], 'j5': ['BL_41', 'BL_29', 'BL_18'], 'j6': ['BL_42', 'BL_30', 'BL_19'], 'j7': ['BL_43', 'BL_31', 'BL_20'], 'j8': ['BL_44', 'BL_32', 'BL_21'], 'j9': ['BL_45', 'BL_33', 'BL_22'], 'j10': ['BL_46', 'BL_34', 'BL_23'], 'j11': ['BL_35', 'BL_24', 'BL_13']},
    #     'k4': {'j1': ['BL_3'], 'j2': ['BL_26', 'BL_15', 'BL_4'], 'j3': ['BL_49'], 'j4': ['BL_28', 'BL_17'], 'j5': ['BL_7'], 'j6': ['BL_8'], 'j7': ['BL_9'], 'j8': ['BL_10'], 'j9': ['BL_11'], 'j10': ['BL_12']},
    #     'k5': {'j3': ['BL_39'], 'j3': ['BL_6']},
    #     'k6': {'j3': ['BL_27', 'BL_16', 'BL_5']}
    # }
    # delete
    # converted_block = Block.index_block(converted_block, rearranged_block)
    # 1-3
    dozer_route_plan = getattr(DozerRoutePlan(block_items, converted_block), f'calc_{sub_type}_route_plan')
    route_plan = dozer_route_plan(rearranged_block, allocated_cells, start_block_name, work_direction, allowable_error_height, s)
    DozerRoutePlan.save_output_csv(file_name(input_path), route_plan)
    
    # PNG
    if output_file_type == 'all':
        DozerRoutePlan.save_output_png(file_name(input_path), route_plan, rearranged_block, converted_block, allocated_cells, f'dozer_{sub_type}')


FUNCTION_MAP = {
    'dozer': execute_dozer,
    'roller': execute_roller,
    'movement': execute_movement,
    'grader' : execute_grader,
    'paver' : execute_paver
}


# python process.py --execute_type all --equip_type dozer --input_path "./input/sample_quadangle2_2.7x3.MDB"  --input_file_type mdb --output_file_type all --start_block_name BL_21 --work_direction 1 --e 10 --s 3 --l 15 --allowable_error_height 0.05 --sub_type fill
if __name__ == "__main__":
    logging.info(f'arguments: {args}')
    logging.info(f'cmd: python process.py --{" --".join([k + " " + str(v) for k, v in args.items()])}')
    start = time.time()
    block_items = read_csv(args.get('input_path')) if args.get('input_file_type') == 'csv' else read_mdb(args.get('input_path'), TABLE_NAME)
    logging.info(f'Read {args.get("input_path")}.. duration: {time.time() - start} sec')
    FUNCTION_MAP.get(args.get('equip_type'))(block_items, **args)
    logging.info(f'total duration: {time.time() - start} sec')
