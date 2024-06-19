#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate routev2
input_file=${1}
output_file=${2}
equipment_width=${3}
attachment_width=${4}
safety_line=${5}
x_min=${6}
turning_radius=${7}
starting_position=${8}
starting_direction=${9}
python /ebim/routev2_test/R_G_route_planner_part1_ver_6.py --input_file $input_file --output_file $output_file --equipment_width $equipment_width --attachment_width $attachment_width --safety_line $safety_line --x_min $x_min --turning_radius $turning_radius --starting_position $starting_position --starting_direction $starting_direction
