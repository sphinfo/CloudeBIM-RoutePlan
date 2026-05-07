#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
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
cycle_num=${10}
line_change_way=${11}
equipment_length=${12}
start_line=${13}
end_line=${14}
blade_front_distance=${15}
obstacles=${16}
echo $obstacles
if [ -z $obstacles ]
then
obstacles=""
else
obstacles="--obstacles $obstacles"
fi

python /data2/ebim/execute/routev2/R_G_route_planner_part2_ver_6.py --input_file $input_file --output_file $output_file --equipment_width $equipment_width --attachment_width $attachment_width --safety_line $safety_line --x_min $x_min --turning_radius $turning_radius --starting_position $starting_position --starting_direction $starting_direction --cycle_num $cycle_num --line_change_way $line_change_way --equipment_length $equipment_length --start_line $start_line --end_line $end_line --blade_front_distance $blade_front_distance $obstacles
