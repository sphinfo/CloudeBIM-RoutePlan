#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate routev2
input_file=${1}
output_file=${2}
equipment_width=${3}
attachment_width=${4}
equipment_length=${5}
starting_position=${6}
starting_direction=${7}
python /data2/ebim/execute/routev2/cell_create.py --input_file $input_file --output_file $output_file --equipment_width $equipment_width --attachment_width $attachment_width --starting_position $starting_position --starting_direction $starting_direction
