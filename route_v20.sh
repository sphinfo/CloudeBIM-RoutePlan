#!/usr/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate route
input_path=${1}
output_file=${2}
equipment_width=${3}
blade_width=${4}
blade_capacity=${5}
obstacle_cell=${6}
if [ -z $obstacle_cell ]
then
obstacle_cell='-'
fi
equipment="${7}"
equipment_length=${8}
repeated_rate=${9}
min_fwdist=${10}
turning_radius=${11}
if [ "-" != $turning_radius ]
then
turning_radius="--turning_radius $turning_radius"
else
turning_radius=""
fi

python /data2/ebim/execute/process_v20.py --input_path $input_path --output_file $output_file --blade_capacity $blade_capacity --blade_width $blade_width --equipment_width $equipment_width --obstacle_cell $obstacle_cell --equipment $equipment --equipment_length $equipment_length --repeated_rate $repeated_rate --min_fwdist $min_fwdist $turning_radius


