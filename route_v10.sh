#!/usr/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate route
input_cell_file=${1}
input_line_file=${2}
output_file=${3}
Equipment_Width=${4}
Blade_Width=${5}
Blade_Capacity=${6}
Start_Line=${7}
Obstacle_Cell=${8}
if [ -z $Obstacle_Cell ]
then
Obstacle_Cell='-'
fi
equipment="${9}"
equipment_length=${10}
Repeated_rate=${11}
Min_Fwdist=${12}
turning_radius=${13}
if [ "-" != $turning_radius ]
then
turning_radius="--turning_radius $turning_radius"
else
turning_radius=""
fi
end_line=${14}
if [ "-" != $end_line ]
then
end_line="--end_line $end_line"
else
end_line=""
fi

python /data2/ebim/execute/process_v10.py --input_cell_file $input_cell_file --input_line_file $input_line_file --output_file $output_file --Blade_Capacity $Blade_Capacity --Blade_Width $Blade_Width --Equipment_Width $Equipment_Width --Start_Line $Start_Line --Obstacle_Cell $Obstacle_Cell --equipment $equipment --equipment_length $equipment_length --Repeated_rate $Repeated_rate --Min_Fwdist $Min_Fwdist $turning_radius $end_line
