#!/usr/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate route
input_cell_file=${1}
input_line_file=${2}
output_file=${3}
Equipment_Width=${4}
Blade_Width=${5}
Blade_Capacity=${6}
Start_Line=${7}
python /data2/ebim/execute/process_v10.py --input_cell_file $input_cell_file --input_line_file $input_line_file --output_file $output_file --Blade_Capacity $Blade_Capacity --Blade_Width $Blade_Width --Equipment_Width $Equipment_Width --Start_Line $Start_Line
