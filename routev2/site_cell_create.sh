#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate routev2
input_file=${1}
idx_a=${2}
idx_b=${3}
divisions=${4}
if [ "-" != $divisions ]
then
divisions="--divisions $divisions"
else
divisions=""
fi
line_change_distance=${5}
blade_width=${6}
overlap_rate=${7}
out_path=${8}
echo "=======cell_splite_visualize======="
python /data2/ebim/execute/routev2/createcell/cell_split_visualize.py --input $input_file --idx_a $idx_a --idx_b $idx_b $divisions --output $out_path
input1="$out_path/region1.csv"
output1="$out_path/grid_cells_LBL.csv"
echo "=======cell_create LBL========"
python /data2/ebim/execute/routev2/createcell/cell_create.py -i $input1 -o $output1 -b $blade_width -r $overlap_rate
input2="$out_path/region2.csv"
output2="$out_path/grid_cells_RBL.csv"
echo "=======cell_create RBL======="
python /data2/ebim/execute/routev2/createcell/cell_create.py -i $input2 -o $output2 -b $blade_width -r $overlap_rate
output3="$out_path/grid_cells_SBL.csv"
echo "=======create_sbl======="
python /data2/ebim/execute/routev2/createcell/create_sbl.py -l $output1 -r $output2 -o $output3 -d $line_change_distance -b $blade_width -w $overlap_rate
