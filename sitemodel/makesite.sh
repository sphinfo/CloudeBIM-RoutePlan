#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate sitemodel
model_json=${1}
angle_ratio=${2}
slope_distance=${3}
output_file=${4}
daylight=${5}
ply_file=${6}
declare -l daylight
daylight=$daylight
echo $daylight
if [ ${daylight} = "false" ]
then
	#1차
	python /data2/ebim/execute/sitemodel/generate_1.py $model_json $angle_ratio $slope_distance $ply_file $output_file
else
	#2차
	python /data2/ebim/execute/sitemodel/generate_2.py $model_json $angle_ratio $ply_file $output_file
fi
