#/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
cd /ebim/route
python process.py --execute_type all --equip_type grader --input_path "./input/sample_simple_2.7x3.csv"  --input_file_type csv --start_block_name BL_9  --work_direction 1 --e 10 --s 3 --l 100 --output_file_type all
