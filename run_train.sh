#!/bin/bash
######################################################################
#RUN_TRAIN ---

# run train file to begin a training exam.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 12 October 2023
######################################################################

######################### Commentary ##################################
##  
######################################################################

export env_name="??"
export python=/home/nxu/anaconda3/envs/env_name/bin/python3
export root_dir="${HOME}/nxu/LEVENs/CAIL_2023/"

##-----------------------------------------------------------------------------------------
export device="7"
export epochs=5
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/?" # todo 
export save_log_path="${root_dir}/log/boring-log.log"
export save_model_path="${pretrained_model_path}/saved_models/${epochs}${lr}${max_sql_length}"
export max_step=20000
echo "--->>>BEGIN TO TRAIN"
${python} train.py \
	--train=1 \
	--max_seq_length=${max_seq_length} \
	--device=${device} \
	--cuda_num=${device} \
	--epochs=${epochs} \
	--batch_size=${max_seq_length} \
	--lr=${lr} \
	--pretrained_model_path=${pretrained_model_path} \
	--save_model_path="${save_model_path}" \
	--max_step=${max_step}
done







echo "RUNNING run_train.sh DONE."
# run_train.sh ends here
