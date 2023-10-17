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

export env_name="Bert-keras"
export python=/home/nxu/anaconda3/envs/${env_name}/bin/python3
export root_dir="${HOME}/LEVENs/CAIL_2023/"

##-----------------------------------------------------------------------------------------
export device="6"
export epochs=5
export batch_size=1
export lr=3e-5
export max_seq_length=768
export pretrained_model_path="${root_dir}/ziliang_test/query_sementics/Lawformer" # todo 
export save_log_path="${root_dir}/log/boring-log.log"
export save_model_path="${pretrained_model_path}/saved_models/manyallretrain_${epochs}${lr}${max_seq_length}"
export max_step=20000

# dataset related
export lblpth="/home/nxu/LEVENs/CAIL_2023/label4train/stage1_train_label.json"
export q_pth="/home/nxu/LEVENs/CAIL_2023/processed_data/stage_1/train_query.json"
export can_dir="/home/nxu/LEVENs/CAIL_2023/processed_data/stage_1/candidate/"

# echo "--->>>BEGIN TO TRAIN"
# ${python} train.py \
# 	--train=1 \
# 	--max_seq_length=${max_seq_length} \
# 	--device=${device} \
# 	--cuda_num=${device} \
# 	--epochs=${epochs} \
# 	--batch_size=${batch_size} \
# 	--lr=${lr} \
# 	--pretrained_model_path=${pretrained_model_path} \
# 	--save_model_path="${save_model_path}" \
# 	--max_step=${max_step} \
# 	--lambdaa=20\
# 	--lblpth=${lblpth}\
# 	--qpth=${q_pth}\
# 	--can_dir=${can_dir}


export save_model_path="${pretrained_model_path}/saved_models/manyallretrain_${epochs}${lr}${max_seq_length}finally"
echo "---> BEGIN to TEST"
${python} train.py \
	--train=0 \
	--max_seq_length=${max_seq_length} \
	--device=${device} \
	--cuda_num=${device} \
	--epochs=${epochs} \
	--batch_size=${batch_size} \
	--lr=${lr} \
	--pretrained_model_path=${pretrained_model_path} \
	--save_model_path="${save_model_path}" \
	--max_step=${max_step} \
	--lambdaa=20\
	--lblpth=${lblpth}\
	--qpth=${q_pth}\
	--can_dir=${can_dir}

echo "RUNNING run_train.sh DONE."
# run_train.sh ends here
