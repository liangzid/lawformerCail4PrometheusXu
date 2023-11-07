#!/bin/bash
######################################################################
#NEWTRAIN --- 

# new train with new dataset formats.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 17 十月 2023
######################################################################

export env_name="Bert-keras"
export python=/home/nxu/anaconda3/envs/${env_name}/bin/python3
export root_dir="${HOME}/LEVENs/CAIL_2023/"

##-----------------------------------------------------------------------------------------
export device="4"
export epochs=5
export batch_size=4
export lr=3e-5
# export max_seq_length=512
export max_seq_length=256
export pretrained_model_path="${root_dir}/ziliang_test/query_sementics/Lawformer" # todo 
export save_log_path="${root_dir}/log/boring-log.log"
export max_step=50000
export using_data2=1
export lambdaa=250
export dataset_type="overall"
# export dataset_type="onlyLinkAll"
# export dataset_type="only_3-2-1-0_all"
# export dataset_type="cosal_3-2-1-0_all"
# export save_model_path="${pretrained_model_path}/saved_models/data2train_${epochs}${lr}${max_seq_length}${using_data2}${dataset_type}${lambdaa}"
export save_model_path="${pretrained_model_path}/saved_models/data2train_${epochs}${lr}${max_seq_length}${using_data2}${dataset_type}${lambdaa}"

# dataset related
export lblpth="/home/nxu/LEVENs/CAIL_2023/label4train/stage1_train_label.json"
export q_pth="/home/nxu/LEVENs/CAIL_2023/processed_data/stage_1/train_query.json"
export can_dir="/home/nxu/LEVENs/CAIL_2023/processed_data/stage_1/candidate/"

echo "--->>>BEGIN TO TRAIN"
${python} train.py \
	--train=1 \
	--max_seq_length=${max_seq_length} \
	--device=${device} \
	--cuda_num=${device} \
	--epochs=${epochs} \
	--batch_size=${batch_size} \
	--lr=${lr} \
	--pretrained_model_path=${pretrained_model_path} \
	--save_model_path="${save_model_path}" \
	--max_step=${max_step} \
	--lambdaa=${lambdaa}\
	--lblpth=${lblpth}\
	--qpth=${q_pth}\
	--can_dir=${can_dir}\
	--using_data2=${using_data2}\
	--dataset_type=${dataset_type}

export save_model_path="${save_model_path}finally"
# export save_model_path="${save_model_path}e0"
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


echo "RUNNING newtrain.sh DONE."
# newtrain.sh ends here
