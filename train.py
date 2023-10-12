"""
======================================================================
TRAIN --- 

train model, blablabla

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 12 October 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import json
import numpy as np

from data import ExtractDataset

import random
import argparse
from tqdm import tqdm
import logging
import os
from os.path import join, exists
from itertools import zip_longest, chain
from datetime import datetime
import pickle

import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import BertTokenizer
from transformers import pipeline

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn


LOGGER = None
PAD_ID = None


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=64,
                        type=int, required=False, help="模型的最大输入长度")
    parser.add_argument("--max_step", default=500,
                        type=int, required=False, help="max step for training.")
    parser.add_argument("--lambdaa", default=0.8,
                        type=int, required=False, help="lambdaa for cosent loss")

    parser.add_argument("--train", default=1, type=int,
                        required=True, help="用以决定是训练模式还是测试模式")
    parser.add_argument('--device', default='6', type=str,
                        required=False, help='设置使用哪些显卡')
    parser.add_argument('--cuda_num', default='6', type=str, required=False)
    
    parser.add_argument('--board_name', default="nothing",
                        type=str, required=False)

    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')

    parser.add_argument('--save_log_path', default='./log/training.log',
                        type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=4, type=int,
                        required=False, help='训练batch size')
    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000,
                        type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int,
                        required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1,
                        type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)

    parser.add_argument('--save_model_path', default='./savemodel/',
                        type=str, required=False, help='对话模型输出路径')
    parser.add_argument('--pretrained_model_path', default='t5-small',
                        type=str, required=True, help='预训练的GPT2模型的路径')

    parser.add_argument('--writer_dir', default='./tensorboard_summary',
                        type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=3933,
                        help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int,
                        default=1, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument("--datasetpath", type=str, required=True, help="设置数据集地址")
    parser.add_argument("--dataset_path_prefix", type=str,
                        required=False, default="/home/liangzi/datasets/soloist/pollution", help="设置数据集地址")

    return parser.parse_args()


def _set_random_seed(seed):
    """
    设置训练的随机种子
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def _create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.save_log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger



def train(model, optimizer, train_loader,args):
    model.train()
    step=0.
    for epoch in range(args.epochs):
        print(f"-------EPOCH {epoch}-------------")
        for i,(q,c3,c2,c15,c1,c05,c0) in enumerate(train_loader):

            ####
            ## link cosine loss
            ## cos(q,c3)>cos(q,c2), cos(q,c2)>cos(q,c15), cos(q,c15>q,c1)
            ## cos(q,c1)>cos(q,c05) cos(q, c05)>cos(q,c0)
            ###
            eq=model(q).pooler_output; # bs, d
            ec3=model(c3).pooler_output;
            ec2=model(c2).pooler_output;
            ec15=model(c15).pooler_output;
            ec1=model(c1).pooler_output;
            ec05=model(c05).pooler_output;
            ec0=model(c0).pooler_output;
            
            
            ######
            ###### calculate the cosent loss
            ######
            # reference: https://spaces.ac.cn/archives/8847
            ## now calculate the variant cosent loss

            # 1. calculate cosine similarity
            cof=torch.nn.CosineSimilarity(dim=1,eps=1e-6);
            co3=cof(eq,ec3);
            co2=cof(eq,ec2);
            co15=cof(eq,ec15);
            co1=cof(eq,ec1);
            co05=cof(eq,ec05);
            co0=cof(eq,ec0);

            ## 2. calculate the 差值 in these orders 
            lambdaa=self.lambdaa
            res=0.
            res+=torch.exp(lambdaa*(co3-co2))
            res+=torch.exp(lambdaa*(co2-co15))
            res+=torch.exp(lambdaa*(co15-co1))
            res+=torch.exp(lambdaa*(co1-co05))
            res+=torch.exp(lambdaa*(co05-co0))

            # now the shape of res should be (bs,1)
            res=torch.sum(res)

            los=torch.log(res+1)
            
            if step%100==0:
                print(f"now the loss is: {los}")

            los.backward()
            if step%args.gradient_accumulation==0:
                alllos+=los.item()
                optimizer.backward()
                optimizer.zero_grad()
            step+=1
            if step>=args.max_step:
                break;

        model.save_pretrained(args.save_model_path +f"e{epoch}")
        tokenizer.save_pretrained(args.save_model_path+f"e{epoch}")
        print(f"save ckpt.")


def main(args):


    global LOGGER
    LOGGER = _create_logger(args)
    # 设置有关设备的问题
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:{}'.format(args.cuda_num) if args.cuda else 'cpu'
    LOGGER.info('using device:  {}'.format(device))
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


    model = AutoModel.from_pretrained(args.pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model = model.to(device)

    if args.seed:
        _set_random_seed(args.seed)

    train_set=ExtractDataset(args,tokenizer,)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    train(model=model, optimizer=optimizer,
          train_loader=train_loader,args)

    
if __name__ == '__main__':
    main()





## running entry
if __name__=="__main__":
    args = setup_train_args()
    # print(type(args.train))
    # print(args.train)

    if args.train == 1:
        main(args)
    else:
        main_test(args)


