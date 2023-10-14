"""
======================================================================
VALID ---

valid the query results of lawformer.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 14 十月 2023
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
from collections import OrderedDict

def valid(tokenizer,model,device,test_query_pth,args):
    model.eval()

    # load top-1000 candidates
    all_candidates_for_query="/home/nxu/LEVENs/CAIL_2023/qwt_test/prediction/combined_prediction_jieba_fact.json"

    # from collections import OrderedDict
    with open(all_candidates_for_query, 'r',encoding='utf8') as f:
        q_c_dict=json.load(f,object_pairs_hook=OrderedDict)


    cof=torch.nn.CosineSimilarity(dim=1,eps=1e-6)
    new_result={}
    # load all query.
    with open(test_query_pth, 'r',encoding='utf8') as f:
        data=f.readlines()
        for x in data:
            if "\n" in x:
                x=x.split("\n")[0]
            adict=json.loads(x,object_pairs_hook=OrderedDict)
            qidx=str(adict["id"])
            top_1000_ls=q_c_dict[qidx]
            print(f"len of top1k ls: {len(top_1000_ls)}")

            # query:
            qtxt=adict["query"]
            can_txt_ls=[]
            can_pth_prefix="/home/nxu/LEVENs/CAIL_2023/processed_data/stage_1/candidate/"
            for cidx in top_1000_ls:
                with open(can_pth_prefix+cidx+".json", 'r',encoding='utf8') as f:
                    data=json.load(f,object_pairs_hook=OrderedDict)
                    can_txt_ls.append(data["qw"])

            # tokenize all candidate texts
            mx=args.max_seq_length
            can_ten=tokenizer(can_txt_ls,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
            # tokenize query:
            q_ten=tokenizer([qtxt],padding="longest",max_length=mx,
                            truncation=True,return_tensors="pt").input_ids
            q_ten=q_ten.to(device)
            with torch.no_grad():
                qe=model(q_ten).pooler_output

            cosls=[]
            for cant in can_ten:
                cant=cant.to(device).unsqueeze(0)
                with torch.no_grad():
                    ce=model(cant).pooler_output
                cosls.append(cof(qe,ce)[0])
                del cant
                del ce
            print(f"cosls: {cosls}")

            # sorted it.
            sorted_cosls = sorted(cosls, reverse=True) 
            indexes = [cosls.index(i) for i in sorted_cosls]

            ranked_can_idx=[top_1000_ls[jjj] for jjj in indexes]
            new_result[qidx]=ranked_can_idx
        del q_ten
            
    with open("temp_test_ourlawformer.json", 'w',encoding='utf8') as f:
        json.dump(new_result,f,ensure_ascii=False,indent=4)
        print("re-ranked idx save done.")
    