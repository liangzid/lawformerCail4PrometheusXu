"""
======================================================================
DATA --- 

    Author: Zi Liang <frostliang@lilith.com>
    Copyright © 2023, lilith, all rights reserved.
    Created: 12 October 2023

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 12 October 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
from torch.utils.data import DataLoader,Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import time
import pickle
import os
import json
import random
from pathlib import Path
import re

class ExtractDataset(Dataset):

    def __init__(self,args,tokenizer,tp="vanilla",mode="train",
                 lblpth="/home/nxu/LEVENs/CAIL_2023/label4train/stage1_train_label.json",
                 q_pth="/mnt/d/backup-deletethis/train_query.json",
                 can_dir="/home/nxu/LEVENs/CAIL_2023/processed_data",
                 ):
        self.lblpth=lblpth

        # from collections import OrderedDict
        with open(self.lblpth, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)

        # query, cand_score3, cand_score2, cand_score1
        self.train_ls=[]

        for query_idx in data:
            idx_s3=-1
            idx_s2=-1
            idx_s15=-1
            idx_s1=-1
            idx_s05=-1
            idx_s0=-1
            candidates=data[query_idx]
            for can in candidates:
                label=candidates[can]
                if idx_s3==-1:
                    if label==3:
                        idx_s3=can
                if idx_s2==-1:
                    if label==2:
                        idx_s2=can
                if idx_s15==-1:
                    if label==2:
                        idx_s15=can
                if idx_s1==-1:
                    if label==1:
                        idx_s1=can
                if idx_s05==-1:
                    if label==2:
                        idx_s05=can
                if idx_s0==-1:
                    if label==1:
                        idx_s0=can
                if idx_s3!=-1 and idx_s2!=-1 and idx_s15!=-1\
                and idx_s1!=-1 and idx_s05!=-1 and idx_s0!=-1:
                    break
            
            if idx_s3!=-1 and idx_s2!=-1 and idx_s15!=-1\
                and idx_s1!=-1 and idx_s05!=-1 and idx_s0!=-1:
                self.train_ls.append((query_idx,idx_s3,idx_s2,idx_s15,
                                  idx_s1,idx_s05,idx_s0))
            else:
                print(">>>> invalid sample with query idx:{query_idx}")
        print(f"all_dataset_len:{len(self.train_ls)}")
        del data

        self.train_dict={}
        for (q,c3,c2,c15,c1,c05,c0) in self.train_ls:
            self.train_dict[q]=(c3,c2,c15,c1,c05,c0)
        
        ## collect the text of both query and candidates, into a new
        # list of tuples


        ##1. first parse the query dataset.
        # from collections import OrderedDict
        self.query_pth=q_pth
        with open(self.query_pth, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)

        self.text_dict={}
        self.qls=[]
        self.c3s=[]
        self.c2s=[]
        self.c15s=[]
        self.c1s=[]
        self.c05s=[]
        self.c0s=[]
        for sample in data:
            if sample["id"] in self.train_dict:
                query=sample["query"]
                # then obtain all candidates text.
                can_txts=[]
                cans=self.train_dict[sample["id"]]
                prefix_can_pth=can_dir
                for c in cans:
                    # from collections import OrderedDict
                    with open(prefix_can_pth+f"{c}.json",
                              'r',encoding='utf8') as f:
                        cd=json.load(f,object_pairs_hook=OrderedDict)
                    can_txts.append(cd["qw"])
                self.text_dict[query]=can_txts
                self.qls.append(query)
                self.c3s.append(can_txts[0])
                self.c2s.append(can_txts[1])
                self.c15s.append(can_txts[2])
                self.c1s.append(can_txts[3])
                self.c05s.append(can_txts[4])
                self.c0s.append(can_txts[5])
            else:
                print(f"WARNING: an index {sample['id']} not seen.")

        ## finally, tokenize it.
        self.tokenizer=tokenizer
        mx=args.max_seq_length

        qls_ten=self.tokenizer(self.qls,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        
        c3_ten=self.tokenizer(self.c3s,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids

        c2_ten=self.tokenizer(self.c2s,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        
        c15_ten=self.tokenizer(self.c15s,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        
        c1_ten=self.tokenizer(self.c1s,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
                        
        c05_ten=self.tokenizer(self.c05s,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids

        c0_ten=self.tokenizer(self.c0s,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids

        self.datals=[]
        for i in range(len(qls_ten)):
            self.datals.append((qls_ten[i],c3_ten[i],c2_ten[i],
                                c15_ten[i],
                                c1_ten[i],c05_ten[i],c0_ten[i]))

    def __getitem__(self,i):
        return self.datals[i]

    def __len__(self):
        return len(self.datals)
                

