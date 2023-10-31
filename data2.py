"""
======================================================================
DATA2 ---

fix the bug of using only-half-of samples in original `data.py` files.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 17 十月 2023
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
from collections import OrderedDict

class ExtractDataset2(Dataset):

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


        # cos(a,b)>cos(c,d)
        self.q3=[]
        self.q2=[]
        self.q15=[]
        self.q1=[]
        self.q05=[]
        self.q0=[]

        all_candidate_idxes=[]
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
                        self.q3.append((query_idx,can))
                        all_candidate_idxes.append(can)
                if idx_s2==-1:
                    if label==2:
                        idx_s2=can
                        self.q2.append((query_idx,can))
                        all_candidate_idxes.append(can)
                if idx_s15==-1:
                    if label==2:
                        idx_s15=can
                        self.q15.append((query_idx,can))
                        all_candidate_idxes.append(can)
                if idx_s1==-1:
                    if label==1:
                        idx_s1=can
                        self.q1.append((query_idx,can))
                        all_candidate_idxes.append(can)
                if idx_s05==-1:
                    if label==2:
                        idx_s05=can
                        self.q05.append((query_idx,can))
                        all_candidate_idxes.append(can)
                if idx_s0==-1:
                    if label==1:
                        idx_s0=can
                        self.q0.append((query_idx,can))
                        all_candidate_idxes.append(can)
                if idx_s3!=-1 and idx_s2!=-1 and idx_s15!=-1\
                and idx_s1!=-1 and idx_s05!=-1 and idx_s0!=-1:
                    break
            
            if idx_s3!=-1 and idx_s2!=-1 and idx_s15!=-1\
                and idx_s1!=-1 and idx_s05!=-1 and idx_s0!=-1:
                self.train_ls.append((query_idx,idx_s3,idx_s2,idx_s15,
                                  idx_s1,idx_s05,idx_s0))
            # else:
            #     print(f">>>> invalid sample with query idx:{query_idx}")

        # print(f"all_dataset_len:{len(self.train_ls)}")
        del data
        print(f"query_score3 num:{len(self.q3)}")
        print(f"query_score2 num:{len(self.q2)}")
        print(f"query_score15 num:{len(self.q15)}")
        print(f"query_score1 num:{len(self.q1)}")
        print(f"query_score05 num:{len(self.q05)}")
        print(f"query_score0 num:{len(self.q0)}")

        ## translate them into text
        self.q_dict={}
        self.can_dict={}

        ##1. first parse the query dataset.
        data=[]
        self.query_pth=q_pth
        with open(self.query_pth, 'r',encoding='utf8') as f:
            # its everyline is a dict.
            lines=f.readlines()
            for l  in lines:
                if "\n" in l:
                    l=l.split("\n")[0]
                data.append(json.loads(l,object_pairs_hook=OrderedDict))
        for i,sample in enumerate(data):
            # if i>2000:
                # break
            # self.q_dict[str(sample["id"])]=sample["fact"]
            self.q_dict[str(sample["id"])]=sample["query"]

        ## parse candidate dirs
        flnes=os.listdir(can_dir)
        for fname in flnes:
            cname=fname.split(".json")[0]
            if cname not in all_candidate_idxes and int(cname)not in all_candidate_idxes:
                continue
            with open(can_dir+fname,
                        'r',encoding='utf8') as f:
                cd=json.load(f,object_pairs_hook=OrderedDict)
            # self.can_dict[cname]=(cd["fact"])
            self.can_dict[cname]=(cd["qw"])

        ## translate index list into text list.
        self.q3=[(self.q_dict[str(q)],self.can_dict[str(c)]) for q,c in self.q3]
        self.q2=[(self.q_dict[str(q)],self.can_dict[str(c)]) for q,c in self.q2]
        self.q15=[(self.q_dict[str(q)],self.can_dict[str(c)]) for q,c in self.q15]
        self.q1=[(self.q_dict[str(q)],self.can_dict[str(c)]) for q,c in self.q1]
        self.q05=[(self.q_dict[str(q)],self.can_dict[str(c)]) for q,c in self.q05]
        self.q0=[(self.q_dict[str(q)],self.can_dict[str(c)]) for q,c in self.q0]

        del self.q_dict
        del self.can_dict

        # save it.
        with open("./save_trainset_query_candidate.json", 'w',encoding='utf8') as f:
            json.dump([self.q3,self.q2,self.q15,self.q1,self.q05,self.q0]
                      ,f,ensure_ascii=False,indent=4)

        alpha=1.

        self.four_ls=[]
        if tp=="overall":
            for a in self.q3:
                for b in self.q2:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q3:
                for b in self.q15:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q3:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q3:
                for b in self.q05:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q3:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q15:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q05:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q15:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q15:
                for b in self.q05:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q15:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q1:
                for b in self.q05:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q1:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q05:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
        elif tp=="onlyLinkAll":
            for a in self.q3:
                for b in self.q2:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q15:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q15:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q1:
                for b in self.q05:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q05:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
        elif tp=="only_3-2-1-0_all":
            for a in self.q3:
                for b in self.q2:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q3:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            print("q3 done.")
            for a in self.q3:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q1:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
        elif tp=="only_3-2-1-0_link":
            for a in self.q3:
                for b in self.q2:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q2:
                for b in self.q1:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
            for a in self.q1:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
        elif tp=="only_3-0":
            for a in self.q3:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))
        else:
            for a in self.q3:
                for b in self.q0:
                    if random.random()>alpha:
                        continue
                    self.four_ls.append((a[0],a[1],b[0],b[1]))

        del self.q3
        del self.q2
        del self.q1
        del self.q0
        del self.q15
        del self.q05

        ## shuffle list and and then sample some of them
        dataset_num=30000
        random.shuffle(self.four_ls)
        self.four_ls=self.four_ls[:dataset_num]

        ## finally, tokenize it.
        self.tokenizer=tokenizer
        mx=args.max_seq_length
        q1,c1,q2,c2=zip(*self.four_ls)
        q1=list(q1)
        q2=list(q2)
        c1=list(c1)
        c2=list(c2)
        print("all sample num",len(self.four_ls))

        print("now begin to tokenize")
        q1=self.tokenizer(q1,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        q2=self.tokenizer(q2,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        c1=self.tokenizer(c1,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        c2=self.tokenizer(c2,padding="longest",
                               max_length=mx,
                               truncation=True,
                               return_tensors="pt").input_ids
        print("tokenize done")

        self.datals=[]
        # print(qls_ten.shape)
        for i in range(len(q1)):
            self.datals.append((q1[i],c1[i],q2[i],c2[i]))
        print("all tuple num",len(self.datals))

    def __getitem__(self,i):
        return self.datals[i]

    def __len__(self):
        return len(self.datals)
                


