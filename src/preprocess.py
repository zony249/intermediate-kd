import os
import sys 
from argparse import ArgumentParser, Namespace 
import wget 

import torch 
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm 
from transformers import T5Tokenizer, T5TokenizerFast  

from utils import mkdir, download
from env import Env

def download_wmt14(args: Namespace):
    mkdir(os.path.join(args.data_dir, "wmt14"))

    download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en", dest_path=os.path.join(args.data_dir, "wmt14", "train.en"))
    download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de", dest_path=os.path.join(args.data_dir, "wmt14", "train.de"))

def preprocess_wmt14(args: Namespace, tok):
    download_wmt14(args) 

    if os.path.exists(os.path.join(args.preprocess, "wmt14", "train")):
        print("WMT14 is already preprocessed. Skipping...")
        return

    mkdir(os.path.join(args.preprocess, "wmt14"))
    mkdir(os.path.join(args.preprocess, "wmt14", "train", "en"))
    mkdir(os.path.join(args.preprocess, "wmt14", "train", "de"))

    with open(os.path.join(args.data_dir, "wmt14", "train.en")) as f:
        line_idx = 0
        file_idx = 0
        lines = []
        for line in tqdm(f, desc="Preprocessing train.en"):
            lines.append(line)
            line_idx += 1
            if line_idx >= args.batch_size:

                tokenized = tok(lines, return_tensors="np", padding="longest")
                input_ids, attention_mask = tokenized.input_ids, tokenized.attention_mask 

                np.save(os.path.join(args.preprocess, "wmt14", "train", "en", str(file_idx) + ".npy"), input_ids)  
                np.save(os.path.join(args.preprocess, "wmt14", "train", "en", str(file_idx) + "_attn.npy"), attention_mask)

                lines = []
                line_idx = 0
                file_idx += 1
    
    with open(os.path.join(args.data_dir, "wmt14", "train.de")) as f:
        line_idx = 0
        file_idx = 0
        lines = []
        for line in tqdm(f, desc="Preprocessing train.de"):
            lines.append(line)
            line_idx += 1
            if line_idx >= args.batch_size:

                tokenized = tok(lines, return_tensors="np", padding="longest")
                input_ids, attention_mask = tokenized.input_ids, tokenized.attention_mask 

                np.save(os.path.join(args.preprocess, "wmt14", "train", "de", str(file_idx) + ".npy"), input_ids)  
                np.save(os.path.join(args.preprocess, "wmt14", "train", "de", str(file_idx) + "_attn.npy"), attention_mask)

                lines = []
                line_idx = 0
                file_idx += 1




if __name__ == "__main__":
    
    Env.add_train_args()
    Env.parse_args()
    Env.info()

    
    tok = T5TokenizerFast.from_pretrained("t5-base") 

    print(tok.is_fast)

    preprocess_wmt14(Env.args, tok)


    pass
