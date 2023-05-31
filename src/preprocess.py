import os
import sys 
from argparse import ArgumentParser, Namespace 
import wget 
import tarfile
import glob
import re
import itertools

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




def download_wmt16(args: Namespace):
    wmt16_dir = os.path.join(args.data_dir, "wmt16")
    tar_save_file = os.path.join(args.data_dir, "wmt16", "ro-en.tgz")

    mkdir(os.path.join(args.data_dir, "wmt16"))
    download("https://www.statmt.org/europarl/v7/ro-en.tgz", dest_path=tar_save_file)
    
    
    if not os.path.exists(os.path.join(wmt16_dir, "train.en")):
        file = tarfile.open(name=tar_save_file, mode="r")
        file.extractall(path=wmt16_dir)
        
        en_file = glob.glob(os.path.join(wmt16_dir, "*.en"))[0]
        ro_file = glob.glob(os.path.join(wmt16_dir, "*.ro"))[0]
        en_file = os.rename(en_file, os.path.join(wmt16_dir, "train.en"))
        ro_file = os.rename(ro_file, os.path.join(wmt16_dir, "train.ro"))


def preprocess_wmt16(args: Namespace, tok):
    preproc_wmt16_path = os.path.join(args.preprocess, "wmt16")
    train_path = os.path.join(preproc_wmt16_path, "train")
    train_en = os.path.join(train_path, "en")
    train_ro = os.path.join(train_path, "ro")

    mkdir(train_en)
    mkdir(train_ro)

    raw_train_en = os.path.join(args.data_dir, "wmt16", "train.en")
    raw_train_ro = os.path.join(args.data_dir, "wmt16", "train.ro")

    tokenizer_dir= os.path.join("tokenizers", args.model_name)
    mkdir(tokenizer_dir) 
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pt")

    with open(raw_train_en) as en, open(raw_train_ro) as ro:
        if not os.path.exists(tokenizer_path): 
            tok.train_new_from_iterator(itertools.chain(en, ro), tok.vocab_size) 
            tok.save_pretrained(tokenizer_path)

        en_list = []
        ro_list = [] 
        idx = 0
        if not os.path.exists(os.path.join(train_path, ".lock")):
            for x, y in tqdm(zip(en, ro), desc=f"Splitting into batches with batch size {args.batch_size}"):
                if re.search('[a-zA-Z]', x) and re.search('[a-zA-Z]', y):
                    en_list.append(x)
                    ro_list.append(y)

                if len(en_list) >= args.batch_size:
                    input_ids_en, attn_mask_en = tok(en_list, padding="longest", return_tensors="np")
                    input_ids_ro, attn_mask_ro = tok(ro_list, padding="longest", return_tensors="np")
                    
                    np.save(os.path.join(train_en, f"{idx}.npy"), input_ids_en)
                    np.save(os.path.join(train_en, f"{idx}_attn.npy"), attn_mask_en)
                    np.save(os.path.join(train_ro, f"{idx}.npy"), input_ids_ro)
                    np.save(os.path.join(train_ro, f"{idx}_attn.npy"), attn_mask_ro)

                    en_list = []
                    ro_list = []
                    idx += 1
            
            with open(os.path.join(train_path, ".lock"), "w") as f:
                f.write("Done preprocessing...")

if __name__ == "__main__":
    
    Env.add_train_args()
    Env.parse_args()
    Env.info()

    
    tok = T5TokenizerFast.from_pretrained("t5-base") 

    print(tok.is_fast)

    # preprocess_wmt14(Env.args, tok)
    download_wmt16(Env.args)
    preprocess_wmt16(Env.args, tok)

    pass
