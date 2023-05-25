import os
import sys 
from argparse import ArgumentParser, Namespace 
import wget 

from tqdm import tqdm 

from utils import mkdir, download

def download_wmt14(args: Namespace):
    mkdir(os.path.join(args.data, "wmt14"))

    download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en", dest_path=os.path.join(args.data, "wmt14", "train.en"))
    download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de", dest_path=os.path.join(args.data, "wmt14", "train.de"))

def preprocess_wmt14(args: Namespace):
    download_wmt14(args) 
    mkdir(os.path.join(args.preprocess, "wmt14"))
    mkdir(os.path.join(args.preprocess, "wmt14", "train", "en"))
    mkdir(os.path.join(args.preprocess, "wmt14", "train", "de"))

    with open(os.path.join(args.data, "wmt14", "train.en")) as f:
        line_idx = 0
        file_idx = 0
        lines = []
        for line in tqdm(f, desc="Preprocessing train.en"):
            lines.append(line)
            line_idx += 1
            if line_idx >= 10000:
                with open(os.path.join(args.preprocess, "wmt14", "train", "en", str(file_idx) + ".txt"), "w") as output: 
                    [output.write(l) for l in lines] 
                    lines = []
                    line_idx = 0
            file_idx += 1
    
    with open(os.path.join(args.data, "wmt14", "train.de")) as f:
        line_idx = 0
        file_idx = 0
        lines = []
        for line in tqdm(f, desc="Preprocessing train.de"):
            lines.append(line)
            line_idx += 1
            if line_idx >= 10000:
                with open(os.path.join(args.preprocess, "wmt14", "train", "de", str(file_idx) + ".txt"), "w") as output: 
                    [output.write(l) for l in lines] 
                    lines = []
                    line_idx = 0
                file_idx += 1




if __name__ == "__main__":
   
    parser = ArgumentParser(description="download and preprocess datasets")
    parser.add_argument("--data", type=str, default="data/", help="data directory")
    parser.add_argument("--preprocess", type=str, default="preprocess", help="directory to preprocessed dataset")


    args = parser.parse_args()
    
    preprocess_wmt14(args)


    pass
