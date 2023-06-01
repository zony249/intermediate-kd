import os
import sys 
from argparse import ArgumentParser
from pprint import pprint

import torch 



class Env:
    parser = ArgumentParser(description="Environment Config")
    args = None


    def __init__(self):
        pass

    @staticmethod
    def info():
        print("Environment: ")
        pprint(vars(Env))

    @staticmethod
    def parse_args():
        Env.parser.add_argument("--device", type=str, default=None, help="Device selection. Defaults to the ideal device for your platform")
        Env.parser.add_argument("--model_dir", type=str, default="models", help="Where to save or load the model checkpoints")
        Env.parser.add_argument("--model_name", type=str, default="default", help="Model name")
        Env.parser.add_argument("--data_dir", type=str, default="data", help="Directory to datasets")
        Env.parser.add_argument("--preprocess", type=str, default="preprocess", help="directory to preprocessed dataset")

        Env.args = Env.parser.parse_args()
        


        # DEVICE
        if Env.args.device is None:
            if torch.cuda.is_available():
                Env.args.device = "cuda"
            elif torch.backends.mps.is_available():
                Env.args.device = "mps"
            else:
                Env.args.device = "cpu"


        setattr(Env, "DEVICE", Env.args.device)

        [setattr(Env, attr, getattr(Env.args, attr)) for attr in vars(Env.args) if attr not in ["device"]]

    @staticmethod 
    def add_train_args():
        Env.parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        Env.parser.add_argument("--batch_size", type=int, default=12, help="Learning rate")
        Env.parser.add_argument("--optim", type=str, default="adamw", help="Optimizer")
        Env.parser.add_argument("--data", type=str, default="wmt16", help="Which dataset to train on. Currently supports wmt14, wmt16")
        Env.parser.add_argument("--max_len", type=int, default=256, help="Max sequence length")
