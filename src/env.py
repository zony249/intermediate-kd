import os
import sys 
from argparse import ArgumentParser
from pprint import pprint

import torch 



class Env:
    @staticmethod
    def info():
        print("Environment: ")
        pprint(vars(Env))

    @staticmethod
    def parse_args():
        parser = ArgumentParser(description="Environment Config")
        parser.add_argument("--device", type=str, default=None, help="Device selection. Defaults to the ideal device for your platform")

        args = parser.parse_args()
        


        # DEVICE
        if args.device is None:
            if torch.cuda.is_available():
                args.device = "cuda"
            elif torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"


        setattr(Env, "DEVICE", args.device)


Env.parse_args()
Env.info()
