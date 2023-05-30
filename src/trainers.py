

import os 
import sys

import torch
from torch.utils.data import Dataset, DataLoader 
from torch.optim import AdamW, Adam, SGD
from transformers import T5ForConditionalGeneration, T5Tokenizer 

from env import Env
from stagedmodel import T5StagedModel
from datasets import WMT14


END_LAYER = 100
INT_MAX = sys.maxsize




class SingleModelTrainer:

    def __init__(self, model, tset, vset=None):
        self.model = model 
        self.tset = tset 
        self.vset = vset 


    def train(self, optim=None, epochs=None, batch_size=16, lr=1e-4, log_interval=100):

        if optim is None:
            optim = "adamw"

        if optim == "adamw":
            optim = AdamW(self.model.parameters(), lr=lr) 
        elif optim == "adam":
            optim = Adam(self.model.parameters(), lr=lr)
        elif optim == "sgd":
            optim = SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"{optim} is not a valid optimizer")


        if epochs is None:
            epochs = INT_MAX

        
        batch_counter = 0

        for epoch in range(epochs):
            steps_per_epoch = len(self.tset) // batch_size    
            for step in range(steps_per_epoch):  
                pass


if __name__ == "__main__":
    Env.add_train_args()
    Env.parse_args()
    Env.info()

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5StagedModel.load_t5("t5-base", [2, 4, 6, 8, 10, END_LAYER], [2, 4, 6, 8, 10, END_LAYER]) 

    tset = WMT14()

    






    
   
    trainer = SingleModelTrainer(model, tset)


