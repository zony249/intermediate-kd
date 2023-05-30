import os
import sys
import glob
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 

from env import Env 

class WMT14(Dataset):
    def __init__(self):
        super().__init__()
    
        assert os.path.exists(os.path.join(Env.preprocess, "wmt14", "train")), "Data folder does not exist. Please run preprocess.py first"
        
        train_path = os.path.join(Env.preprocess, "wmt14", "train") 
        en_path = os.path.join(train_path, "en")
        de_path = os.path.join(train_path, "de")



        self.file_ids = [item.split(".")[0].split("/")[-1] for item in glob.glob(os.path.join(en_path, "*[!_attn].npy"))] 
       
        self.sets = {id:(os.path.join(en_path, id + ".npy"), 
                         os.path.join(en_path, id + "_attn.npy"), 
                         os.path.join(de_path, id + ".npy"), 
                         os.path.join(de_path, id + "_attn.npy")) for id in self.file_ids}
        
    
    def __len__(self):
        return len(self.file_ids) 


    def __getitem__(self, idx):
        input_ids, encoder_attention_mask, target_ids, decoder_attention_mask = self.sets[str(idx)]

        input_ids = np.load(input_ids)
        encoder_attention_mask = np.load(encoder_attention_mask)
        target_ids = np.load(target_ids)
        decoder_attention_mask = np.load(decoder_attention_mask)

        return (torch.from_numpy(input_ids), torch.from_numpy(encoder_attention_mask), torch.from_numpy(target_ids), torch.from_numpy(decoder_attention_mask))


if __name__ == "__main__":

    Env.add_train_args()
    Env.parse_args()
    Env.info()

    
    dataset = WMT14()
    print(dataset[48345])
