import os
import sys
import glob
from pprint import pprint
from random import shuffle

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


class WMT16(Dataset):
    def __init__(self, path_tuples=None):
        super().__init__()

        assert os.path.exists(Env.preprocess, "wmt16", "train"), "Training set does not exist. Please run preprocess."
        assert os.path.exists(Env.preprocess, "wmt16", "test"), "Test set does not exist. Please run preprocess."
        
        self.path_tuples = path_tuples

    def __len__(self):
        assert self.path_tuples is not None, "WMT16 dataset has not been instantiated yet!"
        return len(self.path_tuples)

    def __getitem__(self, idx):
        in_ids, enc_attn_mask, targ_ids, dec_attn_mask = self.path_tuples[idx % len(self)] 

        in_ids = np.load(in_ids)
        enc_attn_mask = np.load(enc_attn_mask)
        targ_ids = np.load(targ_ids)
        dec_attn_mask = np.load(dec_attn_mask)

        return (torch.from_numpy(in_ids), torch.from_numpy(enc_attn_mask), torch.from_numpy(targ_ids), torch.from_numpy(dec_attn_mask))

    @staticmethod 
    def get_train_val_test(val_size=2000):
        
        train_val_path = os.path.join(Env.preprocess, "wmt16", "train")
        train_val_en_path = os.path.join(train_val_path, "en")
        train_val_ro_path = os.path.join(train_val_path, "ro")
        
        test_path = os.path.join(Env.preprocess, "wmt16", "test")
        test_en_path = os.path.join(test_path, "en")
        test_ro_path = os.path.join(test_path, "ro")

        train_val_ids = [item.split(".")[0].split("/")[-1] for item in glob.glob(os.path.join(train_val_en_path, "*[!_attn].npy"))]
        # print(train_val_ids)

        shuffle(train_val_ids)

        test_ids = [item.split(".")[0].split("/")[-1] for item in glob.glob(os.path.join(test_en_path, "*[!_attn].npy"))]
        train_ids = train_val_ids[val_size:]
        val_ids = train_val_ids[:val_size]

        train_tuples = [(os.path.join(train_val_en_path, f"{id}.npy"), 
                        os.path.join(train_val_en_path, f"{id}_attn.npy"),
                        os.path.join(train_val_ro_path, f"{id}.npy"), 
                        os.path.join(train_val_ro_path, f"{id}_attn.npy")) for id in train_ids]


        val_tuples = [(os.path.join(train_val_en_path, f"{id}.npy"), 
                        os.path.join(train_val_en_path, f"{id}_attn.npy"),
                        os.path.join(train_val_ro_path, f"{id}.npy"), 
                        os.path.join(train_val_ro_path, f"{id}_attn.npy")) for id in val_ids]

        test_tuples = [(os.path.join(test_en_path, f"{id}.npy"), 
                        os.path.join(test_en_path, f"{id}_attn.npy"),
                        os.path.join(test_ro_path, f"{id}.npy"), 
                        os.path.join(test_ro_path, f"{id}_attn.npy")) for id in test_ids]


        return WMT16(train_tuples), WMT16(val_tuples), WMT16(test_tuples)


if __name__ == "__main__":

    Env.add_train_args()
    Env.parse_args()
    Env.info()

    
    # dataset = WMT14()
    # print(dataset[48345])

    WMT16.get_train_val_test()
