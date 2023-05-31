

import os 
import sys

import torch
from torch.nn import CrossEntropyLoss, Softmax
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, RandomSampler 
from torch.optim import AdamW, Adam, SGD
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import T5ForConditionalGeneration, T5Tokenizer 

from env import Env
from utils import mkdir
from stagedmodel import T5StagedModel
from datasets import WMT14


END_LAYER = 100
INT_MAX = sys.maxsize

debug=False


class SingleModelTrainer:

    def __init__(self, model, tset, vset=None):
        self.model = model 
        self.tset = tset 
        self.vset = vset 


    def train(self, lossfn=None, optim=None, epochs=None, batch_size=16, lr=1e-4, log_interval=50, save_interval=1000, metrics=[]):

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

        
        tsampler = RandomSampler(self.tset) 
        tloader = DataLoader(self.tset, 
                             sampler=tsampler) 
        
        if self.vset is not None:
            vsampler = RandomSampler(self.vset)
            vloader = DataLoader(self.vset, 
                                 sampler=vsampler)
        batch_counter = 0
        save_counter = 0

        if lossfn is None:
            lossfn = CCE


        if "cuda" in Env.DEVICE:
            scaler = GradScaler()
        
        print("[SINGLEMODELTRAINER]: Start!")
        for epoch in range(epochs):
            steps_per_epoch = len(self.tset) // batch_size    
            for step in range(steps_per_epoch):  
                self.model.zero_grad(set_to_none=True)

                input_ids, encoder_attn_mask, targ_ids, decoder_attn_mask = next(iter(tloader))
               

                # Note that batching is already done during preprocessing, thus, we index these tensers at[0]
                input_ids = input_ids[0].to(Env.DEVICE)
                encoder_attn_mask = encoder_attn_mask[0].to(Env.DEVICE)
                targ_ids = targ_ids[0].to(Env.DEVICE)
    
                targ_ids, targ_ids_shifted = self.shift_right(targ_ids) 


                if "cuda" in Env.DEVICE:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        logits, past_key_values = self.model(input_ids, 
                                                              attention_mask=encoder_attn_mask, 
                                                              target_ids=targ_ids_shifted) 

                        probs = Softmax(dim=-1)(logits)
                        loss = lossfn(probs, targ_ids) 
                    scaler.scale(loss).backward()

                    scaler.step(optim)
                    scaler.update()
                else:

                    logits, past_key_values = self.model(input_ids, 
                                                          attention_mask=encoder_attn_mask, 
                                                          target_ids=targ_ids_shifted) 

                    probs = Softmax(dim=-1)(logits)
                    loss = lossfn(probs, targ_ids) 
                    
                    loss.backward()

                    optim.step()

                if batch_counter%log_interval == 0:
                    logs = {"loss": loss.item()}
                    self.print_log(epoch, batch_counter, **logs, identifier="[SINGLEMODELTRAINER.train]:")
                    torch.cuda.empty_cache()
                batch_counter += 1


                if batch_counter%save_interval==0:  
                    mkdir(os.path.join(Env.model_dir, Env.model_name))
                    torch.save(self.model.state_dict(), os.path.join(Env.model_dir, Env.model_name, f"ckpt_{save_counter}.pt" ))
                    print(f"[SINGLEMODELTRAINER.train]: Saved model checkpoint ckpt_{save_counter}.pt to {os.path.join(Env.model_dir, Env.model_name)}")
                    save_counter += 1 
                    if self.vset is not None:
                        #run vset 

                        pass
                    








    def print_log(self, epochs, batch_number, identifier="", **kwargs):

        print(f"{identifier} E: {epochs}, B: {batch_number}", end="")
        [print(f", {k}: {v:.4f} ", end="") for k, v in kwargs.items()]
        print("")



    def shift_right(self, targ_ids):

        if debug:
            print(targ_ids.shape)
        targ_ids_padded = torch.cat([targ_ids, torch.zeros((targ_ids.shape[0], 1), dtype=torch.long, device=Env.DEVICE)], dim=-1)
        targ_ids_shifted = torch.cat([torch.zeros((targ_ids.shape[0], 1), dtype=torch.long, device=Env.DEVICE), targ_ids], dim=-1)
        
        return targ_ids_padded, targ_ids_shifted





def CCE(y, t):
    """
    Takes in probabilities
    @param y: softmax probabilities (N, S, C)
    @param t: target labels (N, S)
    """
    t = one_hot(t, num_classes=y.shape[-1]) 
    tlogy = torch.sum(-t * torch.log(torch.clip(y, min=1e-6, max=1e6)), axis=-1)
    
    norm_tlogy = torch.sum(tlogy) / y.shape[0] / y.shape[1]
    return norm_tlogy


if __name__ == "__main__":
    Env.add_train_args()
    Env.parse_args()
    Env.info()

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5StagedModel.load_t5("t5-base", [2, 4, 6, 8, 10, END_LAYER], [2, 4, 6, 8, 10, END_LAYER]).to(Env.DEVICE) 

    tset = WMT14()


    trainer = SingleModelTrainer(model, tset=tset)
    trainer.train(lossfn=None, optim=None, epochs=None, batch_size=16, lr=1e-4, log_interval=50, save_interval=1000, metrics=[])
    



