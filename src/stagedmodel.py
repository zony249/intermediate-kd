from __future__ import annotations
import os
import sys 

import torch 
from torch.nn import Module, ModuleList, Sequential 
from torch.nn.utils.rnn import pad_sequence 
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

class Stage(Module): 
    def __init__(self, layers, model="t5"):
        super().__init__()

        self.layers = layers  
        self.model = model 

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None):

        if self.model == "t5":
            return self.t5_forward(hidden_states, 
                                   attention_mask=attention_mask, 
                                   encoder_hidden_states=encoder_hidden_states, 
                                   encoder_attention_mask=encoder_attention_mask)
        
    def t5_forward(self, 
                   hidden_states: torch.Tensor, 
                   attention_mask=None, 
                   encoder_hidden_states=None, 
                   encoder_attention_mask=None):
        for layer in self.layers:
            layer_out = layer(hidden_states, 
                              attention_mask=attention_mask, 
                              encoder_hidden_states=encoder_hidden_states, 
                              encoder_attention_mask=encoder_attention_mask)
            hidden_states = layer_out[0]
            attention_mask = layer_out[1]

        return hidden_states, attention_mask





class StagedModel(Module):
    def __init__(self):
        super().__init__() 

        self.pre_encoder = None 
        self.enc_key_layers = None
        self.post_encoder = None

        self.pre_decoder = None 
        self.dec_key_layers = None
        self.post_decoder = None 

        self.lm = None 

    
    def forward(self, input_ids):
        pass 

    def t5_forward(self, 
                    input_ids, 
                    target_ids=None, 
                    attention_mask=None,
                    encoder_hidden_states=None,
                    decoder_attention_mask=None,
                    past_key_values=None, 
                    return_dict=None, 
                    encoder_return_groups=None, 
                    decoder_return_groups=None):
    
        assert self.pre_encoder is not None, "model not initialized, please load a model first using load_t5 (or any other model loading function)"

        input_embeds = self.pre_encoder[0](input_ids)
        print(input_embeds.shape) 

        enc_hidden_states = input_embeds
       
        # shape is now [batch, None, None, seq_len]
        attention_mask = self.extend_attention_mask_enc(attention_mask)

        
        # Feed through encoder
        for i, enc_block in enumerate(self.enc_key_layers): 
            enc_hidden_states, _ = enc_block(enc_hidden_states, attention_mask=attention_mask)
            if i == encoder_return_groups:
                return enc_hidden_states 
        enc_hidden_states = self.post_encoder(enc_hidden_states)


        
        if target_ids is None:
           target_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long)

        target_embeds = self.pre_decoder[0](target_ids)


        # decoder attention mask
        

        # Feed through decoder
        dec_hidden_states = target_embeds
        for i, dec_block in enumerate(self.dec_key_layers): 
            dec_hidden_states, _ = dec_block(dec_hidden_states, 
                                     attention_mask=decoder_attention_mask, 
                                     encoder_hidden_states=enc_hidden_states, 
                                     encoder_attention_mask=attention_mask)
            if i == decoder_return_groups:
                return dec_hidden_states  

        print(dec_hidden_states.shape) 


    def extend_attention_mask_enc(self, mask):
        extended_attention_mask = mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.get_dtype())  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.get_dtype()).min
        return extended_attention_mask
    
    def get_dtype(self): 
        last_dtype = None
        for t in self.enc_key_layers[0].parameters():
            last_dtype = t.dtype
            if t.is_floating_point():

                # Adding fix for https://github.com/pytorch/xla/issues/4152
                # Fixes issue where the model code passes a value that is out of range for XLA_USE_BF16=1
                # and XLA_DOWNCAST_BF16=1 so the conversion would cast it to -inf
                # NOTE: `is_torch_tpu_available()` is checked last as it induces a graph break in torch dynamo
                if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                    return torch.bfloat16
                if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                    if t.dtype == torch.float:
                        return torch.bfloat16
                    if t.dtype == torch.double:
                        return torch.float32
                return t.dtype


    @staticmethod 
    def load_t5(model_name: str, encoder_staging: list, decoder_staging: list) -> StagedModel:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        staged_model = StagedModel() 

        pre_encoder = [] 
        enc_key_layers = []  
        post_encoder = []

        pre_decoder = []
        dec_key_layers = []
        post_decoder = []


        # initialize encoder 
        pre_encoder.append(model.encoder.embed_tokens)
        prev_staging_idx = 0
        for idx in encoder_staging:
            enc_key_layers.append(Stage(model.encoder.block[prev_staging_idx:idx], model="t5"))
            prev_staging_idx = idx
        post_encoder.append(model.encoder.final_layer_norm)
        post_encoder.append(model.encoder.dropout)


        # initialize decoder
        pre_decoder.append(model.decoder.embed_tokens)
        prev_stading_idx = 0 
        for idx in decoder_staging:
            dec_key_layers.append(Stage(model.decoder.block[prev_staging_idx:idx], model="t5"))
            prev_staging_idx = idx 
        post_decoder.append(model.decoder.final_layer_norm)
        post_decoder.append(model.decoder.dropout)


        staged_model.pre_encoder = ModuleList(pre_encoder)
        staged_model.enc_key_layers = ModuleList(enc_key_layers) 
        staged_model.post_encoder = Sequential(*post_encoder) 

        staged_model.pre_decoder = ModuleList(pre_decoder)
        staged_model.dec_key_layers = ModuleList(dec_key_layers) 
        staged_model.post_decoder = Sequential(*post_decoder) 

        staged_model.lm = model.lm_head 

        return staged_model




if __name__ == "__main__":


    model = StagedModel.load_t5("t5-base", [3, 5, 7, 9, 12], [3, 5, 7, 9, 12])
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tok = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
    
    tokenized = tok(["test one", "test two three"])
    input_ids = [torch.tensor(t) for t in tokenized["input_ids"]] 
    attention_mask = [torch.tensor(t) for t in tokenized["attention_mask"]]

    print(input_ids)

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0) 
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    
    print(input_ids_padded)
    print(attention_mask_padded) 


    model.t5_forward(input_ids=input_ids_padded, attention_mask=attention_mask_padded) 


    # enc = T5ForConditionalGeneration.from_pretrained("t5-base").encoder
    # print(enc(input_ids_padded, attention_mask=attention_mask_padded))

    # test_stage = Stage(T5ForConditionalGeneration.from_pretrained("t5-base").encoder.block[3:6])
    #
    # x = torch.rand((1, 4, 768))
    # test_stage.t5_forward(x)


