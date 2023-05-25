from __future__ import annotations
import os
import sys 

import torch 
from torch.nn import Module, ModuleList, Sequential 
from torch.nn.utils.rnn import pad_sequence 
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

from env import Env


XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

class Stage(Module): 
    def __init__(self, layers, model="t5"):
        super().__init__()

        self.layers = layers  
        self.model = model 
        self.size = len(layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None, 
                past_key_values=None,
                cross_attn_head_mask=None):

        if self.model == "t5":
            return self.t5_forward(hidden_states, 
                                   attention_mask=attention_mask, 
                                   encoder_hidden_states=encoder_hidden_states, 
                                   encoder_attention_mask=encoder_attention_mask, 
                                   past_key_values=past_key_values, 
                                   cross_attn_head_mask=cross_attn_head_mask)
        
    def t5_forward(self, 
                   hidden_states: torch.Tensor, 
                   attention_mask=None, 
                   encoder_hidden_states=None, 
                   encoder_attention_mask=None,
                   past_key_values=None, 
                   cross_attn_head_mask=None):

        present_key_values = [] 

        for i, layer in enumerate(self.layers):
            print("DECODER:", layer.is_decoder, ", I", i)

            # past key value 
            if past_key_values is None:
                past_key_value = None
            else:
                past_key_value = past_key_values[i]
            
            # cross attn 
            if cross_attn_head_mask is None:
                cross_attn_layer_head_mask = None
            else:
                cross_attn_layer_head_mask = cross_attn_head_mask[i]

            layer_out = layer(hidden_states, 
                              attention_mask=attention_mask, 
                              encoder_hidden_states=encoder_hidden_states, 
                              encoder_attention_mask=encoder_attention_mask, 
                              past_key_value=past_key_value, 
                              cross_attn_layer_head_mask=cross_attn_layer_head_mask, 
                              use_cache=True)
            hidden_states = layer_out[0]

            present_key_values.append(layer_out[1])

        return hidden_states, present_key_values





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
                    cross_attn_head_mask=None,
                    past_key_values=None, 
                    return_dict=None, 
                    encoder_return_groups=None, 
                    decoder_return_groups=None):
    
        assert self.pre_encoder is not None, "model not initialized, please load a model first using load_t5 (or any other model loading function)"

        input_embeds = self.pre_encoder[0](input_ids)
        print(input_embeds.shape) 

        enc_hidden_states = input_embeds
       
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape)

        # shape is now [batch, None, None, seq_len]
        attention_mask = self.extend_attention_mask_enc(attention_mask)
 
        # Feed through encoder
        for i, enc_block in enumerate(self.enc_key_layers): 
            enc_hidden_states, _ = enc_block(enc_hidden_states, attention_mask=attention_mask)
            if i == encoder_return_groups:
                return enc_hidden_states, None
        enc_hidden_states = self.post_encoder(enc_hidden_states)

        # set target_ids
        if target_ids is None:
           target_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long)

        target_embeds = self.pre_decoder[0](target_ids)

        # decoder attention mask
        if decoder_attention_mask is not None and target_ids is None:
            raise ValueError("You cannot define decoder_attention_mask without defining target_ids!")
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(target_ids.shape)

        # shape is now [batch, None, None, seq_len]
        decoder_attention_mask = self.extend_attention_mask_dec(decoder_attention_mask, target_ids.shape) 



        present_key_values = []
        pkv_idx = 0
        cross_idx = 0

        # Feed through decoder
        dec_hidden_states = target_embeds
        for i, dec_block in enumerate(self.dec_key_layers): 
            if past_key_values is not None:
                past_key_value = past_key_values[pkv_idx:pkv_idx+dec_block.size]
                pkv_idx += dec_block.size
            else:
                past_key_value = [None] * dec_block.size

            if cross_attn_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_head_mask[cross_idx:cross_idx + dec_block.size]
                cross_attn_layer_head_mask = self._convert_head_mask_to_5d(cross_attn_layer_head_mask, dec_block.size)
                cross_idx += dec_block.size
            else:
                cross_attn_layer_head_mask = [None] * dec_block.size

            dec_hidden_states, present_key_value = dec_block(dec_hidden_states, 
                                     attention_mask=decoder_attention_mask, 
                                     encoder_hidden_states=enc_hidden_states, 
                                     encoder_attention_mask=attention_mask, 
                                     past_key_values=past_key_value, 
                                     cross_attn_head_mask=cross_attn_layer_head_mask)
            present_key_values += present_key_value
            if i == decoder_return_groups:
                return dec_hidden_states, past_key_values 

        print("DECODER HIDDEN STATES SHAPE", dec_hidden_states.shape) 
        
        dec_hidden_states = self.post_decoder(dec_hidden_states)

        lm_logits = self.lm(dec_hidden_states)
        print("LM LOGITS SHAPE", lm_logits.shape)
        return lm_logits, present_key_values 

    
    
    def extend_attention_mask_enc(self, mask):
        """
        Inspired by https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/modeling_utils.py#L818
        """
        extended_attention_mask = mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.get_dtype())  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.get_dtype()).min
        return extended_attention_mask

    def extend_attention_mask_dec(self, mask, input_shape):
        
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=Env.DEVICE)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]

        causal_mask.to(mask.dtype)

        if causal_mask.shape[1] < mask.shape[1]:
            prefix_seq_len = mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=Env.DEVICE, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * mask[:, None, None, :]
        return extended_attention_mask

    
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """
        Taken from https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/modeling_utils.py#L897
        -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        """
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask



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
        prev_staging_idx = 0 
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
    
    tokenized = tok(["test one", "test two three", "testing one", "testing two three four"])
    input_ids = [torch.tensor(t) for t in tokenized["input_ids"]] 
    attention_mask = [torch.tensor(t) for t in tokenized["attention_mask"]]

    print(input_ids)

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0) 
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    
    print(input_ids_padded)
    print(attention_mask_padded) 


    _, past_key_values = model.t5_forward(input_ids=input_ids_padded, attention_mask=attention_mask_padded)
    print("PAST KEY VALUES", past_key_values[0][0].shape)
    model.t5_forward(input_ids=input_ids_padded, attention_mask=attention_mask_padded, past_key_values=past_key_values)

    
    # torch.set_printoptions(profile="short")
    # dec = T5ForConditionalGeneration.from_pretrained("t5-base").decoder
    # print(len(dec(input_ids_padded, attention_mask=attention_mask_padded)['past_key_values'][0]))

    # test_stage = Stage(T5ForConditionalGeneration.from_pretrained("t5-base").encoder.block[3:6])
    #
    # x = torch.rand((1, 4, 768))
    # test_stage.t5_forward(x)


