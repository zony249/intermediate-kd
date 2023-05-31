from __future__ import annotations
import os
import sys
from typing import Optional

import torch 
from torch.nn import Module, ModuleList, Sequential, Softmax 
from torch.nn.utils.rnn import pad_sequence 
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

from env import Env


XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


debug=False


class T5Stage(Module): 
    def __init__(self, layers: ModuleList):
        super().__init__()

        self.layers = layers  
        self.size = len(layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                encoder_hidden_states: Optional[torch.Tensor] = None, 
                encoder_attention_mask: Optional[torch.Tensor] = None, 
                past_key_values: Optional[list] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None):

        present_key_values = [] 

        for i, layer in enumerate(self.layers):
            if debug:
                print("DECODER:", layer.is_decoder, ", I", i)

            # past key value 
            if past_key_values is None:
                past_key_value = None
            else:
                past_key_value = past_key_values[i]
                if past_key_value is not None and debug:
                    print("HIDDEN STATES SHAPE", hidden_states.shape)
                    print("PAST KEY VALUE:",  past_key_value[:2][0].shape)
            
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





class T5StagedModel(Module):
    def __init__(self):
        super().__init__() 

        self.pre_encoder = None 
        self.enc_key_layers = None
        self.post_encoder = None

        self.pre_decoder = None 
        self.dec_key_layers = None
        self.post_decoder = None 

        self.lm = None 

    @staticmethod 
    def load_t5(model_name: str, encoder_staging: list, decoder_staging: list) -> T5StagedModel:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        staged_model = T5StagedModel() 

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
            enc_key_layers.append(T5Stage(model.encoder.block[prev_staging_idx:idx]))
            prev_staging_idx = idx
        post_encoder.append(model.encoder.final_layer_norm)
        post_encoder.append(model.encoder.dropout)


        # initialize decoder
        pre_decoder.append(model.decoder.embed_tokens)
        prev_staging_idx = 0 
        for idx in decoder_staging:
            dec_key_layers.append(T5Stage(model.decoder.block[prev_staging_idx:idx]))
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
    
    def forward(self, 
                input_ids: torch.Tensor, 
                target_ids: Optional[torch.Tensor]=None, 
                attention_mask=None,
                encoder_hidden_states=None,
                decoder_attention_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None, 
                return_dict=None, 
                encoder_groups=None, 
                decoder_groups=None):
        """
        Performs a forward pass of the T5 model. If target_ids are not given, then 
        we assume initial decoder states of zeros. UNLIKE Huggingface's library, 
        defining target_ids will not give you loss. This function also does not automatically 
        shift the target_ids right.
        """
    
        assert self.pre_encoder is not None, "model not initialized, please load a model first using load_t5 (or any other model loading function)"
        
        enc_hidden_states, _, attention_mask = self.encoder_forward(input_ids, 
                                                    attention_mask=attention_mask, 
                                                    run_groups=encoder_groups, 
                                                    return_attention_mask=True)

        if encoder_groups is not None:
            return enc_hidden_states, None

        # set target_ids
        if target_ids is None:
           target_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long)


        decoder_outputs, present_key_values = self.decoder_forward(target_ids, 
                                                                 decoder_attention_mask=decoder_attention_mask, 
                                                                 encoder_hidden_states=enc_hidden_states, 
                                                                 encoder_attention_mask=attention_mask,
                                                                 past_key_values=past_key_values,
                                                                 cross_attn_head_mask=cross_attn_head_mask, 
                                                                 run_groups=decoder_groups)

        return decoder_outputs, present_key_values


    def encoder_forward(self, 
                        input_ids,
                        attention_mask=None, 
                        run_groups=None, 
                        return_attention_mask=False
                        ):
        input_embeds = self.pre_encoder[0](input_ids)
        if debug:
            print(input_embeds.shape) 

        enc_hidden_states = input_embeds
       
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape)

        # shape is now [batch, None, None, seq_len]
        attention_mask = self.extend_attention_mask_enc(attention_mask)

        if run_groups is None:
            run_groups = tuple(range(len(self.enc_key_layers)))

        # Feed through encoder
        for i in run_groups: 
            enc_hidden_states, _ = self.enc_key_layers[i](enc_hidden_states, attention_mask=attention_mask)

        # If we reach the last group, then we also feed through post encoder stages
        if run_groups[-1] == len(self.enc_key_layers)-1:
            enc_hidden_states = self.post_encoder(enc_hidden_states)

        if return_attention_mask:
            return enc_hidden_states, None, attention_mask 
        return enc_hidden_states, None

    def decoder_forward(self, 
                        target_ids,
                        decoder_attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        past_key_values=None,
                        cross_attn_head_mask=None,
                        run_groups=None):

        target_embeds = self.pre_decoder[0](target_ids)
        batch_size, seq_length = target_ids.shape

        # decoder attention mask
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(batch_size, mask_seq_length, device=Env.DEVICE)

        # shape is now [batch, None, None, seq_len]
        decoder_attention_mask = self.extend_attention_mask_dec(decoder_attention_mask, target_ids.shape) 



        present_key_values = []
        pkv_idx = 0
        cross_idx = 0


        if run_groups is None:
            run_groups = tuple(range(len(self.dec_key_layers)))


        # Feed through decoder
        dec_hidden_states = target_embeds
        for i in run_groups:
            dec_block = self.dec_key_layers[i]
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
            
            if debug:
                print("DECODER ATTENTION MASK SHAPE:", decoder_attention_mask.shape)

            dec_hidden_states, present_key_value = dec_block(dec_hidden_states, 
                                     attention_mask=decoder_attention_mask, 
                                     encoder_hidden_states=encoder_hidden_states, 
                                     encoder_attention_mask=encoder_attention_mask, 
                                     past_key_values=past_key_value, 
                                     cross_attn_head_mask=cross_attn_layer_head_mask)
            present_key_values += present_key_value
        
        if debug:
            print("DECODER HIDDEN STATES SHAPE", dec_hidden_states.shape) 

        # If we dont reach the last group, return early
        if run_groups[-1] != len(self.dec_key_layers)-1:
            return dec_hidden_states, present_key_values
        dec_hidden_states = self.post_decoder(dec_hidden_states)

        lm_logits = self.lm(dec_hidden_states)
        if debug:
            print("LM LOGITS SHAPE", lm_logits.shape)
        return lm_logits, present_key_values 

    def greedy_decode(self, input_ids, attention_mask=None, eos_tok=1, max_len=100): 

        encoder_hidden_states, _, encoder_attn_mask = self.encoder_forward(input_ids=input_ids, 
                                                                           attention_mask=attention_mask, 
                                                                           return_attention_mask=True)
        past_key_values = None
        targ_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.int, device=Env.DEVICE)
        
        sequences = None
        
        idx = 0
        while True:
            lm_logits, past_key_values = self.decoder_forward(target_ids=targ_ids,
                                                            encoder_hidden_states=encoder_hidden_states,
                                                            encoder_attention_mask=encoder_attn_mask,
                                                            past_key_values=past_key_values,
                                                            cross_attn_head_mask=None)

            probs = Softmax(dim=-1)(lm_logits)
            tokens = torch.argmax(probs, dim=-1)[:, -1:]

            sequences = tokens if sequences is None else torch.cat([sequences, tokens], dim=1)

            if self.all_eos(tokens, eos_tok):
                return sequences 
            targ_ids = tokens

            if debug:
                print("SEQUENCES:", sequences)
            idx += 1 
            if idx >= max_len:
                return sequences 

    
    def all_eos(self, output_ids: torch.Tensor, eos_tok=1) -> bool:
        num_eos_toks_in_row = torch.sum(output_ids == eos_tok, dim=-1)
        does_zero_exist = torch.sum(num_eos_toks_in_row == 0)
        return does_zero_exist.item() < 1e-6


    
    def extend_attention_mask_enc(self, mask):
        """
        Inspired by https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/modeling_utils.py#L818
        """
        extended_attention_mask = mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.get_dtype())  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.get_dtype()).min
        return extended_attention_mask

    def extend_attention_mask_dec(self, mask, input_shape):
        """
        Taken from https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/modeling_utils.py#L897
        -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        """
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
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.get_dtype())  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.get_dtype()).min
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
        """
        Obtained from huggingface source code
        """
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









if __name__ == "__main__":


    Env.parse_args()
    Env.info()

    model = T5StagedModel.load_t5("t5-base", [3, 5, 7, 9, 12], [3, 5, 7, 9, 12]).to(Env.DEVICE)
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tok = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
    
    tokenized = tok(["translate English to German: Today is a good day."])
    input_ids = [torch.tensor(t) for t in tokenized["input_ids"]] 
    attention_mask = [torch.tensor(t) for t in tokenized["attention_mask"]]

    print(input_ids)

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0).to(Env.DEVICE)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0).to(Env.DEVICE)

    
    print(input_ids_padded)
    print(attention_mask_padded) 


    # _, past_key_values = model(input_ids=input_ids_padded, 
    #                            target_ids=input_ids_padded, 
    #                            attention_mask=attention_mask_padded)
    # print("PAST KEY VALUES", past_key_values[0][0].shape)
    # 
    # for i in range(10):
    #     _, past_key_values = model(input_ids=input_ids_padded, 
    #                                target_ids=input_ids_padded, 
    #                                attention_mask=attention_mask_padded, 
    #                                past_key_values=past_key_values)

   
    all_eos = model.all_eos(torch.tensor([[3, 4, 5, 6, 7, 1, 0, 0], [2, 3, 4, 1, 1, 7, 8, 9], [2, 2, 2, 2, 3, 3, 3, 1]]))
    print(all_eos)

    output_ids = model.greedy_decode(input_ids_padded, attention_mask_padded)
    print("OUTPUT_IDS:", output_ids)
    print(tok.batch_decode(output_ids))


    # torch.set_printoptions(profile="short")
    # dec = T5ForConditionalGeneration.from_pretrained("t5-base").decoder
    # print(len(dec(input_ids_padded, attention_mask=attention_mask_padded)['past_key_values'][0]))

    # test_stage = T5Stage(T5ForConditionalGeneration.from_pretrained("t5-base").encoder.block[3:6])
    #
    # x = torch.rand((1, 4, 768))
    # test_stage.t5_forward(x)


