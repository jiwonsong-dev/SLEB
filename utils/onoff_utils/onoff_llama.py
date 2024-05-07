import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import warnings

class OnOff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.pass_layer = False

    def turn_off(self):
        self.pass_layer = True
    
    def turn_on(self):
        self.pass_layer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # skip this decoder layer
        if self.pass_layer:
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (None,)

            if use_cache:
                outputs += (past_key_value,)

            return outputs

        # else normal forward
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if residual.device != hidden_states.device:

            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
def block_replace(model):

    num_layers = len(model.model.layers)
    for i in range(num_layers):
        model.model.layers[i] = OnOff_LlamaDecoderLayer(model.model.layers[i])
    print("Replacement complete.")

    return model

def turn_off(model, block_idx):

    model.model.layers[block_idx].turn_off()

def turn_on(model, block_idx):

    model.model.layers[block_idx].turn_on()

def scan(model, num_blocks):

    alive_list = []
    skip_list = []

    for i in range(num_blocks):
        if model.model.layers[i].pass_layer == True:
            skip_list.append(i)
        elif model.model.layers[i].pass_layer == False:
            alive_list.append(i)
            
    print(
        f"pass layer: {skip_list}\n"
        f"do layer: {alive_list}"
        )
    


