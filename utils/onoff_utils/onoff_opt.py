import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class OnOff_OPTDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.embed_dim = original_decoder_layer.embed_dim

        self.self_attn = original_decoder_layer.self_attn

        self.do_layer_norm_before = original_decoder_layer.do_layer_norm_before
        self.dropout = original_decoder_layer.dropout
        self.activation_fn = original_decoder_layer.activation_fn

        self.self_attn_layer_norm = original_decoder_layer.self_attn_layer_norm
        self.fc1 = original_decoder_layer.fc1
        self.fc2 = original_decoder_layer.fc2
        self.final_layer_norm = original_decoder_layer.final_layer_norm

        self.pass_layer = False
    
    def turn_off(self):
        self.pass_layer = True
    
    def turn_on(self):
        self.pass_layer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
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
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
def block_replace(model):

    num_layers = len(model.model.decoder.layers)
    for i in range(num_layers):
        model.model.decoder.layers[i] = OnOff_OPTDecoderLayer(model.model.decoder.layers[i])
    print("Replacement complete.")

    return model

def turn_off(model, block_idx):

    model.model.decoder.layers[block_idx].turn_off()

def turn_on(model, block_idx):

    model.model.decoder.layers[block_idx].turn_on()

def scan(model, num_layers):

    alive_list = []
    skip_list = []

    for i in range(num_layers):
        if model.model.decoder.layers[i].pass_layer == True:
            skip_list.append(i)
        elif model.model.decoder.layers[i].pass_layer == False:
            alive_list.append(i)
            
    print(
        f"pass layer: {skip_list}\n"
        f"do layer: {alive_list}"
        )