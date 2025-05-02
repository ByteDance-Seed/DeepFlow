"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Code for Flow-based Models and DeepFlow.

Code block was modified from:

[1] Scalable Diffusion Models with Transformers, ICCV 2023
    William Peebles, Saining Xie

[2] Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers, ECCV 2024
    Nanye Ma, Mark Goldstein, Michael Albergo, Nicholas Boffi, Eric Vanden-Eijnden, Saining Xie.


"""


import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp


# Required Functions
def modulate_func(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using positional embedding
    """
    def __init__(self, hidden_size, frequency_emb_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_emb_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_emb_size = frequency_emb_size
    
    @staticmethod
    def positioning_func(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positioning_func
        t_freq = self.timestep_embedding(t, dim=self.frequency_emb_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

 
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Supporting dropout of label for CFG.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def lbl_token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.lbl_token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



class FlowAttnBlock(nn.Module):
    """
    Block consisting of a self-attention layer, with adaptive layer norm zero (adaLN-Zero) conditioning
    to achieve velocity prediction for flow-matching
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate_func(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_func(self.norm2(x), shift_mlp, scale_mlp))

        return x


class DynLayer(nn.Module):
    """
    The final dynami layer to predict velocity or acceleration.
    - 1. shift and scale from condition
    - 2. modulating normalized dynamic feature using shift and sclae
    - 3. linear projection
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate_func(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x




#################################################################################
#                            Velocity modulation                                #
#################################################################################

class velocity_modulation(nn.Module): 
    """
    The modulation layer for integration of velocity and acceleration.
    """
    def __init__(self, hidden_size, legacy_scaling=True):
        super().__init__()
        final_size = hidden_size // 2
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim,
            out_features=final_size, act_layer=approx_gelu)
            
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.legacy_scaling = legacy_scaling

    def forward(self, x, v, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        if self.legacy_scaling:
            scale = torch.clamp(scale, min=-5, max=5)
        x = modulate_func(self.norm_final(x), shift, scale)
        x = self.mlp(x)
        return x


#################################################################################
#                           Cross-space attention                               #
#################################################################################

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()  # Call the parameter initialization method

    def _reset_parameters(self):
        # Initialize MultiheadAttention weights and biases
        for name, param in self.multihead_attn.named_parameters():
            if 'in_proj_weight' in name or 'out_proj.weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'in_proj_bias' in name or 'out_proj.bias' in name:
                nn.init.zeros_(param)
        # Initialize LayerNorm weights and biases
        if hasattr(self.layer_norm, 'weight') and self.layer_norm.weight is not None:
            nn.init.ones_(self.layer_norm.weight)
        if hasattr(self.layer_norm, 'bias') and self.layer_norm.bias is not None:
            nn.init.zeros_(self.layer_norm.bias)

    def forward(self, query, kv):
        query = self.layer_norm(query)
        key = self.layer_norm(kv)
        value = self.layer_norm(kv)
        attn_output, _ = self.multihead_attn(query, key, value)
        output = query + attn_output
        return output



def build_acc_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim*2),
                nn.SiLU(),
                nn.Linear(projector_dim*2, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


#################################################################################
#                             DeepFlowBlock                                     #
#################################################################################

class DeepFlowBlock(nn.Module):
    """
    A DeepFlow block.
        1. acceleration generation (ACC_MLP)
        2. time-gap condition (velocity_modulation)
        3. cross-space attention: aims to leverage velocity and spatial feature
    """
    def __init__(
            self, hidden_size, acc_hidden_dim, num_heads, legacy_scaling):
        super().__init__()
        
        integrate_size = hidden_size * 2
        # ACC_MLP
        self.acc_mlp = build_acc_mlp(hidden_size, acc_hidden_dim, hidden_size)
        # Modulation
        self.acc_norm = velocity_modulation(
            integrate_size, legacy_scaling=legacy_scaling)
        # Cross-space attention
        self.cross_attention = CrossAttentionBlock(hidden_size, num_heads)

    def forward(self, v, x, dt):
        acc = self.acc_mlp(v)
        refine_vel = self.acc_norm(
            torch.cat([v, acc], dim=-1), v, dt
        )
        xt2 = self.cross_attention(x, refine_vel)

        return acc, xt2 

