# Copyright (c) Meta Platforms, Inc. and affiliates. 
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the CC-BY-NC 
# This file has been modified by Inkyu Shin, ByteDance Ltd.
#
# Original file was released under CC-BY-NC, with the full license text
# available at https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt
#
# This modified file is released under the same license.


"""
DeepFlow Script for Modeling.

Our modeling pipeline is built on top of DiT [1], SiT [2] and REPA [3].

[1] Scalable Diffusion Models with Transformers, ICCV 2023
    William Peebles, Saining Xie

[2] Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers, ECCV 2024
    Nanye Ma, Mark Goldstein, Michael Albergo, Nicholas Boffi, Eric Vanden-Eijnden, Saining Xie.

[3] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think, ICLR 2025
    Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie.


"""


import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from models.blocks import TimestepEmbedder, LabelEmbedder, FlowAttnBlock, \
      DynLayer, DeepFlowBlock
from utils.utils import *


# MLP for SSL_alignment
def build_mlp_projection(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )



class DeepFlow(nn.Module):
    """DeepFlow Model

    DeepFlow aims to improve feature alignment using deep supervision and VeRA block.
    It is seamlessly integrated into SiT block [1] for fundamental flow-based generative models.
    Furthermore, we also supprot REPA [2] for ssl alignment.

    [1] Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers, ECCV 2024
        Nanye Ma, Mark Goldstein, Michael Albergo, Nicholas Boffi, Eric Vanden-Eijnden, Saining Xie.

    [2] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think, ICLR 2025
        Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie.
    """

    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        ssl_align=False,
        z_dims=[768],
        projector_dim=2048,
        df_idxs=[6,],
        legacy_scaling=True,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Flow blocks
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            FlowAttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])

        # SSL alignment
        self.ssl_align = ssl_align
        if ssl_align:
            z_dims = z_dims * len(df_idxs)
            self.projectors = nn.ModuleList([
                build_mlp_projection(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
            
        # DeepFlow
        """Following modules are defined for DeepFlow.
        VeRA block
            1. dt_embedder: time embedder for time-gap.
            2. df_blocks: 
                deepflow blocks containing ACC_MLP, time-gap modulation, and cross-space attention.
        3. final_layer: one final velocity layer for intermediate, the other for final one.
        4. final_acc_layer: we use single acc_layer to predict final acceleration.
        """
        self.df_idxs = df_idxs
        self.num_splits = len(df_idxs)+1
        self.dt_embedder = TimestepEmbedder(hidden_size*2)
        self.df_blocks = nn.ModuleList([
            DeepFlowBlock(
                decoder_hidden_size, projector_dim, num_heads, legacy_scaling) \
                    for _ in range(len(df_idxs))])
        self.final_layer = nn.ModuleList([
            DynLayer(decoder_hidden_size, patch_size, self.out_channels) for _ in range(2)
        ])
        self.final_acc_layer = DynLayer(
            decoder_hidden_size, patch_size, self.out_channels)
    
        # Initialize the modules
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.dt_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.dt_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_acc_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_acc_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_acc_layer.linear.weight, 0)
        nn.init.constant_(self.final_acc_layer.linear.bias, 0)


        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # two separate final layers for intermediate and final branches
        for i in range(2):
            nn.init.constant_(self.final_layer[i].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer[i].adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer[i].linear.weight, 0)
            nn.init.constant_(self.final_layer[i].linear.bias, 0)
            
        for j in range(len(self.df_idxs)):
            nn.init.constant_(self.df_blocks[j].acc_norm.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.df_blocks[j].acc_norm.adaLN_modulation[-1].bias, 0)


    def process_multi_times(self, x, t):
        """processing multiple timesteps for DeepFlow.
        
        Args:
            x: (N, C, H, W) tensor of spatial noisy inputs, e.g., x_t1
            t: List of (N,), tensor of diffusion timestepsm e.g, [t1, t2, ...]
        Returns:
            patch_x_1: tokenized input features added with positional embedding
            t_embed_1: time embedding of t1
            t_embed_list[1:]: time embeddig list of other time-steps
            dt_embed_list: time embedding list of time-gap
        """
        patch_x_1 = self.x_embedder(x) + self.pos_embed
        t_embed_list = []
        dt_embed_list = []
        for i in range(self.num_splits):
            t_embed_list.append(self.t_embedder(t[i]))
            if i < self.num_splits - 1:
                dt_embed_list.append(self.dt_embedder(t[i+1]-t[i]))
        t_embed_1 = t_embed_list[0]
        return patch_x_1, t_embed_1, t_embed_list[1:], dt_embed_list
        
    
    def df_process(self, x_cur, c, xt1, dt, df_num):
        """processing for DeepFlow.

        Args:
            x_cur: (N, T, C), intermediate velocity feature
            c: condition feature with class and time-step
            xt1: original tokenized spatial feature
            dt: time embedding list of time-gap
            df_num: number of DeepFlow process

        Returns:
            acc: detokenized acceleration
            x_cur_2: refined velocity feature
            x_cur_1_final: detokenized intermediate velocity
        """
        x_cur_1_final = self.final_layer[0](x_cur, c)
            
        x_cur_2 = x_cur.clone()
        acc, x_cur_2 = self.df_blocks[df_num](x_cur_2, xt1, dt[df_num])
        
        acc = self.final_acc_layer(acc, c)

        return acc, x_cur_2, x_cur_1_final 


    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    
    def forward(self, x, t, y):
        """Forward pass of DeepFlow.

        Args:
            x: (N, C, H, W) tensor of spatial noisy inputs, e.g., x_t1
            t: List of (N,), tensor of diffusion timestepsm e.g, [t1, t2, ...]
            y: (N,) tensor of class labels
        
        Returns:
            x_cur_list: lists of velocities after deep supervision
            a_to_s_list: lists of generated acceleration
            zs: lists of projected features for ssl alignment
        """

        y = self.y_embedder(y, self.training)

        # process multiple timesteps.
        patch_x, t_embed_1, t_embed, dt_embed = self.process_multi_times(x, t)
        N, T, D = patch_x.shape

        x_cur_list = [] 
        a_to_s_list = []
        zs = []

        c = t_embed_1 + y 
        df_num = 0
        x_cur = patch_x.clone()  
        
        zs = []
        for i, block in enumerate(self.blocks):
            x_cur = block(x_cur, c)  # (N, T, D)
            # DeepFlow
            if (i + 1) in self.df_idxs: 
                acc, x_cur, x_cur_1_final = self.df_process(
                        x_cur, c, patch_x, dt_embed, df_num)
                a_to_s_list.append(self.unpatchify(acc))
                x_cur_list.append(self.unpatchify(x_cur_1_final))
                c = t_embed[df_num] + y
                # SSL Alignment
                if self.ssl_align:
                    zs.append(self.projectors[df_num](x_cur.reshape(-1, D)).reshape(N, T, -1))
                df_num += 1

        x_cur = self.final_layer[1](x_cur, c)
        x_cur_list.append(self.unpatchify(x_cur))

        return x_cur_list, a_to_s_list, zs




## DeepFlow Configurations
## Currently, we support base model and xlarge model.

def DeepFlow_B_2(**kwargs):
    return DeepFlow(hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)
def DeepFlow_XL_2(**kwargs):
    return DeepFlow(hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


DF_models = {
    'DeepFlow-B/2': DeepFlow_B_2,
    'DeepFlow-XL/2': DeepFlow_XL_2,
    }

    