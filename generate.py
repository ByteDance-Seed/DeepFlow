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
DeepFlow Script for Generation.

Generation pipeline was built on top of DiT [1] and SiT [2].


[1] Scalable Diffusion Models with Transformers, ICCV 2023
    William Peebles, Saining Xie

[2] Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers, ECCV 2024
    Nanye Ma, Mark Goldstein, Michael Albergo, Nicholas Boffi, Eric Vanden-Eijnden, Saining Xie.
"""

import torch
import torch.distributed as dist
from models.deepflow import DF_models
from diffusers.models import AutoencoderKL as VAE
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import deepflow_euler_maruyama_sampler
import json


def create_zipped_file_from_samples(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def sampling(args):
    """
    deepflow sampling
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model.
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8

    df_idxs = json.loads(args.df_idxs)
    model = DF_models[args.model](
        depth=args.trans_depth,
        input_size=latent_size,
        num_classes=args.num_classes,
        ssl_align=args.ssl_align,
        df_idxs=df_idxs,
        legacy_scaling=args.legacy_scaling,
        **block_kwargs,
    ).to(device)
    
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']
    
    model.load_state_dict(state_dict)
    model.eval()  # important!

    # Load pretrained VAE.
    vae_module = VAE.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    cfg_use = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_name = args.model.replace("/", "-")
    ckpt_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_name}-{ckpt_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(
        math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, \
        "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, \
        "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    num_inter = len(df_idxs) + 1

    for i in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        if cfg_use:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            num_inter=num_inter,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = deepflow_euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()
            if cfg_use:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            samples = vae_module.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_zipped_file_from_samples(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(DF_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # hyperparameters for DeepFlow
    parser.add_argument("--ssl-align", action="store_true")
    parser.add_argument("--df-idxs", type=str, required=True)
    parser.add_argument("--trans-depth", type=int, default=12)
    parser.add_argument("--legacy-scaling", action="store_true")


    args = parser.parse_args()
    sampling(args)
