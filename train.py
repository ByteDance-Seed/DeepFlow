"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

DeepFlow Training.

The training script of DeepFlow [1] is simple training pipeline,
built on top of SiT [2] and REPA [3].

[1] Deeply Supervised Flow-Based Generative Models, arxiv 2025.
    Inkyu Shin, Chenglin Yang, Liang-Chieh Chen.

[2] Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers, ECCV 2024
    Nanye Ma, Mark Goldstein, Michael Albergo, Nicholas Boffi, Eric Vanden-Eijnden, Saining Xie.

[3] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think, ICLR 2025
    Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie.


"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.deepflow import DF_models
from loss import DFLoss

from utils.repa_utils import load_encoders, preprocess_raw_image
from utils.utils import *

from dataset import CustomDataset
from diffusers.models import AutoencoderKL

import json

logger = get_logger(__name__)


def main(args):    
    # Set accelerator and device
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    assert args.resolution % 8 == 0, \
        "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # Whether to use SSL alignment.
    if args.ssl_align:
        if args.enc_type != 'None':
            encoders, encoder_types, architectures = load_encoders(
                args.enc_type, device, args.resolution)
        else:
            encoders, encoder_types, architectures = [None], [None], [None]

    # Create model.
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    df_idxs = json.loads(args.df_idxs)
    model = DF_models[args.model](
        depth=args.trans_depth,
        input_size=latent_size,
        num_classes=args.num_classes,
        ssl_align=args.ssl_align,
        df_idxs=df_idxs,
        legacy_scaling=args.legacy_scaling,
        **block_kwargs
    )
    model = model.to(device)
    ema = deepcopy(model).to(device) 
    requires_grad(ema, False)
    
    # Load pretrained VAE.
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    # Create loss function.
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    num_scales = len(df_idxs)+1
    loss_fn = DFLoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        num_scales=num_scales,
        scale_weight=args.scale_weight,
        tg_upper_bound=args.tg_upper_bound,
    )


    if accelerator.is_main_process:
        logger.info(
            f"DeepFlow Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data.
    train_dataset = CustomDataset(args.data_dir, args.ssl_align)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")


    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # Resume.
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    # We skip evaluation while training.
    sample_batch_size = 64 // accelerator.num_processes
    _, gt_xs, _ = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias)
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
    max_epochs = args.max_train_steps // len(train_dataloader) + 1


    # Training starts.
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            labels = y
            with torch.no_grad():
                x = sample_posterior(
                    x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    ## load SSL encoder for feat alignment.
                    if args.ssl_align:
                        for encoder, encoder_type, arch \
                                in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, acc_loss, proj_loss = loss_fn(model, x, model_kwargs, zs, vae)
                loss_mean = loss.mean()
                acc_loss_mean = acc_loss.mean()
                loss = loss_mean + acc_loss_mean * args.acc_coeff
                if args.ssl_align:
                    proj_loss_mean = proj_loss.mean()
                    loss += proj_loss_mean * args.proj_coeff 

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                    
                if accelerator.sync_gradients:
                    update_ema(ema, model) 
            
            ## checkpoint & log
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args, 
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "acc_loss": accelerator.gather(acc_loss_mean).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            if args.ssl_align:
                logs["proj_loss"] = accelerator.gather(proj_loss_mean).mean().detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=200000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")

    # hyperparameters for DeepFlow
    parser.add_argument("--tg-upper-bound", type=float, default=0.01)
    parser.add_argument("--scale-weight", nargs='+', help='List of spliting', default=[1.0,])
    parser.add_argument("--ema-param", type=float, default=0.9)
    parser.add_argument("--df-idxs", type=str, required=True)
    parser.add_argument("--acc-coeff", type=float, default=1.0)
    parser.add_argument("--legacy-scaling", action="store_true")


    # hyperparameter for SiT & REPA
    parser.add_argument("--trans-depth", type=int, default=12)
    parser.add_argument("--ssl-align", action="store_true")
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args


if __name__ == "__main__":
    args = parse_args()
    
    main(args)
