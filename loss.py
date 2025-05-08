# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the CC-BY-NC 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#
#     https://github.com/ByteDance-Seed/DeepFlow/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 


"""
DeepFlow Script for Loss Function.
"""

import torch
import numpy as np
import torch.nn.functional as F
import random


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))



class DFLoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            num_scales=1,
            scale_weight=False,
            tg_upper_bound=0.01,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.num_scales = num_scales
        self.tg_upper_bound = tg_upper_bound
        
        self.scale_coeff = scale_weight
        assert len(self.scale_coeff) == self.num_scales

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None, vae=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        time_input_list = []

        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        noises = torch.randn_like(images)
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        init_time = torch.zeros((images.shape[0], 1, 1, 1)).to(
                        device=images[0].device, dtype=images[0].dtype)

        # set upper bound for time gap (beta in our paper).
        tg_upper_bound = self.tg_upper_bound

        # linear interpolant to obtain noisy image
        alpha_t, sigma_t, _, _ = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises
        model_target = noises - images 
        model_input_list = []

        # Obtain different timesteps
        random_number = random.random()
        for _ in range(self.num_scales):
            if random_number < 0.5:
                time_input += init_time
                time_input = torch.clamp(time_input, max=1.0)
            else:
                time_input -= init_time
                time_input = torch.clamp(time_input, min=0.0)
            time_input_list.append(time_input.clone().flatten())
            time_input_init = torch.rand((images.shape[0], 1, 1, 1)).to(
                device=images.device, dtype=images.dtype)
            init_time = time_input_init * tg_upper_bound
            alpha_t, sigma_t, _, _ = self.interpolant(time_input.clone())
            model_input_list.append(alpha_t * images + sigma_t * noises)

        # Run DeepFlow with different time-steps.
        model_outputs, a_to_s_list, zs_tilde = model(model_input, time_input_list, **model_kwargs)

        # Loss: denoising loss + acceleration loss.
        denoising_loss = 0.
        pred_a2s = 0.
        for i, model_output in enumerate(model_outputs):
            denoising_loss += float(self.scale_coeff[i]) * mean_flat((model_output - model_target) ** 2)
            v_cur_t_s = model_outputs[i].detach()
            if i < len(model_outputs) - 1:
                dt = 0.0 - time_input_list[i]
                dt = dt.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                next_sample = model_input_list[i] + v_cur_t_s * dt + 0.5 * a_to_s_list[i] * (dt ** 2)
                pred_a2s += mean_flat((next_sample - images) ** 2)

        # Loss: loss for SSL alignment.
        proj_loss = 0.
        if len(zs) > 0:
            bsz = zs[0].shape[0]
            if len(zs_tilde) > 1:
                zs = zs * len(zs_tilde)
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
            
        return denoising_loss, pred_a2s, proj_loss
 







