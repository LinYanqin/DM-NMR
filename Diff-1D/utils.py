from pathlib import Path

import torch
import os
import logging
import math
import matplotlib.pyplot as plt
import numpy as np


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        betas = betas_for_alpha_bar(num_diffusion_timesteps,
                                    lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
                                    )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_sigma(t, sde):
    """ VE-SDE """
    sigma_t = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t
    return sigma_t


def im2row(im, winSize):
    batch_size, H = im.shape
    num_patches = H - winSize + 1
    patches = torch.zeros((batch_size, num_patches, winSize), dtype=im.dtype)
    for i in range(num_patches):
        patches[:, i, :] = im[:, i:i+winSize]
    return patches


def row2im(patches, size_data, winSize):
    batch_size, num_patches, _ = patches.shape
    H = size_data
    res = torch.zeros((batch_size, H), dtype=patches.dtype)
    count = torch.zeros((batch_size, H), dtype=patches.dtype)

    for i in range(num_patches):
        res[:, i:i + winSize] += patches[:, i, :]
        count[:, i:i + winSize] += 1
    return res / count


def SAKE(x, y, mask, sakeIter=1, keep=6, ksize=32):
    device = x.device
    Trec = ifft(x)
    Trec = real_to_nchw_comp(Trec)
    Trec = torch.permute(Trec, [0, 2, 1]).squeeze(-1) # [batch, H]
    y = real_to_nchw_comp(y)
    y = torch.permute(y, [0, 2, 1]).squeeze(-1)
    mask = torch.permute(mask, [0, 2, 1]).squeeze(-1)
    batch_size, H = Trec.shape
    Krec =  Trec.clone()

    for n in range(sakeIter):

        patches = im2row(Krec, ksize)  # [batch, num_patches, kh]

        U,S,Vh = torch.linalg.svd(patches, full_matrices=False)
        # keep = math.floor(wnthresh * ksize)

        U_trunc = U[:, :, :keep]
        S_trunc = S[:, :keep]
        Vh_trunc = Vh[:, :keep, :]

        S_trunc_complex = torch.diag_embed(torch.complex(S_trunc, torch.zeros_like(S_trunc)))
        A_recon = U_trunc @ S_trunc_complex @ Vh_trunc

        Kcomplex = row2im(A_recon, H, ksize).to(device)

        Krec = Kcomplex * (1- mask) + y * mask

    Trec = torch.unsqueeze(Krec, -1) # [batch, H, 1]
    Trec = torch.permute(Trec, [0, 2, 1])
    Trec = nchw_comp_to_real(Trec)
    Frec = fft(Trec)
    return Frec


def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return np.transpose(x, (1, 0))


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return x



def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)
        logging.error(f"No checkpoint found at {ckpt_dir}. "
                      f"Returned the same state as input")
        FileNotFoundError(f'No such checkpoint: {ckpt_dir} found!')
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        if not skip_optimizer:
            state['optimizer'].load_state_dict(loaded_state['optimizer'])
        loaded_model_state = loaded_state['model']
        if skip_sigma:
            loaded_model_state.pop('module.sigmas')

        state['model'].load_state_dict(loaded_model_state, strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        print(f'loaded checkpoint dir from {ckpt_dir}')
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


"""
Helper functions for new types of inverse problems
"""

def nchw_comp_to_real(x):
    """
    [1, 1, 320] comp --> [1, 2, 320] real
    """
    x = torch.view_as_real(x)
    x = x.squeeze(dim=1)
    x = x.permute(0, 2, 1)
    return x

def real_to_nchw_comp(x):
    """
    [1, 2, 320] real --> [1, 1, 320] comp
    """
    if len(x.shape) == 3:
        x = x[:, 0:1, :] + x[:, 1:2, :] * 1j
    else:
        x = x[0:1, :] + x[1:2, :] * 1j
    return x

def fftshift(x):
    s = (x.shape[-1] // 2)  # 64
    x = torch.cat([x[:, :, s:], x[:, :, :s]], dim=-1)  # exchange right and left
    return x

def ifftshift(x):
    s = (x.shape[-1] // 2)  # 64
    x = torch.cat([x[:, :, s:], x[:, :, :s]], dim=-1)  # exchange right and left
    return x

def fft(x):
  """ FFT with shifting DC to the center of the image"""
  complex_tensor = real_to_nchw_comp(x)
  fft_result = torch.fft.fft(complex_tensor)
  return nchw_comp_to_real(fft_result)

def ifft(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  complex_tensor = real_to_nchw_comp(x)
  ifft_result = torch.fft.ifft(complex_tensor)
  return nchw_comp_to_real(ifft_result)

