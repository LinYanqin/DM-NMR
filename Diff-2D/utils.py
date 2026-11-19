import math
from pathlib import Path

import torch
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev


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
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas



def im2row(im, winSize):
    batch_size, H, W, C = im.shape
    kh, kw = winSize
    num_patches = (H - kh + 1) * (W - kw + 1)
    patches = torch.zeros((batch_size, num_patches, kh*kw*C), dtype=im.dtype)
    count = 0
    for i in range(H - kh + 1):
        for j in range(W - kw + 1):
            patch = im[:, i:i+kh, j:j+kw, :]    # [batch, kh, kw, C]
            patches[:, count, :] = patch.reshape(batch_size, -1)
            count = count + 1
    return patches


def row2im(patches, size_data, winSize):
    batch_size, num_patches, _ = patches.shape
    H, W, C = size_data
    kh, kw = winSize
    res = torch.zeros((batch_size, H, W, C), dtype=patches.dtype)
    W_count = torch.zeros((batch_size, H, W, C), dtype=patches.dtype)

    count = 0

    for i in range(H - kh + 1):
        for j in range(W - kw + 1):
            patch = patches[:, count, :].view(batch_size, kh, kw, C)
            res[:, i:i + kh, j:j + kw, :] += patch
            W_count[:, i:i + kh, j:j + kw, :] += 1
            count = count + 1
    return res / W_count


def SAKE(x, y, mask, sakeIter=1, wnthresh=0.2, ksize=8, keep=25):
    device = x.device
    Trec = ifft2(x)
    Trec = four_to_nchw_comp(Trec)
    Trec = torch.permute(Trec, [0, 2, 3, 1])  # [batch, H, W, 2]
    y = four_to_nchw_comp(y)
    y = torch.permute(y, [0, 2, 3, 1])
    mask = torch.permute(mask, [0, 2, 3, 1])
    batch_size, H, W, C = Trec.shape
    Krec =  Trec.clone()

    for n in range(sakeIter):

        patches = im2row(Krec, [ksize, ksize])  # [batch, num_patches, kh*kw*C]
        A = patches.view(batch_size, -1, patches.shape[-1])
        # A_tensor = torch.view_as_real(A)# [batch, num_patches, kh*kw*C]

        U,S,Vh = torch.linalg.svd(A, full_matrices=False)
        # keep = math.floor(wnthresh * ksize * ksize * C)


        U_trunc = U[:, :, :keep]
        S_trunc = S[:, :keep]
        Vh_trunc = Vh[:, :keep, :]

        S_trunc_complex = torch.diag_embed(torch.complex(S_trunc, torch.zeros_like(S_trunc)))
        A_recon = U_trunc @ S_trunc_complex @ Vh_trunc
        # A_recon = torch.view_as_complex(A_recon)

        Kcomplex = row2im(A_recon, (H, W, C), [ksize, ksize]).to(device)

        Krec = Kcomplex * (1- mask) + y * mask

    Trec = torch.permute(Krec, [0, 3, 1, 2])
    Trec = nchw_comp_to_four(Trec)
    Frec = fft2(Trec)
    return Frec


def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return np.transpose(x, (1, 2, 0))


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
    [1, 1, 320, 320] comp --> [1, 2, 320, 320] real
    """
    x = torch.view_as_real(x)
    x = x.squeeze(dim=1)
    x = x.permute(0, 3, 1, 2)
    return x

def real_to_nchw_comp(x):
    """
    [1, 2, 320, 320] real --> [1, 1, 320, 320] comp
    """
    if len(x.shape) == 4:
        x = x[:, 0:1, :, :] + x[:, 1:2, :, :] * 1j
    elif len(x.shape) == 3:
        x = x[0:1, :, :] + x[1:2, :, :] * 1j
    return x

def nchw_comp_to_four(x):
    """
    [1, 1, 320, 320] comp --> [1, 2, 320, 320] real
    """
    x_1 = nchw_comp_to_real(x[:, 0:1, :, :])
    x_2 = nchw_comp_to_real(x[:, 1:2, :, :])
    return torch.cat([x_1, x_2], dim=1)

def four_to_nchw_comp(x):
    """
    [1, 2, 320, 320] real --> [1, 1, 320, 320] comp
    """
    if len(x.shape) == 4:
        x_1 = real_to_nchw_comp(x[:, 0:2, :, :])
        x_2 = real_to_nchw_comp(x[:, 2:4, :, :])
        return torch.cat([x_1, x_2], dim=1)
    elif len(x.shape) == 3:
        x_1 = real_to_nchw_comp(x[0:2, :, :])
        x_2 = real_to_nchw_comp(x[2:4, :, :])
        return torch.cat([x_1, x_2], dim=0)
    return x

def fftshift(x):
    s0 = (x.shape[-2] // 2)  # 32
    s1 = (x.shape[-1] // 2)  # 32
    x = torch.cat([x[:, :, s0:, :], x[:, :, :s0, :]], dim=-2)  # exchange down and up
    x = torch.cat([x[:, :, :, s1:], x[:, :, :, :s1]], dim=-1)  # exchange right and left
    return x

def ifftshift(x):
    s0 = (x.shape[-2] // 2)  # 32
    s1 = (x.shape[-1] // 2)  # 32
    x = torch.cat([x[:, :, s0:, :], x[:, :, :s0, :]], dim=-2)  # exchange down and up
    x = torch.cat([x[:, :, :, s1:], x[:, :, :, :s1]], dim=-1)  # exchange right and left
    return x

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  x_1 = x[:, 0:2, :, :]
  x_2 = x[:, 2:4, :, :]
  complex_tensor_1 = real_to_nchw_comp(x_1)
  fft_result_1 = fftshift(torch.fft.fft2(complex_tensor_1))
  real_tensor_1 = nchw_comp_to_real(fft_result_1)
  complex_tensor_2 = real_to_nchw_comp(x_2)
  fft_result_2 = fftshift(torch.fft.fft2(complex_tensor_2))
  real_tensor_2 = nchw_comp_to_real(fft_result_2)
  return torch.cat([real_tensor_1, real_tensor_2], dim=1)
  # return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])

def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  x_1 = x[:, 0:2, :, :]
  x_2 = x[:, 2:4, :, :]
  complex_tensor_1 = real_to_nchw_comp(x_1)
  ifft_result_1 = torch.fft.ifft2(ifftshift(complex_tensor_1))
  real_tensor_1 = nchw_comp_to_real(ifft_result_1)
  complex_tensor_2 = real_to_nchw_comp(x_2)
  ifft_result_2 = torch.fft.ifft2(ifftshift(complex_tensor_2))
  real_tensor_2 = nchw_comp_to_real(ifft_result_2)
  return torch.cat([real_tensor_1, real_tensor_2], dim=1)
  # return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))



