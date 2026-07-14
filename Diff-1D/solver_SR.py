import os
import json
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.script_util import create_model

from pathlib import Path
import time

from physics.nmr import NMR_two_channel_time, NMR_two_channel_freq
from utils import clear, get_beta_schedule, ncwh_to_one_abs, SAKE
from functools import partial
import scipy.io as sio



class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.args.image_folder = Path(self.args.image_folder)
        (self.args.image_folder / "recon" / "progress").mkdir(exist_ok=True, parents=True)
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        ckpt = os.path.join(self.args.exp, self.args.ckpt_load_name)
        
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"Model ckpt loaded from {ckpt}")
        model.to(self.device)
        model.eval()

        print('Run DDS.',
            f'{self.args.T_sampling} sampling steps.',
            f'Task: nmr_dc.'
            )
        self.dds(model)
            
            
    def dds(self, model):
        args, config = self.args, self.config
        print(f"Dataset path: {self.args.dataset_path}")
        root = Path(self.args.dataset_path) / 'SR'
        
        # parameters to be moved to args
        
        # read all data
        for sample_n in range(1,11):
            filename = f'{root}/{sample_n}/input_data.mat'
            fname_list = [filename]

            print(f'Beginning inference')
            tic = time.time()
            for m, fname in enumerate(fname_list):
                x_orig = sio.loadmat(fname)
                x_orig = x_orig[f'input_data']
                if len(x_orig.shape) == 2:
                    x_orig = np.expand_dims(x_orig, axis=0)
                x_orig = np.transpose(x_orig, [0, 2, 1])
                x_orig = np.asarray(x_orig, dtype=np.float32)
                # x_orig, x_min, x_max = get_data_scaler(x_orig)
                x_orig = torch.from_numpy(x_orig)
                x_orig = x_orig.to(self.device)

                mask = sio.loadmat(f'{root}/{sample_n}/mask.mat')
                mask = mask['mask_data']
                mask = mask[0, :, :]
                mask = torch.from_numpy(mask)
                mask = mask.view(-1, 1, config.data.image_size)
                mask = mask.to(config.device)
                if "Time" in args.nmr_domain:
                    chunk_size = 370
                    A_funcs = NMR_two_channel_time(config.data.image_size, mask)
                else:
                    chunk_size = 370
                    A_funcs = NMR_two_channel_freq(config.data.image_size, mask)
                A = lambda z: A_funcs.A(z)
                AT = lambda z: A_funcs.AT(z)
                Ap = lambda z: A_funcs.A_dagger(z)

                res = np.zeros((x_orig.shape[0], config.data.channels, config.data.image_size))
                num_chunks = (x_orig.shape[0] + chunk_size - 1) // chunk_size

                for idx in range(num_chunks):

                    start_idx = idx * chunk_size
                    end_idx = min((idx + 1) * chunk_size, x_orig.shape[0])
                    print(f'start:{start_idx}, end:{end_idx}')
                    x_slice = x_orig[start_idx:end_idx, :, :]

                    y = A(x_slice)
                    Apy = Ap(y)
                    ATy = AT(y)

                    plt.plot(np.linspace(0, config.data.image_size, config.data.image_size),
                             ncwh_to_one_abs(clear(y), "Time"))
                    plt.savefig(str(self.args.image_folder / "recon" / f"progress" / f"input_{str(idx).zfill(3)}.tif"),
                                dpi=300, bbox_inches='tight')
                    plt.close()

                    # init x_T
                    x = torch.randn(
                        x_slice.shape[0],
                        config.data.channels,
                        config.data.image_size,
                        device=self.device,
                    )

                    with torch.no_grad():
                        skip = config.diffusion.num_diffusion_timesteps//args.T_sampling
                        n = x.size(0)
                        x0_preds = []
                        xs = [x]

                        # generate time schedule
                        times = range(0, 1000, skip)
                        times_next = [-1] + list(times[:-1])
                        times_pair = zip(reversed(times), reversed(times_next))

                        # reverse diffusion sampling
                        for i, j in tqdm.tqdm(times_pair, total=len(times)):
                            t = (torch.ones(n) * i).to(x.device)
                            next_t = (torch.ones(n) * j).to(x.device)

                            at = compute_alpha(self.betas, t.long())
                            at_next = compute_alpha(self.betas, next_t.long())
                            sigma_t = (1 - at_next ** 2).sqrt()

                            xt = xs[-1].to('cuda')

                            # 0. NFE
                            et = model(xt, t)
                            if et.shape[1] == 4:
                                et = et[:, :et.size(1)//2]

                            # 1. Tweedie
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()


                            if j % 50 == 0:
                                plt.figure(figsize=(8,4))
                                plt.plot(np.linspace(0, config.data.image_size, config.data.image_size),
                                         ncwh_to_one_abs(clear(x0_t), args.nmr_domain))
                                plt.savefig(str(self.args.image_folder / "recon" / f"progress" / f"reco_{str(j).zfill(3)}.tif"), dpi=300, bbox_inches='tight')
                                plt.close()

                            # 2. DC
                            if self.args.sigma_y == 0.0:
                                lambda_t = 1.
                                gamma_t = 1.
                            else:
                                if sigma_t[0,0,0].item() >= at_next[0,0,0].item() * self.args.sigma_y:
                                    lambda_t = 1.
                                    gamma_t = (sigma_t ** 2 - (at_next * self.args.sigma_y) ** 2).sqrt()
                                else:
                                    lambda_t = sigma_t / (at_next * self.args.sigma_y)
                                    gamma_t = 0.

                            # x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)

                            if j % 100 == 0:
                                x0_t_hat = SAKE(x0_t, y, mask, wnthresh=0.2)
                            else:
                                x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)

                            zeta = self.args.zeta

                            c1 = (1 - at_next).sqrt() * (zeta ** 0.5)
                            c2 = (1 - at_next).sqrt() * ((1 - zeta) ** 0.5)

                            # DDIM sampling
                            if j != 0:
                                xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t_hat) + c2 * et)
                            # Final step
                            else:
                                xt_next = x0_t_hat

                            x0_preds.append(x0_t.to('cpu'))
                            xs.append(xt_next.to('cpu'))
                        x = xs[-1]

                    x_sv = clear(x)
                    # x_sv = get_data_inverse_scaler(clear(x), x_min, x_max)
                    res[start_idx:end_idx, :, :] = x_sv

                sio.savemat(str(self.args.dataset_path / "SR" / f"{sample_n}" / f"res{m+1}.mat"), {f'res': res})
            toc = time.time() - tic
            print(f'The sample: {sample_n}. Time took for recon: {toc} secs.')



def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a
