"""
Train a diffusion model on images.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import torch
from torch_utils import distributed as dist
from torch_utils import misc
from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dist.init()
    model_dir = os.path.join("/data4/cwh/ADM-NMR-1D/model/",args.model_path)
    logger.configure(dir=model_dir)

    logger.log("creating model and diffusion...")
    logger.log(args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([args.batch_size, 2, 256], device=device)
            sigma = torch.ones([args.batch_size], device=device)
            misc.print_module_summary(model, [images, sigma], max_nesting=2)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/data4/cwh/DPSdata/20250325_1130/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="ADM_20250325_T_64", help="model path"
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
