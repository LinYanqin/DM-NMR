import argparse
import traceback
import logging
import yaml
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import numpy as np

from solver_SR import Diffusion as Diffusion_1d
from pathlib import Path

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default="nmr_20250325.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--seed", type=int, default=256, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="./model/ADM_20250325/", help="Path for saving running related data."
    )
    parser.add_argument(
        "--ckpt_load_name", type=str, default="model100000.pt", help="Load pre-trained ckpt"
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0.00, help="sigma_y"
    )
    parser.add_argument(
        "--zeta", type=float, default=0.1, help="Zeta"
    )
    parser.add_argument(
        "--T_sampling", type=int, default=1000, help="Total number of sampling steps"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="./results/",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./Process_data", help="The folder of the dataset"
    )
    parser.add_argument(
        "--mask_type", type=str, default="Rand", help="Undersampling type"
    )
    parser.add_argument(
        "--sample_rate", type=int, default="25", help="Sample rate"
    )
    parser.add_argument(
        "--nmr_domain", type=str, default="Freq", help="NMR domain"
    )
    parser.add_argument(
        "--nmr_type", type=str, default="15430", help="NMR type"
    )


    args = parser.parse_args()

    # parse config file
    with open(os.path.join("config/", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    args.dataset_path = Path(args.dataset_path) / "nmr_dc" / f"{args.mask_type}_SR_{args.sample_rate}_{args.nmr_domain}_{args.nmr_type}"

    args.image_folder = Path(args.image_folder) / "nmr_dc" / f"{args.mask_type}_SR_{args.sample_rate}_{args.nmr_domain}_{args.nmr_type}"
                            
    args.image_folder.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    runner = Diffusion_1d(args, config)
    runner.sample()
    return 0


if __name__ == "__main__":
    sys.exit(main())
