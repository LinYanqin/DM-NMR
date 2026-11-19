import argparse
import traceback
import logging
import yaml
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import numpy as np

from solver_2d import Diffusion as Diffusion_2d
from pathlib import Path
from preprocess import preprocess_BMRB

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default="nmr.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--seed", type=int, default=256, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="./model/", help="Path for saving running related data."
    )
    parser.add_argument(
        "--ckpt_load_name", type=str, default="model.pt", help="Load pre-trained ckpt"
    )
    parser.add_argument(
        "--T_sampling", type=int, default=1000, help="Total number of sampling steps"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./Process_data", help="The folder of the dataset"
    )
    parser.add_argument(
        "--mask_type", type=str, default="Pois", help="Under sampling scheme type"
    )
    parser.add_argument(
        "--sample_rate", type=int, default="3", help="Sample rate"
    )
    parser.add_argument(
        "--nmr_type", type=str, default="16101", help="NMR name"
    )
    parser.add_argument(
        "--target_path", type=str, default="./nmrpipe_data/fully_sampled/fid_temp.dat", help="target_path"
    )
    parser.add_argument(
        "--mask_path", type=str, default="./mask/3/mask1.txt", help="mask_path"
    )
    parser.add_argument(
        "--direct_dim", type=int, default=280, help="direct_dim"
    )


    args = parser.parse_args()

    # parse config file
    with open(os.path.join("config/", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    args.dataset_path = Path(args.dataset_path) / "nmr" / f"{args.mask_type}_SR_{args.sample_rate}_Type_{args.nmr_type}"

    args.output_path = Path("./Res_data") / "nmr" / f"{args.mask_type}_SR_{args.sample_rate}_Type_{args.nmr_type}"

    args.dataset_path.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    SNR = preprocess_BMRB(args.dataset_path, args.target_path, args.mask_path, new_config.data.row_size,
                          new_config.data.col_size, args.direct_dim)

    args.output_path.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

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
    runner = Diffusion_2d(args, config)
    runner.sample()
    return 0


if __name__ == "__main__":
    sys.exit(main())
