import argparse
import numpy as np
import os
import random

from glob import glob

from nr4seg.utils import load_yaml

parser = argparse.ArgumentParser(description="Generates a data split file.")

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
default_exp_path = os.path.join(
    curr_dir_path, "../../cfg/exp/pretrain_scannet_25k_deeplabv3.yml")
parser.add_argument("--config",
                    type=str,
                    default=default_exp_path,
                    help="Path to config file.")

args = parser.parse_args()

cfg = load_yaml(args.config)
cfg = cfg["data_module"]

train_all = glob(cfg["root"] + cfg["data_preprocessing"]["image_regex"])
random.shuffle(train_all)
val = train_all[:int(len(train_all) * cfg["data_preprocessing"]["val_ratio"])]
train = train_all[int(len(train_all) * cfg["data_preprocessing"]["val_ratio"]):]
test = val
train, val, test = map(sorted, [train, val, test])

split_cl = {"train_cl": train}

split_dict = {"train": train, "test": test, "val": val}
env_name = os.environ["ENV_WORKSTATION_NAME"]
env = load_yaml(os.path.join("cfg/env", env_name + ".yml"))
scannet_25k_dir = env["scannet_frames_25k"]
out_file = os.path.join(scannet_25k_dir,
                        cfg["data_preprocessing"]["split_file_cl"])
np.savez(out_file, **split_cl)
