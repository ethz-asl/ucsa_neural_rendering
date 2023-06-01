"""Continual learning protocal for calling vis4d in multiple stages."""
import argparse
import os
import sys
import wandb

from nr4seg import ROOT_DIR
from nr4seg.utils import load_yaml
from scripts.train_joint import train

SCENE_ORDER = [
    "scene0000_00",
    "scene0001_00",
    "scene0002_00",
    "scene0003_00",
    "scene0004_00",
    "scene0005_00",
    "scene0006_00",
    "scene0007_00",
    "scene0008_00",
    "scene0009_00",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="cfg/exp/finetune/deeplabv3_s0.yml",
        help=
        "Experiment yaml file path relative to template_project_name/cfg/exp directory.",
    )
    parser.add_argument(
        "--exp_name",
        default="debug",
        help="overall experiment of this continual learning experiment.",
    )
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument(
        "--fix_nerf",
        action="store_true",
        help="whether or not to fix nerf during joint training",
    )

    parser.add_argument("--project_name", default="test_one_by_one")
    parser.add_argument("--nerf_train_epoch", default=10, type=int)

    parser.add_argument("--joint_train_epoch", default=10, type=int)
    args = parser.parse_args()
    return args


def main():
    """Main function."""
    env_cfg_path = os.path.join(ROOT_DIR, "cfg/env",
                                os.environ["ENV_WORKSTATION_NAME"] + ".yml")
    env = load_yaml(env_cfg_path)
    os.chdir(ROOT_DIR)
    args = parse_args()
    exp_cfg_path = os.path.join(ROOT_DIR, args.exp)
    exp = load_yaml(exp_cfg_path)
    exp_name = args.exp_name
    exp["exp_name"] = exp_name

    prev_stage = ""
    stage = "init"
    exp["scenes"] = []

    for i, new_scene in enumerate(SCENE_ORDER):
        exp["scenes"].append(new_scene)
        prev_stage = stage
        stage = f"stage_{i}"
        exp["general"]["name"] = f"{exp_name}/{stage}"

        # train on new class
        exp["trainer"]["resume_from_checkpoint"] = False
        exp["trainer"]["load_from_checkpoint"] = True
        if i == 0:
            exp["general"]["load_pretrain"] = True
            old_model_path = exp["general"]["checkpoint_load"]
        else:
            exp["general"]["load_pretrain"] = False
            old_model_path = os.path.join("experiments", exp_name, prev_stage,
                                          "deeplab.ckpt")

        exp["general"]["checkpoint_load"] = old_model_path

        print(f"training on: {new_scene}")

        train(exp, env, exp_cfg_path, env_cfg_path, args)
        wandb.finish()


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(1)
