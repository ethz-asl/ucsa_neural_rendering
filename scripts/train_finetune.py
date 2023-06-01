import argparse
import os
from pathlib import Path
import torch
import shutil

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.profiler import AdvancedProfiler

from nr4seg import ROOT_DIR
from nr4seg.lightning import FineTuneDataModule, SemanticsLightningNet
from nr4seg.utils import flatten_dict, get_wandb_logger, load_yaml


def train(exp, env, exp_cfg_path, env_cfg_path, args) -> float:
    seed_everything(args.seed)

    ####################################################################################################################

    ############################################  CREAT EXPERIMENT FOLDER  #############################################
    model_path = os.path.join(env["results"], exp["general"]["name"])
    if exp["general"]["clean_up_folder_if_exists"]:
        shutil.rmtree(model_path, ignore_errors=True)

    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Copy config files
    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]
    print(f"Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}")
    shutil.copy(exp_cfg_path, f"{model_path}/{exp_cfg_fn}")
    shutil.copy(env_cfg_path, f"{model_path}/{env_cfg_fn}")
    exp["general"]["name"] = model_path
    ####################################################################################################################

    #################################################  CREATE LOGGER  ##################################################

    logger = get_wandb_logger(
        exp=exp,
        env=env,
        exp_p=exp_cfg_path,
        env_p=env_cfg_path,
        project_name=args.project_name,
        save_dir=model_path,
    )
    ex = flatten_dict(exp)
    logger.log_hyperparams(ex)

    ####################################################################################################################

    ###########################################  CREAET NETWORK AND DATASET  ###########################################
    model = SemanticsLightningNet(exp, env)
    datamodule = FineTuneDataModule(exp, env, prev_exp_name=args.prev_exp_name)
    ####################################################################################################################

    #################################################  TRAINER SETUP  ##################################################
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    cb_ls = [lr_monitor]

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        save_last=True,
        save_top_k=1,
    )
    cb_ls.append(checkpoint_callback)

    # set gpus
    if (exp["trainer"]).get("gpus", -1) == -1:
        nr = torch.cuda.device_count()
        print(f"Set GPU Count for Trainer to {nr}!")
        for i in range(nr):
            print(f"Device {i}: ", torch.cuda.get_device_name(i))
        exp["trainer"]["gpus"] = nr

    # profiler
    if exp["trainer"].get("profiler", False):
        exp["trainer"]["profiler"] = AdvancedProfiler(
            output_filename=os.path.join(model_path, "profile.out"))
    else:
        exp["trainer"]["profiler"] = False

    # check if restore checkpoint
    if exp["trainer"]["resume_from_checkpoint"] is True:
        exp["trainer"]["resume_from_checkpoint"] = exp["general"][
            "checkpoint_load"]
    else:
        del exp["trainer"]["resume_from_checkpoint"]

    if exp["trainer"]["load_from_checkpoint"] is True:
        checkpoint = torch.load(exp["general"]["checkpoint_load"])
        checkpoint = checkpoint["state_dict"]
        # remove any aux classifier stuff
        removekeys = [
            key for key in checkpoint.keys()
            if key.startswith("_model._model.aux_classifier")
        ]
        for key in removekeys:
            del checkpoint[key]
        model.load_state_dict(checkpoint, strict=True)

    del exp["trainer"]["load_from_checkpoint"]

    trainer = Trainer(
        **exp["trainer"],
        plugins=DDPPlugin(find_unused_parameters=False),
        default_root_dir=model_path,
        callbacks=cb_ls,
        logger=logger,
    )
    ####################################################################################################################
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    os.chdir(ROOT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="cfg/exp/finetune/deeplabv3_s0.yml",
        help=
        "Experiment yaml file path relative to template_project_name/cfg/exp directory.",
    )
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--project_name", default="scannet_debug")
    parser.add_argument("--prev_exp_name", default="one_step_nerf_only")
    args = parser.parse_args()
    exp_cfg_path = os.path.join(ROOT_DIR, args.exp)
    exp = load_yaml(exp_cfg_path)
    env_cfg_path = os.path.join(ROOT_DIR, "cfg/env",
                                os.environ["ENV_WORKSTATION_NAME"] + ".yml")
    env = load_yaml(env_cfg_path)

    train(exp, env, exp_cfg_path, env_cfg_path, args)
