import argparse
import os
import shutil
import torch

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from nr4seg import ROOT_DIR
from nr4seg.lightning import JointTrainDataModule, JointTrainLightningNet
from nr4seg.utils import flatten_dict, get_wandb_logger, load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="cfg/exp/finetune/deeplabv3_s0.yml",
        help=
        ("Experiment yaml file path relative to template_project_name/cfg/exp "
         "directory."),
    )
    parser.add_argument(
        "--exp_name",
        default="debug",
        help="overall experiment of this continual learning experiment.",
    )

    parser.add_argument(
        "--fix_nerf",
        action="store_true",
        help="whether or not to fix nerf during joint training",
    )

    parser.add_argument("--seed", default=123, type=int)

    parser.add_argument("--project_name", default="test_one_by_one")
    parser.add_argument("--nerf_train_epoch", default=10, type=int)

    parser.add_argument("--joint_train_epoch", default=10, type=int)
    args = parser.parse_args()
    return args


def train(exp, env, exp_cfg_path, env_cfg_path, args) -> float:
    seed_everything(args.seed)
    exp["exp_name"] = args.exp_name
    exp["fix_nerf"] = args.fix_nerf

    # Create experiment folder.
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

    # Create logger.
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

    # Create network and dataset.
    model = JointTrainLightningNet(exp, env)
    datamodule = JointTrainDataModule(exp, env)
    datamodule.setup()

    # Trainer setup.
    # - Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    cb_ls = [lr_monitor]

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        save_last=True,
        save_top_k=1,
    )
    cb_ls.append(checkpoint_callback)
    # - Set GPUs.
    if (exp["trainer"]).get("gpus", -1) == -1:
        nr = torch.cuda.device_count()
        print(f"Set GPU Count for Trainer to {nr}!")
        for i in range(nr):
            print(f"Device {i}: ", torch.cuda.get_device_name(i))
        exp["trainer"]["gpus"] = nr

    # - Check whether to restore checkpoint.
    if exp["trainer"]["resume_from_checkpoint"] is True:
        exp["trainer"]["resume_from_checkpoint"] = exp["general"][
            "checkpoint_load"]
    else:
        del exp["trainer"]["resume_from_checkpoint"]

    if exp["trainer"]["load_from_checkpoint"] is True:
        if exp["general"]["load_pretrain"]:
            checkpoint = torch.load(exp["general"]["checkpoint_load"])
            checkpoint = checkpoint["state_dict"]
            # remove any aux classifier stuff
            removekeys = [
                key for key in checkpoint.keys()
                if key.startswith("_model._model.aux_classifier")
            ]
            for key in removekeys:
                del checkpoint[key]

            seg_model_state_dict = {}
            for key in checkpoint.keys():
                seg_model_key = key.split(".", 1)[1]
                seg_model_state_dict[seg_model_key] = checkpoint[key]

            model.seg_model.load_state_dict(seg_model_state_dict, strict=True)
        else:
            checkpoint = torch.load(exp["general"]["checkpoint_load"])
            checkpoint = checkpoint["state_dict"]
            model.seg_model.load_state_dict(checkpoint)

    del exp["trainer"]["load_from_checkpoint"]

    # - Add distributed plugin.
    if exp["trainer"]["gpus"] > 1:
        if (exp["trainer"]["accelerator"] == "ddp" or
                exp["trainer"]["accelerator"] is None):
            ddp_plugin = DDPPlugin(find_unused_parameters=exp["trainer"].get(
                "find_unused_parameters", False))
        exp["trainer"]["plugins"] = [ddp_plugin]

    exp["trainer"]["max_epochs"] = args.nerf_train_epoch

    trainer_nerf = Trainer(
        **exp["trainer"],
        default_root_dir=model_path,
        logger=logger,
        callbacks=cb_ls,
    )

    exp["trainer"]["check_val_every_n_epoch"] = 10
    exp["trainer"]["max_epochs"] = args.joint_train_epoch
    trainer_joint = Trainer(
        **exp["trainer"],
        default_root_dir=model_path,
        logger=logger,
        callbacks=cb_ls,
    )

    # Train NeRF.
    model.joint_train = False
    trainer_nerf.fit(model,
                     train_dataloaders=datamodule.train_dataloader_nerf())
    # test initial nerf performance on the training set
    trainer_joint.test(model, dataloaders=datamodule.test_dataloader_nerf())
    # # test initial seg performance on the validation set
    trainer_joint.validate(model, dataloaders=datamodule.val_dataloader())
    # joint train + old scenes
    model.joint_train = True
    # trainer_seg.fit(model, train_dataloaders=datamodule.train_dataloader_seg(), val_dataloaders=datamodule.val_dataloader())
    trainer_joint.fit(
        model,
        train_dataloaders=datamodule.train_dataloader_joint(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    # test nerf performance on the training set after joint training + test generalization performance on scannet 25k
    trainer_joint.test(model, dataloaders=datamodule.train_dataloader_nerf())
    # predict pseudo labels
    trainer_joint.predict(model, dataloaders=datamodule.predict_dataloader())
    # save checkpoint of the deeplab model
    torch.save(
        {"state_dict": model.seg_model.state_dict()},
        os.path.join(model_path, "deeplab.ckpt"),
    )


if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    args = parse_args()
    exp_cfg_path = os.path.join(ROOT_DIR, args.exp)
    exp = load_yaml(exp_cfg_path)
    exp["general"]["load_pretrain"] = True
    env_cfg_path = os.path.join(ROOT_DIR, "cfg/env",
                                os.environ["ENV_WORKSTATION_NAME"] + ".yml")
    env = load_yaml(env_cfg_path)
    train(exp, env, exp_cfg_path, env_cfg_path, args)
