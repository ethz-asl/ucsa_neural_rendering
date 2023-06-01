import os

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger

from nr4seg.utils import flatten_dict

__all__ = ["get_neptune_logger", "get_tensorboard_logger", "get_wandb_logger"]


def log_important_params(exp):
    dic = {}
    dic = flatten_dict(exp)
    return dic


def get_neptune_logger(exp, env, exp_p, env_p, project_name=""):
    params = log_important_params(exp)

    name_full = exp["general"]["name"]
    name_short = "__".join(name_full.split("/")[-2:])

    return NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=project_name,
        name=name_short,
        tags=[
            os.environ["ENV_WORKSTATION_NAME"],
            name_full.split("/")[-2],
            name_full.split("/")[-1],
        ],
    )


def get_wandb_logger(exp, env, exp_p, env_p, project_name, save_dir):
    params = log_important_params(exp)
    name_full = exp["general"]["name"]
    name_short = "__".join(name_full.split("/")[-2:])
    return WandbLogger(
        name=name_short,
        project=project_name,
        save_dir=save_dir,
    )


def get_tensorboard_logger(exp, env, exp_p, env_p):
    params = log_important_params(exp)
    return TensorBoardLogger(
        save_dir=exp["general"]["name"],
        name="tensorboard",
        default_hp_metric=params,
    )
