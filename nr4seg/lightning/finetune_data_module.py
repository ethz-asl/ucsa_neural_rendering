import numpy as np
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional

from nr4seg.dataset import ScanNet, ScanNetCL, ScanNetNGP


class FineTuneDataModule(pl.LightningDataModule):

    def __init__(
        self,
        exp: dict,
        env: dict,
        prev_exp_name: str = "one_step_nerf_only",
    ):
        super().__init__()

        self.env = env
        self.exp = exp
        self.cfg_loader = self.exp["data_module"]
        self.prev_exp_name = prev_exp_name

    def setup(self, stage: Optional[str] = None) -> None:
        ## test adaption (last 20% of the new scenes)
        finetune_seqs = self.exp["scenes"]
        self.scannet_test_ada = ScanNetNGP(
            root=self.env["scannet"],
            mode="val",  # val
            val_mode="gtgt",
            scene_list=finetune_seqs,
        )
        ## test generation
        scannet_25k_dir = self.env["scannet_frames_25k"]
        split_file = os.path.join(
            scannet_25k_dir,
            self.cfg_loader["data_preprocessing"]["split_file"],
        )
        img_list = np.load(split_file)
        self.scannet_test_gen = ScanNet(
            root=self.cfg_loader["root"],
            img_list=img_list["test"],
            mode="test",
        )
        ## train (all Torch-NGP generated labels)
        # random select from

        if not self.exp["cl"]["active"]:
            self.scannet_train = ScanNetNGP(
                root=self.env["scannet"],
                train_image=self.cfg_loader["train_image"],
                train_label=self.cfg_loader["train_label"],
                mode="train",
                scene_list=finetune_seqs,
                prev_exp_name=self.prev_exp_name,
            )
        else:
            scannet_ngp = ScanNetNGP(
                root=self.env["scannet"],
                train_image=self.cfg_loader["train_image"],
                train_label=self.cfg_loader["train_label"],
                mode="train",
                scene_list=finetune_seqs,
                prev_exp_name=self.prev_exp_name,
            )
            split_file_cl = os.path.join(
                scannet_25k_dir,
                self.cfg_loader["data_preprocessing"]["split_file_cl"],
            )
            img_list_cl = np.load(split_file_cl)["train_cl"]
            img_list_cl = img_list_cl[:int(self.exp["cl"]["25k_fraction"] *
                                           len(img_list_cl))]
            scannet_25k = ScanNet(
                root=self.cfg_loader["root"],
                img_list=img_list_cl,
                mode="train",
            )
            self.scannet_train = ScanNetCL(
                scannet_25k,
                scannet_ngp,
                ngp_25k_ratio=self.exp["cl"]["ngp_25k_ratio"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_train,
            batch_size=self.cfg_loader["batch_size"],
            drop_last=True,
            shuffle=True,  # only true in train_dataloader
            collate_fn=self.scannet_train.collate
            if self.exp["cl"]["active"] else None,
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test_ada,
            batch_size=
            1,  ## set bs=1 to ensure a batch always has frames from the same scene
            drop_last=False,
            shuffle=False,  # false
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test_gen,
            batch_size=self.cfg_loader["batch_size"],
            drop_last=False,
            shuffle=False,  # false
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )
