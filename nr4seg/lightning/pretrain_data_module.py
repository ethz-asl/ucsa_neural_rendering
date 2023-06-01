import numpy as np
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional

from nr4seg.dataset import ScanNet


class PretrainDataModule(pl.LightningDataModule):

    def __init__(self, env: dict, cfg_dm: dict):
        super().__init__()

        self.cfg_dm = cfg_dm
        self.env = env

    def setup(self, stage: Optional[str] = None) -> None:
        split_file = os.path.join(
            self.cfg_dm["root"],
            self.cfg_dm["data_preprocessing"]["split_file"],
        )
        img_list = np.load(split_file)
        self.scannet_test = ScanNet(root=self.cfg_dm["root"],
                                    img_list=img_list["test"],
                                    mode="test")
        self.scannet_train = ScanNet(root=self.cfg_dm["root"],
                                     img_list=img_list["train"],
                                     mode="train")
        self.scannet_val = ScanNet(root=self.cfg_dm["root"],
                                   img_list=img_list["val"],
                                   mode="val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_train,
            batch_size=self.cfg_dm["batch_size"],
            drop_last=self.cfg_dm["drop_last"],
            shuffle=self.cfg_dm["shuffle"],
            pin_memory=self.cfg_dm["pin_memory"],
            num_workers=self.cfg_dm["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_val,
            batch_size=self.cfg_dm["batch_size"],
            drop_last=self.cfg_dm["drop_last"],
            shuffle=False,
            pin_memory=self.cfg_dm["pin_memory"],
            num_workers=self.cfg_dm["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test,
            batch_size=self.cfg_dm["batch_size"],
            drop_last=self.cfg_dm["drop_last"],
            shuffle=False,
            pin_memory=self.cfg_dm["pin_memory"],
            num_workers=self.cfg_dm["num_workers"],
        )
