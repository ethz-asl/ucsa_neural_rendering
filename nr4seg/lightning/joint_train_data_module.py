import numpy as np
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional

from nr4seg.dataset import ScanNet, ScanNetCLJoint, ScanNetNGPJoint


class JointTrainDataModule(pl.LightningDataModule):

    def __init__(
        self,
        exp: dict,
        env: dict,
        split_ratio: float = 0.2,
    ):
        super().__init__()

        self.env = env
        self.exp = exp
        self.cfg_loader = self.exp["data_module"]

        self.split_ratio = split_ratio
        self.test_sz = 0
        self.val_sz = 0
        self.train_sz = 0

    def setup(self, stage: Optional[str] = None) -> None:
        finetune_seqs = self.exp["scenes"]
        self.scannet_val_ada = ScanNetNGPJoint(
            root=self.env["scannet"],
            mode="val",  # val
            scene_list=finetune_seqs,
            exp_name=self.exp["exp_name"],
            only_new_scene=False,
        )
        self.scannet_train_val_ada = ScanNetNGPJoint(
            root=self.env["scannet"],
            mode="train_val",  # train_val
            scene_list=finetune_seqs,
            exp_name=self.exp["exp_name"],
            only_new_scene=False,
        )
        self.scannet_predict_ada = ScanNetNGPJoint(
            root=self.env["scannet"],
            mode="predict",  # val
            scene_list=finetune_seqs,
            exp_name=self.exp["exp_name"],
            use_novel_viewpoints=self.exp["cl"]["use_novel_viewpoints"],
            only_new_scene=True,
        )
        ## test generation
        scannet_25k_dir = self.env["scannet_frames_25k"]
        split_file = os.path.join(
            scannet_25k_dir,
            self.cfg_loader["data_preprocessing"]["split_file"],
        )
        img_list = np.load(split_file)
        self.scannet_test_gen = ScanNet(
            root=self.env["scannet_frames_25k"],
            img_list=img_list["test"],
            mode="test",
        )

        self.scannet_train_nerf = ScanNetNGPJoint(
            root=self.env["scannet"],
            mode="train",
            scene_list=finetune_seqs,
            exp_name=self.exp["exp_name"],
            only_new_scene=True,
        )
        print("\033[93mNOTE: By default, the replay buffer is set to have size "
              "100.\033[0m")
        self.scannet_train_joint = ScanNetNGPJoint(
            root=self.env["scannet"],
            mode="train",
            scene_list=finetune_seqs,
            exp_name=self.exp["exp_name"],
            only_new_scene=False,
            # NOTE: This is referred to whether the previous scenes (if any)
            # used for replay were generated from novel viewpoints.
            use_novel_viewpoints=self.exp["cl"]["use_novel_viewpoints"],
            fix_nerf=False,
            replay_buffer_size=self.exp["cl"]["replay_buffer_size"],
        )
        ## train (all Torch-NGP generated labels)
        # random select from
        if self.exp["cl"]["active"]:
            split_file_cl = os.path.join(
                scannet_25k_dir,
                self.cfg_loader["data_preprocessing"]["split_file_cl"],
            )
            img_list_cl = np.load(split_file_cl)["train_cl"]
            img_list_cl = img_list_cl[:int(self.exp["cl"]["25k_fraction"] *
                                           len(img_list_cl))]
            scannet_25k = ScanNet(
                root=self.env["scannet_frames_25k"],
                img_list=img_list_cl,
                mode="train",
            )
            self.scannet_train_joint = ScanNetCLJoint(
                scannet_25k,
                self.scannet_train_joint,
                ngp_25k_ratio=self.exp["cl"]["ngp_25k_ratio"],
            )

        self.test_sz = len(self.scannet_test_gen)
        self.val_sz = len(self.scannet_val_ada)
        self.train_sz = len(self.scannet_train_joint)
        print("Train/Val/Test/Total: {}/{}/{}/{}".format(
            self.train_sz,
            self.val_sz,
            self.test_sz,
            self.train_sz + self.val_sz + self.test_sz,
        ))

    def train_dataloader_nerf(self) -> DataLoader:
        return DataLoader(
            self.scannet_train_nerf,
            batch_size=1,
            drop_last=False,
            shuffle=True,  # only true in train_dataloader
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def train_dataloader_joint(self) -> DataLoader:
        return DataLoader(
            self.scannet_train_joint,
            batch_size=self.cfg_loader["batch_size"],
            drop_last=True,
            shuffle=True,  # only true in train_dataloader
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
            collate_fn=ScanNetNGPJoint.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.scannet_val_ada,
                # Set bs=1 to ensure a batch always has frames from the same
                # scene.
                batch_size=1,
                drop_last=False,
                shuffle=False,  # false
                pin_memory=self.cfg_loader["pin_memory"],
                num_workers=self.cfg_loader["num_workers"],
            ),
            DataLoader(
                self.scannet_train_val_ada,
                # Set bs=1 to ensure a batch always has frames from the same
                # scene.
                batch_size=1,
                drop_last=False,
                shuffle=False,  # false
                pin_memory=self.cfg_loader["pin_memory"],
                num_workers=self.cfg_loader["num_workers"],
            ),
        ]

    def test_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.scannet_train_nerf,
                batch_size=1,
                drop_last=False,
                shuffle=False,  # false
                pin_memory=self.cfg_loader["pin_memory"],
                num_workers=self.cfg_loader["num_workers"],
            ),
            DataLoader(
                self.scannet_test_gen,
                batch_size=4,
                drop_last=False,
                shuffle=False,  # false
                pin_memory=self.cfg_loader["pin_memory"],
                num_workers=self.cfg_loader["num_workers"],
            ),
        ]

    def test_dataloader_nerf(self) -> DataLoader:
        return DataLoader(
            self.scannet_train_nerf,
            batch_size=1,
            drop_last=False,
            shuffle=False,  # false
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_predict_ada,
            batch_size=1,
            drop_last=False,
            shuffle=False,  # false
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )
