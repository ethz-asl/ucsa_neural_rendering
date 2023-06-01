import os
import random
import torch

from glob import glob
from torch.utils.data import Dataset

__all__ = ["ScanNetCL"]


class ScanNetCL(Dataset):

    def __init__(
        self,
        scannet_25k,
        scannet_ngp,
        ngp_25k_ratio=1,
    ):
        """
        Dataset dosent know if it contains replayed or normal samples !

        Some images are stored in 640x480 other ins 1296x968
        Warning scene0088_03 has wrong resolution -> Ignored
        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """

        super(ScanNetCL, self).__init__()
        self.scannet_25k = scannet_25k
        self.scannet_ngp = scannet_ngp
        self.ngp_25k_ratio = ngp_25k_ratio

    def get_image_pths(self, scene_list, val_ratio=0.2):
        img_list = []
        for scene_name in scene_list:
            all_imgs = glob(self.root + "/" + scene_name + "/color/*jpg")
            all_imgs = sorted(all_imgs,
                              key=lambda x: int(os.path.basename(x)[:-4]))
            val_imgs = all_imgs[-int(len(all_imgs) * val_ratio):]
            train_imgs = all_imgs[:-int(len(all_imgs) * val_ratio)]
            if self._mode == "train":
                img_list.extend(train_imgs[::self._sub])
            else:
                img_list.extend(val_imgs[::self._sub])

        return img_list

    def __getitem__(self, index):
        # Read Image and Label
        ret_ngp = self.scannet_ngp.__getitem__(index)
        ret_25k = []

        for _ in range(self.ngp_25k_ratio):
            rand_id = random.randint(0, self.scannet_25k.__len__() - 1)
            ret_25k.append(self.scannet_25k.__getitem__(rand_id))

        return (ret_ngp, ret_25k)

    @staticmethod
    def collate(batch):
        img = []
        label = []
        img_ori = []

        for bb in batch:
            img.append(bb[0][0])
            label.append(bb[0][1])
            img_ori.append(bb[0][2])
            for bbb in bb[1]:
                img.append(bbb[0])
                label.append(bbb[1])
                img_ori.append(bbb[2])

        img = torch.stack(img, dim=0)
        label = torch.stack(label, dim=0)
        img_ori = torch.stack(img_ori, dim=0)
        return img, label, img_ori

    def __len__(self):
        return self.scannet_ngp.__len__()
