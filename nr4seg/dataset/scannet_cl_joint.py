import random
import torch
from torch.utils.data import Dataset

__all__ = ["ScanNetCLJoint"]


class ScanNetCLJoint(Dataset):

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

        super(ScanNetCLJoint, self).__init__()
        self.scannet_25k = scannet_25k
        self.scannet_ngp = scannet_ngp
        self.ngp_25k_ratio = ngp_25k_ratio

    def __getitem__(self, index):
        # Read Image and Label
        ret_dict = self.scannet_ngp.__getitem__(index)
        ret_25k = {"replay_img": [], "replay_label": []}

        for _ in range(self.ngp_25k_ratio):
            rand_id = random.randint(0, self.scannet_25k.__len__() - 1)
            img, label, _ = self.scannet_25k.__getitem__(rand_id)
            ret_25k["replay_img"].append(img)
            ret_25k["replay_label"].append(label)

        for key in ret_25k.keys():
            ret_25k[key] = torch.stack(ret_25k[key], dim=0)

        ret_dict.update(ret_25k)
        return ret_dict

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
        return batch_new, batch_old

    def __len__(self):
        return self.scannet_ngp.__len__()

    def __len__(self):
        return self.scannet_ngp.__len__()
