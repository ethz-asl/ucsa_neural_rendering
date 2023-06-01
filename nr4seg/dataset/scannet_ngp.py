import cv2
import os
import random
import torch

from glob import glob
from torch.utils.data import Dataset

try:
    from .helper import AugmentationList
except Exception:
    from helper import AugmentationList

__all__ = ["ScanNetNGP"]


class ScanNetNGP(Dataset):

    def __init__(
        self,
        root,
        scene_list,
        prev_exp_name="one_step_nerf_only",
        mode="train",
        train_image="nerf",
        train_label="nerf",
        val_mode="gtgt",
        output_trafo=None,
        output_size=(240, 320),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        sub=1,
        data_augmentation=True,
        label_setting="default",
        confidence_aux=0,
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

        super(ScanNetNGP, self).__init__()
        self._sub = sub
        self._mode = mode
        self._confidence_aux = confidence_aux

        self.H = output_size[0]
        self.W = output_size[1]

        self._label_setting = label_setting
        self.root = root
        self.image_pths, self.img_num = self.get_image_pths(scene_list)

        self.image_gt_pths = self.image_pths
        self.image_nerf_pths = [
            p.replace("color_scaled",
                      prev_exp_name + "/nerf_image").replace("jpg", "png")
            for p in self.image_pths
        ]
        self.label_nerf_pths = [
            p.replace("color_scaled",
                      prev_exp_name + "/nerf_label").replace("jpg", "png")
            for p in self.image_pths
        ]
        self.label_mapping_pths = [
            p.replace("color_scaled", "mapping_label").replace("jpg", "png")
            for p in self.image_pths
        ]
        self.label_gt_pths = [
            p.replace("color_scaled", "label_scaled").replace("jpg", "png")
            for p in self.image_pths
        ]

        self.length = len(self.image_pths)
        self._augmenter = AugmentationList(output_size, degrees, flip_p,
                                           jitter_bcsh)
        self._output_trafo = output_trafo
        self._data_augmentation = data_augmentation
        self.train_image = train_image
        self.train_label = train_label
        self.val_mode = val_mode

    def get_image_pths(self, scene_list, val_ratio=0.2):
        img_list = []
        img_num = []
        for scene_name in scene_list:
            all_imgs = glob(self.root + "/" + scene_name + "/color_scaled/*jpg")
            all_imgs = sorted(all_imgs,
                              key=lambda x: int(os.path.basename(x)[:-4]))
            val_imgs = all_imgs[-int(len(all_imgs) * val_ratio):]
            # val_imgs = val_imgs[: len(val_imgs)//4*4]
            train_imgs = all_imgs[:-int(len(all_imgs) * val_ratio)]
            if self._mode == "train":
                img_list.extend(train_imgs[::self._sub])
                img_num.append(len(train_imgs[::self._sub]))
            else:
                img_list.extend(val_imgs[::self._sub])

        return img_list, img_num

    def __getitem__(self, index):
        # Read Image and Label

        use_nerf_label = False
        if self._mode == "train":
            if self.train_image == "gt":
                img = cv2.imread(self.image_gt_pths[index],
                                 cv2.IMREAD_UNCHANGED)
            elif self.train_image == "nerf":
                img = cv2.imread(self.image_nerf_pths[index],
                                 cv2.IMREAD_UNCHANGED)
            elif self.train_image == "half":
                img = (cv2.imread(self.image_gt_pths[index],
                                  cv2.IMREAD_UNCHANGED)
                       if random.random() > 0.5 else cv2.imread(
                           self.image_nerf_pths[index], cv2.IMREAD_UNCHANGED))
            else:
                raise NotImplementedError
        else:
            if self.val_mode == "gtgt":
                img = cv2.imread(self.image_gt_pths[index],
                                 cv2.IMREAD_UNCHANGED)
            elif self.val_mode in ["nerfgt", "nerfnerf"]:
                img = cv2.imread(self.image_nerf_pths[index],
                                 cv2.IMREAD_UNCHANGED)
            else:
                raise NotImplementedError

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = (torch.from_numpy(img).type(torch.float32).permute(2, 0, 1)
              )  # C H W range 0-1

        if self._mode == "train":
            if self.train_label == "nerf":
                label = cv2.imread(self.label_nerf_pths[index],
                                   cv2.IMREAD_UNCHANGED)
            else:
                label = cv2.imread(self.label_mapping_pths[index],
                                   cv2.IMREAD_UNCHANGED)
            use_nerf_label = self.train_label == "nerf"
        else:
            if self.val_mode in ["gtgt", "nerfgt"]:
                label = cv2.imread(self.label_gt_pths[index],
                                   cv2.IMREAD_UNCHANGED)
            elif self.val_mode == "nerfnerf":
                label = cv2.imread(self.label_nerf_pths[index],
                                   cv2.IMREAD_UNCHANGED)
                use_nerf_label = True

        label = cv2.resize(label, (self.W, self.H),
                           interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label).type(
            torch.float32)[None, :, :]  # C H W -> contains 0-40

        # if use nerf labels + 1
        if use_nerf_label:
            label = label + 1

        label = [label]

        if self._mode.find("train") != -1 and self._data_augmentation:
            img, label = self._augmenter.apply(img, label)
        else:
            img, label = self._augmenter.apply(img, label, only_crop=True)

        label[0] = label[0] - 1

        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)

        ret = (img, label[0].type(torch.int64)[0, :, :])
        ret += (img_ori,)

        if self._mode != "train":
            current_scene_name = os.path.normpath(self.image_pths[index]).split(
                os.path.sep)[-3]
            ret += (current_scene_name,)

        return ret

    def __len__(self):
        return self.length

    def __str__(self):
        string = "=" * 90
        string += "\nScannet Dataset: \n"
        length = len(self)
        string += f"    Total Samples: {length}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"  »  DataAug: {self._data_augmentation}"
        string += "=" * 90
        return string
