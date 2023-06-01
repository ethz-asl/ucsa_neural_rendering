import imageio
import numpy as np
import os
import random
import torch

from torch.utils.data import Dataset

try:
    from .helper import AugmentationList
except Exception:
    from helper import AugmentationList

from .label_loader import LabelLoaderAuto

__all__ = ["ScanNet"]


class ScanNet(Dataset):

    def __init__(
        self,
        root,
        img_list,
        mode="train",
        output_trafo=None,
        output_size=(240, 320),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        sub=10,
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

        super(ScanNet, self).__init__()
        self._sub = sub
        self._mode = mode
        self._confidence_aux = confidence_aux

        self._label_setting = label_setting
        self.image_pths = img_list
        self.label_pths = [
            p.replace("color", "label").replace("jpg", "png") for p in img_list
        ]
        self.length = len(self.image_pths)
        self.aux_labels = False
        self._augmenter = AugmentationList(output_size, degrees, flip_p,
                                           jitter_bcsh)
        self._output_trafo = output_trafo
        self._data_augmentation = data_augmentation
        self.unique = False

        self.aux_labels_fake = False

        self._label_loader = LabelLoaderAuto(root_scannet=root,
                                             confidence=self._confidence_aux)
        if self.aux_labels:
            self._preprocessing_hack()

    def set_aux_labels_fake(self, flag=True):
        self.aux_labels_fake = flag
        self.aux_labels = flag

    def __getitem__(self, index):
        # Read Image and Label
        label, _ = self._label_loader.get(self.label_pths[index])
        label = torch.from_numpy(label).type(
            torch.float32)[None, :, :]  # C H W -> contains 0-40
        label = [label]
        if self.aux_labels and not self.aux_labels_fake:
            _p = self.aux_label_pths[index]
            if os.path.isfile(_p):
                aux_label, _ = self._label_loader.get(_p)
                aux_label = torch.from_numpy(aux_label).type(
                    torch.float32)[None, :, :]
                label.append(aux_label)
            else:
                if _p.find("_.png") != -1:
                    print(_p)
                    print("Processed not found")
                    _p = _p.replace("_.png", ".png")
                    aux_label, _ = self._label_loader.get(_p)
                    aux_label = torch.from_numpy(aux_label).type(
                        torch.float32)[None, :, :]
                    label.append(aux_label)

        img = imageio.imread(self.image_pths[index])
        img = (torch.from_numpy(img).type(torch.float32).permute(2, 0, 1) / 255
              )  # C H W range 0-1

        if self._mode.find("train") != -1 and self._data_augmentation:
            img, label = self._augmenter.apply(img, label)
        else:
            img, label = self._augmenter.apply(img, label, only_crop=True)

        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)

        for k in range(len(label)):
            label[k] = label[k] - 1  # 0 == chairs 39 other prop  -1 invalid

        # REJECT LABEL
        if (label[0] != -1).sum() < 10:
            idx = random.randint(0, len(self) - 1)
            if not self.unique:
                return self[idx]
            else:
                return False

        ret = (img, label[0].type(torch.int64)[0, :, :])
        if self.aux_labels:
            if self.aux_labels_fake:
                ret += (
                    label[0].type(torch.int64)[0, :, :],
                    torch.tensor(False),
                )
            else:
                ret += (
                    label[1].type(torch.int64)[0, :, :],
                    torch.tensor(True),
                )

        ret += (img_ori,)
        return ret

    def __len__(self):
        return self.length

    def __str__(self):
        string = "=" * 90
        string += "\nScannet Dataset: \n"
        length = len(self)
        string += f"    Total Samples: {length}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"    Replay: {self.replay}"
        string += f"  »  DataAug: {self._data_augmentation}"
        string += (
            f"  »  DataAug Replay: {self._data_augmentation_for_replay}\n")
        string += "=" * 90
        return string

    def _preprocessing_hack(self, force=False):
        """
        If training with aux_labels ->
        generates label for fast loading with a fixed certainty.
        """

        # check if this has already been performed
        aux_label, method = self._label_loader.get(
            self.aux_label_pths[self.global_to_local_idx[0]])
        print("Meethod ", method)
        print(
            "self.global_to_local_idx[0] ",
            self.global_to_local_idx[0],
            self.aux_label_pths[self.global_to_local_idx[0]],
        )
        if method == "RGBA":

            # This should always evaluate to true
            if (self.aux_label_pths[self.global_to_local_idx[0]].find("_.png")
                    == -1):
                print(
                    "self.aux_label_pths[self.global_to_local_idx[0]]",
                    self.aux_label_pths[self.global_to_local_idx[0]],
                    self.global_to_local_idx[0],
                )
                if (os.path.isfile(self.aux_label_pths[
                        self.global_to_local_idx[0]].replace(".png", "_.png"))
                        and os.path.isfile(self.aux_label_pths[
                            self.global_to_local_idx[-1]].replace(
                                ".png", "_.png")) and not force):
                    # only perform simple renaming
                    print("Only do renanming")
                    self.aux_label_pths = [
                        a.replace(".png", "_.png") for a in self.aux_label_pths
                    ]
                else:
                    print("Start multithread preprocessing of images")

                    def parallel(gtli, aux_label_pths, label_loader):
                        print("Start take care of: ", gtli[0], " - ", gtli[-1])
                        for i in gtli:
                            aux_label, method = label_loader.get(
                                aux_label_pths[i])
                            imageio.imwrite(
                                aux_label_pths[i].replace(".png", "_.png"),
                                np.uint8(aux_label),
                            )

                    def parallel2(aux_pths, label_loader):
                        for a in aux_pths:
                            aux_label, method = label_loader.get(a)
                            imageio.imwrite(
                                a.replace(".png", "_.png"),
                                np.uint8(aux_label),
                            )

                    cores = 16
                    tasks = [
                        t.tolist() for t in np.array_split(
                            np.array(self.global_to_local_idx), cores)
                    ]

                    from multiprocessing import Process

                    for i in range(cores):
                        p = Process(
                            target=parallel2,
                            args=(
                                np.array(self.aux_label_pths)[np.array(
                                    tasks[i])].tolist(),
                                self._label_loader,
                            ),
                        )
                        p.start()
                    p.join()
                    print("Done multithread preprocessing of images")
                    self.aux_label_pths = [
                        a.replace(".png", "_.png") for a in self.aux_label_pths
                    ]
