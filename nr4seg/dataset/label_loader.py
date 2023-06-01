import imageio
import numpy as np
import os
import pandas
import torch

__all__ = ["LabelLoaderAuto"]


class LabelLoaderAuto:

    def __init__(self, root_scannet=None, confidence=0, H=968, W=1296):
        assert root_scannet is not None
        self._get_mapping(root_scannet)
        self._confidence = confidence
        # return label between 0-40

        self.max_classes = 40
        self.label = np.zeros((H, W, self.max_classes))
        iu16 = np.iinfo(np.uint16)
        mask = np.full((H, W), iu16.max, dtype=np.uint16)
        self.mask_low = np.right_shift(mask, 6, dtype=np.uint16)

    def get(self, path):
        img = imageio.imread(path)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                H, W, _ = img.shape
                self.label = np.zeros((H, W, self.max_classes))
                for i in range(3):
                    prob = np.bitwise_and(img[:, :, i], self.mask_low) / 1023
                    cls = np.right_shift(img[:, :, i], 10, dtype=np.uint16)
                    m = np.eye(self.max_classes)[cls] == 1
                    self.label[m] = prob.reshape(-1)
                m = np.max(self.label, axis=2) < self._confidence
                self.label = np.argmax(self.label, axis=2).astype(np.int32) + 1
                self.label[m] = 0
                method = "RGBA"
            else:
                raise Exception("Type not know")
        elif len(img.shape) == 2 and img.dtype == np.uint8:
            self.label = img.astype(np.int32)
            method = "FAST"
        elif len(img.shape) == 2 and img.dtype == np.uint16:
            self.label = torch.from_numpy(img.astype(np.int32)).type(
                torch.float32)[None, :, :]
            sa = self.label.shape
            self.label = self.label.flatten()
            self.label = self.mapping[self.label.type(torch.int64)]
            self.label = self.label.reshape(sa).numpy().astype(np.int32)[0]
            method = "MAPPED"
        else:
            raise Exception("Type not know")
        return self.label, method

    def get_probs(self, path):
        img = imageio.imread(path)
        assert len(img.shape) == 3
        assert img.shape[2] == 4
        H, W, _ = img.shape
        probs = np.zeros((H, W, self.max_classes))
        for i in range(3):
            prob = np.bitwise_and(img[:, :, i], self.mask_low) / 1023
            cls = np.right_shift(img[:, :, i], 10, dtype=np.uint16)
            m = np.eye(self.max_classes)[cls] == 1
            probs[m] = prob.reshape(-1)

        return probs

    def _get_mapping(self, root):
        tsv = os.path.join(root, "scannetv2-labels.combined.tsv")
        df = pandas.read_csv(tsv, sep="\t")
        mapping_source = np.array(df["id"])
        mapping_target = np.array(df["nyu40id"])

        self.mapping = torch.zeros((int(mapping_source.max() + 1)),
                                   dtype=torch.int64)
        for so, ta in zip(mapping_source, mapping_target):
            self.mapping[so] = ta
