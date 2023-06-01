import random
import PIL
import torch
from torchvision import transforms as tf
from torchvision.transforms import functional as F
import warnings

__all__ = ["Augmentation", "AugmentationList", "get_output_size"]


def get_output_size(output_size):
    if type(output_size) == list or type(output_size) == tuple:
        if len(output_size) == 2:
            output_size = tuple(output_size)
        elif len(output_size) == 1:
            output_size = (output_size[0], output_size[0])
    elif type(output_size) == int:
        output_size = (output_size, output_size)
    return output_size


class Augmentation:

    def __init__(
        self,
        output_size=400,
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
    ):

        # training transforms
        output_size = get_output_size(output_size)
        self._output_size = output_size
        self._crop = tf.RandomCrop(self._output_size)
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            self._rot = tf.RandomRotation(degrees=degrees,
                                          resample=PIL.Image.BILINEAR)
        self._flip_p = flip_p
        self._degrees = degrees

        self._jitter = tf.ColorJitter(
            brightness=jitter_bcsh[0],
            contrast=jitter_bcsh[1],
            saturation=jitter_bcsh[2],
            hue=jitter_bcsh[3],
        )
        self._crop_center = tf.CenterCrop(self._output_size)

    def apply(self, img, label, only_crop=False):
        # LABEL is now a list of labels

        scale = False
        # Check if rescaling is neccessary based on image height and output height
        if img.shape[1] >= 2 * self._output_size[0]:
            sf = float(self._output_size[0] / img.shape[1]) * 1.2
            sf2 = float(self._output_size[1] / img.shape[2]) * 1.2
            sf = max(sf, sf2)

            scale = True
        elif (img.shape[1] < self._output_size[0] or
              img.shape[2] < self._output_size[1]):
            sf1 = float(self._output_size[0] / img.shape[1]) * 1.2
            sf2 = float(self._output_size[1] / img.shape[2]) * 1.2
            sf = max(sf1, sf2)
            scale = True

        if scale:
            img = torch.nn.functional.interpolate(
                img[None],
                scale_factor=(sf, sf),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )[0]
            label = torch.nn.functional.interpolate(
                label[None],
                scale_factor=(sf, sf),
                mode="nearest",
                recompute_scale_factor=False,
            )[0]
        if not only_crop:
            # Color Jitter
            img = self._jitter(img)

            # Rotate
            angle = random.uniform(-self._degrees, self._degrees)
            with warnings.catch_warnings():
                # this will suppress all warnings in this block
                warnings.simplefilter("ignore")
                img = F.rotate(
                    img,
                    angle,
                    resample=PIL.Image.BILINEAR,
                    expand=False,
                    center=None,
                    fill=0,
                )
                label = F.rotate(
                    label,
                    angle,
                    resample=PIL.Image.NEAREST,
                    expand=False,
                    center=None,
                    fill=0,
                )

            # Crop
            i, j, h, w = self._crop.get_params(img, self._output_size)
            img = F.crop(img, i, j, h, w)
            label = F.crop(label, i, j, h, w)

            # Flip
            if torch.rand(1) < self._flip_p:
                img = F.hflip(img)
                label = F.hflip(label)

        # Performes center crop
        img = self._crop_center(img)
        label = self._crop_center(label)

        return img, label


class AugmentationList:

    def __init__(
        self,
        output_size=400,
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
    ):

        # training transforms
        output_size = get_output_size(output_size)
        self._output_size = output_size
        self._crop = tf.RandomCrop(self._output_size)
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            self._rot = tf.RandomRotation(degrees=degrees,
                                          resample=PIL.Image.BILINEAR)
        self._flip_p = flip_p
        self._degrees = degrees

        self._jitter = tf.ColorJitter(
            brightness=jitter_bcsh[0],
            contrast=jitter_bcsh[1],
            saturation=jitter_bcsh[2],
            hue=jitter_bcsh[3],
        )
        self._crop_center = tf.CenterCrop(self._output_size)

    def apply(self, img, label, only_crop=False):
        scale = False
        # Check if rescaling is neccessary based on image height and output height
        if img.shape[1] >= 2 * self._output_size[0]:
            sf = float(self._output_size[0] / img.shape[1]) * 1.2
            sf2 = float(self._output_size[1] / img.shape[2]) * 1.2
            sf = max(sf, sf2)

            scale = True
        elif (img.shape[1] < self._output_size[0] or
              img.shape[2] < self._output_size[1]):
            sf1 = float(self._output_size[0] / img.shape[1]) * 1.2
            sf2 = float(self._output_size[1] / img.shape[2]) * 1.2
            sf = max(sf1, sf2)
            scale = True

        if scale:
            img = torch.nn.functional.interpolate(
                img[None],
                scale_factor=(sf, sf),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )[0]
            for _i, l in enumerate(label):
                label[_i] = torch.nn.functional.interpolate(
                    l[None],
                    scale_factor=(sf, sf),
                    mode="nearest",
                    recompute_scale_factor=False,
                )[0]
        if not only_crop:
            # Color Jitter
            img = self._jitter(img)

            # Rotate
            angle = random.uniform(-self._degrees, self._degrees)
            with warnings.catch_warnings():
                # this will suppress all warnings in this block
                warnings.simplefilter("ignore")
                img = F.rotate(
                    img,
                    angle,
                    resample=PIL.Image.BILINEAR,
                    expand=False,
                    center=None,
                    fill=0,
                )
                for _i, l in enumerate(label):
                    label[_i] = F.rotate(
                        l,
                        angle,
                        resample=PIL.Image.NEAREST,
                        expand=False,
                        center=None,
                        fill=0,
                    )

            # Crop
            i, j, h, w = self._crop.get_params(img, self._output_size)
            img = F.crop(img, i, j, h, w)
            for _i, l in enumerate(label):
                label[_i] = F.crop(l, i, j, h, w)

            # Flip
            if torch.rand(1) < self._flip_p:
                img = F.hflip(img)
                for _i, l in enumerate(label):
                    label[_i] = F.hflip(l)

        # Performes center crop
        img = self._crop_center(img)
        for _i, l in enumerate(label):
            label[_i] = self._crop_center(l)

        return img, label
