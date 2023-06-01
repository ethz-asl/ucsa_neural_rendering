import cv2
import imageio
import numpy as np
import os
import skimage
import wandb

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw

from nr4seg.visualizer import BINARY_COLORS, SCANNET_CLASSES, SCANNET_COLORS

__all__ = ["Visualizer"]


def get_img_from_fig(fig, dpi=180):
    """
    Converts matplot figure to np.array
    """

    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a np array
    buf = np.asarray(buf)
    buf = Image.fromarray(buf)
    buf = buf.convert("RGB")
    return buf


def image_functionality(func):
    """
    Decorator to allow for logging functionality.
    Should be added to all plotting functions of visualizer.
    The plot function has to return a np.uint8

    @image_functionality
    def plot_segmentation(self, seg, **kwargs):

        return np.zeros((H, W, 3), dtype=np.uint8)

    not_log [optional, bool, default: false] : the decorator is ignored if set to true
    epoch [optional, int, default: visualizer.epoch ] : overwrites visualizer, epoch used to log the image
    store [optional, bool, default: visualizer.store ] : overwrites visualizer, flag if image is stored to disk
    tag [optinal, str, default: tag_not_defined ] : stores image as: visulaiter._p_visu{epoch}_{tag}.png
    """

    def wrap(*args, **kwargs):
        img = func(*args, **kwargs)

        if not kwargs.get("not_log", False):
            log_exp = args[0]._pl_model.logger is not None
            tag = kwargs.get("tag", "tag_not_defined")

            if kwargs.get("store", None) is not None:
                store = kwargs["store"]
            else:
                store = args[0]._store

            if kwargs.get("epoch", None) is not None:
                epoch = kwargs["epoch"]
            else:
                epoch = args[0]._epoch

            # Store to disk
            if store:
                p = os.path.join(args[0]._p_visu, f"{tag}_epoch_{epoch}.png")
                imageio.imwrite(p, img)

            if log_exp:
                H, W, C = img.shape
                ds = cv2.resize(
                    img,
                    dsize=(int(W / 2), int(H / 2)),
                    interpolation=cv2.INTER_CUBIC,
                )
                if args[0]._pl_model.logger is not None:
                    args[0]._pl_model.logger.experiment.log(
                        {tag: [wandb.Image(ds, caption=tag)]}, commit=False)
        return func(*args, **kwargs)

    return wrap


class Visualizer:

    def __init__(self, p_visu, store, pl_model, epoch=0, num_classes=22):
        self._p_visu = p_visu
        self._pl_model = pl_model
        self._epoch = epoch
        self._store = store

        os.makedirs(os.path.join(self._p_visu, "train_vis"), exist_ok=True)
        os.makedirs(os.path.join(self._p_visu, "val_vis"), exist_ok=True)
        os.makedirs(os.path.join(self._p_visu, "test_vis"), exist_ok=True)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, store):
        self._store = store

    @image_functionality
    def plot_segmentation(self, seg, **kwargs):
        try:
            seg = seg.clone().cpu().numpy()
        except:
            pass

        if seg.dtype == np.bool:
            col_map = BINARY_COLORS
        else:
            col_map = SCANNET_COLORS

        H, W = seg.shape[:2]
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for i, color in enumerate(col_map):
            img[seg == i] = color[:3]

        return img

    @image_functionality
    def plot_image(self, img, **kwargs):
        """
        ----------
        img : CHW HWC accepts torch.tensor or numpy.array
              Range 0-1 or 0-255
        """
        try:
            img = img.clone().cpu().numpy()
        except:
            pass

        if img.shape[2] == 3:
            pass
        elif img.shape[0] == 3:
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        else:
            raise Exception("Invalid Shape")
        if img.max() <= 1:
            img = img * 255

        img = np.uint8(img)
        return img

    @image_functionality
    def plot_detectron(
        self,
        img,
        label,
        text_off=False,
        alpha=0.5,
        draw_bound=True,
        shift=2.5,
        font_size=12,
        **kwargs,
    ):
        """
        ----------
        img : CHW HWC accepts torch.tensor or numpy.array
              Range 0-1 or 0-255
        label: HW accepts torch.tensor or numpy.array
        """

        img = self.plot_image(img, not_log=True)
        try:
            label = label.clone().cpu().numpy()
        except:
            pass
        label = label.astype(np.long)

        H, W, C = img.shape
        uni = np.unique(label)
        overlay = np.zeros_like(img)

        centers = []
        for u in uni:
            m = label == u
            col = SCANNET_COLORS[u]
            overlay[m] = col
            labels_mask = skimage.measure.label(m)
            regions = skimage.measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            cen = np.mean(regions[0].coords, axis=0).astype(np.uint32)[::-1]

            centers.append((SCANNET_CLASSES[u], cen))

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)),
                                        Image.fromarray(np.uint8(fore)))
        draw = ImageDraw.Draw(img_new)

        if not text_off:

            for i in centers:
                pose = i[1]
                pose[0] -= len(str(i[0])) * shift
                pose[1] -= font_size / 2
                draw.text(tuple(pose), str(i[0]), fill=(255, 255, 255, 128))

        img_new = img_new.convert("RGB")
        mask = skimage.segmentation.mark_boundaries(np.array(img_new),
                                                    label,
                                                    color=(255, 255, 255))
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)
