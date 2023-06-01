import cv2
import numpy as np
import os
import PIL
import pytorch_lightning as pl
import random
import shutil
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf

from torchvision import transforms as tf

from nr4seg.dataset.ngp_utils import custom_meshgrid, nyu40_colour_code
from nr4seg.nerf.network_tcnn_semantics import SemanticNeRFNetwork
from nr4seg.network import DeepLabV3
from nr4seg.utils.metrics import SemanticsMeter
from nr4seg.visualizer import Visualizer


class JointTrainLightningNet(pl.LightningModule):

    def __init__(self, exp, env):
        super().__init__()
        self.num_classes = exp["model"]["num_classes"]

        ## setup models
        self.seg_model = DeepLabV3(exp["model"])
        self.nerf_model = SemanticNeRFNetwork(
            encoding="hashgrid",
            bound=4,
            cuda_ray=False,
            density_scale=1,
            num_semantic_classes=self.num_classes,
        )
        ## setup losses
        self.criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=-1,
                                                       reduction="none")
        self.criterion_nerf_rgb = torch.nn.MSELoss(reduction="none")
        self.criterion_nerf_semantics = torch.nn.NLLLoss(ignore_index=-1,
                                                         reduction="none")
        self.criterion_nerf_depth = torch.nn.L1Loss(reduction="none")

        self.weight_depth = 0.1
        self.weight_semantics = 0.04
        self.nerf_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.automatic_optimization = False
        self.joint_train = False
        self.fix_nerf = exp["fix_nerf"]
        self.root_new_scene = os.path.join(env["scannet"], exp["scenes"][-1],
                                           exp["exp_name"])

        # setup misc
        self.prev_scene_name = None

        self._visualizer = Visualizer(
            os.path.join(exp["general"]["name"], "visu"),
            exp["visualizer"]["store"],
            self,
        )

        self._meter = {
            "train_nerf":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "train_seg":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "train_nerf_seg":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "train_seg_nerf":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "val_seg":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "train_val_seg":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "test_nerf":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "test_25k":
                SemanticsMeter(number_classes=exp["model"]["num_classes"]),
        }

        self._visu_count = {"val": 0, "test": 0, "train": 0}

        self._exp = exp
        self._env = env
        self._mode = "train"
        self.length_train_dataloader = 10000

        ### online data aug setup
        self._output_size = (240, 320)
        self._crop = tf.RandomCrop((240, 320))
        self._rot = tf.RandomRotation(degrees=10, resample=PIL.Image.BILINEAR)
        self._flip_p = 0.5
        self._degrees = 10

        self._jitter = tf.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        )
        self._crop_center = tf.CenterCrop((240, 320))

        # Stored in case ground-truth images are not available to determine
        # height and width (as in the case of renderings from novel viewpoints).
        self._default_H = None
        self._default_W = None

    @torch.cuda.amp.autocast(enabled=False)
    def get_rays_train(self, batch, bs, N=4096):
        """get rays
        Args:
            poses: [B, 4, 4], cam2world
            intrinsics: [4]
            H, W, N: int
            error_map: [B, 128 * 128], sample probability based on training
                error
        Returns:
            rays_o, rays_d: [B, N, 3]
            direction_norms: [B, N, 1]
            inds: [B, N]
        """

        poses = batch["pose"][[bs], ...]

        device = poses.device
        B = poses.shape[0]
        fx, fy, cx, cy = batch["intrinsics"][bs]
        H = batch["H"][bs]
        W = batch["W"][bs]

        i, j = custom_meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
        )
        i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
        j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

        N = min(N, H * W)

        # weighted sample on a low-reso grid
        inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
        inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        zs = torch.ones_like(i)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        directions = torch.stack((xs, ys, zs), dim=-1)
        direction_norms = torch.norm(directions, dim=-1, keepdim=True)
        directions = directions / direction_norms
        rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

        rays_o = poses[..., :3, 3]  # [B, 3]
        rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]
        return rays_o, rays_d, direction_norms, inds

    def forward_seg(self, batch, image=None) -> torch.Tensor:
        if image is None:
            image = batch["img"]
        output = self.seg_model(image)
        pred = F.softmax(output["out"], dim=1)
        pred_argmax = torch.argmax(pred, dim=1)
        return {"seg_semantics": pred_argmax, "seg_semantics_raw": pred}

    @torch.cuda.amp.autocast(enabled=True)
    def forward_nerf_train(self, batch, output_seg, bs) -> torch.Tensor:
        rays_o, rays_d, direction_norms, inds = self.get_rays_train(batch, bs)
        # use fp16 image
        images = batch["img_fp16"][[bs], ...]  # [B, N, 3/4]
        label_nerf = output_seg["seg_semantics"][[bs], ...]
        depths = batch["depth"][[bs], ...]
        one_m_to_scene_uom = batch["one_m_to_scene_uom"][bs]
        B, C, H, W = images.shape
        # - Store for later.
        self._default_H = H
        self._default_W = W

        gt_rgb = torch.gather(
            images.reshape(B, C, -1).permute(0, 2, 1),
            1,
            torch.stack(C * [inds], -1),
        )  # [B, N, 3/4]
        labels = torch.gather(label_nerf.view(B, -1), 1, inds)
        gt_depth = torch.gather(depths.view(B, -1), 1, inds)

        outputs = self.nerf_model.render(
            rays_o,
            rays_d,
            direction_norms=direction_norms,
            staged=False,
            bg_color=None,
            perturb=True,
            epoch=self.current_epoch,
        )
        pred_rgb = outputs["image"]
        semantics = outputs["semantics"]
        pred_depth = outputs["depth"]

        ## normalize

        invalid_semantics = torch.sum(semantics, dim=-1) == 0
        semantics[invalid_semantics] = 1
        semantics = semantics / torch.sum(semantics, dim=-1, keepdim=True)
        labels[invalid_semantics] = -1

        loss_color = self.criterion_nerf_rgb(
            pred_rgb, gt_rgb).mean()  # [B, N, 3] --> [B, N]
        # no valid case: no gradient flow
        if torch.sum(invalid_semantics) == semantics.shape[1]:
            loss_semantics = None
        else:
            semantics = torch.log(semantics + 1e-15).permute(0, 2, 1)
            loss_semantics = self.criterion_nerf_semantics(
                semantics, labels).mean()  # [B, N, C]

        loss_depth = self.criterion_nerf_depth(
            pred_depth[gt_depth != 0] / one_m_to_scene_uom,
            gt_depth[gt_depth != 0],
        ).mean(-1)

        return loss_color, loss_semantics, loss_depth

    @torch.cuda.amp.autocast(enabled=True)
    def forward_nerf_test(self, batch) -> torch.Tensor:
        rays_o = batch["rays_o"]  # [B, N, 3]
        rays_d = batch["rays_d"]  # [B, N, 3]
        direction_norms = batch["direction_norms"]
        if batch["viewpoint_is_novel"][0]:
            B = len(batch["viewpoint_is_novel"])
            H = self._default_H
            W = self._default_W
        else:
            img = batch["img"]  # [B, H, W, 3/4]
            B, C, H, W = img.shape
        outputs = self.nerf_model.render(
            rays_o,
            rays_d,
            direction_norms=direction_norms,
            staged=True,
            bg_color=1,
            perturb=False,
        )

        pred_rgb = outputs["image"].reshape(B, H, W, 3)
        semantics = outputs["semantics"].reshape(B, H, W, self.num_classes)
        invalid_semantics = torch.sum(semantics, dim=-1) == 0
        semantics[invalid_semantics] = 1
        semantics = semantics / torch.sum(semantics, dim=-1, keepdim=True)
        pred_semantics = torch.argmax(semantics, dim=-1)

        return {
            "nerf_rgb": pred_rgb.permute(0, 3, 1, 2),
            "nerf_semantics": pred_semantics,
            "nerf_semantics_raw": semantics,
        }

    def data_aug(self, img, label):
        # Color Jitter
        label = label[None, :, :]
        # add one so unknown has a label of 0
        label = label + 1
        img = self._jitter(img)
        # Rotate
        angle = random.uniform(-self._degrees, self._degrees)

        img = tvf.rotate(
            img,
            angle,
            interpolation=tvf.InterpolationMode.BILINEAR,
            expand=False,
            center=None,
            fill=0,
        )

        label = tvf.rotate(
            label,
            angle,
            interpolation=tvf.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
        )

        # Crop
        i, j, h, w = self._crop.get_params(img, self._output_size)
        img = tvf.crop(img, i, j, h, w)
        label = tvf.crop(label, i, j, h, w)

        # Flip
        if torch.rand(1) < self._flip_p:
            img = tvf.hflip(img)
            label = tvf.hflip(label)

        # Performes center crop
        img = self._crop_center(img)
        label = self._crop_center(label)
        label = label - 1
        label = label[0, :, :]

        return img, label

    def visu(self,
             gt_image,
             target,
             pred_seg=None,
             nerf_image=None,
             pred_nerf=None):
        if not (self._visu_count[self._mode]
                < self._exp["visualizer"]["store_n"][self._mode]):
            return

        for b in range(gt_image.shape[0]):
            if (self._visu_count[self._mode]
                    < self._exp["visualizer"]["store_n"][self._mode]):
                c = self._visu_count[self._mode]
                self._visualizer.plot_image(
                    gt_image[b], tag=f"{self._mode}_vis/gt_image_{c}")
                if nerf_image is not None:
                    self._visualizer.plot_image(
                        nerf_image[b], tag=f"{self._mode}_vis/nerf_image_{c}")

                if pred_seg is not None:
                    self._visualizer.plot_segmentation(
                        pred_seg[b], tag=f"{self._mode}_vis/pred_seg_{c}")
                if pred_nerf is not None:
                    self._visualizer.plot_segmentation(
                        pred_nerf[b], tag=f"{self._mode}_vis/pred_nerf_{c}")
                self._visualizer.plot_segmentation(
                    target[b], tag=f"{self._mode}_vis/target_{c}")

                self._visualizer.plot_detectron(
                    gt_image[b],
                    target[b],
                    tag=f"{self._mode}_vis/detectron_{c}",
                )

                self._visu_count[self._mode] += 1
            else:
                break

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._meter["train_seg"].clear()
        self._meter["train_nerf"].clear()
        self._meter["train_nerf_seg"].clear()
        self._meter["train_seg_nerf"].clear()
        if (self.current_epoch +
                1) % 10 == 0 and self.joint_train:  # manually call predict loop
            print("Start visualization, current joint training epoch: "
                  f"{self.current_epoch}")
            self.on_predict_epoch_start_copy()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.joint_train:
            self.training_step_joint(batch)
        else:
            self.training_step_nerf(batch)

    def training_step_joint(self, batch):
        optimizer_seg, optimizer_nerf = self.optimizers(use_pl_optimizer=False)
        batch_old, batch_new, batch_cl = batch
        # train nerf model using only batch_new
        if batch_new is not None:
            with torch.no_grad():
                self.nerf_model.eval()
                output_nerf = self.forward_nerf_test(batch_new)
                self.nerf_model.train()

        if not self.fix_nerf and batch_new is not None:
            self.seg_model.eval()
            if (batch_new["img"].shape[0]
                    > 1):  # only enable BN training when the batch size > 1
                for m in self.seg_model.modules():
                    if m.__class__.__name__.startswith("BatchNorm"):
                        m.train()
            output_seg = self.forward_seg(batch_new)
            self.seg_model.train()
            for bs in range(batch_new["img"].shape[0]):
                (
                    loss_color,
                    loss_semantics,
                    loss_depth,
                ) = self.forward_nerf_train(batch_new, output_seg, bs)
                self.log(
                    f"{self._mode}/loss_nerf_rgb",
                    loss_color.mean().item(),
                    on_step=self._mode == "train",
                    on_epoch=self._mode != "train",
                )
                self.log(
                    f"{self._mode}/loss_depth",
                    loss_depth.mean().item(),
                    on_step=self._mode == "train",
                    on_epoch=self._mode != "train",
                )
                self.log(
                    f"{self._mode}/loss_nerf_semantics",
                    loss_semantics.mean().item(),
                    on_step=self._mode == "train",
                    on_epoch=self._mode != "train",
                )
                total_nerf_loss = loss_color
                if loss_semantics is not None:
                    total_nerf_loss += loss_semantics * self.weight_semantics
                if loss_depth is not None:
                    total_nerf_loss += loss_depth * self.weight_depth

                optimizer_nerf.zero_grad()
                total_nerf_loss = self.nerf_scaler.scale(total_nerf_loss)
                self.manual_backward(total_nerf_loss)
                self.nerf_scaler.step(optimizer_nerf)
                self.nerf_scaler.update()

        ### use nerf as label
        with torch.no_grad():
            if batch_new is not None:
                ## data augmentation
                nerf_rgb, nerf_semantics = [], []
                for bs in range(batch_new["img"].shape[0]):
                    nerf_rgb_, nerf_semantics_ = self.data_aug(
                        output_nerf["nerf_rgb"][bs, ...],
                        output_nerf["nerf_semantics"][bs, ...],
                    )
                    nerf_rgb.append(nerf_rgb_)
                    nerf_semantics.append(nerf_semantics_)
                rgb_seg = torch.stack(nerf_rgb, dim=0)
                label_seg = torch.stack(nerf_semantics, dim=0)
            else:
                rgb_seg, label_seg = None, None
            if batch_old is not None:
                old_nerf_rgb, old_nerf_semantics = (
                    batch_old["img"],
                    batch_old["nerf_label"],
                )
                if rgb_seg is None:
                    rgb_seg, label_seg = old_nerf_rgb, old_nerf_semantics
                else:
                    rgb_seg = torch.concat([rgb_seg, old_nerf_rgb], dim=0)
                    label_seg = torch.concat([label_seg, old_nerf_semantics],
                                             dim=0)

            ## if there are cl labels, also stack cl labels and images
            if batch_cl is not None:
                replay_img = batch_cl["replay_img"]
                _, _, C, H, W = replay_img.shape
                replay_img = replay_img.reshape(-1, C, H, W)
                replay_label = batch_cl["replay_label"]
                replay_label = replay_label.reshape(-1, H, W)
                rgb_seg = torch.concat([rgb_seg, replay_img], dim=0)
                label_seg = torch.concat([label_seg, replay_label], dim=0)

        output_seg_grad = self.forward_seg(batch, rgb_seg)
        pred = output_seg_grad["seg_semantics_raw"]
        loss = self.criterion_seg(pred, label_seg)
        optimizer_seg.zero_grad()
        self.manual_backward(loss.mean())
        optimizer_seg.step()
        self.log(
            f"{self._mode}/loss_seg",
            loss.mean().item(),
            on_step=self._mode == "train",
            on_epoch=self._mode != "train",
        )
        if (self.current_epoch + 1
           ) % 10 == 0 and batch_new is not None:  # manually call predict loop
            with torch.no_grad():
                self.predict_step_copy(batch_new)

    def training_step_nerf(self, batch):
        optimizer_seg, optimizer_nerf = self.optimizers(use_pl_optimizer=False)
        with torch.no_grad():
            self.seg_model.eval()
            output_seg = self.forward_seg(batch)
            self.seg_model.train()

        ### simple strategy: use seg as label and train nerf
        for bs in range(batch["img"].shape[0]):
            loss_color, loss_semantics, loss_depth = self.forward_nerf_train(
                batch, output_seg, bs)
            self.log(
                f"{self._mode}/loss_nerf_rgb",
                loss_color.mean().item(),
                on_step=self._mode == "train",
                on_epoch=self._mode != "train",
            )
            self.log(
                f"{self._mode}/loss_depth",
                loss_depth.mean().item(),
                on_step=self._mode == "train",
                on_epoch=self._mode != "train",
            )
            self.log(
                f"{self._mode}/loss_nerf_semantics",
                loss_semantics.mean().item(),
                on_step=self._mode == "train",
                on_epoch=self._mode != "train",
            )

            total_nerf_loss = loss_color
            if loss_semantics is not None:
                total_nerf_loss += loss_semantics * self.weight_semantics
            if loss_depth is not None:
                total_nerf_loss += loss_depth * self.weight_depth

            optimizer_nerf.zero_grad()
            total_nerf_loss = self.nerf_scaler.scale(total_nerf_loss)
            self.manual_backward(total_nerf_loss)
            self.nerf_scaler.step(optimizer_nerf)
            self.nerf_scaler.update()

    def on_train_epoch_end(self):
        for net_name in ["seg", "nerf", "nerf_seg", "seg_nerf"]:
            if self._meter[f"train_{net_name}"].conf_mat is not None:
                m_iou, total_acc, m_acc = self._meter[
                    f"train_{net_name}"].measure()
                self.log(
                    f"train/{net_name}_total_accuracy",
                    total_acc,
                    rank_zero_only=True,
                )
                self.log(
                    f"train/{net_name}_mean_accuracy",
                    m_acc,
                    rank_zero_only=True,
                )
                self.log(f"train/{net_name}_mean_IoU",
                         m_iou,
                         rank_zero_only=True)
                self._meter[f"train_{net_name}"].clear()

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0
        self._meter["val_seg"].clear()

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        output_seg = self.forward_seg(batch)
        scene_name = batch["current_scene_name"][0]
        mode = "val" if dataloader_idx == 0 else "train_val"
        if (self.prev_scene_name != None and
                self.prev_scene_name != scene_name and
                self._meter[f"{mode}_seg"].conf_mat is not None):
            for net_name in ["seg"]:
                m_iou, total_acc, m_acc = self._meter[
                    f"{mode}_{net_name}"].measure()
                self.log(
                    f"{mode}/{net_name}_total_accuracy_{self.prev_scene_name}",
                    total_acc,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{mode}/{net_name}_mean_accuracy_{self.prev_scene_name}",
                    m_acc,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{mode}/{net_name}_mean_IoU_{self.prev_scene_name}",
                    m_iou,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
                self._meter[f"{mode}_{net_name}"].clear()
        self.prev_scene_name = scene_name
        self._meter[f"{mode}_seg"].update(output_seg["seg_semantics"],
                                          batch["label"])
        # Compute Loss
        loss = F.cross_entropy(
            output_seg["seg_semantics_raw"],
            batch["label"],
            ignore_index=-1,
            reduction="none",
        )
        # Visu
        self.visu(
            batch["img"],
            batch["label"] + 1,
            pred_seg=output_seg["seg_semantics"] + 1,
        )
        # Loss loggging
        self.log(
            f"{self._mode}/loss",
            loss.mean().item(),
            on_step=self._mode == "train",
            on_epoch=self._mode != "train",
        )
        return loss.mean()

    def on_validation_epoch_end(self):
        for net_name in ["seg"]:
            m_iou, total_acc, m_acc = self._meter[f"val_{net_name}"].measure()
            self.log(
                f"val/{net_name}_total_accuracy_{self.prev_scene_name}",
                total_acc,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/{net_name}_mean_accuracy_{self.prev_scene_name}",
                m_acc,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/{net_name}_mean_IoU_{self.prev_scene_name}",
                m_iou,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            self._meter[f"val_{net_name}"].clear()
        for net_name in ["seg"]:
            m_iou, total_acc, m_acc = self._meter[
                f"train_val_{net_name}"].measure()
            self.log(
                f"train_val/{net_name}_total_accuracy_{self.prev_scene_name}",
                total_acc,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"train_val/{net_name}_mean_accuracy_{self.prev_scene_name}",
                m_acc,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"train_val/{net_name}_mean_IoU_{self.prev_scene_name}",
                m_iou,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            self._meter[f"train_val_{net_name}"].clear()
        self.prev_scene_name = None

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0
        self._meter["test_nerf"].clear()
        self._meter["test_25k"].clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        if dataloader_idx == 0:
            output_nerf = self.forward_nerf_test(batch)
            self._meter["test_nerf"].update(output_nerf["nerf_semantics"],
                                            batch["label"])
            # Visu
            self.visu(
                batch["img"],
                batch["label"] + 1,
                pred_seg=None,
                nerf_image=output_nerf["nerf_rgb"],
                pred_nerf=output_nerf["nerf_semantics"] + 1,
            )
        else:
            image, target, ori_image = batch
            output = self.seg_model(image)
            pred = F.softmax(output["out"], dim=1)
            pred_argmax = torch.argmax(pred, dim=1)
            all_pred_argmax = self.all_gather(pred_argmax)
            all_target = self.all_gather(target)
            self._meter["test_25k"].update(all_pred_argmax, all_target)

    def on_test_epoch_end(self):
        for net_name in ["nerf", "25k"]:
            if self._meter[f"test_{net_name}"].conf_mat is not None:
                m_iou, total_acc, m_acc = self._meter[
                    f"test_{net_name}"].measure()
                self.log(
                    f"test/{net_name}_total_accuracy",
                    total_acc,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"test/{net_name}_mean_accuracy",
                    m_acc,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"test/{net_name}_mean_IoU",
                    m_iou,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
                self._meter[f"test_{net_name}"].clear()

    def on_predict_epoch_start(self):
        self.root_folder = self.root_new_scene
        os.makedirs(self.root_folder, exist_ok=True)
        # Create the subfolders also for optional novel viewpoints.
        for output_folder in ["", "novel_viewpoints"]:
            for curr_subfolder_name in [
                    "nerf_image",
                    "nerf_label",
                    "nerf_label_vis",
                    "seg_label",
                    "seg_label_vis",
            ]:
                curr_subfolder_path = os.path.join(self.root_folder,
                                                   output_folder,
                                                   curr_subfolder_name)
                if os.path.exists(curr_subfolder_path):
                    shutil.rmtree(curr_subfolder_path)
                os.makedirs(curr_subfolder_path)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Current viewpoint.
        output_nerf = self.forward_nerf_test(batch)
        if batch["viewpoint_is_novel"][0]:
            output_seg = self.forward_seg(batch, image=output_nerf["nerf_rgb"])
        else:
            output_seg = self.forward_seg(batch)

        subfolder_name = ("novel_viewpoints"
                          if batch["viewpoint_is_novel"][0] else "")

        nerf_image_path = os.path.join(
            self.root_folder,
            subfolder_name,
            "nerf_image",
            batch["current_index"][0] + ".png",
        )
        nerf_label_path = os.path.join(
            self.root_folder,
            subfolder_name,
            "nerf_label",
            batch["current_index"][0] + ".png",
        )
        nerf_label_vis_path = os.path.join(
            self.root_folder,
            subfolder_name,
            "nerf_label_vis",
            batch["current_index"][0] + ".png",
        )
        seg_label_path = os.path.join(
            self.root_folder,
            subfolder_name,
            "seg_label",
            batch["current_index"][0] + ".png",
        )
        seg_label_vis_path = os.path.join(
            self.root_folder,
            subfolder_name,
            "seg_label_vis",
            batch["current_index"][0] + ".png",
        )
        cv2.imwrite(
            nerf_image_path,
            cv2.cvtColor(
                (output_nerf["nerf_rgb"][0].permute(
                    1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )
        nerf_label = output_nerf["nerf_semantics"] + 1
        nerf_label_vis = nyu40_colour_code[nerf_label[0].detach().cpu().numpy()]
        cv2.imwrite(
            nerf_label_path,
            nerf_label[0].detach().cpu().numpy().astype(np.uint8),
        )
        cv2.imwrite(
            nerf_label_vis_path,
            cv2.cvtColor(nerf_label_vis.astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        seg_label = output_seg["seg_semantics"] + 1
        seg_label_vis = nyu40_colour_code[seg_label[0].detach().cpu().numpy()]
        cv2.imwrite(
            seg_label_path,
            seg_label[0].detach().cpu().numpy().astype(np.uint8),
        )
        cv2.imwrite(
            seg_label_vis_path,
            cv2.cvtColor(seg_label_vis.astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

    def on_predict_epoch_start_copy(self):
        self.root_folder = (
            f"{self.root_new_scene}_epoch_{self.current_epoch+1}")
        if os.path.exists(self.root_folder):
            shutil.rmtree(self.root_folder)
        os.makedirs(self.root_folder)
        # Create the subfolders also for optional novel viewpoints.
        for output_folder in ["", "novel_viewpoints"]:
            os.makedirs(
                os.path.join(self.root_folder, output_folder, "nerf_image"))
            os.makedirs(
                os.path.join(self.root_folder, output_folder, "nerf_label"))
            os.makedirs(
                os.path.join(self.root_folder, output_folder, "nerf_label_vis"))
            os.makedirs(
                os.path.join(self.root_folder, output_folder, "seg_label"))
            os.makedirs(
                os.path.join(self.root_folder, output_folder, "seg_label_vis"))

    def predict_step_copy(self, batch):
        # Current viewpoint.
        self.nerf_model.eval()
        self.seg_model.eval()
        output_nerf = self.forward_nerf_test(batch)
        output_seg = self.forward_seg(batch)
        subfolder_name = ("novel_viewpoints"
                          if batch["viewpoint_is_novel"][0] else "")
        for i in range(len(batch["current_index"])):
            nerf_image_path = os.path.join(
                self.root_folder,
                subfolder_name,
                "nerf_image",
                batch["current_index"][i] + ".png",
            )
            nerf_label_path = os.path.join(
                self.root_folder,
                subfolder_name,
                "nerf_label",
                batch["current_index"][i] + ".png",
            )
            nerf_label_vis_path = os.path.join(
                self.root_folder,
                subfolder_name,
                "nerf_label_vis",
                batch["current_index"][i] + ".png",
            )
            seg_label_path = os.path.join(
                self.root_folder,
                subfolder_name,
                "seg_label",
                batch["current_index"][i] + ".png",
            )
            seg_label_vis_path = os.path.join(
                self.root_folder,
                subfolder_name,
                "seg_label_vis",
                batch["current_index"][i] + ".png",
            )
            cv2.imwrite(
                nerf_image_path,
                cv2.cvtColor(
                    (output_nerf["nerf_rgb"][i].permute(
                        1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            nerf_label = output_nerf["nerf_semantics"] + 1
            nerf_label_vis = nyu40_colour_code[
                nerf_label[i].detach().cpu().numpy()]
            cv2.imwrite(
                nerf_label_path,
                nerf_label[i].detach().cpu().numpy().astype(np.uint8),
            )
            cv2.imwrite(
                nerf_label_vis_path,
                cv2.cvtColor(nerf_label_vis.astype(np.uint8),
                             cv2.COLOR_RGB2BGR),
            )
            seg_label = output_seg["seg_semantics"] + 1
            seg_label_vis = nyu40_colour_code[
                seg_label[i].detach().cpu().numpy()]
            cv2.imwrite(
                seg_label_path,
                seg_label[i].detach().cpu().numpy().astype(np.uint8),
            )
            cv2.imwrite(
                seg_label_vis_path,
                cv2.cvtColor(seg_label_vis.astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
        self.nerf_model.train()
        self.seg_model.train()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_seg = self._exp["optimizer"]["name"]
        lr_seg = self._exp["optimizer"]["lr_seg"]
        if optimizer_seg == "Adam":
            optimizer_seg = torch.optim.Adam(self.seg_model.parameters(),
                                             lr=lr_seg)
        if optimizer_seg == "SGD":
            sgd_cfg = self._exp["optimizer"]["sgd_cfg"]
            optimizer_seg = torch.optim.SGD(
                self.seg_model.parameters(),
                lr=lr_seg,
                momentum=sgd_cfg["momentum"],
                weight_decay=sgd_cfg["weight_decay"],
            )
        if optimizer_seg == "Adadelta":
            optimizer_seg = torch.optim.Adadelta(self.seg_model.parameters(),
                                                 lr=lr_seg)
        if optimizer_seg == "RMSprop":
            optimizer_seg = torch.optim.RMSprop(self.seg_model.parameters(),
                                                momentum=0.9,
                                                lr=lr_seg)
        lr_nerf = self._exp["optimizer"]["lr_nerf"]

        optimizer_nerf = torch.optim.Adam(
            [
                {
                    "name": "encoding",
                    "params": list(self.nerf_model.encoder.parameters()),
                },
                {
                    "name":
                        "net",
                    "params":
                        list(self.nerf_model.sigma_net.parameters()) +
                        list(self.nerf_model.color_net.parameters()) +
                        list(self.nerf_model.semantics_net.parameters()),
                    "weight_decay":
                        1e-6,
                },
            ],
            lr=lr_nerf,
            betas=(0.9, 0.99),
            eps=1e-15,
        )

        return optimizer_seg, optimizer_nerf
