import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nr4seg.network import DeepLabV3
from nr4seg.utils.metrics import SemanticsMeter
from nr4seg.visualizer import Visualizer


class SemanticsLightningNet(pl.LightningModule):

    def __init__(self, exp, env):
        super().__init__()
        self._model = DeepLabV3(exp["model"])
        self.prev_scene_name = None

        self._visualizer = Visualizer(
            os.path.join(exp["general"]["name"], "visu"),
            exp["visualizer"]["store"],
            self,
        )

        self._meter = {
            "val_1": SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "val_2": SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "val_3": SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "test": SemanticsMeter(number_classes=exp["model"]["num_classes"]),
            "train": SemanticsMeter(number_classes=exp["model"]["num_classes"]),
        }

        self._visu_count = {"val": 0, "test": 0, "train": 0}

        self._exp = exp
        self._env = env
        self._mode = "train"
        self.length_train_dataloader = 10000

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._model(image)

    def visu(self, image, target, pred):
        if not (self._visu_count[self._mode]
                < self._exp["visualizer"]["store_n"][self._mode]):
            return

        for b in range(image.shape[0]):
            if (self._visu_count[self._mode]
                    < self._exp["visualizer"]["store_n"][self._mode]):
                c = self._visu_count[self._mode]
                self._visualizer.plot_image(image[b],
                                            tag=f"{self._mode}_vis/image_{c}")
                self._visualizer.plot_segmentation(
                    pred[b], tag=f"{self._mode}_vis/pred_{c}")
                self._visualizer.plot_segmentation(
                    target[b], tag=f"{self._mode}_vis/target_{c}")

                self._visualizer.plot_detectron(
                    image[b], target[b], tag=f"{self._mode}_vis/detectron_{c}")

                self._visu_count[self._mode] += 1
            else:
                break

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._meter["train"].clear()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        image, target, ori_image = batch
        output = self(image)
        pred = F.softmax(output["out"], dim=1)
        pred_argmax = torch.argmax(pred, dim=1)
        all_pred_argmax = self.all_gather(pred_argmax)
        all_target = self.all_gather(target)
        self._meter[self._mode].update(all_pred_argmax, all_target)
        # Compute Loss
        loss = F.cross_entropy(pred, target, ignore_index=-1, reduction="none")
        # Visu
        # self.visu(ori_image, target+1, pred_argmax+1)
        # Loss loggging
        self.log(
            f"{self._mode}/loss",
            loss.mean().item(),
            on_step=self._mode == "train",
            on_epoch=self._mode != "train",
        )
        return loss.mean()

    def on_train_epoch_end(self):
        m_iou, total_acc, m_acc = self._meter["train"].measure()
        self.log(f"train/total_accuracy", total_acc, rank_zero_only=True)
        self.log(f"train/mean_accuracy", m_acc, rank_zero_only=True)
        self.log(f"train/mean_IoU", m_iou, rank_zero_only=True)

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0
        self._meter["val_1"].clear()
        self._meter["val_2"].clear()
        self._meter["val_3"].clear()

    def validation_step(self, batch, batch_idx: int) -> None:
        dataloader_idx = 0  # TODO: modify back
        image, target, ori_image, scene_name = batch
        scene_name = scene_name[0]
        output = self(image)
        pred = F.softmax(output["out"], dim=1)
        pred_argmax = torch.argmax(pred, dim=1)
        all_pred_argmax = self.all_gather(pred_argmax)
        all_target = self.all_gather(target)

        self.prev_scene_name = scene_name
        self._meter[f"val_{dataloader_idx+1}"].update(all_pred_argmax,
                                                      all_target)

        # Compute Loss
        loss = F.cross_entropy(pred, target, ignore_index=-1, reduction="none")
        # Visu
        # self.visu(ori_image, target+1, pred_argmax+1)
        # Loss loggging
        self.log(
            f"{self._mode}/loss",
            loss.mean().item(),
            on_step=self._mode == "train",
            on_epoch=self._mode != "train",
        )
        return loss.mean()

    def on_validation_epoch_end(self):
        m_iou_1, total_acc_1, m_acc_1 = self._meter["val_1"].measure()
        self.log(f"val/total_accuracy_gg", total_acc_1, rank_zero_only=True)
        self.log(f"val/mean_accuracy_gg", m_acc_1, rank_zero_only=True)
        self.log(f"val/mean_IoU_gg", m_iou_1, rank_zero_only=True)
        self.prev_scene_name = None

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0
        self._meter["test"].clear()

    def test_step(self, batch, batch_idx: int) -> None:
        return self.training_step(batch, batch_idx)

    def on_test_epoch_end(self):
        m_iou, total_acc, m_acc = self._meter["test"].measure()
        self.log(f"test/total_accuracy", total_acc, rank_zero_only=True)
        self.log(f"test/mean_accuracy", m_acc, rank_zero_only=True)
        self.log(f"test/mean_IoU", m_iou, rank_zero_only=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self._exp["optimizer"]["name"]
        lr = self._exp["optimizer"]["lr"]
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if optimizer == "SGD":
            sgd_cfg = self._exp["optimizer"]["sgd_cfg"]
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=lr,
                momentum=sgd_cfg["momentum"],
                weight_decay=sgd_cfg["weight_decay"],
            )
        if optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(self._model.parameters(), lr=lr)
        if optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(self._model.parameters(),
                                            momentum=0.9,
                                            lr=lr)
        if self._exp["lr_scheduler"]["active"]:
            scheduler = self._exp["lr_scheduler"]["name"]
            if scheduler == "POLY":
                init_lr = (self._exp["optimizer"]["lr"],)
                max_epochs = self._exp["lr_scheduler"]["poly_cfg"]["max_epochs"]
                target_lr = self._exp["lr_scheduler"]["poly_cfg"]["target_lr"]
                power = self._exp["lr_scheduler"]["poly_cfg"]["power"]
                lambda_lr = (lambda epoch: (
                    ((max_epochs - min(max_epochs, epoch)) / max_epochs)**
                    (power)) + (1 - ((
                        (max_epochs - min(max_epochs, epoch)) / max_epochs)**
                                     (power))) * target_lr / init_lr)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                              lambda_lr,
                                                              last_epoch=-1,
                                                              verbose=True)
                interval = "epoch"
            lr_scheduler = {"scheduler": scheduler, "interval": interval}
            ret = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            ret = [optimizer]
        return ret
