import torchvision

from torch import nn


class DeepLabV3(nn.Module):

    def __init__(self, cfg_model):
        super().__init__()
        self._model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=cfg_model["pretrained"],
            pretrained_backbone=cfg_model["pretrained_backbone"],
            progress=True,
            num_classes=cfg_model["num_classes"],
            aux_loss=None,
        )

    def forward(self, data):
        return self._model(data)
