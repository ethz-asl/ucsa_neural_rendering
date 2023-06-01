import numpy as np
import torch

from sklearn.metrics import confusion_matrix


def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()


class SemanticsMeter:

    def __init__(self, number_classes):
        self.conf_mat = None
        self.number_classes = number_classes

    def clear(self):
        self.conf_mat = None

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds = preds.flatten()
        truths = truths.flatten()
        valid_pix_ids = truths != -1
        preds = preds[valid_pix_ids]
        truths = truths[valid_pix_ids]
        conf_mat_current = confusion_matrix(truths,
                                            preds,
                                            labels=list(
                                                range(self.number_classes)))
        if self.conf_mat is None:
            self.conf_mat = conf_mat_current
        else:
            self.conf_mat += conf_mat_current

    def measure(self):
        conf_mat = self.conf_mat
        norm_conf_mat = np.transpose(
            np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

        missing_class_mask = np.isnan(norm_conf_mat.sum(
            1))  # missing class will have NaN at corresponding class
        exsiting_class_mask = ~missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)
        ious = np.zeros(self.number_classes)
        for class_id in range(self.number_classes):
            ious[class_id] = conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id])
        miou_valid_class = np.mean(ious[exsiting_class_mask])
        return miou_valid_class, total_accuracy, class_average_accuracy
