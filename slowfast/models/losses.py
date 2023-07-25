#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial
import torch
import torch.nn as nn

from pytorchvideo.losses.soft_target_cross_entropy import (
    SoftTargetCrossEntropyLoss,
)


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(
            inputs, targets
        )
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


class FocalLoss(nn.Module):

    def __init_(self, gamma = 2, alpha = 0.25, label_smoothing = 0.0, apply_class_balancing = False):

        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.apply_class_balancing = apply_class_balancing
        self.loss_fct = nn.BCEWithLogitsLoss(reduction = "none")
        self.prob_fct = nn.Softmax()

    def forward(self, inputs, targets):

        num_classes = labels.shape[-1]
        labels = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        probs = self.prob_fct(inputs)
        p_t = labels * probs + (1 - labels) * (1 - probs)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        bce = self.loss_fct(inputs, labels)
        focal_bce = focal_factor * bce

        if (self.apply_class_balancing):

            weight = labels * self.alpha + (1 - labels) * (1 - self.alpha)
            focal_bce = weight * focal_bce

        loss = focal_bce.mean()

        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "focal_loss": FocalLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
