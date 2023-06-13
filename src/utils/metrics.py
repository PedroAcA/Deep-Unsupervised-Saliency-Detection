#!/usr/bin/env python

import torch
from utils.meter import MetricLogger


def mae(pred, gt):
    """
    Both pred and gt should be binary float tensors
    pred: torch.tensor (B, 1, H, W)
    gt: torch.tensor(B, 1, H, W)
    """
    mae = torch.mean(torch.abs(pred-gt), dim=(-2, -1))
    return mae


def precision(pred, gt):
    """
    Defined as portion of correctly classified fg pixels.
    tp / tp + fp
    Both pred and gt should be binary float tensors
    pred: torch.tensor (B, 1, H, W)
    gt: torch.tensor(B, 1, H, W)
    """
    intersection = pred * gt
    return torch.nan_to_num(torch.count_nonzero(intersection, dim=(-2, -1))/torch.count_nonzero(pred, dim=(-2, -1)), nan=0)


def recall(pred, gt):
    """
    Defined as the portion of
    tp / tp + fn
    pred: np.array size (B, 1, H, W)
    gt: np.array size (B, 1, H, W)
    """
    intersection = pred * gt
    return torch.nan_to_num(torch.count_nonzero(intersection, dim=(-2, -1))/torch.count_nonzero(gt, dim=(-2, -1)), nan=0)


def f1(precision, recall, beta):
    """
    precision: float
    recall: float
    beta: float
    """
    return (1+beta**2) * (precision*recall) / ((beta**2 * precision) + recall)


def binary_jaccard(pred, gt):
    i = torch.count_nonzero(pred * gt, dim=(-2, -1))
    u = torch.count_nonzero(pred + gt, dim=(-2, -1))
    return i / u


def dice_index(pred, gt):
    numerator = torch.count_nonzero(pred*gt, dim=(-2, -1))
    denominator = torch.count_nonzero(pred, dim=(-2, -1)) + torch.count_nonzero(gt, dim=(-2, -1))
    return (2*numerator)/denominator


def log_metrics(pred, gt, avg_meter, bin_th=0.5):
    bin_pred = (pred > bin_th).to(torch.float)
    bin_gt = (gt> bin_th).to(torch.float)
    avg_meter.update(precision=precision(bin_pred, bin_gt))
    avg_meter.update(recall=recall(bin_pred, bin_gt))
    avg_meter.update(mae=mae(bin_pred, bin_gt))
    avg_meter.update(dice=dice_index(bin_pred, bin_gt))
    avg_meter.update(jaccard=binary_jaccard(bin_pred, bin_gt))
    # avg_meter.update(f_beta=f1(prec, rec, beta=0.3))

