# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
from tqdm import tqdm

import torch


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, max_norm: float = 0):
    model.train()
    criterion.train()
    step = 0
    loss_value = 0

    for samples, targets in tqdm(data_loader, leave=False):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        step += 1
        loss_value += losses.item()

        # print("Epoch: {}, Step: {}/{}, Avarge Loss: {}".format(epoch, step, steps, loss_value/step))

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    # print("giou_loss:", loss_dict['loss_giou'].item())
    # print("cls_loss: ", loss_dict['loss_ce'].item())
    # print("bbox_loss:", loss_dict['loss_bbox'].item())

    return loss_value/step


def valid_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, device: torch.device):
    model.eval()
    criterion.eval()
    step = 0
    loss_value = 0

    for samples, targets in tqdm(data_loader, leave=False):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        step += 1
        loss_value += losses.item()

    return loss_value/step
