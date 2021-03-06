# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import yaml
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.engine import train_one_epoch, valid_one_epoch
from models.detr import build
from util.dataset import selfDataset, collateFunction


def main(cfg):
    device = torch.device(cfg['device'])

    # fix the seed for reproducibility
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build(cfg)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg['lr_backbone'],
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['lr_drop'])

    # load data
    dataset = selfDataset(
        cfg['train_dir'], cfg['scaled_width'], cfg['scaled_height'], cfg['num_class'])
    dataLoader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collateFunction,
                            pin_memory=True, num_workers=cfg['num_workers'])
    dataset_val = selfDataset(
        cfg['val_dir'], cfg['scaled_width'], cfg['scaled_height'], cfg['num_class'])
    dataLoader_val = DataLoader(dataset_val, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collateFunction,
                                pin_memory=True, num_workers=cfg['num_workers'])
    # steps = int(dataset.__len__() / cfg['batch_size'])

    if cfg['frozen_weights'] is not None:
        checkpoint = torch.load(cfg['frozen_weights'], map_location='cpu')
        model.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(cfg['output_dir'])

    if cfg['resume']:
        checkpoint = torch.load(cfg['resume'], map_location='cuda')
        model.load_state_dict(checkpoint['model'])

        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            cfg['start_epoch'] = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(cfg['start_epoch'], cfg['epochs']):
        loss_train = train_one_epoch(
            model, criterion, dataLoader, optimizer, device,
            cfg['clip_max_norm'])
        lr_scheduler.step()

        with torch.no_grad():
            loss_val = valid_one_epoch(model, criterion, dataLoader_val, device)

        print("Epoch: {}, Avarge Train Loss: {:.2f}, Avarge Valid Loss: {:.2f}".format(epoch, loss_train, loss_val))

        with (output_dir / "log.txt").open('a') as f:
            f.write("Epoch: {}, Train_Loss: {}".format(
                epoch, loss_train) + "\n" + "Epoch: {}, Valid_Loss: {}".format(epoch, loss_val) + "\n")

        with (output_dir / "log.csv").open('ab') as f:
            np.savetxt(f, np.array(
                [[epoch, loss_train, loss_val]]), delimiter=",")

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': cfg['epochs'],
        'cfgs': cfg,
    }, '{}/checkpoint.pth'.format(output_dir))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    torch.save(model.state_dict(), '{}/wt.pt'.format(cfg['output_dir']))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    with open('config/cfg.yaml', 'r') as loadfile:
        config = yaml.load_all(loadfile, Loader=yaml.FullLoader)
        config_all = [x for x in config]

    # train mode
    config = config_all[0]

    if config['output_dir']:
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    main(config)
