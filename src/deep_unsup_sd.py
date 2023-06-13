#!/usr/bin/env python

import argparse
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
import torch
from torch import optim as optim
from torch.utils.tensorboard import SummaryWriter
from checkpoint import Checkpointer
from config import cfg
from dataloader import get_loader
from model import BaseModule, NoiseModule
from utils.basic import set_seeds, custom_lr
from utils.save import save_config
from utils.setup_logger import setup_logger
from training_and_testing_routines import train, test

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='path of config file', default=None, type=str)
    parser.add_argument('--mode', help='Would you like to train, test or train and test the model?',
                        choices=["train", "test", "train_test"], type=str, required=True)
    parser.add_argument('--clean_run', help='run from scratch', default=False, type=bool)
    parser.add_argument('--chkpt_file', help='Optional. Checkpoint file to load when test mode is enabled', default='',
                        type=str, required=False)
    parser.add_argument('--test_batch_size', help='Optional. Batch size used for tests', default=1,
                        type=int, required=False)
    parser.add_argument('--test_type', help='Optional. Would you like to do a qualitative or quantitative test?',
                        choices=["qualitative", "quantitative", "all"],
                        default="quantitative", type=str, required=False)
    parser.add_argument('opts', help='modify arguments', default=None, nargs=argparse.REMAINDER)
    return parser

def create_optimizers_and_schedulers(cfg, prediction_model):
    lr = cfg.SOLVER.LR
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    betas = cfg.SOLVER.BETAS
    step_size = cfg.SOLVER.STEP_SIZE
    decay_factor = cfg.SOLVER.FACTOR
    # Optimizer
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    if cfg.SOLVER.SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_factor)
    elif cfg.SOLVER.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=cfg.SOLVER.FACTOR,
                                                         # min_lr=cfg.SOLVER.MIN_LR,
                                                         patience=cfg.SOLVER.PAITENCE,
                                                         cooldown=cfg.SOLVER.COOLDOWN,
                                                         threshold=cfg.SOLVER.THRESHOLD,
                                                         eps=1e-24)

    poly_decay = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr(maxiter=cfg.SOLVER.EPOCHS, power=10))
    return optimizer, scheduler, poly_decay


def main():
    # argparse
    parser = create_parser()
    args = parser.parse_args()
    if args.mode!= 'train' and args.test_type!='quantitative':
        assert args.test_batch_size==1, "In order to save images, test_batch_size should be 1, but received {} instead".format(args.test_batch_size)

    # config setup
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None: cfg.merge_from_list(args.opts)
    if args.clean_run and args.mode != "test":
        if os.path.exists(f'{cfg.SAVE_ROOT}{cfg.SYSTEM.EXP_NAME}'):
            shutil.rmtree(f'{cfg.SAVE_ROOT}{cfg.SYSTEM.EXP_NAME}', ignore_errors=True)
        if os.path.exists(f'{cfg.SAVE_ROOT}runs/{cfg.SYSTEM.EXP_NAME}'):
            shutil.rmtree(f'{cfg.SAVE_ROOT}runs/{cfg.SYSTEM.EXP_NAME}', ignore_errors=True)
            # Note!: Sleeping to make tensorboard delete it's cache.
            time.sleep(5)

    assert not (cfg.DATA.TRAIN.GT_ROOT and cfg.DATA.TRAIN.NOISE_ROOT), "Cannot simultaneosly handle ground truth and noisy labels at training phase. Please set only one of them"
    assert not(cfg.DATA.VAL.GT_ROOT and cfg.DATA.VAL.NOISE_ROOT), "Cannot simultaneosly handle ground truth and noisy labels at validation phase. Please set only one of them"
    cfg.SYSTEM.DEVICE = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    search = defaultdict()
    search['lr'], search['momentum'], search['factor'], search['step_size'] = [True]*4
    if cfg.SYSTEM.SEED: #If introduced by Pedro Aurelio Coelho de Almeida on 21st of September, 2021 to use deterministic behaviour only when needed
        set_seeds(cfg)

    logdir, chk_dir = save_config(cfg.SAVE_ROOT, cfg)
    writer = SummaryWriter(log_dir=logdir)
    # setup logger
    logger_dir = Path(chk_dir).parent
    logger = setup_logger(cfg.SYSTEM.EXP_NAME, save_dir=logger_dir)
    # Model
    prediction_model = BaseModule(cfg)
    # load the data
    train_loader = get_loader(cfg, 'train')
    use_validation = cfg.DATA.VAL.NOISE_ROOT or cfg.DATA.VAL.GT_ROOT
    val_loader = get_loader(cfg, 'val') if use_validation else None
    optimizer, scheduler, poly_decay = create_optimizers_and_schedulers(cfg, prediction_model)
    # checkpointer
    chkpt = Checkpointer(prediction_model, optimizer, scheduler=scheduler, save_dir=chk_dir, logger=logger,
                         save_to_disk=True)
    offset = 0
    if cfg.MODEL.TRANSFER_LEARNING and args.mode != "test" and args.chkpt_file:
        # Load only model weights, leaving optimizer, scheduler and offset unchanged
        Checkpointer(prediction_model, logger=logger).load(f=args.chkpt_file, use_latest=False)
        checkpointer = {}
    elif args.mode == "test" and args.chkpt_file:
        checkpointer = chkpt.load(f=args.chkpt_file, use_latest=False)
    else:
        checkpointer = chkpt.load()

    if not checkpointer == {}:
        offset = checkpointer.pop('epoch')
    print(f'Same optimizer, {scheduler.optimizer == optimizer}')
    print(cfg)
    if args.mode != "test":
        noise_model = NoiseModule(cfg)
        loader = [train_loader, val_loader]
        model = [prediction_model, noise_model]
        train(cfg, model, optimizer, scheduler, poly_decay, loader, chkpt, writer, offset)
    if args.mode != "train":
        print("Test type {}".format(args.test_type))
        test_loader = get_loader(cfg, 'test', test_batch_size=args.test_batch_size)
        samples = ["3_94_94612", "7_201_201231", "0_24_24196", "4_140_140326",
                   "2_68_68857", "5_154_154732", "6_181_181363", "9_17208",
                   "0_13_13027", "3_94_94173", "1_63_63372", "3_112_112202"]
        # samples = ['JPCNN075', 'JPCNN037', 'JPCLN027', 'JPCLN038']
        test(cfg, prediction_model, test_loader, args.test_type, logger, samples)


if __name__ == "__main__":
    main()
