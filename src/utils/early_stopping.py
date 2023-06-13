#!/usr/bin/env python
import logging

import torch

from utils.meter import MetricLogger
from utils.metrics import log_metrics
import sys
sys.path.append("..")


class EarlyStopping:
    def __init__(self, model, noise_module, val_loader, cfg):
        self.device = cfg.SYSTEM.DEVICE
        self.noise_module = noise_module
        self.model = model
        self.solver_lambda = cfg.SOLVER.LAMBDA
        self.val_loader = val_loader
        self.best_tracking_metric = None
        self.cfg = cfg
        self.logger = logging.getLogger(str(cfg.SYSTEM.EXP_NAME) + '.utils.early_stopping')
        self.paitence = cfg.SOLVER.PAITENCE
        self.wait = 0
        self.tracking_metric = None
        self.converged = False
        self.threshold = cfg.SOLVER.THRESHOLD
        self.val_loss = MetricLogger()
        self.val_metrics = MetricLogger()
        self.num_maps = cfg.SOLVER.NUM_MAPS
        self.h, self.w = cfg.SOLVER.IMG_SIZE
        self.batch_size = 1

    def reset(self):
        self.converged = False
        self.wait = 0
        self.best_tracking_metric = None

    def check_cgt(self):
        print('self.best_tracking_metric: {} self.tracking_metric: {} \n'.format(self.best_tracking_metric, self.tracking_metric))
        if self.best_tracking_metric is None:
            self.best_tracking_metric = self.tracking_metric
        elif self.tracking_metric >= (self.best_tracking_metric-self.threshold):
            if self.wait >= self.paitence:
                self.converged = True
                self.wait = 0
            else:
                self.wait += 1
        else:
            self.wait = 0
            self.best_tracking_metric = self.tracking_metric

    def validate(self, writer, epoch_idx):
        self.model.eval()
        pred_loss = torch.nn.BCELoss()
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader):
                x = data['sal_image'].to(self.device)
                pred = self.model(x)
                if self.cfg.DATA.VAL.GT_ROOT:
                    y = data['sal_label'].to(self.device)
                    y_pred = pred
                else:
                    y = data['sal_noisy_label'].to(self.device)
                    y = torch.reshape(y, (self.batch_size, self.num_maps, self.h, self.w))
                    y_pred = self.noise_module.add_noise_to_prediction(pred, data['idx'])

                loss = pred_loss(y_pred, y)
                if not loss.shape:
                    # reshape loss into a 1 dimensional tensor to enable concatenation if its shape was 0 dimensional
                    loss = torch.tensor([loss], device=loss.device, dtype=loss.dtype)

                self.val_loss.update(**{'val_loss': loss})
                if idx % self.cfg.SYSTEM.LOG_FREQ == 0:
                    self.logger.debug(f'Validation Loss: {loss}, Avg: {self.val_loss.meters["val_loss"].avg}')

                if self.cfg.DATA.VAL.GT_ROOT:
                    log_metrics(y_pred, y, self.val_metrics)

        self.tracking_metric = self.val_loss.meters["val_loss"].avg

        if self.cfg.DATA.VAL.GT_ROOT:
            writer.add_scalars('val_metrics', {'val_precision': self.val_metrics.meters['precision'].avg,
                                               'val_recall': self.val_metrics.meters['recall'].avg,
                                                #'val_f1': metrics.meters['f_beta'].avg,
                                               'val_mae': self.val_metrics.meters['mae'].avg,
                                               'val_samples':self.val_metrics.meters['precision'].num_samples}, epoch_idx)

        writer.add_scalar('val_loss', loss, epoch_idx)
        self.check_cgt()
