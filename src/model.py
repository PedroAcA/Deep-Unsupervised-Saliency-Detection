#!/usr/bin/env python
import logging
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.distributions.normal import Normal
import torch, torchvision


class NoiseModule:
	def __init__(self, cfg):
		"""
		Noise Module implements the noise mdoel from the paper.
		It maintains a variance for each pixel of each image.
		sd = sqrt(var), we store the variances for each of the prior distributions
		Sampling is done using var * N(0, 1)
		It is responsible for sampling from the noise distribution, loss calculation and updating the variance.
		Args
		---
		cfg: (yacs.CfgNode) base configuration for the experiment.
		"""
		super(NoiseModule, self).__init__()
		self.cfg = cfg
		self.logger = logging.getLogger(str(cfg.SYSTEM.EXP_NAME) + ".noise_module")
		self.num_imgs = cfg.SOLVER.NUM_IMG
		self.num_maps = cfg.SOLVER.NUM_MAPS
		self.h, self.w = cfg.SOLVER.IMG_SIZE
		self.batch_size = cfg.SOLVER.BATCH_SIZE
		self.alpha = cfg.NOISE.ALPHA
		self.device = cfg.SYSTEM.DEVICE
		# a priori standard deviation of the noise distribution
		self.noise_std = torch.zeros((self.num_imgs, self.h, self.w), device=self.device)
		self.small_ct = torch.finfo(self.noise_std.dtype).eps #smaller representable number between 0 and 1
		self.round = 1

	def get_batch_prior_distribution(self, idxs):
		batch_std = torch.clamp(self.noise_std[idxs], min=self.small_ct)
		batch_mean = torch.zeros_like(batch_std)
		return Normal(batch_mean, batch_std)

	def sample_noise(self, idxs):
		noise = torch.zeros(len(idxs), self.num_maps, self.h, self.w)
		batch_prior_distribution = self.get_batch_prior_distribution(idxs)
		for i in range(self.num_maps):
			noise[:,i] = batch_prior_distribution.sample()
		return noise

	def add_noise_to_prediction(self, pred, item_idxs):
		if self.round == 1:
			# pred: In round 1 no sampling is made because variance is zero.
			# Here the prediction is repeated along the noisy maps dimension to have the same size as noisy mapss
			y_pred = torch.repeat_interleave(pred, repeats=self.cfg.SOLVER.NUM_MAPS, dim=1)
		# Round > 1
		else:
			# noise_prior: Sampled Noise from Noise Module, (None, NUM_MAPS, 128, 128)
			noise_prior = self.sample_noise(item_idxs).to(self.device)
			# noisy_pred: Noisy predictions after adding noise to predictions, (None, NUM_MAPS, 128, 128)
			y_pred = pred + noise_prior
			# truncate to lie in range[0, 1] see 3.2 after Eq 4
			y_pred = torch.clamp(y_pred, min=0.0, max=1.0)

		return y_pred

	def update(self, pred_model, train_loader):
		"""
		Updates the prior variance for each pixel of each image by emp variance
		Args
		---
		pred_model:  the backbone network.
		train_loader: torch.utils.data.dataloader for training data
		"""
		self.logger.info("Updating Noise at round {}".format(self.round))
		self.logger.info("----------------------------------------")
		with torch.no_grad():
			pred_model.eval()
			for batch_idx, data in enumerate(train_loader):
				# x : input data, (None, 3, 128, 128)
				x = data['sal_image'].to(self.device)
				item_idxs = data['idx']
				# y_noise: Unsup labels, (None, NUM_MAPS, H, W)
				y_noise = torch.reshape(data['sal_noisy_label'].to(self.device),
										(self.batch_size, self.num_maps, self.h, self.w))
				# pred: (None, 1, H, W)
				pred = pred_model(x)
				# emp_std: Empirical Standard Deviation for each pixel for each image, (None, H, W)
				emp_std = torch.std(y_noise - pred, 1, unbiased=True)
				emp_var = torch.square(emp_std)
				prior_var = torch.square(self.noise_std[item_idxs])
				prior_var = prior_var + self.alpha * (emp_var - prior_var)
				self.noise_std[item_idxs] = torch.sqrt(prior_var)

		self.round = self.round + 1
		self.logger.info(f'Max: {torch.max(self.noise_std)}, Min :{torch.min(self.noise_std)}')
		self.logger.info(f"Noise Standard Deviation: {self.noise_std}")
		self.logger.info("----------------------------------------")


class BaseModule(nn.Module):
	def __init__(self, cfg):
		super(BaseModule, self).__init__()
		self.model = deeplabv3_resnet101(pretrained=cfg.MODEL.PRE_TRAINED, pretrained_backbone=cfg.MODEL.PRE_TRAINED)
		self.sigmoid_layer = nn.Sigmoid()
		self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
		if cfg.MODEL.PRE_TRAINED:
			self.model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

		device = cfg.SYSTEM.DEVICE
		self.model = self.model.to(device)
		self.sigmoid_layer = self.sigmoid_layer.to(device)

	def forward(self, input):
		x = self.model(input)
		x = self.sigmoid_layer(x["out"])
		return x
