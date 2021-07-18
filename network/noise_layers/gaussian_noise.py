import numpy as np
import torch
import torch.nn as nn


class GN(nn.Module):

	def __init__(self, var, mean=0):
		super(GN, self).__init__()
		self.var = var
		self.mean = mean

	def gaussian_noise(self, image, mean, var):
		noise = torch.Tensor(np.random.normal(mean, var ** 0.5, image.shape)).to(image.device)
		out = image + noise
		return out

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return self.gaussian_noise(image, self.mean, self.var)
