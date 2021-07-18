import torch.nn as nn
from kornia.filters import GaussianBlur2d


class GF(nn.Module):

	def __init__(self, sigma, kernel=7):
		super(GF, self).__init__()
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return self.gaussian_filter(image)

