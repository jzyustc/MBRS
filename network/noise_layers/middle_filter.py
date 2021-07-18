import torch.nn as nn
from kornia.filters import MedianBlur


class MF(nn.Module):

	def __init__(self, kernel):
		super(MF, self).__init__()
		self.middle_filter = MedianBlur((kernel, kernel))

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return self.middle_filter(image)

