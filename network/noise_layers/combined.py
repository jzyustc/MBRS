from . import Identity
import torch.nn as nn
from . import get_random_int


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	def forward(self, image_and_cover):
		id = get_random_int([0, len(self.list) - 1])
		return self.list[id](image_and_cover)
